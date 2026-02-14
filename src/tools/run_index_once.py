# ============================================================================
# HybridRAG — Main Indexing Runner (src/tools/run_index_once.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   This is the "main engine" that indexes your documents. When you run
#   "rag-index" from PowerShell, THIS is the file that executes.
#
# THE INDEXING PIPELINE:
#   1. Load configuration (paths, model settings, chunk sizes)
#   2. Validate config and create directories if needed
#   3. Generate a unique run_id for this indexing session (audit trail)
#   4. Connect to the SQLite database
#   5. Load the embedding model (~80MB, takes ~10 seconds first time)
#   6. Walk through every file in the source folder
#   7. For each file: parse -> chunk -> embed -> store in database
#   8. After all files: rebuild the FTS5 keyword index
#   9. Clean up "stale" chunks from files that were deleted
#   10. Record the run as complete in the tracking table
#   11. Close all resources (BUG-003 fix)
#
# SAFETY FEATURES:
#   - Anti-sleep: Prevents Windows from sleeping during long runs
#   - Crash-safe: If interrupted (Ctrl+C), progress is saved
#   - Run tracking: Every run gets a UUID and is recorded in the database
#   - Error isolation: One bad file doesn't crash the entire run
#   - FTS rebuild: Keyword search index is auto-rebuilt after every run
#   - Stale cleanup: Chunks from deleted files are automatically removed
#   - Resource cleanup: All connections and models are closed in finally block
#
# INTERNET ACCESS:
#   - None during indexing (embedding model runs locally)
#   - The embedding model downloads once on first run (~80MB)
#
# BUGS FIXED (2026-02-08):
#   BUG-003: Added indexer.close() and vs.close() in the finally block
#            to release SQLite connections, memmap handles, and the
#            embedding model from RAM after indexing completes.
#
# USAGE:
#   . .\start_hybridrag.ps1    # Activate environment
#   rag-index                   # This runs run_index_once.py
# ============================================================================

from __future__ import annotations

import os
import sys
import time
import sqlite3
import ctypes
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from uuid import uuid4

PROJ_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


def prevent_sleep():
    """
    Tell Windows NOT to go to sleep while indexing is running.

    Uses the SetThreadExecutionState API with flags:
    - ES_CONTINUOUS: Keep the setting active until cleared
    - ES_SYSTEM_REQUIRED: Don't sleep the computer
    - ES_DISPLAY_REQUIRED: Don't turn off the display
    """
    if sys.platform == 'win32':
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )


def allow_sleep():
    """Re-enable normal Windows sleep behavior after indexing finishes."""
    if sys.platform == 'win32':
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger("hybridrag.indexer")

from src.core.config import Config, load_config, validate_config, ensure_directories
from src.core.vector_store import VectorStore
from src.core.embedder import Embedder
from src.core.chunker import Chunker, ChunkerConfig
from src.core.indexer import Indexer, IndexingProgressCallback


class RunTracker:
    """Tracks indexing runs in SQLite for audit trail and monitoring."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS index_runs (
                run_id          TEXT PRIMARY KEY,
                start_time      TEXT NOT NULL,
                end_time        TEXT,
                status          TEXT NOT NULL DEFAULT 'running',
                source_folder   TEXT,
                files_scanned   INTEGER DEFAULT 0,
                files_indexed   INTEGER DEFAULT 0,
                files_skipped   INTEGER DEFAULT 0,
                chunks_added    INTEGER DEFAULT 0,
                errors_count    INTEGER DEFAULT 0,
                elapsed_seconds REAL DEFAULT 0,
                error_message   TEXT
            );
        """)
        self.conn.commit()

    def start_run(self, run_id, source_folder):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            INSERT OR REPLACE INTO index_runs
                (run_id, start_time, status, source_folder)
            VALUES (?, ?, 'running', ?)
        """, (run_id, now, source_folder))
        self.conn.commit()
        log.info("run_started", run_id=run_id, source_folder=source_folder)

    def complete_run(self, run_id, result, error_count=0):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            UPDATE index_runs SET
                end_time = ?,
                status = 'completed',
                files_scanned = ?,
                files_indexed = ?,
                files_skipped = ?,
                chunks_added = ?,
                errors_count = ?,
                elapsed_seconds = ?
            WHERE run_id = ?
        """, (
            now,
            result.get("total_files_scanned", 0),
            result.get("total_files_indexed", 0),
            result.get("total_files_skipped", 0),
            result.get("total_chunks_added", 0),
            error_count,
            result.get("elapsed_seconds", 0),
            run_id,
        ))
        self.conn.commit()
        log.info("run_completed", run_id=run_id, **result)

    def fail_run(self, run_id, error):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("""
            UPDATE index_runs SET
                end_time = ?,
                status = 'failed',
                error_message = ?
            WHERE run_id = ?
        """, (now, str(error)[:2000], run_id))
        self.conn.commit()
        log.error("run_failed", run_id=run_id, error=str(error)[:500])

    def close(self):
        if self.conn:
            self.conn.close()


class LoggingProgress(IndexingProgressCallback):
    """Progress callback that logs to structlog (terminal output)."""

    def __init__(self):
        self.start_time = time.time()
        self.errors = []
        self._file_start_time = 0.0

    def on_file_start(self, file_path, file_num, total_files):
        self._file_start_time = time.time()
        name = Path(file_path).name
        elapsed = time.time() - self.start_time
        if file_num > 1:
            avg_per_file = elapsed / (file_num - 1)
            remaining = avg_per_file * (total_files - file_num + 1)
            eta_min = remaining / 60
        else:
            eta_min = 0.0
        log.info("file_start", file=name, progress=f"{file_num}/{total_files}",
                 elapsed_min=round(elapsed / 60, 1), eta_min=round(eta_min, 1))

    def on_file_complete(self, file_path, chunks_created):
        name = Path(file_path).name
        file_elapsed = time.time() - self._file_start_time
        log.info("file_complete", file=name, chunks=chunks_created,
                 file_seconds=round(file_elapsed, 1))

    def on_file_skipped(self, file_path, reason):
        name = Path(file_path).name
        log.debug("file_skipped", file=name, reason=reason)

    def on_indexing_complete(self, total_chunks, elapsed_seconds):
        cps = total_chunks / elapsed_seconds if elapsed_seconds > 0 else 0
        log.info("indexing_complete", total_chunks=total_chunks,
                 elapsed_min=round(elapsed_seconds / 60, 1),
                 chunks_per_sec=round(cps, 1), errors=len(self.errors))

    def on_error(self, file_path, error):
        name = Path(file_path).name
        self.errors.append(f"{name}: {error}")
        log.error("file_error", file=name, error=error[:300])


def main():
    prevent_sleep()
    log.info("hybridrag_indexing_start", version="v3")

    config = load_config(str(PROJ_ROOT))
    errors = validate_config(config)
    if errors:
        for e in errors:
            log.error("config_error", message=e)
        log.error("config_invalid", hint="Fix config/default_config.yaml or env vars")
        sys.exit(1)

    ensure_directories(config)

    source_folder = config.paths.source_folder
    if not source_folder:
        log.error("config_error", message="No source folder configured",
                  hint="Set HYBRIDRAG_INDEX_FOLDER env var")
        sys.exit(1)
    if not os.path.isdir(source_folder):
        log.error("config_error", message=f"Source folder not found: {source_folder}")
        sys.exit(1)

    log.info("config_loaded", source_folder=source_folder,
             database=config.paths.database,
             embed_model=config.embedding.model_name,
             embed_batch=config.embedding.batch_size,
             chunk_size=config.chunking.chunk_size,
             chunk_overlap=config.chunking.overlap, mode=config.mode)

    run_id = str(uuid4())
    log.info("run_id_generated", run_id=run_id)

    tracker = RunTracker(config.paths.database)
    tracker.start_run(run_id, source_folder)

    vs = VectorStore(db_path=config.paths.database,
                     embedding_dim=config.embedding.dimension)
    vs.connect()
    stats = vs.get_stats()
    log.info("existing_data", chunks=stats.get("chunk_count", 0),
             files=stats.get("source_count", 0),
             embeddings=stats.get("embedding_count", 0))

    log.info("loading_embedding_model", model=config.embedding.model_name)
    embedder = Embedder(config.embedding.model_name)
    log.info("embedding_model_ready", dimension=embedder.dimension)

    chunker = Chunker(ChunkerConfig(
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap,
        max_heading_len=config.chunking.max_heading_len))

    indexer = Indexer(config, vs, embedder, chunker)
    progress = LoggingProgress()

    try:
        result = indexer.index_folder(source_folder, progress_callback=progress)
        tracker.complete_run(run_id, result, error_count=len(progress.errors))

        if progress.errors:
            log.warning("files_with_errors", count=len(progress.errors),
                        first_10=progress.errors[:10])

        final_stats = vs.get_stats()
        log.info("final_store_stats", chunks=final_stats.get("chunk_count", 0),
                 files=final_stats.get("source_count", 0),
                 embeddings=final_stats.get("embedding_count", 0))

        # --- Auto-rebuild FTS5 keyword index after indexing ---
        try:
            vs.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            vs.conn.commit()
            log.info("fts_rebuilt")
        except Exception as fts_err:
            log.warning("fts_rebuild_failed", error=str(fts_err))

        # --- Stale file cleanup: remove chunks for deleted source files ---
        try:
            stale_rows = vs.conn.execute(
                "SELECT DISTINCT source_path FROM chunks"
            ).fetchall()
            stale_count = 0
            for (sp,) in stale_rows:
                if not Path(sp).exists():
                    vs.conn.execute(
                        "DELETE FROM chunks WHERE source_path = ?", (sp,)
                    )
                    stale_count += 1
                    log.info("stale_file_removed", path=sp)
            if stale_count > 0:
                vs.conn.commit()
                vs.conn.execute(
                    "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"
                )
                vs.conn.commit()
                log.info("stale_cleanup_done", files_removed=stale_count)
        except Exception as cleanup_err:
            log.warning("stale_cleanup_failed", error=str(cleanup_err))

    except KeyboardInterrupt:
        log.warning("indexing_interrupted",
                    message="Ctrl+C received. Progress saved. Restart to resume.")
        tracker.fail_run(run_id, "Interrupted by user (KeyboardInterrupt)")
        sys.exit(0)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log.error("indexing_fatal_error", error=error_msg)
        tracker.fail_run(run_id, error_msg)
        raise
    finally:
        # =============================================================
        # BUG-003 FIX: Close all resources to prevent memory/handle leaks
        # =============================================================
        # Previously, this finally block only closed the tracker and
        # re-enabled sleep. The VectorStore (SQLite connection + memmap
        # file handle) and Embedder (~100MB model in RAM) were never
        # closed, leaking resources over long runs.
        #
        # ORDER MATTERS:
        #   1. Close tracker first (its own SQLite connection)
        #   2. Close indexer (which closes embedder + vector_store)
        #   3. Re-enable Windows sleep
        #
        # If indexer.close() also closes vs, calling vs.close() again
        # is safe — the close() methods are all idempotent (no-op if
        # already closed).
        # =============================================================
        tracker.close()
        try:
            indexer.close()
        except Exception:
            pass  # Best-effort cleanup — don't crash during shutdown
        try:
            vs.close()
        except Exception:
            pass
        try:
            embedder.close()
        except Exception:
            pass
        allow_sleep()

    log.info("hybridrag_indexing_end", run_id=run_id)


if __name__ == "__main__":
    main()
