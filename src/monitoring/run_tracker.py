# ============================================================================
# HybridRAG - Run Tracker (src/monitoring/run_tracker.py)
# ============================================================================
# What this file does (plain English):
# - Creates an auditable "run_id" for a long indexing job
# - Writes run information and per-file events into SQLite
# - Tracks progress: stages, MB processed, text produced, chunks, embeddings
# - Estimates "tokens" (as an estimate) and ETA based on observed throughput
#
# Why this exists:
# - Week-long runs need audit trails + resumability + clear status
# - "Auditable logging" usually means:
#     - every file has a hash
#     - every run has a unique ID
#     - you can prove what happened and when (events + errors)
#     - evidence is reproducible (config + versions recorded)
#
# IMPORTANT:
# - This does NOT sanitize PII (PII is query-only later)
# - This does NOT store file contents in logs; it stores metrics + errors only
# ============================================================================

from __future__ import annotations

import os
import sqlite3
import time
import uuid
import platform
import getpass
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, Tuple


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def estimate_tokens_from_text(text: str) -> int:
    """
    Token estimate (fast + consistent).
    Rule of thumb: ~4 characters per token in English text.
    We label it as an estimate.
    """
    # Quick estimate: English text averages ~4 characters per token.
    # This isn't exact (tokenizers are more complex) but it's good enough
    # for progress reporting and cost estimates.
    if not text:
        return 0
    n_chars = len(text)
    return (n_chars + 3) // 4  # Integer division, rounding up


@dataclass
class RunConfigSnapshot:
    """
    A small config snapshot stored with the run.

    We store only key knobs, not secrets.
    """
    project_root: str
    data_dir: str
    source_dir: str
    profile: str
    embed_batch: str
    retrieval_block_rows: str
    ocr_fallback: str
    poppler_path: str
    tesseract_cmd: str
    hash_mode: str


class RunTracker:
    """
    Tracks a single indexing run.
    Writes to SQLite tables:
      - index_runs
      - doc_events

    This class is intentionally independent of VectorStore so we can use it
    before schema refactors, and it remains stable for audits.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.run_id = str(uuid.uuid4())

        # Totals (we update these as we go)
        self.start_ts = time.time()
        self.files_total = 0
        self.files_done = 0
        self.files_failed = 0
        self.files_skipped = 0

        self.bytes_in_total = 0
        self.bytes_in_done = 0

        self.text_chars_total = 0
        self.text_chars_done = 0

        self.chunks_added = 0
        self.embeddings_added = 0

        # Rolling throughput for ETA
        self._last_eta_calc_ts = self.start_ts
        self._last_bytes_done = 0
        self._bytes_per_sec_ema = None  # exponential moving average

        # Ensure tables exist
        self._ensure_tables()

        # Create run row
        self._insert_run_row()

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the tracking database with safe settings."""
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _ensure_tables(self) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS index_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL, -- running|finished|failed|aborted
                    host TEXT NOT NULL,
                    user TEXT NOT NULL,
                    project_root TEXT,
                    data_dir TEXT,
                    source_dir TEXT,
                    profile TEXT,
                    notes TEXT
                );
                """
            )

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    source_path TEXT,
                    stage TEXT NOT NULL, -- scan|hash|parse|ocr|chunk|embed|store|done|skip|error
                    message TEXT,
                    bytes_in INTEGER,
                    text_chars INTEGER,
                    chunks INTEGER,
                    embeddings INTEGER,
                    elapsed_ms REAL,
                    FOREIGN KEY(run_id) REFERENCES index_runs(run_id)
                );
                """
            )

            # Index for faster status queries
            con.execute("CREATE INDEX IF NOT EXISTS idx_doc_events_run_ts ON doc_events(run_id, ts);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_doc_events_run_stage ON doc_events(run_id, stage);")

            con.commit()
        finally:
            con.close()

    def _insert_run_row(self) -> None:
        con = self._connect()
        try:
            snap = self._snapshot_config()
            con.execute(
                """
                INSERT INTO index_runs (
                    run_id, created_at, started_at, status,
                    host, user, project_root, data_dir, source_dir, profile, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    utc_now_iso(),
                    utc_now_iso(),
                    "running",
                    platform.node(),
                    getpass.getuser(),
                    snap.project_root,
                    snap.data_dir,
                    snap.source_dir,
                    snap.profile,
                    self._format_notes(snap),
                ),
            )
            con.commit()
        finally:
            con.close()

    def _format_notes(self, snap: RunConfigSnapshot) -> str:
        """
        Store a readable snapshot.
        Do NOT store secrets here.
        """
        d = asdict(snap)
        lines = [f"{k}={v}" for k, v in d.items()]
        return "\n".join(lines)

    def _snapshot_config(self) -> RunConfigSnapshot:
        project_root = os.getenv("HYBRIDRAG_PROJECT_ROOT", "")
        data_dir = os.getenv("HYBRIDRAG_DATA_DIR", "")
        source_dir = os.getenv("HYBRIDRAG_INDEX_FOLDER", "")
        profile = os.getenv("HYBRIDRAG_PROFILE", "laptop_safe")

        return RunConfigSnapshot(
            project_root=project_root,
            data_dir=data_dir,
            source_dir=source_dir,
            profile=profile,
            embed_batch=os.getenv("HYBRIDRAG_EMBED_BATCH", ""),
            retrieval_block_rows=os.getenv("HYBRIDRAG_RETRIEVAL_BLOCK_ROWS", ""),
            ocr_fallback=os.getenv("HYBRIDRAG_OCR_FALLBACK", "0"),
            poppler_path=os.getenv("HYBRIDRAG_POPPLER_PATH", ""),
            tesseract_cmd=os.getenv("HYBRIDRAG_TESSERACT_CMD", ""),
            hash_mode=os.getenv("HYBRIDRAG_HASH_MODE", "sha256"),
        )

    # ------------------------------------------------------------------
    # Public API: set totals, log events, finalize
    # ------------------------------------------------------------------

    def set_discovery_totals(self, files_total: int, bytes_total: int) -> None:
        """Called at the start of indexing after scanning the source folder.
        Records how many files and bytes we expect to process (for ETA calculation)."""
        self.files_total = max(0, int(files_total))
        self.bytes_in_total = max(0, int(bytes_total))
        self.event(stage="scan", source_path=None, message=f"discovered files_total={self.files_total}, bytes_total={self.bytes_in_total}")

    def event(
        self,
        stage: str,
        source_path: Optional[str],
        message: Optional[str] = None,
        bytes_in: Optional[int] = None,
        text_chars: Optional[int] = None,
        chunks: Optional[int] = None,
        embeddings: Optional[int] = None,
        elapsed_ms: Optional[float] = None,
    ) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO doc_events (
                    run_id, ts, source_path, stage, message,
                    bytes_in, text_chars, chunks, embeddings, elapsed_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    utc_now_iso(),
                    source_path,
                    stage,
                    message,
                    bytes_in,
                    text_chars,
                    chunks,
                    embeddings,
                    elapsed_ms,
                ),
            )
            con.commit()
        finally:
            con.close()

    def mark_file_done(self, source_path: str, bytes_in: int, text_chars: int,
                       chunks_added: int, embeddings_added: int, elapsed_ms: float) -> None:
        """Called after each file is successfully indexed. Updates all running counters."""
        self.files_done += 1
        self.bytes_in_done += max(0, int(bytes_in))
        self.text_chars_done += max(0, int(text_chars))
        self.chunks_added += max(0, int(chunks_added))
        self.embeddings_added += max(0, int(embeddings_added))

        self.event(
            stage="done",
            source_path=source_path,
            message="file_indexed_ok",
            bytes_in=bytes_in,
            text_chars=text_chars,
            chunks=chunks_added,
            embeddings=embeddings_added,
            elapsed_ms=elapsed_ms,
        )

    def mark_file_skipped(self, source_path: str, reason: str, bytes_in: int = 0) -> None:
        self.files_skipped += 1
        self.bytes_in_done += max(0, int(bytes_in))
        self.event(stage="skip", source_path=source_path, message=reason, bytes_in=bytes_in)

    def mark_file_failed(self, source_path: str, error: str, bytes_in: int = 0) -> None:
        self.files_failed += 1
        self.event(stage="error", source_path=source_path, message=error, bytes_in=bytes_in)

    def eta_seconds(self) -> Optional[float]:
        """
        Estimate remaining seconds based on observed bytes/sec.
        Uses an EMA so OCR-heavy spikes don't totally break the estimate.
        """
        if self.bytes_in_total <= 0:
            return None

        now = time.time()
        dt = max(0.001, now - self._last_eta_calc_ts)
        dbytes = max(0, self.bytes_in_done - self._last_bytes_done)

        inst_rate = dbytes / dt  # bytes/sec this interval

        # EMA smoothing (Exponential Moving Average)
        # EMA gives more weight to recent measurements. Alpha=0.25 means
        # 25% weight to current measurement, 75% to previous average.
        # This prevents OCR-heavy files from wildly skewing the ETA.
        alpha = 0.25
        if self._bytes_per_sec_ema is None:
            self._bytes_per_sec_ema = inst_rate
        else:
            self._bytes_per_sec_ema = alpha * inst_rate + (1 - alpha) * self._bytes_per_sec_ema

        self._last_eta_calc_ts = now
        self._last_bytes_done = self.bytes_in_done

        rate = self._bytes_per_sec_ema
        if rate is None or rate <= 1e-6:
            return None

        remaining = max(0, self.bytes_in_total - self.bytes_in_done)
        return remaining / rate

    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_ts
        eta = self.eta_seconds()

        return {
            "run_id": self.run_id,
            "elapsed_seconds": elapsed,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "bytes_total": self.bytes_in_total,
            "bytes_done": self.bytes_in_done,
            "mb_total": self.bytes_in_total / (1024 * 1024) if self.bytes_in_total else 0.0,
            "mb_done": self.bytes_in_done / (1024 * 1024) if self.bytes_in_done else 0.0,
            "text_chars_done": self.text_chars_done,
            "tokens_est_done": estimate_tokens_from_text("x" * min(self.text_chars_done, 1_000_000)) * (self.text_chars_done // 1_000_000 + 1)
            if self.text_chars_done > 0
            else 0,
            "chunks_added": self.chunks_added,
            "embeddings_added": self.embeddings_added,
            "eta_seconds": eta,
        }

    def finish(self, status: str = "finished") -> None:
        """Mark this indexing run as complete in the database.
        Called at the very end of indexing (or with status='failed' on crash)."""
        con = self._connect()
        try:
            con.execute(
                """
                UPDATE index_runs
                SET finished_at=?, status=?
                WHERE run_id=?
                """,
                (utc_now_iso(), status, self.run_id),
            )
            con.commit()
        finally:
            con.close()
