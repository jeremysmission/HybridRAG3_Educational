# ============================================================================
# HybridRAG -- Transfer Manifest Database (src/tools/transfer_manifest.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   A SQLite database that tracks every single file encountered during a
#   bulk transfer from network drives to HybridRAG's source folder.
#
#   Think of it like a shipping company's manifest -- when a cargo ship
#   arrives at port, the manifest says what's on board, what was supposed
#   to be on board, and what's missing. This does the same for files.
#
#   It answers three critical questions after any transfer:
#     1. What did I get? (successfully transferred and verified files)
#     2. What did I miss? (skipped, locked, permission denied, failed)
#     3. What changed since last time? (delta sync: new, modified, deleted,
#        renamed files detected by comparing manifests between runs)
#
#   The manifest is the "ground truth" -- every file in the source must
#   be accounted for. Nothing disappears silently.
#
# WHY THIS MATTERS:
#   Without a manifest, you have no idea if a transfer actually got
#   everything. A 10,000-file transfer might silently drop 50 files
#   due to permission errors, locked files, or network hiccups. You
#   would never know unless you manually counted. The manifest makes
#   these invisible failures visible.
#
# HOW IT WORKS (non-programmer summary):
#   1. When a transfer starts, a new "run" is created (like opening
#      a new page in a logbook)
#   2. As the engine discovers files on the network drive, every single
#      file is recorded in the source_manifest table -- even files we
#      will not transfer (wrong type, too big, locked, etc.)
#   3. For files we DO transfer, the transfer_log table records the
#      result (success, failed, hash mismatch) with timing data
#   4. For files we SKIP, the skipped_files table records why
#   5. At the end, the verification report checks:
#      (transferred + skipped) = total discovered. If not, something
#      fell through the cracks (a "gap").
#
# TABLES (what each database table stores):
#   transfer_runs   -- One row per transfer run (start/end/status)
#   source_manifest -- Every file discovered at source (the ground truth)
#   transfer_log    -- Per-file transfer result (success/fail/skip + timing)
#   skipped_files   -- Files not transferred, with full path and reason
#
# DELTA SYNC (incremental transfers):
#   On the second run, the manifest compares current files against the
#   previous run's manifest to detect:
#     - New files (exist now but not before)
#     - Deleted files (existed before but not now)
#     - Modified files (same path, different hash)
#     - Renamed files (same hash, different path)
#   This allows transferring ONLY what changed, saving hours on re-runs.
#
# THREAD SAFETY:
#   Multiple transfer workers write to this database simultaneously.
#   A threading lock prevents them from stepping on each other, and
#   SQLite's WAL mode allows readers and writers to work in parallel.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TransferManifest:
    """
    SQLite database tracking every file in a bulk transfer operation.

    Thread-safe. All timestamps stored in UTC ISO-8601.

    NON-PROGRAMMER NOTE:
      This class is a "database wrapper" -- it hides the ugly SQL
      commands behind simple Python method calls. Instead of writing
      raw SQL, the rest of the code calls methods like
      record_source_file() or record_skip().
    """

    def __init__(self, db_path: str) -> None:
        # ------------------------------------------------------------------
        # Open (or create) the SQLite database file at db_path.
        #
        # PRAGMA journal_mode=WAL:
        #   WAL = "Write-Ahead Log". Normal SQLite locks the entire
        #   database when writing. WAL mode lets readers continue
        #   reading while a write is happening. Critical for 8-thread
        #   parallel transfers.
        #
        # PRAGMA synchronous=NORMAL:
        #   Tells SQLite "you don't need to flush to disk after every
        #   single write." Faster, and safe with WAL mode. The tiny
        #   risk: if the power cuts mid-write, the last ~50 rows might
        #   be lost. Acceptable because the transfer would restart anyway.
        #
        # check_same_thread=False:
        #   SQLite normally complains if Thread A opens a connection
        #   and Thread B tries to use it. This flag says "I know what
        #   I'm doing, I have my own lock." (We do -- see self._lock.)
        # ------------------------------------------------------------------
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()

        # Batch commit counter: instead of writing to disk after every
        # single row, we accumulate 50 rows and write them all at once.
        # This is ~10x faster for bulk inserts.
        self._pending_writes = 0

        self._create_tables()

    def _create_tables(self) -> None:
        """
        Create the four database tables if they don't already exist.

        NON-PROGRAMMER NOTE:
          "CREATE TABLE IF NOT EXISTS" means "make this table, but
          don't complain if it already exists from a previous run."
          This makes the database self-initializing -- no separate
          setup step needed.
        """
        with self._lock:
            c = self.conn.cursor()
            c.executescript("""
                -- =====================================================
                -- transfer_runs: one row per transfer session
                -- Think of this as the "header page" of the logbook.
                -- =====================================================
                CREATE TABLE IF NOT EXISTS transfer_runs (
                    run_id         TEXT PRIMARY KEY,   -- Timestamp-based ID
                    started_at     TEXT NOT NULL,       -- When did it start
                    finished_at    TEXT,                -- When did it finish
                    source_paths   TEXT,                -- JSON list of source dirs
                    dest_path      TEXT,                -- Where files go
                    account        TEXT DEFAULT '',     -- Who ran it
                    status         TEXT DEFAULT 'running',  -- running / complete
                    config_json    TEXT DEFAULT '{}'    -- Settings snapshot
                );

                -- =====================================================
                -- source_manifest: the GROUND TRUTH.
                -- Every file discovered at the source, even ones we skip.
                -- If a file exists on the network drive, it has a row here.
                -- =====================================================
                CREATE TABLE IF NOT EXISTS source_manifest (
                    source_path    TEXT NOT NULL,       -- Full path on network drive
                    run_id         TEXT NOT NULL,       -- Which transfer run
                    file_size      INTEGER DEFAULT 0,   -- Bytes
                    file_mtime     REAL DEFAULT 0,      -- Last modified time
                    file_ctime     REAL DEFAULT 0,      -- Creation time
                    extension      TEXT DEFAULT '',      -- .pdf, .docx, etc.
                    is_hidden      INTEGER DEFAULT 0,   -- Windows hidden flag
                    is_system      INTEGER DEFAULT 0,   -- Windows system flag
                    is_readonly    INTEGER DEFAULT 0,   -- Read-only flag
                    is_symlink     INTEGER DEFAULT 0,   -- Symlink/junction
                    is_accessible  INTEGER DEFAULT 1,   -- Can we read it?
                    path_length    INTEGER DEFAULT 0,   -- Chars in path
                    encoding_issue INTEGER DEFAULT 0,   -- Bad filename chars
                    owner          TEXT DEFAULT '',      -- File owner
                    content_hash   TEXT DEFAULT '',      -- SHA-256 of contents
                    PRIMARY KEY (source_path, run_id)   -- One entry per file per run
                );

                -- =====================================================
                -- transfer_log: what happened to each file we TRIED to copy.
                -- Records timing, hash verification, retry count, errors.
                -- =====================================================
                CREATE TABLE IF NOT EXISTS transfer_log (
                    source_path       TEXT NOT NULL,    -- Original location
                    dest_path         TEXT DEFAULT '',  -- Where it ended up
                    run_id            TEXT NOT NULL,
                    file_size_source  INTEGER DEFAULT 0, -- Size at source
                    file_size_dest    INTEGER DEFAULT 0, -- Size at destination
                    hash_source       TEXT DEFAULT '',  -- SHA-256 before copy
                    hash_dest         TEXT DEFAULT '',  -- SHA-256 after copy
                    hash_match        INTEGER DEFAULT 0, -- 1 = match, 0 = mismatch
                    transfer_start    TEXT DEFAULT '',  -- When copy began
                    transfer_end      TEXT DEFAULT '',  -- When copy finished
                    duration_sec      REAL DEFAULT 0,   -- How long it took
                    speed_mbps        REAL DEFAULT 0,   -- MB/sec for this file
                    result            TEXT DEFAULT 'pending', -- success/failed/locked/etc
                    retry_count       INTEGER DEFAULT 0,  -- How many retries
                    error_message     TEXT DEFAULT '',  -- What went wrong (if anything)
                    PRIMARY KEY (source_path, run_id)
                );

                -- =====================================================
                -- skipped_files: files we CHOSE not to transfer, with why.
                -- This is how you answer "what did I miss?"
                -- =====================================================
                CREATE TABLE IF NOT EXISTS skipped_files (
                    source_path    TEXT NOT NULL,       -- Full path of skipped file
                    run_id         TEXT NOT NULL,
                    file_size      INTEGER DEFAULT 0,
                    extension      TEXT DEFAULT '',
                    reason         TEXT NOT NULL,       -- Category (locked, hidden, etc.)
                    detail         TEXT DEFAULT '',     -- Human-readable explanation
                    logged_at      TEXT DEFAULT ''      -- When we logged it
                );

                -- Indexes speed up common queries (like "find by hash"
                -- or "show all skipped files for this run").
                CREATE INDEX IF NOT EXISTS idx_manifest_hash
                    ON source_manifest(content_hash);
                CREATE INDEX IF NOT EXISTS idx_manifest_run
                    ON source_manifest(run_id);
                CREATE INDEX IF NOT EXISTS idx_transfer_result
                    ON transfer_log(result);
                CREATE INDEX IF NOT EXISTS idx_transfer_run
                    ON transfer_log(run_id);
                CREATE INDEX IF NOT EXISTS idx_skipped_reason
                    ON skipped_files(reason);
            """)
            self.conn.commit()

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def start_run(
        self, run_id: str, source_paths: List[str], dest_path: str,
        account: str = "", config_json: str = "{}",
    ) -> None:
        """
        Record the start of a new transfer run.

        NON-PROGRAMMER NOTE:
          Like writing the date and "STARTED" at the top of a new
          logbook page. The run_id is a timestamp string so runs sort
          chronologically.
        """
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO transfer_runs "
                "(run_id, started_at, source_paths, dest_path, account, "
                "status, config_json) VALUES (?, ?, ?, ?, ?, 'running', ?)",
                (run_id, _utc_now(), json.dumps(source_paths),
                 dest_path, account, config_json),
            )
            self.conn.commit()

    def finish_run(self, run_id: str) -> None:
        """Mark a transfer run as complete with a finish timestamp."""
        with self._lock:
            self.conn.execute(
                "UPDATE transfer_runs SET finished_at=?, status='complete' "
                "WHERE run_id=?",
                (_utc_now(), run_id),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Source manifest (ground truth)
    # ------------------------------------------------------------------

    def record_source_file(
        self, run_id: str, source_path: str, file_size: int = 0,
        file_mtime: float = 0, file_ctime: float = 0, extension: str = "",
        is_hidden: bool = False, is_system: bool = False,
        is_readonly: bool = False, is_symlink: bool = False,
        is_accessible: bool = True, path_length: int = 0,
        encoding_issue: bool = False, owner: str = "",
        content_hash: str = "",
    ) -> None:
        """
        Record a single file in the source manifest (ground truth).

        NON-PROGRAMMER NOTE:
          This is called for EVERY file found on the network drive,
          regardless of whether we plan to transfer it. Even .exe files
          we will never copy get recorded here. This is what makes the
          "zero-gap" verification possible -- we know exactly what was
          out there.
        """
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO source_manifest "
                "(source_path, run_id, file_size, file_mtime, file_ctime, "
                "extension, is_hidden, is_system, is_readonly, is_symlink, "
                "is_accessible, path_length, encoding_issue, owner, "
                "content_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (source_path, run_id, file_size, file_mtime, file_ctime,
                 extension, int(is_hidden), int(is_system), int(is_readonly),
                 int(is_symlink), int(is_accessible), path_length,
                 int(encoding_issue), owner, content_hash),
            )
            self._batch_commit()

    # ------------------------------------------------------------------
    # Transfer log (per-file result with timing)
    # ------------------------------------------------------------------

    def record_transfer(
        self, run_id: str, source_path: str, dest_path: str = "",
        file_size_source: int = 0, file_size_dest: int = 0,
        hash_source: str = "", hash_dest: str = "",
        transfer_start: str = "", transfer_end: str = "",
        duration_sec: float = 0, speed_mbps: float = 0,
        result: str = "pending", retry_count: int = 0,
        error_message: str = "",
    ) -> None:
        """
        Record the outcome of transferring one file.

        NON-PROGRAMMER NOTE:
          After copying a file, this logs whether it succeeded or
          failed, how long it took, and whether the hash matched.
          Think of it like a shipping receipt with a signature
          confirming the package arrived intact.
        """
        hash_match = 1 if (hash_source and hash_source == hash_dest) else 0
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO transfer_log "
                "(source_path, dest_path, run_id, file_size_source, "
                "file_size_dest, hash_source, hash_dest, hash_match, "
                "transfer_start, transfer_end, duration_sec, speed_mbps, "
                "result, retry_count, error_message) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (source_path, dest_path, run_id, file_size_source,
                 file_size_dest, hash_source, hash_dest, hash_match,
                 transfer_start, transfer_end, duration_sec, speed_mbps,
                 result, retry_count, error_message),
            )
            self._batch_commit()

    # ------------------------------------------------------------------
    # Skipped files (missed file log)
    # ------------------------------------------------------------------

    def record_skip(
        self, run_id: str, source_path: str, file_size: int = 0,
        extension: str = "", reason: str = "", detail: str = "",
    ) -> None:
        """
        Record a file we intentionally did NOT transfer.

        NON-PROGRAMMER NOTE:
          Every skipped file gets a reason ("locked", "hidden",
          "unsupported_extension", "path_too_long", etc.) and a
          human-readable detail string. After a transfer, you can
          query this table to see exactly what was missed and why.
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO skipped_files "
                "(source_path, run_id, file_size, extension, reason, "
                "detail, logged_at) VALUES (?,?,?,?,?,?,?)",
                (source_path, run_id, file_size, extension, reason,
                 detail, _utc_now()),
            )
            self._batch_commit()

    # ------------------------------------------------------------------
    # Delta sync queries
    # ------------------------------------------------------------------

    def get_previous_manifest(self, run_id: str) -> Dict[str, str]:
        """
        Return {source_path: content_hash} from the most recent
        completed run before this one.

        NON-PROGRAMMER NOTE:
          This is the heart of "delta sync." By comparing the current
          file list against the previous run's list, we can tell which
          files are new, which were deleted, and which were modified.
          Files that haven't changed can be skipped entirely, saving
          potentially hours of re-copying.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT run_id FROM transfer_runs "
                "WHERE status='complete' AND run_id < ? "
                "ORDER BY run_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if not row:
                return {}
            prev_run = row[0]
            rows = self.conn.execute(
                "SELECT source_path, content_hash FROM source_manifest "
                "WHERE run_id=?",
                (prev_run,),
            ).fetchall()
            return {r[0]: r[1] for r in rows}

    def is_already_transferred(self, source_path: str) -> bool:
        """
        Check if this exact file was already successfully transferred
        in a previous run. Used for resume/restart scenarios.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM transfer_log "
                "WHERE source_path=? AND result='success' LIMIT 1",
                (source_path,),
            ).fetchone()
            return row is not None

    def find_by_hash(self, content_hash: str) -> Optional[str]:
        """
        Find an already-transferred file with this content hash.

        NON-PROGRAMMER NOTE:
          If two files have the same SHA-256 hash, they are identical.
          This is used for deduplication: if file B has the same hash
          as file A (which was already copied), we can skip copying B
          entirely. The SHA-256 collision probability is astronomically
          small (1 in 2^256) -- effectively zero.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT dest_path FROM transfer_log "
                "WHERE hash_source=? AND result='success' LIMIT 1",
                (content_hash,),
            ).fetchone()
            return row[0] if row else None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_verification_report(self, run_id: str) -> str:
        """
        Zero-gap verification report -- every file accounted for.

        NON-PROGRAMMER NOTE:
          This is the "audit report" you run after a transfer. It tells
          you: "I found X files on the source. I successfully copied Y.
          I skipped Z (here's why). The total is X = Y + Z. Zero gap."

          If Y + Z does NOT equal X, something fell through the cracks
          and the report will flag it as [WARN] UNACCOUNTED.
        """
        with self._lock:
            manifest_count = self.conn.execute(
                "SELECT COUNT(*) FROM source_manifest WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]

            results = self.conn.execute(
                "SELECT result, COUNT(*) FROM transfer_log "
                "WHERE run_id=? GROUP BY result ORDER BY COUNT(*) DESC",
                (run_id,),
            ).fetchall()

            skip_results = self.conn.execute(
                "SELECT reason, COUNT(*), SUM(file_size) "
                "FROM skipped_files WHERE run_id=? "
                "GROUP BY reason ORDER BY COUNT(*) DESC",
                (run_id,),
            ).fetchall()

            # Pull sample files for each skip reason (up to 5 examples)
            skip_samples: Dict[str, List[Tuple]] = {}
            for reason, _, _ in skip_results:
                samples = self.conn.execute(
                    "SELECT source_path, file_size, detail "
                    "FROM skipped_files WHERE run_id=? AND reason=? LIMIT 5",
                    (run_id, reason),
                ).fetchall()
                skip_samples[reason] = samples

            # Pull sample failed files (up to 20 examples)
            failed = self.conn.execute(
                "SELECT source_path, error_message, file_size_source "
                "FROM transfer_log WHERE run_id=? AND result='failed' "
                "LIMIT 20",
                (run_id,),
            ).fetchall()
            failed_total = self.conn.execute(
                "SELECT COUNT(*) FROM transfer_log "
                "WHERE run_id=? AND result='failed'",
                (run_id,),
            ).fetchone()[0]

        # Build the report text
        lines = [
            "", "=" * 70,
            "  TRANSFER VERIFICATION REPORT",
            "=" * 70, "",
            f"  Files in source manifest:  {manifest_count:,}",
        ]
        transfer_total = sum(r[1] for r in results)
        skip_total = sum(r[1] for r in skip_results)
        accounted = transfer_total + skip_total
        lines.append(f"  Files in transfer log:     {transfer_total:,}")
        lines.append(f"  Files in skip log:         {skip_total:,}")
        lines.append(f"  Total accounted:           {accounted:,}")
        gap = manifest_count - accounted
        if gap == 0:
            lines.append("  GAP:                       0 (ZERO-GAP VERIFIED)")
        else:
            lines.append(f"  GAP:                       {gap:,} [WARN] UNACCOUNTED")
        lines.append("")

        for result, count in results:
            lines.append(f"  [{result.upper()}] {count:,}")
        lines.append("")

        if skip_results:
            lines.append("  SKIPPED FILES:")
            for reason, count, size_sum in skip_results:
                sz = (size_sum or 0) / (1024 * 1024)
                lines.append(f"    [{reason}] {count:,} ({sz:.1f} MB)")
                for path, fsize, detail in skip_samples.get(reason, []):
                    d = f" -- {detail}" if detail else ""
                    lines.append(f"      {path}{d}")
                if count > 5:
                    lines.append(f"      ... and {count - 5} more")
            lines.append("")

        if failed:
            lines.append(f"  FAILED FILES ({failed_total:,} total):")
            for path, err, sz in failed:
                lines.append(f"    {path} -- {err[:80]}")
            if failed_total > 20:
                lines.append(f"    ... and {failed_total - 20} more")
            lines.append("")

        lines.extend(["=" * 70, ""])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _batch_commit(self) -> None:
        """
        Accumulate writes and commit every 50 rows.

        NON-PROGRAMMER NOTE:
          Imagine writing checks -- you could drive to the bank after
          each check, or you could save up 50 and make one trip. Same
          idea. SQLite disk commits are expensive; batching them makes
          the transfer engine ~10x faster during discovery.
        """
        self._pending_writes += 1
        if self._pending_writes >= 50:
            self.conn.commit()
            self._pending_writes = 0

    def flush(self) -> None:
        """Force all pending writes to disk immediately."""
        with self._lock:
            self.conn.commit()
            self._pending_writes = 0

    def close(self) -> None:
        """Commit any remaining writes and close the database."""
        with self._lock:
            self.conn.commit()
            self.conn.close()


# ============================================================================
# Module-level helpers
# ============================================================================

def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string (e.g. '2026-02-20T14:30:00+00:00')."""
    return datetime.now(timezone.utc).isoformat()
