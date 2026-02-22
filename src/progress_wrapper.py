# ============================================================================
# HybridRAG -- Progress Wrapper (src/progress_wrapper.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Tracks indexing progress (files done, bytes processed, chunks created)
#   and writes it to a SQLite table so other tools can read it.
#
# WHY THIS EXISTS:
#   When indexing thousands of files over several hours, you need to know:
#   - How many files are done?
#   - How fast is it going?
#   - When will it finish? (ETA)
#
#   This wrapper sits "outside" the indexer -- it doesn't modify the indexer
#   code at all. It just watches what happens and records progress.
#
# SAFETY DESIGN:
#   - Never throws exceptions (all writes are try/except wrapped)
#   - Never blocks or slows down the indexer
#   - Only writes to SQLite every N seconds (configurable, default 2s)
#   - If the database is busy, it silently skips the update
#
# HOW IT WORKS:
#   1. Indexer calls wrapper.file_done(size) after each file
#   2. Wrapper updates its in-memory counters (instant, no I/O)
#   3. Periodically (every 2 seconds), it flushes to SQLite
#   4. Other tools can read the run_progress table to show status
#
# INTERNET ACCESS: None
# ============================================================================

import os
import time
import sqlite3


class ProgressWrapper:
    """
    External progress tracker that wraps indexing runs without modifying indexer code.

    Safe for long runs:
    - never throws (all database writes are wrapped in try/except)
    - never blocks indexing (skips writes if DB is busy)
    - writes progress periodically (not after every single file)
    """

    def __init__(self, db_path, run_id, total_files, total_bytes):
        self.db_path = db_path
        self.run_id = run_id

        # Expected totals (set at the start, used for percentage/ETA calculation)
        self.total_files = total_files or 0
        self.total_bytes = total_bytes or 0

        # Running counters (updated after each file/chunk/embedding)
        self.files_done = 0
        self.bytes_done = 0
        self.chunks_done = 0
        self.embeddings_done = 0

        # Timing for ETA calculation
        self.start_time = time.time()
        self.last_flush = 0  # Timestamp of last database write

        # How often to write progress to SQLite (in seconds)
        # Configurable via environment variable
        self.flush_every_seconds = float(
            os.getenv("HYBRIDRAG_PROGRESS_EVERY_S", "2.0")
        )

        # Create the progress tracking table if it doesn't exist
        self._init_row()

    def _init_row(self):
        """
        Create the run_progress table and insert a row for this run.

        The table is simple: one row per indexing run with counters.
        INSERT OR IGNORE means if the row already exists, skip it.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS run_progress (
                        run_id TEXT PRIMARY KEY,
                        files_done INTEGER,
                        bytes_done INTEGER,
                        chunks_done INTEGER,
                        embeddings_done INTEGER,
                        eta_seconds INTEGER,
                        last_update_time TEXT
                    )
                """)
                conn.execute("""
                    INSERT OR IGNORE INTO run_progress
                    (run_id, files_done, bytes_done, chunks_done, embeddings_done)
                    VALUES (?,0,0,0,0)
                """, (self.run_id,))
        except Exception:
            # If database is busy or locked, silently skip
            pass

    def file_done(self, file_size):
        """Called by the indexer after each file is processed."""
        self.files_done += 1
        self.bytes_done += int(file_size or 0)

    def chunks_added(self, n):
        """Called when new chunks are created from a file."""
        self.chunks_done += int(n or 0)

    def embeddings_added(self, n):
        """Called when new embedding vectors are created."""
        self.embeddings_done += int(n or 0)

    def flush(self):
        """
        Write current progress to SQLite.

        Calculates ETA based on average files-per-second so far.
        This is called periodically, not after every file (to avoid
        hammering the database during fast indexing).
        """
        try:
            # Calculate files-per-second rate
            elapsed = max(1.0, time.time() - self.start_time)
            rate = self.files_done / elapsed

            # Estimate remaining time
            eta = None
            if self.total_files and rate > 0:
                eta = int((self.total_files - self.files_done) / rate)

            # Write to SQLite
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE run_progress
                    SET files_done=?,
                        bytes_done=?,
                        chunks_done=?,
                        embeddings_done=?,
                        eta_seconds=?,
                        last_update_time=datetime('now')
                    WHERE run_id=?
                """, (
                    self.files_done,
                    self.bytes_done,
                    self.chunks_done,
                    self.embeddings_done,
                    eta,
                    self.run_id,
                ))
        except Exception:
            # If write fails (DB busy, locked, etc.), silently skip
            pass

    def maybe_flush(self):
        """
        Flush to database only if enough time has passed.

        This is the method the indexer should call frequently (e.g., after
        every file). It checks the clock and only actually writes to the
        database if flush_every_seconds has elapsed since the last write.
        """
        now = time.time()
        if now - self.last_flush >= self.flush_every_seconds:
            self.flush()
            self.last_flush = now
