# ============================================================================
# HybridRAG - Index Status Tool (src/tools/index_status.py)
# ============================================================================
# What this tool does:
# - Prints the most recent indexing run_id
# - Shows totals: done/failed/skipped
# - Shows latest stage events (tail)
#
# Usage:
#   .\.venv\Scripts\python.exe .\src\tools\index_status.py
# ============================================================================

from __future__ import annotations

import os
import sqlite3
from datetime import datetime


def get_db_path() -> str:
    data_dir = os.getenv("HYBRIDRAG_DATA_DIR", "")
    return os.path.join(data_dir, "hybridrag.sqlite3")


def main() -> None:
    db = get_db_path()
    if not os.path.exists(db):
        print(f"DB not found: {db}")
        return

    con = sqlite3.connect(db)
    try:
        run = con.execute(
            "SELECT run_id, started_at, finished_at, status, host, user, profile FROM index_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()

        if not run:
            print("No index_runs found yet.")
            return

        run_id, started_at, finished_at, status, host, user, profile = run
        print(f"DB: {db}")
        print(f"Latest run_id: {run_id}")
        print(f"Status: {status}")
        print(f"Host/User: {host} / {user}")
        print(f"Profile: {profile}")
        print(f"Started: {started_at}")
        print(f"Finished: {finished_at or '(still running)'}")
        print("")

        # Totals by stage
        # Each file goes through stages: scan → hash → parse → chunk → embed → done
        # This query counts how many events occurred at each stage
        totals = con.execute(
            """
            SELECT stage, COUNT(*)
            FROM doc_events
            WHERE run_id=?
            GROUP BY stage
            ORDER BY COUNT(*) DESC
            """,
            (run_id,),
        ).fetchall()

        print("Event totals (by stage):")
        for stage, cnt in totals:
            print(f"  {stage:>8} : {cnt}")
        print("")

        # Show the 15 most recent events (like "tail -f" on a log file)
        # Reversed so newest appears at the bottom (natural reading order)
        tail = con.execute(
            """
            SELECT ts, stage, source_path, message
            FROM doc_events
            WHERE run_id=?
            ORDER BY id DESC
            LIMIT 15
            """,
            (run_id,),
        ).fetchall()

        print("Latest events:")
        for ts, stage, source_path, message in reversed(tail):
            src = source_path or ""
            msg = message or ""
            print(f"{ts} | {stage:>6} | {msg} | {src}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
