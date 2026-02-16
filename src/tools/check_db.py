#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Deep Database Integrity Check
# ============================================================================
# FILE: src/tools/check_db.py
#
# WHAT THIS DOES (plain English):
#   Runs PRAGMA quick_check on the SQLite database. This reads every
#   page in the database file and verifies that the B-tree structure
#   is internally consistent. On a ~1GB database this takes about
#   2 minutes.
#
# WHEN TO USE:
#   - After a crash or unexpected shutdown during indexing
#   - After copying or moving the database file
#   - If queries return unexpected results
#   - Before a major demo or delivery
#   - Anytime you suspect corruption from concurrent access
#
# HOW TO RUN:
#   rag-check-db                  (after sourcing start_hybridrag.ps1)
#   python src\tools\check_db.py  (direct)
#
# INTERNET ACCESS: NONE -- pure local file operation
# ============================================================================

import os
import sys
import time
import sqlite3


def main():
    # Find the database
    data_dir = os.environ.get("HYBRIDRAG_DATA_DIR", "")
    if data_dir:
        db = os.path.join(data_dir, "hybridrag.sqlite3")
    else:
        db = ""

    if not db or not os.path.exists(db):
        print(f"  [FAIL] Database not found: {db}")
        print(f"         Is HYBRIDRAG_DATA_DIR set?")
        print(f"         Run: . .\\start_hybridrag.ps1")
        sys.exit(1)

    size_mb = os.path.getsize(db) / (1024 * 1024)
    print()
    print(f"  Database: {db}")
    print(f"  Size:     {size_mb:.1f} MB")
    print()
    print(f"  Running PRAGMA quick_check (reads every page)...")
    print(f"  This may take 1-2 minutes on a large database.")
    print()

    t0 = time.time()
    conn = sqlite3.connect(db)
    result = conn.execute("PRAGMA quick_check;").fetchone()[0]
    elapsed = time.time() - t0
    conn.close()

    if result == "ok":
        print(f"  [OK] Database integrity verified in {elapsed:.1f}s")
        print(f"       All B-tree pages are consistent.")
    else:
        print(f"  [FAIL] Integrity check failed: {result}")
        print(f"         Database may be corrupted.")
        print(f"         Consider restoring from backup or re-indexing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
