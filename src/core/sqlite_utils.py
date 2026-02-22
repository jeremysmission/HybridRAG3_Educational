"""
src/core/sqlite_utils.py

===========================================================
PURPOSE
===========================================================

This file contains SQLite tuning ("PRAGMA") settings.

Why we keep this in src/core:
- Your code imports from src.core.* modules.
- Keeping this here avoids import/path problems.
- This file is portable to any Windows machine.

===========================================================
WHAT IS A PRAGMA?
===========================================================

A PRAGMA is a special SQLite command that changes how SQLite behaves.
Think of these as "database engine settings".

These settings are safe for a local laptop RAG system.
"""

from __future__ import annotations

import sqlite3


def apply_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    """
    Apply recommended settings for HybridRAG.

    These settings help:
    - indexing speed
    - query performance
    - reduce "database is locked" errors
    - stability on laptops
    """

    # WAL journaling improves concurrency and reduces locking issues
    conn.execute("PRAGMA journal_mode=WAL;")

    # NORMAL = good safety/performance balance for desktops/laptops
    conn.execute("PRAGMA synchronous=NORMAL;")

    # temp tables/operations in RAM (faster; usually small)
    conn.execute("PRAGMA temp_store=MEMORY;")

    # cache_size negative means KB; -200000 ~= 200MB cache
    # If your laptop is very low RAM, reduce this number later.
    conn.execute("PRAGMA cache_size=-200000;")

    # Keep relational integrity on
    conn.execute("PRAGMA foreign_keys=ON;")

    # Wait briefly if DB is busy rather than immediately failing
    conn.execute("PRAGMA busy_timeout=5000;")
