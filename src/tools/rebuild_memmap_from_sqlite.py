r"""
src/tools/rebuild_memmap_from_sqlite.py

===========================================================
PURPOSE
===========================================================

FORCED rebuild of the Step 4 memmap embedding store from SQLite chunk text.

Use this when:
- SQLite already has embedding_row values (not NULL),
  BUT the memmap store is empty or missing.

This script:
1) Deletes/creates memmap store fresh (you delete files before running)
2) Reads ALL chunks from SQLite (chunk_id + text)
3) Embeds in laptop-safe batches
4) Appends vectors to memmap store
5) Overwrites embedding_row for every chunk_id in SQLite

===========================================================
WHY THIS IS SAFE
===========================================================

- It does NOT re-parse PDFs.
- It does NOT re-chunk files.
- It only re-embeds the stored chunk text already in SQLite.
- After this, retrieval works immediately.

===========================================================
IMPORTANT
===========================================================

Run this AFTER deleting:
    \embeddings.f16.dat
    \embeddings_meta.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import sqlite3
import numpy as np

# ------------------------------------------------------------
# Make "src" imports work when running directly
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.vector_store import VectorStore
from src.core.embedder import Embedder
from src.core.health_checks import check_memmap_ready


def fetch_all_chunks(conn: sqlite3.Connection, limit: int, offset: int) -> List[Tuple[str, str]]:
    """
    Fetch chunk_id + text for ALL chunks.

    We page using LIMIT/OFFSET so memory usage stays low.
    """
    rows = conn.execute(
        """
        SELECT chunk_id, text
        FROM chunks
        WHERE text IS NOT NULL
          AND LENGTH(text) > 0
        ORDER BY chunk_pk
        LIMIT ? OFFSET ?;
        """,
        (limit, offset),
    ).fetchall()

    return [(str(r[0]), str(r[1])) for r in rows]


def update_embedding_rows(conn: sqlite3.Connection, updates: List[Tuple[int, str]]) -> None:
    """
    Overwrite embedding_row for each chunk_id.
    updates: list of (embedding_row, chunk_id)
    """
    conn.executemany(
        """
        UPDATE chunks
        SET embedding_row = ?
        WHERE chunk_id = ?;
        """,
        updates,
    )


def main():
    data_dir = os.getenv("HYBRIDRAG_DATA_DIR", r"")
    db_path = os.path.join(data_dir, "hybridrag.sqlite3")

    print("Using DB:", db_path)
    print("Using data dir:", data_dir)

    # Create VectorStore (this will re-create empty memmap meta if missing)
    vs = VectorStore(db_path=db_path)
    vs.connect()

    conn = vs.conn
    if conn is None:
        print("ERROR: Could not open SQLite connection.")
        return

    # Count total chunks we can embed
    total = conn.execute(
        """
        SELECT COUNT(*)
        FROM chunks
        WHERE text IS NOT NULL AND LENGTH(text) > 0;
        """
    ).fetchone()[0]
    total = int(total)

    if total == 0:
        print("No chunks found in SQLite. You must run indexing first.")
        return

    print(f"Total chunks in SQLite: {total}")

    # Embedder (SentenceTransformer)
    emb = Embedder()

    # Laptop-safe knobs (override via env vars)
    page_size = int(os.getenv("HYBRIDRAG_REBUILD_PAGE_SIZE", "256"))
    embed_batch = int(os.getenv("HYBRIDRAG_EMBED_BATCH", "32"))

    print("Page size:", page_size)
    print("Embed batch:", embed_batch)

    # IMPORTANT:
    # We process chunks in the same order as they appear in SQLite (by chunk_pk).
    # This gives each chunk a predictable, sequential embedding_row number.
    # We want a clean sequential embedding_row mapping.
    # We'll assign rows in the exact order we fetch from SQLite.
    migrated = 0
    offset = 0

    # OPTIONAL: wipe embedding_row first (clarity, not strictly required)
    # Reset all embedding_row values to NULL before rebuilding.
    # This ensures a clean mapping -- every chunk gets a fresh row number.
    print("Clearing existing embedding_row values in SQLite...")
    conn.execute("UPDATE chunks SET embedding_row = NULL;")
    conn.commit()

    while True:
        rows = fetch_all_chunks(conn, limit=page_size, offset=offset)
        if not rows:
            break

        chunk_ids = [cid for (cid, _txt) in rows]
        texts = [txt for (_cid, txt) in rows]

        # Embed in small sub-batches to reduce RAM spikes
        page_emb_parts = []
        for i in range(0, len(texts), embed_batch):
            batch_texts = texts[i : i + embed_batch]
            e = emb.embed_batch(batch_texts)  # float32 normalized
            page_emb_parts.append(e)

        emb_matrix = np.vstack(page_emb_parts).astype(np.float32, copy=False)

        # Append to memmap -> get assigned row range
        start_row, end_row = vs.mem_store.append_batch(emb_matrix)

        # Update SQLite pointers
        updates: List[Tuple[int, str]] = []
        for j, cid in enumerate(chunk_ids):
            updates.append((start_row + j, cid))

        update_embedding_rows(conn, updates)
        conn.commit()

        migrated += len(rows)
        offset += len(rows)

        print(f"Rebuilt {migrated}/{total} chunks... (memmap rows now end at {end_row})")

    ok, msg = check_memmap_ready(vs)
    print("Rebuild complete.")
    print("Memmap check:", msg)


if __name__ == "__main__":
    main()
