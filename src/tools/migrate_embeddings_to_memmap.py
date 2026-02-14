"""
src/tools/migrate_embeddings_to_memmap.py

===========================================================
PURPOSE
===========================================================

One-time migration script to populate the Step 4 memmap embedding store.

Why this exists:
- Before Step 4, embeddings may have been stored in SQLite, or not at all.
- Step 4 expects embeddings in a memmap file:
    \embeddings.f16.dat
    \embeddings_meta.json

This script:
1) Reads chunk text from SQLite
2) Embeds chunks in small batches (RAM-safe)
3) Writes embeddings to memmap store (append-only)
4) Writes embedding_row pointers back into SQLite

This avoids re-parsing PDFs and re-chunking.

===========================================================
SAFETY / IMPORTANT NOTES
===========================================================

- This script assumes your SQLite chunks table exists and has:
    chunk_id, text, embedding_row

- If embedding_row is already set for some rows, we skip them.
- If memmap already contains vectors, we APPEND more, and continue.

If you want a "fresh rebuild", delete these first:
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
# Make sure "src" imports work when running this file directly
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.vector_store import VectorStore
from src.core.embedder import Embedder
from src.core.health_checks import check_memmap_ready


def fetch_chunks_missing_embedding_row(conn: sqlite3.Connection, limit: int, offset: int) -> List[Tuple[str, str]]:
    """
    Fetch chunk_id + text for chunks that have no embedding_row.

    We page through the database using LIMIT/OFFSET so we never load everything at once.
    """
    rows = conn.execute(
        """
        SELECT chunk_id, text
        FROM chunks
        WHERE embedding_row IS NULL
          AND text IS NOT NULL
          AND LENGTH(text) > 0
        ORDER BY chunk_pk
        LIMIT ? OFFSET ?;
        """,
        (limit, offset),
    ).fetchall()

    return [(str(r[0]), str(r[1])) for r in rows]


def update_embedding_rows(conn: sqlite3.Connection, updates: List[Tuple[int, str]]) -> None:
    """
    Apply embedding_row updates back into SQLite.

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
    # Data location is controlled by env var, defaulting to 
    data_dir = os.getenv("HYBRIDRAG_DATA_DIR", r"")
    db_path = os.path.join(data_dir, "hybridrag.sqlite3")

    print("Using DB:", db_path)
    print("Using data dir:", data_dir)

    # Open VectorStore normally (it sets up schema and memmap store)
    vs = VectorStore(db_path=db_path)
    vs.connect()

    # Embedder (SentenceTransformer)
    emb = Embedder()

    # Batch sizes:
    # - Keep these small for laptop safety
    page_size = int(os.getenv("HYBRIDRAG_MIGRATE_PAGE_SIZE", "256"))
    embed_batch = int(os.getenv("HYBRIDRAG_EMBED_BATCH", "32"))

    print("Page size:", page_size)
    print("Embed batch:", embed_batch)

    conn = vs.conn
    if conn is None:
        print("ERROR: VectorStore did not open SQLite connection.")
        return

    # Count how many chunks in the database don't have an embedding vector yet.
    # "embedding_row IS NULL" means the chunk's text hasn't been converted
    # to a vector yet — that's what this migration script fixes.
    total_missing = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding_row IS NULL AND text IS NOT NULL AND LENGTH(text) > 0;"
    ).fetchone()[0]
    total_missing = int(total_missing)

    if total_missing == 0:
        ok, msg = check_memmap_ready(vs)
        print("Nothing to migrate. Memmap check:", msg)
        return

    print(f"Chunks missing embedding_row: {total_missing}")

    migrated = 0
    offset = 0

    # We loop until no more rows are returned
    while True:
        rows = fetch_chunks_missing_embedding_row(conn, limit=page_size, offset=offset)
        if not rows:
            break

        # Extract texts to embed
        chunk_ids = [cid for (cid, _txt) in rows]
        texts = [txt for (_cid, txt) in rows]

        # Embed in smaller batches for RAM safety
        # Processing all text at once could use several GB of RAM.
        # By doing it in batches of 32, we keep memory under control.
        # We will build a list of embeddings for this page
        page_embeddings = []

        for i in range(0, len(texts), embed_batch):
            batch_texts = texts[i : i + embed_batch]

            # Embedder returns float32 normalized vectors
            e = emb.embed_batch(batch_texts)  # shape (B, D)
            page_embeddings.append(e)

        # Combine embeddings for this page into one matrix
        emb_matrix = np.vstack(page_embeddings).astype(np.float32, copy=False)

        # Append the new embedding vectors to the memmap file.
        # The memmap store returns the row range where they were written.
        # Example: if we had 1000 vectors and added 256 more,
        # start_row=1000, end_row=1256
        start_row, end_row = vs.mem_store.append_batch(emb_matrix)

        # Prepare updates: each chunk_id gets its assigned embedding_row
        updates: List[Tuple[int, str]] = []
        for j, cid in enumerate(chunk_ids):
            updates.append((start_row + j, cid))

        # Write embedding_row pointers back into SQLite
        update_embedding_rows(conn, updates)
        conn.commit()

        migrated += len(rows)
        offset += len(rows)

        print(f"Migrated {migrated}/{total_missing} chunks... (memmap rows now end at {end_row})")

    ok, msg = check_memmap_ready(vs)
    print("Migration complete.")
    print("Memmap check:", msg)


if __name__ == "__main__":
    main()
