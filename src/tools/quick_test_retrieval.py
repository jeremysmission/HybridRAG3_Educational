"""
src/tools/quick_test_retrieval.py

===========================================================
PURPOSE
===========================================================

A simple test you can run from PowerShell to confirm:
- VectorStore connects
- memmap embedding store exists and has vectors
- Retriever returns results
- We print top sources + relevance

IMPORTANT:
This project does NOT export a Config class from src.core.config.
So we do NOT import Config here.

Instead:
- We try to load config via get_config('.')
- If that fails, we fallback to a tiny config object
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ------------------------------------------------------------
# FIX FOR "ModuleNotFoundError: No module named 'src'"
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.vector_store import VectorStore
from src.core.embedder import Embedder
from src.core.retriever import Retriever
from src.core.health_checks import check_memmap_ready


# ============================================================================
# Fallback Config -- used when the main config can't be loaded
# ============================================================================
# The Retriever needs a config object to know things like "how many results
# to return" (top_k) and "how many rows to scan at once" (block_rows).
# If we can't load the real config, we use these safe defaults.
# ============================================================================
class _FallbackConfig:
    """
    Minimal config fallback for Retriever.

    Retriever mainly needs:
    - retrieval.top_k
    - retrieval.block_rows

    If your Retriever ignores config, this is still harmless.
    """
    class retrieval:
        top_k = 8
        block_rows = 25000


def _load_config(project_root: str):
    """
    Load config using your project's preferred API.
    """
    try:
        from src.core.config import get_config
        return get_config(project_root)
    except Exception:
        return _FallbackConfig()


def main():
    data_dir = os.getenv("HYBRIDRAG_DATA_DIR", "")
    db_path = os.path.join(data_dir, "hybridrag.sqlite3")

    print("DB path:", db_path)

    cfg = _load_config(".")

    vs = VectorStore(db_path=db_path)
    vs.connect()

    # Check if the embedding file exists and has vectors in it
    # If the memmap is empty, search can't work because there are no vectors to compare against
    ok, msg = check_memmap_ready(vs)
    print("Memmap check:", msg)
    if not ok:
        print("Run indexing first to create embeddings.")
        return

    emb = Embedder()
    r = Retriever(vs, emb, cfg)

    q = input("Enter a test query: ").strip()
    if not q:
        print("No query entered.")
        return

    # Run the full hybrid search pipeline:
    # 1. Embed the query text into a vector
    # 2. Search for similar vectors in the memmap store
    # 3. Run BM25 keyword search in FTS5
    # 4. Fuse results with Reciprocal Rank Fusion
    # 5. Optionally rerank with cross-encoder
    results = r.search(q)

    if not results:
        print("No results found.")
        return

    print("\nTop results:")
    for i, res in enumerate(results[:8], start=1):
        preview = res.text[:200].replace("\n", " ")
        print(f"{i}. score={res.score:.4f}  source={res.source_path}")
        print(f"   text preview: {preview}")
        print()

    sources = r.get_sources(results)
    print("Source summary:")
    for s in sources[:5]:
        print(f"- {s['path']}  chunks={s['chunks']}  avg_relevance={s['avg_relevance']:.4f}")


if __name__ == "__main__":
    main()
