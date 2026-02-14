# ============================================================================
# HybridRAG — Health Checks (src/core/health_checks.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Quick validation that the system is ready to run.
#   Used by:
#     - start_hybridrag.ps1 (rag-status command)
#     - run_index_once.py (pre-flight check before a long indexing run)
#     - Future GUI (green/red status indicators)
#
# WHY SEPARATE FROM VECTOR STORE:
#   The VectorStore class shouldn't know about "is the system healthy?"
#   That's a higher-level concern. Keeping health checks in their own
#   file means any tool or script can import and use them without
#   pulling in the entire storage layer.
#
# DESIGN PRINCIPLE:
#   Every check returns (ok: bool, message: str).
#   This makes it trivial to display results in any format —
#   terminal, GUI, JSON API, whatever.
# ============================================================================

from __future__ import annotations

from typing import Tuple


def check_memmap_ready(vector_store) -> Tuple[bool, str]:
    """
    Check if the memmap embedding store has data and is ready for search.

    Parameters
    ----------
    vector_store : VectorStore
        A VectorStore instance (should already be .connect()'d).

    Returns
    -------
    (ok, message) where:
        ok : bool — True if memmap is ready for search
        message : str — human-readable status description
    """
    # Make sure the vector store actually has a mem_store attribute
    try:
        ms = vector_store.mem_store
    except AttributeError:
        return False, "VectorStore has no mem_store attribute."

    try:
        # Check that both files exist on disk
        ok, msg = ms.paths_ok()
        if not ok:
            return False, f"Memmap files missing: {msg}"

        # Check that there's actually data in the store
        if ms.count <= 0:
            return False, "Embedding store is empty (count=0). Run indexing first."

        return True, f"Memmap ready ({ms.count:,} embeddings, {ms.dim} dimensions)."

    except Exception as e:
        return False, f"Memmap check failed: {type(e).__name__}: {e}"


def check_sqlite_ready(vector_store) -> Tuple[bool, str]:
    """
    Check if the SQLite database has chunks and is queryable.

    Parameters
    ----------
    vector_store : VectorStore
        A VectorStore instance (should already be .connect()'d).

    Returns
    -------
    (ok, message)
    """
    if vector_store.conn is None:
        return False, "SQLite connection not open. Call vector_store.connect() first."

    try:
        row = vector_store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        count = row[0] if row else 0

        if count == 0:
            return False, "Chunks table is empty. Run indexing first."

        row2 = vector_store.conn.execute(
            "SELECT COUNT(DISTINCT source_path) FROM chunks"
        ).fetchone()
        sources = row2[0] if row2 else 0

        return True, f"SQLite ready ({count:,} chunks from {sources:,} files)."

    except Exception as e:
        return False, f"SQLite check failed: {type(e).__name__}: {e}"


def check_all(vector_store) -> Tuple[bool, str]:
    """
    Run all health checks and return a combined result.

    Returns (True, summary) if everything is healthy.
    Returns (False, summary) if anything is wrong.
    """
    results = []

    mm_ok, mm_msg = check_memmap_ready(vector_store)
    results.append(("Memmap", mm_ok, mm_msg))

    sq_ok, sq_msg = check_sqlite_ready(vector_store)
    results.append(("SQLite", sq_ok, sq_msg))

    all_ok = all(r[1] for r in results)

    lines = []
    for name, ok, msg in results:
        status = "OK" if ok else "FAIL"
        lines.append(f"  [{status}] {name}: {msg}")

    summary = "\n".join(lines)
    return all_ok, summary