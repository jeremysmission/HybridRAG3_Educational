# ============================================================================
# HybridRAG v3 -- Diagnostic: Performance Benchmarks
# ============================================================================
# FILE: src/diagnostic/perf_benchmarks.py
#
# WHAT THIS FILE DOES:
#   Measures how fast each pipeline stage runs:
#     Config load, SQLite query, chunker throughput, embedder throughput,
#     vector search, FTS5 keyword search, hybrid search.
#
#   Each function returns a PerfMetric with min/max/avg/stddev stats
#   and memory delta. All benchmarks are non-destructive (read-only).
#
# WHY MEASURE PERFORMANCE:
#   - Detect regressions after code changes
#   - Identify bottlenecks (is search slow because of vector or FTS5?)
#   - Track trends over time with --json-file
#   - Know your baseline before a week-long indexing run
# ============================================================================

from __future__ import annotations

import os
import json
import sqlite3
from typing import Optional

from . import PerfMetric, benchmark, PROJ_ROOT


def _get_db_path() -> str:
    from src.core.config import load_config
    return getattr(load_config(str(PROJ_ROOT)).paths, "database", "")


def perf_config_load(iters: int = 3) -> PerfMetric:
    """How long does loading the full config take?"""
    from src.core.config import load_config
    return benchmark(lambda: load_config(str(PROJ_ROOT)),
                     iters, "config_load", "Config", "ms")


def perf_sqlite_query(iters: int = 3) -> PerfMetric:
    """How long does SQLite connect + simple query take?"""
    db = _get_db_path()
    if not os.path.exists(db):
        return PerfMetric("sqlite_query", "Database", 0, "ms",
                          details={"skip": "no database"})

    def run():
        c = sqlite3.connect(db)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("SELECT COUNT(*) FROM chunks").fetchone()
        c.close()

    return benchmark(run, iters, "sqlite_connect_query", "Database", "ms")


def perf_chunker(iters: int = 3) -> PerfMetric:
    """How many chunks per second can the chunker produce?"""
    from src.core.chunker import Chunker, ChunkerConfig
    from src.core.config import load_config

    cfg = load_config(str(PROJ_ROOT))
    chunker = Chunker(ChunkerConfig(
        chunk_size=cfg.chunking.chunk_size, overlap=cfg.chunking.overlap))
    # ~100KB synthetic text (realistic paragraph density)
    sample = "\n\n".join([f"Paragraph {i}. " * 20 for i in range(500)])

    return benchmark(
        lambda: chunker.chunk_text(sample), iters,
        "chunker_throughput", "Chunker", "chunks/sec",
        value_extractor=lambda el, r: len(r) / el if el > 0 else 0)


def perf_embedder(iters: int = 3) -> Optional[PerfMetric]:
    """How fast is the embedding model? (requires model to be loaded)"""
    try:
        from src.core.embedder import Embedder
        from src.core.config import load_config

        cfg = load_config(str(PROJ_ROOT))
        embedder = Embedder(cfg.embedding.model_name)
        bs = cfg.embedding.batch_size
        texts = [f"Test sentence {i} for benchmark." for i in range(bs)]

        m = benchmark(
            lambda: embedder.embed_batch(texts), iters,
            f"embedder_batch_{bs}", "Embedder", "embeddings/sec",
            value_extractor=lambda el, _: bs / el if el > 0 else 0)
        m.details["batch_size"] = bs
        m.details["model"] = cfg.embedding.model_name
        return m
    except Exception as e:
        return PerfMetric("embedder", "Embedder", 0, "embeddings/sec",
                          details={"error": str(e)})


def perf_vector_search(iters: int = 3) -> PerfMetric:
    """How fast is raw vector similarity search?"""
    try:
        from src.core.config import load_config
        from src.core.vector_store import VectorStore
        import numpy as np

        cfg = load_config(str(PROJ_ROOT))
        db = cfg.paths.database
        if not os.path.exists(db):
            return PerfMetric("vector_search", "Search", 0, "ms/query",
                              details={"skip": "no database"})

        vs = VectorStore(db_path=db, embedding_dim=cfg.embedding.dimension)
        vs.connect()
        if vs.get_stats().get("chunk_count", 0) == 0:
            return PerfMetric("vector_search", "Search", 0, "ms/query",
                              details={"skip": "no indexed data"})

        qv = np.random.randn(cfg.embedding.dimension).astype(np.float32)
        qv /= np.linalg.norm(qv)

        m = benchmark(lambda: vs.search(qv, top_k=5), iters,
                      "vector_search_top5", "Search", "ms/query")
        m.details["chunks"] = vs.get_stats().get("chunk_count", 0)
        return m
    except Exception as e:
        return PerfMetric("vector_search", "Search", 0, "ms/query",
                          details={"error": str(e)})


def perf_fts5_search(iters: int = 3) -> PerfMetric:
    """How fast is FTS5 keyword search?"""
    try:
        db = _get_db_path()
        if not os.path.exists(db):
            return PerfMetric("fts5_search", "Search", 0, "ms/query",
                              details={"skip": "no database"})

        def run():
            c = sqlite3.connect(db)
            c.execute(
                "SELECT rowid, rank FROM chunks_fts "
                "WHERE chunks_fts MATCH 'frequency OR range' "
                "ORDER BY rank LIMIT 5"
            ).fetchall()
            c.close()

        return benchmark(run, iters, "fts5_keyword_search", "Search", "ms/query")
    except Exception as e:
        return PerfMetric("fts5_search", "Search", 0, "ms/query",
                          details={"error": str(e)})


def perf_hybrid_search(iters: int = 3) -> Optional[PerfMetric]:
    """How fast is full hybrid search (vector + BM25 + RRF)?"""
    try:
        from src.core.config import load_config
        from src.core.vector_store import VectorStore
        from src.core.embedder import Embedder
        from src.core.retriever import Retriever

        cfg = load_config(str(PROJ_ROOT))
        db = cfg.paths.database
        if not os.path.exists(db):
            return PerfMetric("hybrid_search", "Search", 0, "ms/query",
                              details={"skip": "no database"})

        vs = VectorStore(db_path=db, embedding_dim=cfg.embedding.dimension)
        vs.connect()
        if vs.get_stats().get("chunk_count", 0) == 0:
            return PerfMetric("hybrid_search", "Search", 0, "ms/query",
                              details={"skip": "no data"})

        embedder = Embedder(cfg.embedding.model_name)
        retriever = Retriever(vs, embedder, cfg)
        q = "What is the operating frequency range?"

        m = benchmark(lambda: retriever.search(q), iters,
                      "hybrid_search_e2e", "Search", "ms/query")
        m.details["query"] = q
        return m
    except Exception as e:
        return PerfMetric("hybrid_search", "Search", 0, "ms/query",
                          details={"error": str(e)})
