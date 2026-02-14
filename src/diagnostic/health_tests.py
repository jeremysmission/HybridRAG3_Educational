# ============================================================================
# HybridRAG v3 -- Diagnostic: Pipeline Health Tests
# ============================================================================
# FILE: src/diagnostic/health_tests.py
#
# WHAT THIS FILE DOES:
#   Contains every pipeline health check, organized by subsystem:
#     Config -> Database -> Indexer -> Parsers -> Chunker -> Embedder ->
#     Storage -> Security
#
#   Each function is self-contained, takes no complex arguments, and
#   returns a TestResult. If it crashes, the safe wrapper in __init__
#   catches it and reports an ERROR without stopping other tests.
#
# HOW TO ADD A NEW TEST:
#   1. Write a function that returns TestResult(name, category, status, message)
#   2. Register it in the run list inside hybridrag_diagnostic.py main()
#   That's it -- the reporting and JSON export handle everything else.
# ============================================================================

from __future__ import annotations

import os
import sys
import re
import json
import time
import inspect
import sqlite3
from pathlib import Path
from typing import Dict, Any

from . import TestResult, PROJ_ROOT


def _get_db_path() -> str:
    """Load config and return database path."""
    from src.core.config import load_config
    return getattr(load_config(str(PROJ_ROOT)).paths, "database", "")


# ============================================================================
# CONFIG TESTS
# ============================================================================

def test_config_load() -> TestResult:
    """Can we load and validate the full configuration?"""
    try:
        from src.core.config import load_config, validate_config
        config = load_config(str(PROJ_ROOT))
        errors = validate_config(config)
        if errors:
            return TestResult("config_load", "Config", "FAIL",
                f"Validation failed: {'; '.join(errors)}", {"errors": errors},
                fix_hint="Check config/default_config.yaml and env vars.")
        return TestResult("config_load", "Config", "PASS",
            f"Config OK (mode={config.mode})",
            {"mode": config.mode,
             "database": getattr(config.paths, "database", "?"),
             "source_folder": getattr(config.paths, "source_folder", "?"),
             "embed_model": getattr(config.embedding, "model_name", "?"),
             "chunk_size": getattr(config.chunking, "chunk_size", "?")})
    except ImportError as e:
        return TestResult("config_load", "Config", "ERROR",
            f"Cannot import config: {e}",
            fix_hint="Run from HybridRAG project root.")


def test_config_paths() -> TestResult:
    """Do configured paths actually exist on disk?"""
    try:
        from src.core.config import load_config
        config = load_config(str(PROJ_ROOT))
        issues, d = [], {}
        db = getattr(config.paths, "database", "")
        if db:
            d["database"] = db
            d["db_exists"] = os.path.exists(db)
            dd = os.path.dirname(db)
            if dd and not os.path.isdir(dd):
                issues.append(f"DB dir missing: {dd}")
        else:
            issues.append("Database path empty")
        sf = getattr(config.paths, "source_folder", "")
        if sf:
            d["source_folder"] = sf
            d["src_exists"] = os.path.isdir(sf)
            if os.path.isdir(sf):
                d["file_count"] = sum(1 for _ in Path(sf).rglob("*") if _.is_file())
                if d["file_count"] == 0:
                    issues.append("Source folder empty")
            else:
                issues.append(f"Source folder missing: {sf}")
        else:
            issues.append("Source folder not set (HYBRIDRAG_INDEX_FOLDER)")
        if issues:
            return TestResult("config_paths", "Config",
                "WARN" if d.get("db_exists") else "FAIL",
                f"Path issues: {'; '.join(issues)}", d,
                fix_hint="Check start_hybridrag.ps1 env vars.")
        return TestResult("config_paths", "Config", "PASS",
            f"Paths OK ({d.get('file_count', 0)} source files)", d)
    except Exception as e:
        return TestResult("config_paths", "Config", "ERROR", f"{e}")


# ============================================================================
# DATABASE / SCHEMA TESTS
# ============================================================================

def test_sqlite_connection() -> TestResult:
    """Can we open SQLite and is it healthy?"""
    try:
        db = _get_db_path()
        if not db or not os.path.exists(db):
            return TestResult("sqlite_conn", "Database", "SKIP",
                f"Database not found: {db}", fix_hint="Run rag-index first.")
        conn = sqlite3.connect(db)
        jm = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        integ = conn.execute("PRAGMA quick_check;").fetchone()[0]
        conn.close()
        d = {"path": db, "journal": jm, "integrity": integ,
             "size_mb": round(os.path.getsize(db) / (1024*1024), 2)}
        if integ != "ok":
            return TestResult("sqlite_conn", "Database", "FAIL",
                f"Integrity failed: {integ}", d, fix_hint="DB may be corrupted.")
        return TestResult("sqlite_conn", "Database", "PASS",
            f"SQLite OK ({d['size_mb']}MB, {jm} mode)", d)
    except Exception as e:
        return TestResult("sqlite_conn", "Database", "ERROR", f"{e}")


def test_schema_chunks_table() -> TestResult:
    """Does chunks table have expected columns? Detects BUG-001 (missing file_hash)."""
    try:
        db = _get_db_path()
        if not db or not os.path.exists(db):
            return TestResult("schema_chunks", "Database", "SKIP", "No DB")
        conn = sqlite3.connect(db)
        cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(chunks);").fetchall()}
        idxs = [r[1] for r in conn.execute("PRAGMA index_list(chunks);").fetchall()]
        conn.close()
        expected = ["chunk_pk", "chunk_id", "source_path", "chunk_index",
                    "text", "text_length", "created_at", "embedding_row"]
        missing = [c for c in expected if c not in cols]
        d = {"columns": list(cols.keys()), "has_file_hash": "file_hash" in cols,
             "indexes": idxs, "missing": missing}
        if "file_hash" not in cols:
            return TestResult("schema_chunks", "Database", "FAIL",
                "BUG-001: MISSING file_hash column -- _file_changed() silently fails", d,
                fix_hint="ALTER TABLE chunks ADD COLUMN file_hash TEXT;")
        if missing:
            return TestResult("schema_chunks", "Database", "FAIL",
                f"Missing columns: {missing}", d)
        return TestResult("schema_chunks", "Database", "PASS",
            f"Schema OK ({len(cols)} cols, {len(idxs)} indexes)", d)
    except Exception as e:
        return TestResult("schema_chunks", "Database", "ERROR", f"{e}")


def test_schema_fts5() -> TestResult:
    """Is FTS5 full-text search index present and in sync?"""
    try:
        db = _get_db_path()
        if not db or not os.path.exists(db):
            return TestResult("schema_fts5", "Database", "SKIP", "No DB")
        conn = sqlite3.connect(db)
        tables = [t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "chunks_fts" not in tables:
            conn.close()
            return TestResult("schema_fts5", "Database", "FAIL",
                "FTS5 table missing -- keyword search broken",
                fix_hint="Rebuild: INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        cc = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fc = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        fts_ok = True
        try:
            conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'test' LIMIT 1")
        except Exception:
            fts_ok = False
        conn.close()
        d = {"chunks": cc, "fts_rows": fc, "fts_query_works": fts_ok}
        if cc > 0 and fc == 0:
            return TestResult("schema_fts5", "Database", "FAIL",
                f"FTS5 EMPTY but {cc} chunks exist", d)
        if cc > 0 and abs(cc - fc) > 10:
            return TestResult("schema_fts5", "Database", "WARN",
                f"FTS5 out of sync: {fc} vs {cc} chunks", d)
        return TestResult("schema_fts5", "Database", "PASS",
            f"FTS5 OK ({fc} rows, query: {fts_ok})", d)
    except Exception as e:
        return TestResult("schema_fts5", "Database", "ERROR", f"{e}")


def test_data_integrity() -> TestResult:
    """Are chunks, embeddings, and source paths consistent?"""
    try:
        db = _get_db_path()
        if not db or not os.path.exists(db):
            return TestResult("data_integrity", "Database", "SKIP", "No DB")
        conn = sqlite3.connect(db)
        cc = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sc = conn.execute("SELECT COUNT(DISTINCT source_path) FROM chunks").fetchone()[0]
        max_er = conn.execute("SELECT MAX(embedding_row) FROM chunks").fetchone()[0]
        null_er = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding_row IS NULL").fetchone()[0]
        sources = conn.execute(
            "SELECT source_path, COUNT(*) FROM chunks GROUP BY source_path ORDER BY 2 DESC"
        ).fetchall()
        conn.close()
        data_dir = os.path.dirname(db)
        meta_p = os.path.join(data_dir, "embeddings_meta.json")
        mc = 0
        if os.path.exists(meta_p):
            with open(meta_p) as f:
                mc = json.load(f).get("count", 0)
        d = {"chunks": cc, "sources": sc, "memmap": mc, "max_emb_row": max_er,
             "null_emb_rows": null_er,
             "source_list": [{"file": Path(s[0]).name, "n": s[1]} for s in sources]}
        issues = []
        if max_er is not None and mc > 0 and max_er >= mc:
            issues.append(f"Max embedding_row ({max_er}) >= memmap count ({mc})")
        if mc > 0 and cc > 0 and mc / cc > 2.0:
            issues.append(f"Memmap {mc} vs {cc} chunks ({mc/cc:.1f}x orphaned)")
        if null_er > 0:
            issues.append(f"{null_er} chunks NULL embedding_row")
        if issues:
            return TestResult("data_integrity", "Database", "WARN",
                "; ".join(issues), d)
        return TestResult("data_integrity", "Database", "PASS",
            f"OK ({cc} chunks, {mc} embeddings, {sc} sources)", d)
    except Exception as e:
        return TestResult("data_integrity", "Database", "ERROR", f"{e}")


# ============================================================================
# INDEXER CODE INSPECTION (detects bugs without running indexer)
# ============================================================================

def test_indexer_change_detection() -> TestResult:
    """BUG-002: Is _file_changed() actually wired into index_folder()?"""
    try:
        from src.core.indexer import Indexer
        has_fc = hasattr(Indexer, "_file_changed")
        src = inspect.getsource(Indexer.index_folder)
        calls_fc = "_file_changed" in src
        calls_ai = "_file_already_indexed" in src
        d = {"has_method": has_fc, "called_in_index_folder": calls_fc,
             "uses_already_indexed": calls_ai}
        if has_fc and not calls_fc:
            return TestResult("change_detection", "Indexer", "FAIL",
                "BUG-002: _file_changed() exists but NEVER CALLED", d,
                fix_hint="Wire _file_changed() into skip logic.")
        if calls_ai and not calls_fc:
            return TestResult("change_detection", "Indexer", "WARN",
                "Skip only checks existence -- edited files won't re-index", d,
                fix_hint="Add: if indexed AND NOT changed: skip; else re-index.")
        return TestResult("change_detection", "Indexer", "PASS",
            "Change detection wired in", d)
    except Exception as e:
        return TestResult("change_detection", "Indexer", "ERROR", f"{e}")


def test_resource_cleanup() -> TestResult:
    """BUG-003: Do key components have close() methods?"""
    try:
        issues, d = [], {}
        try:
            from src.core.vector_store import VectorStore
            d["vs_close"] = hasattr(VectorStore, "close") or hasattr(VectorStore, "__del__")
            if not d["vs_close"]:
                issues.append("VectorStore no close()")
        except ImportError:
            d["vs_import"] = False
        try:
            from src.core.embedder import Embedder
            d["emb_close"] = hasattr(Embedder, "close") or hasattr(Embedder, "__del__")
            if not d["emb_close"]:
                issues.append("Embedder no close()")
        except ImportError:
            d["emb_import"] = False
        if issues:
            return TestResult("resource_cleanup", "Indexer", "WARN",
                f"BUG-003: {'; '.join(issues)}", d,
                fix_hint="Add close() methods. Call in finally block.")
        return TestResult("resource_cleanup", "Indexer", "PASS",
            "Cleanup methods present", d)
    except Exception as e:
        return TestResult("resource_cleanup", "Indexer", "ERROR", f"{e}")

