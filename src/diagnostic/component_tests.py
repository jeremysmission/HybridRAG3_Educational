# ============================================================================
# HybridRAG v3 -- Diagnostic: Component & Security Health Tests
# ============================================================================
# FILE: src/diagnostic/component_tests.py
#
# WHAT THIS FILE DOES:
#   Health checks for the "processing pipeline" components and security:
#     Parsers -> Chunker -> Embedder -> Memmap Storage -> Security
#
# SPLIT FROM health_tests.py TO KEEP FILES UNDER 500 LINES.
# ============================================================================

from __future__ import annotations

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any

from . import TestResult, PROJ_ROOT


# ============================================================================
# PARSERS, CHUNKER, EMBEDDER, MEMMAP
# ============================================================================

def test_parser_registry() -> TestResult:
    """Are all parsers registered and importable?"""
    try:
        from src.parsers.registry import REGISTRY
        exts = REGISTRY.supported_extensions()
        expected = [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".eml"]
        missing = [e for e in expected if e not in exts]
        fails = {}
        for ext in exts:
            try:
                REGISTRY.get(ext).parser_cls()
            except Exception as e:
                fails[ext] = str(e)
        d = {"count": len(exts), "extensions": exts, "failures": fails}
        if fails:
            return TestResult("parser_registry", "Parsers", "WARN",
                f"Failed to instantiate: {list(fails.keys())}", d,
                fix_hint="Install missing libs (python-docx, pdfplumber, etc.)")
        if missing:
            return TestResult("parser_registry", "Parsers", "WARN",
                f"Missing: {missing}", d)
        return TestResult("parser_registry", "Parsers", "PASS",
            f"Registry OK ({len(exts)} types)", d)
    except Exception as e:
        return TestResult("parser_registry", "Parsers", "ERROR", f"{e}")


def test_parse_file(file_path: str) -> TestResult:
    """Parse a specific file -- checks for BUG-004 binary garbage."""
    try:
        p = Path(file_path)
        if not p.exists():
            return TestResult("parse_file", "Parsers", "FAIL", f"Not found: {file_path}")
        from src.parsers.registry import REGISTRY
        info = REGISTRY.get(p.suffix.lower())
        d = {"file": str(p), "ext": p.suffix, "size": p.stat().st_size}
        if not info:
            return TestResult("parse_file", "Parsers", "FAIL", f"No parser for {p.suffix}", d)
        parser = info.parser_cls()
        t0 = time.perf_counter()
        text = parser.parse(str(p)) if not hasattr(parser, "parse_with_details") \
            else parser.parse_with_details(str(p))[0]
        ms = (time.perf_counter() - t0) * 1000
        d.update({"ms": round(ms, 1), "chars": len(text or ""),
                  "chars_per_ms": round(len(text or "") / ms, 1) if ms > 0 else 0})
        if not text or not text.strip():
            return TestResult("parse_file", "Parsers", "FAIL",
                f"No text from {p.name}", d, fix_hint="May need OCR or file is corrupted.")
        sample = text[:5000]
        bad = sum(1 for c in sample if not c.isprintable() and c not in "\n\r\t")
        ratio = bad / len(sample) if sample else 0
        d["garbage_ratio"] = round(ratio, 4)
        if ratio > 0.10:
            return TestResult("parse_file", "Parsers", "WARN",
                f"BUG-004: {ratio:.1%} non-printable in {p.name}", d)
        return TestResult("parse_file", "Parsers", "PASS",
            f"{p.name}: {len(text):,} chars in {ms:.0f}ms", d)
    except Exception as e:
        return TestResult("parse_file", "Parsers", "ERROR", f"{e}")


def test_chunker() -> TestResult:
    """Does the chunker produce correct overlapping chunks?"""
    try:
        from src.core.chunker import Chunker, ChunkerConfig
        from src.core.config import load_config
        cfg = load_config(str(PROJ_ROOT))
        chunker = Chunker(ChunkerConfig(
            chunk_size=cfg.chunking.chunk_size, overlap=cfg.chunking.overlap))
        sample = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(50)])
        chunks = chunker.chunk_text(sample)
        if not chunks:
            return TestResult("chunker", "Chunker", "FAIL", "Zero chunks")
        lens = [len(c) for c in chunks]
        d = {"input": len(sample), "count": len(chunks), "avg": round(sum(lens)/len(lens)),
             "min": min(lens), "max": max(lens), "empty": sum(1 for c in chunks if not c.strip())}
        issues = []
        if d["empty"] > 0:
            issues.append(f"{d['empty']} empty")
        if d["max"] > cfg.chunking.chunk_size * 1.5:
            issues.append(f"Max {d['max']} > 1.5x config")
        if issues:
            return TestResult("chunker", "Chunker", "WARN", "; ".join(issues), d)
        return TestResult("chunker", "Chunker", "PASS",
            f"OK ({d['count']} chunks, avg {d['avg']})", d)
    except Exception as e:
        return TestResult("chunker", "Chunker", "ERROR", f"{e}")


def test_embedder(live: bool = False) -> TestResult:
    """Test embedding model. --test-embed for live test (~30s first time)."""
    try:
        from src.core.embedder import Embedder
        from src.core.config import load_config
        cfg = load_config(str(PROJ_ROOT))
        d = {"model": cfg.embedding.model_name, "dim": cfg.embedding.dimension, "live": live}
        if not live:
            return TestResult("embedder", "Embedder", "PASS",
                f"Importable ({cfg.embedding.model_name}). Use --test-embed for live.", d)
        import numpy as np
        t0 = time.perf_counter()
        emb = Embedder(cfg.embedding.model_name)
        d["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        texts = ["Frequency range 2-30 MHz.", "Digisonde 4D system.", "Ionospheric sounding."]
        t0 = time.perf_counter()
        out = emb.embed_batch(texts)
        d["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        d["shape"] = list(out.shape)
        if out.shape[1] != cfg.embedding.dimension:
            return TestResult("embedder", "Embedder", "FAIL",
                f"Dim mismatch: {out.shape[1]} vs {cfg.embedding.dimension}", d)
        if np.any(np.linalg.norm(out, axis=1) < 0.01):
            return TestResult("embedder", "Embedder", "WARN", "Near-zero embeddings", d)
        return TestResult("embedder", "Embedder", "PASS",
            f"OK (dim={out.shape[1]}, load={d['load_ms']:.0f}ms, embed={d['embed_ms']:.0f}ms)", d)
    except Exception as e:
        return TestResult("embedder", "Embedder", "ERROR", f"{e}")


def test_memmap_store() -> TestResult:
    """Is the memmap embedding file healthy?"""
    try:
        from src.core.config import load_config
        cfg = load_config(str(PROJ_ROOT))
        dd = os.path.dirname(cfg.paths.database)
        dat = os.path.join(dd, "embeddings.f16.dat")
        mp = os.path.join(dd, "embeddings_meta.json")
        if not os.path.exists(dat):
            return TestResult("memmap", "Storage", "SKIP", "Not found", fix_hint="Run rag-index.")
        meta = json.load(open(mp)) if os.path.exists(mp) else {}
        cnt, dim = meta.get("count", 0), meta.get("dim", 384)
        exp = cnt * dim * 2
        act = os.path.getsize(dat)
        d = {"count": cnt, "dim": dim, "expected": exp, "actual": act,
             "mb": round(act / (1024*1024), 2)}
        if exp != act:
            return TestResult("memmap", "Storage", "FAIL",
                f"Size mismatch: meta={exp:,}B file={act:,}B", d)
        import numpy as np
        if cnt > 0:
            mm = np.memmap(dat, dtype=np.float16, mode="r", shape=(cnt, dim))
            s = mm[:min(10, cnt)].astype(np.float32)
            del mm
            if np.any(np.isnan(s)) or np.any(np.isinf(s)):
                return TestResult("memmap", "Storage", "WARN", "NaN/Inf found", d)
        return TestResult("memmap", "Storage", "PASS",
            f"OK ({cnt:,} emb, {dim}d, {d['mb']}MB)", d)
    except Exception as e:
        return TestResult("memmap", "Storage", "ERROR", f"{e}")


# ============================================================================
# SECURITY TESTS
# ============================================================================

def test_security_endpoint() -> TestResult:
    """SEC-001: Is API endpoint safe for restricted environments?"""
    try:
        from src.core.config import load_config
        cfg = load_config(str(PROJ_ROOT))
        d = {"mode": cfg.mode, "endpoint": cfg.api.endpoint}
        public = cfg.api.endpoint == "https://api.openai.com/v1/chat/completions"
        d["is_public"] = public
        # Scan source code for hardcoded API keys (a security risk)
        # We look for patterns like 'sk-abc123...' which are OpenAI keys
        keys, yaml_key = [], False
        src_dir = PROJ_ROOT / "src"
        if src_dir.exists():
            for f in src_dir.rglob("*.py"):
                try:
                    if re.search(r'["\']sk-[a-zA-Z0-9]{20,}["\']',
                                 f.read_text(errors="ignore")):
                        keys.append(str(f.relative_to(PROJ_ROOT)))
                except Exception:
                    pass
        yp = PROJ_ROOT / "config" / "default_config.yaml"
        if yp.exists():
            try:
                yaml_key = bool(re.search(r'sk-[a-zA-Z0-9]{20,}', yp.read_text(errors="ignore")))
            except Exception:
                pass
        d.update({"hardcoded_keys": keys, "key_in_yaml": yaml_key})
        issues = []
        if public:
            issues.append("Endpoint defaults to public OpenAI")
        if keys:
            issues.append(f"Hardcoded keys in: {keys}")
        if yaml_key:
            issues.append("Key in YAML -- use env var")
        if issues:
            return TestResult("security_endpoint", "Security",
                "FAIL" if public else "WARN", "; ".join(issues), d,
                fix_hint="Default to empty endpoint. Add URL allowlist. Use env vars.")
        return TestResult("security_endpoint", "Security", "PASS",
            "Endpoint not public default", d)
    except Exception as e:
        return TestResult("security_endpoint", "Security", "ERROR", f"{e}")


def test_security_network() -> TestResult:
    """Audit network-capable code and verify NetworkGate is in control."""
    try:
        d = {"net_files": [], "urls": [], "gate_present": False}
        src_dir = PROJ_ROOT / "src"
        if not src_dir.exists():
            return TestResult("security_net", "Security", "SKIP", "No src/")
        # Patterns that indicate network-capable code
        # We scan for HTTP library imports to find files that can reach the internet
        pats = [r'import\s+httpx', r'import\s+requests', r'httpx\.', r'requests\.(get|post)']
        url_re = re.compile(r'https?://[^\s\'"\\]+')
        for f in src_dir.rglob("*.py"):
            try:
                t = f.read_text(errors="ignore")
                rp = str(f.relative_to(PROJ_ROOT))
                if any(re.search(p, t) for p in pats):
                    d["net_files"].append(rp)
                for u in url_re.findall(t):
                    d["urls"].append({"file": rp, "url": u.rstrip('",)')})
                # Check for NetworkGate usage (the centralized access control)
                if "network_gate" in t or "NetworkGate" in t or "check_allowed" in t:
                    d["gate_present"] = True
            except Exception:
                pass
        d["net_files"] = sorted(set(d["net_files"]))
        n = len(d["net_files"])
        if n > 0 and not d["gate_present"]:
            return TestResult("security_net", "Security", "WARN",
                f"{n} network files, NetworkGate not found", d,
                fix_hint="Ensure network_gate.py is imported and check_allowed() is called.")
        return TestResult("security_net", "Security", "PASS",
            f"Audit: {n} files, gate_present={d['gate_present']}", d)
    except Exception as e:
        return TestResult("security_net", "Security", "ERROR", f"{e}")

def test_parser_smoke() -> TestResult:
    """
    Smoke test: Can each critical parser import its library?
    Catches the #1 portability problem: package in requirements.txt
    but not installed in .venv.
    """
    checks = [
        ("PDF (pdfplumber)",    "pdfplumber",            "pdfplumber"),
        ("PDF (pypdf)",         "pypdf",                 "pypdf"),
        ("DOCX (python-docx)",  "docx",                  "python-docx"),
        ("PPTX (python-pptx)",  "pptx",                  "python-pptx"),
        ("XLSX (openpyxl)",     "openpyxl",              "openpyxl"),
        ("HTTP (httpx)",        "httpx",                  "httpx"),
        ("ML (transformers)",   "transformers",           "transformers"),
        ("ML (sentence_transformers)", "sentence_transformers", "sentence-transformers"),
        ("Config (yaml)",       "yaml",                   "PyYAML"),
        ("Logging (structlog)", "structlog",              "structlog"),
        ("Images (PIL)",        "PIL",                    "pillow"),
    ]

    passed = []
    failed = []
    details = {}

    for name, module, pip_name in checks:
        try:
            __import__(module)
            passed.append(name)
            details[name] = "OK"
        except ImportError:
            failed.append(name)
            details[name] = f"MISSING -- fix: pip install {pip_name}"
        except Exception as e:
            failed.append(name)
            details[name] = f"ERROR: {type(e).__name__}: {e}"

    d = {
        "total": len(checks),
        "passed": len(passed),
        "failed": len(failed),
        "details": details,
    }

    if failed:
        fix_packages = [p for n, m, p in checks if n in failed]
        fix_cmd = "pip install " + " ".join(fix_packages)
        return TestResult(
            "parser_smoke", "Parsers", "FAIL",
            f"{len(failed)} missing: {', '.join(failed)}", d,
            fix_hint=f"Run: {fix_cmd}")

    return TestResult(
        "parser_smoke", "Parsers", "PASS",
        f"All {len(checks)} critical libraries importable", d)
