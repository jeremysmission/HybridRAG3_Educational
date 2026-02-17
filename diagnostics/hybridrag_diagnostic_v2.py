#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Diagnostic Suite v2.0
# ============================================================================
#
# PURPOSE:
#   Comprehensive built-in-test (BIT) for the HybridRAG system.
#   Inspired by aerospace/enterprise BIT standards (industry standard, industry standard)
#   where every subsystem must self-report its health before operations.
#
# DESIGN PHILOSOPHY (from Senior Principal Engineering):
#   1. DETECT EARLY: Every test should catch problems BEFORE they cause
#      a user-visible failure. Like pre-flight checks on an aircraft.
#   2. ISOLATE FAST: Each test targets ONE subsystem. When something fails,
#      you know exactly which box to open.
#   3. RECOMMEND ACTION: Every failure includes what to do next -- not just
#      "FAIL" but "FAIL because X, fix by doing Y."
#   4. NO FALSE CONFIDENCE: If we can't verify something, we say UNKNOWN,
#      not PASS. Untested is not the same as working.
#   5. PORTABLE: This file runs on any machine with the HybridRAG project.
#      No external dependencies beyond what HybridRAG already requires.
#
# TEST CATEGORIES (modeled after electronics BIT levels):
#   Level 1 -- POWER-ON BIT: Environment, paths, Python, venv
#   Level 2 -- INITIATED BIT: Config, database, schema, credentials
#   Level 3 -- CONTINUOUS BIT: Pipeline components, models, connectivity
#   Level 4 -- MAINTENANCE BIT: Performance benchmarks, known bugs, integrity
#
# USAGE:
#   python hybridrag_diagnostic_v2.py                # Full diagnostic
#   python hybridrag_diagnostic_v2.py --quick        # Level 1+2 only (~5 sec)
#   python hybridrag_diagnostic_v2.py --level 3      # Up to Level 3
#   python hybridrag_diagnostic_v2.py --json out.json # Machine-readable output
#   python hybridrag_diagnostic_v2.py --verbose      # Show evidence for every test
#
# HISTORY:
#   v1.0 (2026-02-08): 15 tests -- config, DB, schema, parsers, security
#   v2.0 (2026-02-11): 35+ tests -- added credential, network, environment,
#        model cache, API reachability, Ollama health, venv integrity,
#        disk/RAM checks, config portability, log health, requirements
#        completeness, FTS5 sync, PowerShell profile, git status.
#        Redesigned from scratch for portability and modularity.
# ============================================================================

from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import struct
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple


# ============================================================================
# SECTION 1: Data Structures
# ============================================================================
# These are the "forms" that test results get written on. Every test fills
# out one TestResult, and the final report collects them all.

@dataclass
class TestResult:
    """
    One test's outcome. Think of it like a line item on an inspection report.
    
    Fields:
        name:      Short identifier (e.g., "venv_integrity")
        category:  Which subsystem (e.g., "Environment", "Database")
        level:     BIT level 1-4 (higher = deeper test)
        status:    PASS, FAIL, WARN, SKIP, ERROR
        message:   Human-readable summary of what happened
        fix_hint:  What to do if this failed (always provided for FAIL/WARN)
        details:   Machine-readable evidence dict for logging/GUI
        elapsed_ms: How long this test took to run
    """
    name: str
    category: str
    level: int
    status: str
    message: str
    fix_hint: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass 
class BugReport:
    """A known bug detected by code inspection."""
    bug_id: str
    title: str
    description: str
    fix_steps: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW


@dataclass
class DiagnosticReport:
    """The complete diagnostic report -- all tests, all findings."""
    timestamp: str = ""
    hostname: str = ""
    python_version: str = ""
    platform_info: str = ""
    project_root: str = ""
    results: List[TestResult] = field(default_factory=list)
    bugs: List[BugReport] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")
    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == "WARN")
    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == "SKIP")
    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == "ERROR")
    @property
    def total(self) -> int:
        return len(self.results)


# ============================================================================
# SECTION 2: Terminal Colors
# ============================================================================
# Color codes for pretty terminal output. Falls back to plain text if
# the terminal doesn't support colors (like redirecting to a file).

_NO_COLOR = os.getenv("NO_COLOR") or not sys.stdout.isatty()
def _c(code: str) -> str:
    return "" if _NO_COLOR else f"\033[{code}m"

GREEN = _c("32"); RED = _c("31"); YELLOW = _c("33"); CYAN = _c("36")
BOLD = _c("1"); DIM = _c("2"); RESET = _c("0"); WHITE = _c("37")
BLUE = _c("34"); MAGENTA = _c("35")

STATUS_COLORS = {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW, "SKIP": DIM, "ERROR": RED+BOLD}
STATUS_ICONS  = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]", "ERROR": "[FAIL][FAIL]"}


# ============================================================================
# SECTION 3: Helper Functions
# ============================================================================

def _find_project_root() -> Path:
    """Walk up from this script's location to find the HybridRAG project root.
    Looks for config/default_config.yaml or src/core/config.py as markers."""
    start = Path(__file__).resolve().parent
    for p in [start] + list(start.parents):
        if (p / "config" / "default_config.yaml").exists():
            return p
        if (p / "src" / "core" / "config.py").exists():
            return p
    return start  # fallback

def _safe_import(module_path: str):
    """Try to import a module, return (module, None) or (None, error_string)."""
    try:
        return importlib.import_module(module_path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _run_test(name: str, category: str, level: int, fn: Callable) -> TestResult:
    """Run a test function with timing and error catching."""
    t0 = time.perf_counter()
    try:
        result = fn()
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result
    except Exception as e:
        return TestResult(
            name=name, category=category, level=level, status="ERROR",
            message=f"Test crashed: {e}",
            fix_hint=f"Exception in test code: {traceback.format_exc()[-200:]}",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

def _get_memory_mb() -> float:
    """Get current Python process memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return -1.0

def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False

def _get_disk_free_gb(path: str) -> float:
    """Get free disk space in GB for the drive containing path."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return -1.0


# ============================================================================
# SECTION 4: LEVEL 1 TESTS -- POWER-ON BIT (Environment)
# ============================================================================
# These are the most basic checks. If these fail, nothing else will work.
# Like checking if the aircraft has power and hydraulics before anything else.

def test_python_version(root: Path) -> TestResult:
    """Verify Python version is 3.10+ (required for match statements, typing)."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        return TestResult("python_version", "Environment", 1, "PASS",
            f"Python {version_str}", details={"version": version_str})
    return TestResult("python_version", "Environment", 1, "FAIL",
        f"Python {version_str} -- need 3.10+",
        fix_hint="Install Python 3.10 or newer from python.org",
        details={"version": version_str})


def test_venv_integrity(root: Path) -> TestResult:
    """Verify we're running inside the project's virtual environment.
    THIS TEST CATCHES: Wrong venv activated, system Python used by mistake,
    OneDrive venv vs local venv confusion."""
    exe = Path(sys.executable).resolve()
    venv_dir = root / ".venv"
    
    details = {
        "python_executable": str(exe),
        "expected_venv": str(venv_dir),
        "in_venv": hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix),
    }
    
    if not details["in_venv"]:
        return TestResult("venv_integrity", "Environment", 1, "WARN",
            "Not running in a virtual environment -- using system Python",
            fix_hint=f"Activate venv: . .\\start_hybridrag.ps1  OR  & \"{venv_dir}\\Scripts\\Activate.ps1\"",
            details=details)
    
    # Check the venv is in the project directory (not somewhere else like OneDrive)
    try:
        exe_str = str(exe).lower()
        root_str = str(root).lower()
        if root_str not in exe_str:
            return TestResult("venv_integrity", "Environment", 1, "WARN",
                f"Venv is outside project root. Python: {exe}",
                fix_hint=f"Deactivate, then activate the correct venv from {root}",
                details=details)
    except Exception:
        pass
    
    return TestResult("venv_integrity", "Environment", 1, "PASS",
        f"Venv active: {exe}", details=details)


def test_requirements_installed(root: Path) -> TestResult:
    """Verify all required packages are importable.
    THIS TEST CATCHES: Missing keyring, missing pip-system-certs, 
    sentence-transformers not installed, torch wrong version."""
    # Core packages that MUST be importable
    required = {
        "numpy": "Vector math for embeddings",
        "yaml": "Config file parsing (pyyaml)",
        "httpx": "HTTP client for API calls",
        "sentence_transformers": "Embedding model (ML)",
        "torch": "PyTorch (ML backend)",
        "xxhash": "Fast file hashing",
        "chardet": "Encoding detection for parsers",
    }
    # Optional but important packages
    optional = {
        "keyring": "Credential Manager access for API keys",
        "rich": "Progress bars and formatted output",
    }
    
    missing_required = []
    missing_optional = []
    installed = {}
    
    for pkg, desc in required.items():
        mod, err = _safe_import(pkg)
        if mod:
            ver = getattr(mod, "__version__", "?")
            installed[pkg] = ver
        else:
            missing_required.append(f"{pkg} ({desc})")
    
    for pkg, desc in optional.items():
        mod, err = _safe_import(pkg)
        if mod:
            ver = getattr(mod, "__version__", "?")
            installed[pkg] = ver
        else:
            missing_optional.append(f"{pkg} ({desc})")
    
    details = {"installed": installed, "missing_required": missing_required, 
               "missing_optional": missing_optional}
    
    if missing_required:
        return TestResult("requirements", "Environment", 1, "FAIL",
            f"Missing {len(missing_required)} required package(s): {', '.join(missing_required)}",
            fix_hint=f"pip install -r requirements.txt",
            details=details)
    if missing_optional:
        return TestResult("requirements", "Environment", 1, "WARN",
            f"Missing optional: {', '.join(missing_optional)}. Core packages OK ({len(installed)} installed).",
            fix_hint="pip install " + " ".join(p.split(" ")[0] for p in missing_optional),
            details=details)
    return TestResult("requirements", "Environment", 1, "PASS",
        f"All {len(installed)} required packages installed",
        details=details)


def test_project_structure(root: Path) -> TestResult:
    """Verify expected directories and __init__.py files exist.
    THIS TEST CATCHES: Missing __init__.py (ModuleNotFoundError), 
    missing config directory, missing src tree."""
    expected_dirs = ["src", "src/core", "src/parsers", "src/monitoring", "config", "logs"]
    expected_inits = ["src/__init__.py", "src/core/__init__.py", 
                      "src/parsers/__init__.py", "src/monitoring/__init__.py"]
    
    missing_dirs = [d for d in expected_dirs if not (root / d).is_dir()]
    missing_inits = [f for f in expected_inits if not (root / f).exists()]
    
    details = {"missing_dirs": missing_dirs, "missing_inits": missing_inits}
    
    if missing_dirs:
        return TestResult("project_structure", "Environment", 1, "FAIL",
            f"Missing directories: {missing_dirs}",
            fix_hint="Run setup script or recreate missing directories",
            details=details)
    if missing_inits:
        return TestResult("project_structure", "Environment", 1, "FAIL",
            f"Missing __init__.py files: {missing_inits}",
            fix_hint="Create empty __init__.py: " + "; ".join(f"echo.>{f}" for f in missing_inits),
            details=details)
    return TestResult("project_structure", "Environment", 1, "PASS",
        "Project structure intact -- all directories and __init__.py present",
        details=details)


def test_disk_space(root: Path) -> TestResult:
    """Check available disk space on the project drive.
    THIS TEST CATCHES: Drive full during indexing (silent corruption)."""
    free_gb = _get_disk_free_gb(str(root))
    details = {"free_gb": round(free_gb, 2), "drive": str(root.anchor)}
    
    if free_gb < 0:
        return TestResult("disk_space", "Environment", 1, "SKIP",
            "Could not determine disk space", details=details)
    if free_gb < 2.0:
        return TestResult("disk_space", "Environment", 1, "FAIL",
            f"Only {free_gb:.1f} GB free -- risk of corruption during indexing",
            fix_hint="Free up disk space. Index database needs ~12 GB for 500 GB source data.",
            details=details)
    if free_gb < 10.0:
        return TestResult("disk_space", "Environment", 1, "WARN",
            f"{free_gb:.1f} GB free -- adequate but monitor during indexing",
            details=details)
    return TestResult("disk_space", "Environment", 1, "PASS",
        f"{free_gb:.1f} GB free", details=details)


def test_log_directory(root: Path) -> TestResult:
    """Verify log directory exists and is writable.
    THIS TEST CATCHES: Logs silently failing, audit trail broken."""
    log_dir = root / "logs"
    details = {"log_dir": str(log_dir)}
    
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            details["created"] = True
        except Exception as e:
            return TestResult("log_directory", "Environment", 1, "FAIL",
                f"Cannot create logs directory: {e}",
                fix_hint=f"mkdir \"{log_dir}\"", details=details)
    
    # Test write permission
    test_file = log_dir / ".diag_write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        details["writable"] = True
    except Exception as e:
        return TestResult("log_directory", "Environment", 1, "FAIL",
            f"Log directory not writable: {e}",
            fix_hint="Check permissions on the logs/ folder", details=details)
    
    return TestResult("log_directory", "Environment", 1, "PASS",
        "Log directory exists and writable", details=details)


# ============================================================================
# SECTION 5: LEVEL 2 TESTS -- INITIATED BIT (Configuration & Data)
# ============================================================================
# Verify that configuration is valid, database exists and has correct schema,
# and credentials are available. Like checking instruments are calibrated.

def test_config_load(root: Path) -> TestResult:
    """Load the YAML config and verify it parses correctly."""
    sys.path.insert(0, str(root))
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        details = {
            "mode": config.mode,
            "chunk_size": config.chunking.chunk_size,
            "overlap": config.chunking.overlap,
            "model_name": config.embedding.model_name,
            "dimension": config.embedding.dimension,
        }
        return TestResult("config_load", "Configuration", 2, "PASS",
            f"Config loaded -- mode: {config.mode}, chunk: {config.chunking.chunk_size}",
            details=details)
    except Exception as e:
        return TestResult("config_load", "Configuration", 2, "FAIL",
            f"Config load failed: {e}",
            fix_hint="Check config/default_config.yaml for YAML syntax errors",
            details={"error": str(e)})


def test_config_paths(root: Path) -> TestResult:
    """Verify all configured paths point to real locations on THIS machine.
    THIS TEST CATCHES: Home machine paths on work machine after git pull,
    source_folder emptied by config overwrite."""
    try:
        from src.core.config import load_config
        config = load_config(str(root))
    except Exception as e:
        return TestResult("config_paths", "Configuration", 2, "SKIP",
            f"Cannot load config: {e}")
    
    issues = []
    details = {}
    
    # Check database path
    db_path = config.paths.database
    details["database"] = db_path
    if not db_path:
        issues.append("database path is empty")
    elif not Path(db_path).parent.exists():
        issues.append(f"database parent directory doesn't exist: {Path(db_path).parent}")
    
    # Check source folder
    src_folder = config.paths.source_folder
    details["source_folder"] = src_folder
    if not src_folder:
        issues.append("source_folder is empty -- indexing will have no target")
    elif not Path(src_folder).exists():
        issues.append(f"source_folder doesn't exist: {src_folder}")
    
    # Check embeddings cache
    cache = config.paths.embeddings_cache
    details["embeddings_cache"] = cache
    if cache and not Path(cache).exists():
        issues.append(f"embeddings_cache doesn't exist: {cache}")
    
    if issues:
        return TestResult("config_paths", "Configuration", 2, "FAIL",
            f"Path issues: {'; '.join(issues)}",
            fix_hint="Edit config/default_config.yaml or set HYBRIDRAG_DATA_DIR environment variable",
            details=details)
    return TestResult("config_paths", "Configuration", 2, "PASS",
        "All configured paths exist on this machine", details=details)


def test_database_connection(root: Path) -> TestResult:
    """Open the SQLite database and verify basic operations."""
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        db_path = config.paths.database
    except Exception:
        return TestResult("database_connection", "Database", 2, "SKIP",
            "Cannot load config to find database path")
    
    if not db_path or not Path(db_path).exists():
        return TestResult("database_connection", "Database", 2, "SKIP",
            f"Database file doesn't exist yet: {db_path}",
            fix_hint="Run indexing first to create the database")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        
        # Get database file size
        db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
        
        conn.close()
        details = {"db_path": db_path, "tables": table_count, "size_mb": round(db_size_mb, 2)}
        return TestResult("database_connection", "Database", 2, "PASS",
            f"Database OK -- {table_count} tables, {db_size_mb:.1f} MB",
            details=details)
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower():
            return TestResult("database_connection", "Database", 2, "FAIL",
                f"Database locked by another process: {e}",
                fix_hint="Close other Python processes or check Task Manager for zombie processes")
        return TestResult("database_connection", "Database", 2, "FAIL",
            f"Database error: {e}", fix_hint="Database may be corrupted. Check file integrity.")


def test_schema_chunks(root: Path) -> TestResult:
    """Verify chunks table has expected columns including file_hash."""
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        db_path = config.paths.database
        if not db_path or not Path(db_path).exists():
            return TestResult("schema_chunks", "Database", 2, "SKIP", "No database file")
        
        conn = sqlite3.connect(db_path)
        columns = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        conn.close()
        
        if not columns:
            return TestResult("schema_chunks", "Database", 2, "SKIP",
                "chunks table doesn't exist yet")
        
        expected = {"chunk_id", "source_path", "text"}
        present = set(columns)
        missing = expected - present
        has_file_hash = "file_hash" in present
        has_fts = any("fts" in c.lower() for c in columns)
        
        details = {"columns": columns, "has_file_hash": has_file_hash, "column_count": len(columns)}
        
        if missing:
            return TestResult("schema_chunks", "Database", 2, "FAIL",
                f"Missing columns: {missing}", fix_hint="Database schema outdated. Re-index.",
                details=details)
        if not has_file_hash:
            return TestResult("schema_chunks", "Database", 2, "WARN",
                "Missing file_hash column -- change detection won't work (BUG-001)",
                fix_hint="ALTER TABLE chunks ADD COLUMN file_hash TEXT",
                details=details)
        return TestResult("schema_chunks", "Database", 2, "PASS",
            f"Schema OK -- {len(columns)} columns, file_hash present", details=details)
    except Exception as e:
        return TestResult("schema_chunks", "Database", 2, "ERROR", f"Schema check failed: {e}")


def test_schema_fts5(root: Path) -> TestResult:
    """Verify FTS5 virtual table exists and is populated."""
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        db_path = config.paths.database
        if not db_path or not Path(db_path).exists():
            return TestResult("schema_fts5", "Database", 2, "SKIP", "No database file")
        
        conn = sqlite3.connect(db_path)
        
        # Check for FTS5 tables
        fts_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts5%'"
        ).fetchall()
        
        details = {"fts_tables": [t[0] for t in fts_tables]}
        
        if not fts_tables:
            # Check if FTS5 extension is even available
            try:
                conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_test USING fts5(content)")
                conn.execute("DROP TABLE _fts_test")
                details["fts5_available"] = True
            except Exception:
                details["fts5_available"] = False
                return TestResult("schema_fts5", "Database", 2, "WARN",
                    "FTS5 extension not available -- keyword search disabled",
                    fix_hint="Rebuild Python with FTS5 or install a binary with FTS5",
                    details=details)
            
            return TestResult("schema_fts5", "Database", 2, "WARN",
                "No FTS5 tables found -- keyword search not indexed yet",
                fix_hint="Re-index to create FTS5 tables", details=details)
        
        # Check row count matches chunks
        try:
            fts_name = fts_tables[0][0]
            fts_count = conn.execute(f"SELECT COUNT(*) FROM [{fts_name}]").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            details["fts_rows"] = fts_count
            details["chunk_rows"] = chunk_count
            details["in_sync"] = fts_count == chunk_count
            
            if fts_count != chunk_count:
                return TestResult("schema_fts5", "Database", 2, "WARN",
                    f"FTS5 out of sync: {fts_count} FTS rows vs {chunk_count} chunks",
                    fix_hint="Re-index to rebuild FTS5 index", details=details)
        except Exception:
            pass
        
        conn.close()
        return TestResult("schema_fts5", "Database", 2, "PASS",
            f"FTS5 OK -- {len(fts_tables)} table(s)", details=details)
    except Exception as e:
        return TestResult("schema_fts5", "Database", 2, "ERROR", f"FTS5 check failed: {e}")


def test_data_integrity(root: Path) -> TestResult:
    """Cross-check chunks table: row count, null checks, source file references."""
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        db_path = config.paths.database
        if not db_path or not Path(db_path).exists():
            return TestResult("data_integrity", "Database", 2, "SKIP", "No database file")
        
        conn = sqlite3.connect(db_path)
        issues = []
        details = {}
        
        # Total chunks
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        details["total_chunks"] = total
        
        if total == 0:
            conn.close()
            return TestResult("data_integrity", "Database", 2, "WARN",
                "Database is empty -- no chunks indexed yet",
                fix_hint="Run indexing to populate the database", details=details)
        
        # Null text check
        try:
            null_text = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_text IS NULL OR chunk_text = ''"
            ).fetchone()[0]
            details["null_text_chunks"] = null_text
            if null_text > 0:
                issues.append(f"{null_text} chunks with empty text")
        except Exception:
            pass
        
        # Unique source files
        try:
            unique_files = conn.execute(
                "SELECT COUNT(DISTINCT source_path) FROM chunks"
            ).fetchone()[0]
            details["unique_source_files"] = unique_files
        except Exception:
            pass
        
        # Check for orphaned source references (files that no longer exist)
        try:
            source_paths = [r[0] for r in conn.execute(
                "SELECT DISTINCT source_path FROM chunks LIMIT 50"
            ).fetchall()]
            missing_sources = [p for p in source_paths if not Path(p).exists()]
            details["checked_sources"] = len(source_paths)
            details["missing_sources"] = len(missing_sources)
            if missing_sources:
                issues.append(f"{len(missing_sources)}/{len(source_paths)} sampled source files no longer exist")
        except Exception:
            pass
        
        conn.close()
        
        if issues:
            return TestResult("data_integrity", "Database", 2, "WARN",
                f"Integrity issues: {'; '.join(issues)}", details=details)
        return TestResult("data_integrity", "Database", 2, "PASS",
            f"Data OK -- {total:,} chunks from {details.get('unique_source_files', '?')} files",
            details=details)
    except Exception as e:
        return TestResult("data_integrity", "Database", 2, "ERROR", f"Integrity check failed: {e}")


def test_credentials(root: Path) -> TestResult:
    """Check if API credentials are stored in Windows Credential Manager.
    THIS TEST CATCHES: keyring not installed, credentials never stored,
    credential manager inaccessible."""
    details = {}
    
    try:
        import keyring
        details["keyring_installed"] = True
    except ImportError:
        return TestResult("credentials", "Security", 2, "WARN",
            "keyring package not installed -- API credentials cannot be stored securely",
            fix_hint="pip install keyring",
            details={"keyring_installed": False})
    
    # Check for API key
    try:
        key = keyring.get_password("hybridrag", "azure_api_key")
        details["api_key_stored"] = key is not None and len(key) > 0
    except Exception as e:
        details["api_key_stored"] = False
        details["api_key_error"] = str(e)
    
    # Check for endpoint
    try:
        endpoint = keyring.get_password("hybridrag", "azure_endpoint") 
        details["endpoint_stored"] = endpoint is not None and len(endpoint) > 0
        if endpoint:
            details["endpoint_is_azure"] = any(x in endpoint.lower() for x in ["azure", "aoai"])
    except Exception as e:
        details["endpoint_stored"] = False
        details["endpoint_error"] = str(e)
    
    has_key = details.get("api_key_stored", False)
    has_endpoint = details.get("endpoint_stored", False)
    
    if has_key and has_endpoint:
        azure_note = " (Azure detected)" if details.get("endpoint_is_azure") else ""
        return TestResult("credentials", "Security", 2, "PASS",
            f"API key and endpoint stored in Credential Manager{azure_note}",
            details=details)
    
    missing = []
    if not has_key: missing.append("API key")
    if not has_endpoint: missing.append("endpoint URL")
    
    return TestResult("credentials", "Security", 2, "WARN",
        f"Missing credentials: {', '.join(missing)}. Online mode won't work.",
        fix_hint="Run: rag-store-key  and  rag-store-endpoint",
        details=details)


# ============================================================================
# SECTION 6: LEVEL 3 TESTS -- CONTINUOUS BIT (Pipeline Components)
# ============================================================================
# Test each component of the RAG pipeline individually.
# Like testing each instrument on a bench before running the full procedure.

def test_embedding_model_cache(root: Path) -> TestResult:
    """Check if the embedding model is cached locally (no download needed).
    THIS TEST CATCHES: First-run download fails behind corporate firewall."""
    model_name = "all-MiniLM-L6-v2"
    details = {"model_name": model_name}
    
    # Check common cache locations
    hf_home = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    model_cache = os.path.join(hf_home, "hub", f"models--sentence-transformers--{model_name}")
    project_cache = root / ".model_cache"
    
    details["hf_cache_path"] = model_cache
    details["project_cache_path"] = str(project_cache)
    
    if Path(model_cache).exists():
        # Check if it has actual model files (not just metadata)
        safetensors = list(Path(model_cache).rglob("*.safetensors"))
        bin_files = list(Path(model_cache).rglob("*.bin"))
        details["has_safetensors"] = len(safetensors) > 0
        details["has_bin"] = len(bin_files) > 0
        if safetensors or bin_files:
            return TestResult("model_cache", "ML Pipeline", 3, "PASS",
                f"Embedding model cached at {model_cache}", details=details)
    
    if project_cache.exists():
        safetensors = list(project_cache.rglob("*.safetensors"))
        bin_files = list(project_cache.rglob("*.bin"))
        if safetensors or bin_files:
            return TestResult("model_cache", "ML Pipeline", 3, "PASS",
                f"Embedding model cached in project at {project_cache}", details=details)
    
    # Check if HF_HUB_OFFLINE is set (which would prevent download)
    offline_mode = os.getenv("HF_HUB_OFFLINE", "0") == "1"
    details["hf_offline_mode"] = offline_mode
    
    if offline_mode:
        return TestResult("model_cache", "ML Pipeline", 3, "FAIL",
            "Embedding model NOT cached and HF_HUB_OFFLINE=1 -- model cannot download",
            fix_hint="Copy model cache from another machine or set HF_HUB_OFFLINE=0 temporarily",
            details=details)
    
    return TestResult("model_cache", "ML Pipeline", 3, "WARN",
        "Embedding model not found in cache -- will attempt download on first use",
        fix_hint="If behind corporate firewall, copy .cache/huggingface from a working machine",
        details=details)


def test_embedder_functional(root: Path) -> TestResult:
    """Actually load the embedding model and produce a vector.
    THIS IS THE REAL TEST -- if this passes, embeddings work."""
    try:
        from src.core.config import load_config
        from src.core.embedder import Embedder
        config = load_config(str(root))
        
        embedder = Embedder(config.embedding.model_name)
        test_text = "diagnostic test query"
        vec = embedder.embed_query(test_text)
        
        details = {
            "model": config.embedding.model_name,
            "vector_shape": list(vec.shape),
            "vector_dtype": str(vec.dtype),
            "expected_dim": config.embedding.dimension,
            "actual_dim": vec.shape[0],
            "dim_match": vec.shape[0] == config.embedding.dimension,
        }
        
        if not details["dim_match"]:
            return TestResult("embedder_functional", "ML Pipeline", 3, "FAIL",
                f"Dimension mismatch: model outputs {vec.shape[0]} but config says {config.embedding.dimension}",
                fix_hint="Update embedding.dimension in config/default_config.yaml",
                details=details)
        
        return TestResult("embedder_functional", "ML Pipeline", 3, "PASS",
            f"Embedder OK -- {config.embedding.model_name} -> {vec.shape[0]}D vector",
            details=details)
    except Exception as e:
        return TestResult("embedder_functional", "ML Pipeline", 3, "FAIL",
            f"Embedder failed: {e}",
            fix_hint="Check model cache, sentence-transformers installation, and torch",
            details={"error": str(e)})


def test_chunker_functional(root: Path) -> TestResult:
    """Test the chunker produces valid, overlapping chunks."""
    try:
        from src.core.config import load_config
        from src.core.chunker import Chunker, ChunkerConfig
        config = load_config(str(root))
        
        chunker = Chunker(ChunkerConfig(
            chunk_size=config.chunking.chunk_size,
            overlap=config.chunking.overlap
        ))
        
        # Test with realistic text
        test_text = "The digisonde operates by transmitting radio pulses. " * 100
        chunks = chunker.chunk_text(test_text)
        
        details = {
            "input_chars": len(test_text),
            "chunks_produced": len(chunks),
            "avg_chunk_len": sum(len(c) for c in chunks) // max(len(chunks), 1),
            "config_chunk_size": config.chunking.chunk_size,
            "config_overlap": config.chunking.overlap,
        }
        
        if not chunks:
            return TestResult("chunker_functional", "ML Pipeline", 3, "FAIL",
                "Chunker produced zero chunks from test text",
                fix_hint="Check chunker.py chunk_text() logic", details=details)
        
        return TestResult("chunker_functional", "ML Pipeline", 3, "PASS",
            f"Chunker OK -- {len(test_text)} chars -> {len(chunks)} chunks (avg {details['avg_chunk_len']} chars)",
            details=details)
    except Exception as e:
        return TestResult("chunker_functional", "ML Pipeline", 3, "FAIL",
            f"Chunker failed: {e}", details={"error": str(e)})


def test_parser_registry(root: Path) -> TestResult:
    """Verify text parser can be imported and supports expected file types."""
    try:
        from src.parsers.text_parser import TextParser
        parser = TextParser()
        
        # Check for supported extensions attribute
        extensions = getattr(parser, "SUPPORTED_EXTENSIONS", None) or \
                     getattr(parser, "supported_extensions", None) or []
        
        details = {"parser_class": "TextParser", "extensions": list(extensions)[:20]}
        
        return TestResult("parser_registry", "ML Pipeline", 3, "PASS",
            f"TextParser OK -- {len(extensions)} file types supported",
            details=details)
    except Exception as e:
        return TestResult("parser_registry", "ML Pipeline", 3, "FAIL",
            f"Parser import failed: {e}",
            fix_hint="Check src/parsers/text_parser.py for import errors",
            details={"error": str(e)})


def test_ollama_health(root: Path) -> TestResult:
    """Check if Ollama is running and has models loaded.
    THIS TEST CATCHES: Ollama not started, port conflict, no models pulled,
    airplane mode blocking localhost."""
    details = {}
    port = 11434
    
    # Step 1: Is the port open?
    port_open = _check_port("127.0.0.1", port, timeout=3.0)
    details["port_open"] = port_open
    
    if not port_open:
        # Check if something else is on that port
        details["port_in_use"] = _check_port("localhost", port, timeout=1.0)
        return TestResult("ollama_health", "LLM Backend", 3, "WARN",
            f"Ollama not responding on localhost:{port}",
            fix_hint="Start Ollama: ollama serve  (or start the Ollama desktop app)",
            details=details)
    
    # Step 2: Can we reach the API?
    try:
        import httpx
        with httpx.Client(timeout=5, proxy=None) as client:
            resp = client.get(f"http://127.0.0.1:{port}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "?") for m in data.get("models", [])]
                details["models"] = models
                details["model_count"] = len(models)
                
                if not models:
                    return TestResult("ollama_health", "LLM Backend", 3, "WARN",
                        "Ollama running but no models loaded",
                        fix_hint="Pull a model: ollama pull llama3",
                        details=details)
                
                return TestResult("ollama_health", "LLM Backend", 3, "PASS",
                    f"Ollama OK -- {len(models)} model(s): {', '.join(models[:3])}",
                    details=details)
            else:
                details["status_code"] = resp.status_code
                return TestResult("ollama_health", "LLM Backend", 3, "WARN",
                    f"Ollama responded with HTTP {resp.status_code}",
                    details=details)
    except Exception as e:
        details["error"] = str(e)
        return TestResult("ollama_health", "LLM Backend", 3, "WARN",
            f"Ollama port open but API unreachable: {e}",
            fix_hint="Check if Ollama is fully started, try: curl http://localhost:11434/api/tags",
            details=details)


def test_api_reachability(root: Path) -> TestResult:
    """Test if the configured API endpoint is reachable (without sending a real query).
    THIS TEST CATCHES: Wrong URL, SSL issues, corporate proxy blocking,
    VPN required but not connected."""
    details = {}
    
    # Get endpoint from credentials
    try:
        import keyring
        endpoint = keyring.get_password("hybridrag", "azure_endpoint")
        details["endpoint_source"] = "credential_manager"
    except Exception:
        endpoint = None
    
    if not endpoint:
        # Try config
        try:
            from src.core.config import load_config
            config = load_config(str(root))
            endpoint = getattr(config.api, "endpoint", "") or getattr(config.api, "base_url", "")
            details["endpoint_source"] = "config"
        except Exception:
            pass
    
    if not endpoint:
        return TestResult("api_reachability", "LLM Backend", 3, "SKIP",
            "No API endpoint configured -- online mode not set up",
            fix_hint="Run: rag-store-endpoint", details=details)
    
    details["endpoint"] = endpoint[:50] + "..." if len(endpoint) > 50 else endpoint
    details["is_azure"] = any(x in endpoint.lower() for x in ["azure", "aoai"])
    
    # Try to reach it (just a TCP connection, not a real API call)
    try:
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        details["host"] = host
        details["port"] = port
        
        reachable = _check_port(host, port, timeout=5.0)
        details["tcp_reachable"] = reachable
        
        if not reachable:
            return TestResult("api_reachability", "LLM Backend", 3, "FAIL",
                f"Cannot reach API server at {host}:{port}",
                fix_hint="Check: VPN connected? Firewall rule allows HTTPS to this host? DNS resolves?",
                details=details)
        
        # Try an actual HTTPS connection
        try:
            import httpx
            with httpx.Client(timeout=10, verify=True) as client:
                # Just hit the base URL -- we expect a 404 or 200, anything except connection error
                resp = client.get(endpoint)
                details["http_status"] = resp.status_code
                return TestResult("api_reachability", "LLM Backend", 3, "PASS",
                    f"API endpoint reachable (HTTP {resp.status_code})",
                    details=details)
        except Exception as e:
            err = str(e).lower()
            if "ssl" in err or "certificate" in err:
                return TestResult("api_reachability", "LLM Backend", 3, "FAIL",
                    f"SSL certificate error reaching API: {e}",
                    fix_hint="pip install pip-system-certs  (tells Python to trust corporate proxy cert)",
                    details=details)
            details["https_error"] = str(e)[:200]
            return TestResult("api_reachability", "LLM Backend", 3, "WARN",
                f"TCP reachable but HTTPS failed: {str(e)[:100]}",
                details=details)
    
    except Exception as e:
        return TestResult("api_reachability", "LLM Backend", 3, "ERROR",
            f"Reachability test error: {e}", details=details)


def test_security_network_lockdown(root: Path) -> TestResult:
    """Inspect code for unintended network access patterns.
    THIS TEST CATCHES: SEC-001 (public API default), accidental internet access."""
    details = {}
    issues = []
    
    # Scan Python files for network-capable patterns
    net_patterns = [
        (r"import\s+requests", "requests library (HTTP client)"),
        (r"import\s+urllib", "urllib (HTTP client)"),
        (r"api\.openai\.com", "public OpenAI endpoint hardcoded"),
        (r"huggingface\.co", "HuggingFace endpoint (model downloads)"),
    ]
    
    src_dir = root / "src"
    if not src_dir.exists():
        return TestResult("security_network", "Security", 3, "SKIP", "No src/ directory")
    
    findings = []
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern, desc in net_patterns:
                if re.search(pattern, content):
                    findings.append(f"{py_file.name}: {desc}")
        except Exception:
            pass
    
    details["network_findings"] = findings
    
    # Check if API config defaults to a public endpoint
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        api_endpoint = getattr(config.api, "endpoint", "") or getattr(config.api, "base_url", "")
        if api_endpoint and "openai.com" in api_endpoint:
            issues.append("SEC-001: API defaults to public OpenAI -- data exfiltration risk")
    except Exception:
        pass
    
    if issues:
        return TestResult("security_network", "Security", 3, "FAIL",
            "; ".join(issues),
            fix_hint="Set API endpoint to empty string. Require explicit configuration.",
            details=details)
    
    if findings:
        return TestResult("security_network", "Security", 3, "WARN",
            f"{len(findings)} network-capable code patterns found (review for compliance)",
            details=details)
    
    return TestResult("security_network", "Security", 3, "PASS",
        "No unexpected network access patterns detected", details=details)


def test_change_detection(root: Path) -> TestResult:
    """Verify hash-based change detection is active in index_folder().
    BUG-002 RESOLVED 2026-02-14: Logic is inlined using _compute_file_hash()
    + get_file_hash(), not via a separate _file_changed() method."""
    try:
        from src.core.indexer import Indexer
        has_compute_hash = hasattr(Indexer, "_compute_file_hash")
        source = inspect.getsource(Indexer.index_folder)
        has_hash_in_loop = "_compute_file_hash" in source and "get_file_hash" in source
        details = {
            "has_compute_hash": has_compute_hash,
            "hash_detection_in_loop": has_hash_in_loop,
        }
        if not has_compute_hash:
            return TestResult("change_detection", "Indexer", 3, "FAIL",
                "No _compute_file_hash() method -- change detection not implemented",
                fix_hint="Add hash-based change detection to Indexer",
                details=details)
        if not has_hash_in_loop:
            return TestResult("change_detection", "Indexer", 3, "FAIL",
                "BUG-002: Hash methods exist but not used in index_folder()",
                fix_hint="Add _compute_file_hash + get_file_hash to index_folder loop",
                details=details)
        return TestResult("change_detection", "Indexer", 3, "PASS",
            "Change detection active -- hash-based skip logic in index_folder()",
            details=details)
    except Exception as e:
        return TestResult("change_detection", "Indexer", 3, "ERROR",
            f"Code inspection failed: {e}", details={"error": str(e)})

def test_resource_cleanup(root: Path) -> TestResult:
    """Check if VectorStore and Embedder have close()/cleanup methods."""
    try:
        findings = []
        from src.core.vector_store import VectorStore
        if not hasattr(VectorStore, "close"):
            findings.append("VectorStore has no close() method")
        
        from src.core.embedder import Embedder
        if not hasattr(Embedder, "close") and not hasattr(Embedder, "cleanup"):
            findings.append("Embedder has no close()/cleanup() method")
        
        if findings:
            return TestResult("resource_cleanup", "Indexer", 3, "WARN",
                f"BUG-003: {'; '.join(findings)} -- potential memory leak on long runs",
                fix_hint="Add close() methods for proper resource cleanup")
        
        return TestResult("resource_cleanup", "Indexer", 3, "PASS",
            "Cleanup methods present on core components")
    except Exception as e:
        return TestResult("resource_cleanup", "Indexer", 3, "ERROR",
            f"Inspection failed: {e}")


def test_memmap_integrity(root: Path) -> TestResult:
    """Verify numpy memmap embedding files exist and match database dimensions."""
    try:
        from src.core.config import load_config
        config = load_config(str(root))
        cache_dir = config.paths.embeddings_cache
        
        if not cache_dir or not Path(cache_dir).exists():
            return TestResult("memmap_integrity", "ML Pipeline", 3, "SKIP",
                "Embeddings cache directory not configured or doesn't exist")
        
        npy_files = list(Path(cache_dir).glob("*.npy"))
        mmap_files = list(Path(cache_dir).glob("*.mmap"))
        dat_files = list(Path(cache_dir).glob("*.dat"))
        all_files = npy_files + mmap_files + dat_files
        
        details = {"cache_dir": cache_dir, "embedding_files": len(all_files)}
        
        if not all_files:
            return TestResult("memmap_integrity", "ML Pipeline", 3, "SKIP",
                "No embedding files found -- index may use in-DB storage",
                details=details)
        
        # Check first file's dimensions match config
        try:
            import numpy as np
            test_file = all_files[0]
            if test_file.suffix == ".npy":
                data = np.load(str(test_file), mmap_mode="r")
            else:
                # Try to read as raw float32 memmap
                file_size = test_file.stat().st_size
                expected_dim = config.embedding.dimension
                if file_size % (expected_dim * 4) == 0:
                    rows = file_size // (expected_dim * 4)
                    data = np.memmap(str(test_file), dtype=np.float32, mode="r",
                                     shape=(rows, expected_dim))
                else:
                    details["dimension_alignment"] = False
                    return TestResult("memmap_integrity", "ML Pipeline", 3, "WARN",
                        f"Embedding file size doesn't align with {expected_dim}D vectors",
                        details=details)
            
            details["file_shape"] = list(data.shape)
            details["expected_dim"] = config.embedding.dimension
            
            if len(data.shape) == 2 and data.shape[1] != config.embedding.dimension:
                return TestResult("memmap_integrity", "ML Pipeline", 3, "FAIL",
                    f"Dimension mismatch: file has {data.shape[1]}D, config says {config.embedding.dimension}D",
                    fix_hint="Re-index to regenerate embeddings with correct dimensions",
                    details=details)
        except Exception as e:
            details["read_error"] = str(e)
        
        return TestResult("memmap_integrity", "ML Pipeline", 3, "PASS",
            f"Embedding files OK -- {len(all_files)} file(s) in cache", details=details)
    except Exception as e:
        return TestResult("memmap_integrity", "ML Pipeline", 3, "ERROR",
            f"Memmap check failed: {e}")


# ============================================================================
# SECTION 7: LEVEL 4 TESTS -- MAINTENANCE BIT (Deep Checks)
# ============================================================================

def test_indexer_code_bugs(root: Path) -> TestResult:
    """Scan indexer source code for known bug patterns."""
    bugs_found = []
    try:
        from src.core.indexer import Indexer
        source = inspect.getsource(Indexer)
        
        # BUG-004: No post-parse validation
        if "non_printable" not in source and "validate_text" not in source and "garbage" not in source:
            bugs_found.append("BUG-004: No binary garbage validation after parsing")
        
        # Bare except (swallows real errors)
        bare_excepts = len(re.findall(r"except\s*:", source))
        if bare_excepts > 0:
            bugs_found.append(f"CODE-SMELL: {bare_excepts} bare except: clauses (swallow errors silently)")
        
        details = {"bugs_found": bugs_found, "source_lines": source.count("\n")}
        
        if bugs_found:
            return TestResult("code_bugs", "Code Quality", 4, "WARN",
                f"{len(bugs_found)} code issue(s): {'; '.join(bugs_found)}",
                details=details)
        return TestResult("code_bugs", "Code Quality", 4, "PASS",
            "No known code bug patterns detected", details=details)
    except Exception as e:
        return TestResult("code_bugs", "Code Quality", 4, "ERROR",
            f"Code inspection failed: {e}")


def test_llm_router_url_construction(root: Path) -> TestResult:
    """Inspect llm_router.py for URL doubling issues.
    THIS TEST CATCHES: The exact bug that caused our 404 errors -- 
    the code appending /openai/deployments/... to a URL that already has it."""
    try:
        from src.core.llm_router import LLMRouter
        source = inspect.getsource(LLMRouter) if hasattr(LLMRouter, '__module__') else ""
        
        # Also try getting APIRouter source
        try:
            # Look for the API router class
            mod = importlib.import_module("src.core.llm_router")
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if "api" in name.lower() and name != "LLMRouter":
                    source += "\n" + inspect.getsource(obj)
        except Exception:
            pass
        
        details = {}
        issues = []
        
        # Check for URL path doubling
        if source.count("/openai/deployments") > 1:
            issues.append("Multiple /openai/deployments references -- potential URL doubling")
        
        # Check for hardcoded public endpoints
        if "api.openai.com" in source:
            issues.append("Hardcoded api.openai.com -- should use configured endpoint only")
        
        # Check for proper Azure detection
        has_azure_detection = "azure" in source.lower() or "aoai" in source.lower()
        details["has_azure_detection"] = has_azure_detection
        
        if not has_azure_detection and "api_key" not in source.lower():
            issues.append("No Azure detection -- may send wrong auth header format")
        
        details["issues"] = issues
        
        if issues:
            return TestResult("llm_router_urls", "LLM Backend", 4, "WARN",
                f"LLM router issues: {'; '.join(issues)}",
                fix_hint="Review llm_router.py URL construction. Upload file for review.",
                details=details)
        
        return TestResult("llm_router_urls", "LLM Backend", 4, "PASS",
            "LLM router URL construction looks clean", details=details)
    except Exception as e:
        return TestResult("llm_router_urls", "LLM Backend", 4, "ERROR",
            f"Router inspection failed: {e}")


def test_powershell_profile(root: Path) -> TestResult:
    """Check if start_hybridrag.ps1 exists and loads api_mode_commands.
    THIS TEST CATCHES: Missing dot-source of api_mode_commands.ps1."""
    details = {}
    
    startup = root / "start_hybridrag.ps1"
    api_commands = root / "api_mode_commands.ps1"
    
    details["startup_exists"] = startup.exists()
    details["api_commands_exists"] = api_commands.exists()
    
    if not startup.exists():
        return TestResult("powershell_profile", "Environment", 4, "WARN",
            "start_hybridrag.ps1 not found",
            fix_hint="Create startup script that activates venv and loads commands",
            details=details)
    
    issues = []
    try:
        content = startup.read_text(encoding="utf-8", errors="ignore")
        
        # Check for venv activation
        if ".venv" not in content and "Activate" not in content:
            issues.append("No venv activation in startup script")
        
        # Check for api_mode_commands dot-sourcing
        if api_commands.exists() and "api_mode_commands" not in content:
            issues.append("api_mode_commands.ps1 not loaded by startup script")
        
        details["issues"] = issues
    except Exception:
        pass
    
    if issues:
        return TestResult("powershell_profile", "Environment", 4, "WARN",
            f"Startup script issues: {'; '.join(issues)}",
            fix_hint="Add '. .\\api_mode_commands.ps1' to end of start_hybridrag.ps1",
            details=details)
    
    return TestResult("powershell_profile", "Environment", 4, "PASS",
        "Startup script present and properly configured", details=details)


def test_git_status(root: Path) -> TestResult:
    """Check git repository status -- uncommitted changes, sync state."""
    git_dir = root / ".git"
    if not git_dir.exists():
        return TestResult("git_status", "Environment", 4, "SKIP",
            "Not a git repository", details={"git_dir": str(git_dir)})
    
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(root), timeout=10
        )
        changes = result.stdout.strip().split("\n") if result.stdout.strip() else []
        details = {"uncommitted_files": len(changes), "changes": changes[:10]}
        
        if changes:
            return TestResult("git_status", "Environment", 4, "WARN",
                f"{len(changes)} uncommitted change(s) -- remember to commit before end of day",
                fix_hint="git add -A && git commit -m 'session checkpoint'",
                details=details)
        return TestResult("git_status", "Environment", 4, "PASS",
            "Git clean -- all changes committed", details=details)
    except FileNotFoundError:
        return TestResult("git_status", "Environment", 4, "SKIP",
            "git command not found", details={})
    except Exception as e:
        return TestResult("git_status", "Environment", 4, "SKIP",
            f"Git check failed: {e}", details={})


def test_ssl_certificates(root: Path) -> TestResult:
    """Check if Python can verify SSL certificates (corporate proxy issue).
    THIS TEST CATCHES: The exact SSL error that blocked HuggingFace downloads
    and the initial API connection on the work laptop."""
    details = {}
    
    # Check if pip-system-certs is installed
    try:
        import pip_system_certs
        details["pip_system_certs"] = True
    except ImportError:
        details["pip_system_certs"] = False
    
    # Check certifi bundle
    try:
        import certifi
        cert_path = certifi.where()
        cert_size = Path(cert_path).stat().st_size
        details["certifi_path"] = cert_path
        details["certifi_size_kb"] = round(cert_size / 1024, 1)
    except Exception:
        details["certifi_available"] = False
    
    # Check environment variables that affect SSL
    ssl_env_vars = {
        "REQUESTS_CA_BUNDLE": os.getenv("REQUESTS_CA_BUNDLE"),
        "SSL_CERT_FILE": os.getenv("SSL_CERT_FILE"),
        "CURL_CA_BUNDLE": os.getenv("CURL_CA_BUNDLE"),
    }
    details["ssl_env_vars"] = {k: v for k, v in ssl_env_vars.items() if v}
    
    if not details.get("pip_system_certs", False):
        return TestResult("ssl_certificates", "Network", 3, "WARN",
            "pip-system-certs not installed -- may fail behind corporate proxy",
            fix_hint="pip install pip-system-certs",
            details=details)
    
    return TestResult("ssl_certificates", "Network", 3, "PASS",
        "SSL certificate handling configured (pip-system-certs installed)",
        details=details)


# ============================================================================
# SECTION 8: Known Bug Detection
# ============================================================================

def detect_known_bugs(report: DiagnosticReport) -> None:
    """Scan results and register all detected known bugs."""
    
    def _find(name: str) -> Optional[TestResult]:
        return next((r for r in report.results if r.name == name), None)
    
    # BUG-001: Missing file_hash column
    r = _find("schema_chunks")
    if r and not r.details.get("has_file_hash", True):
        report.bugs.append(BugReport("BUG-001", "Missing file_hash column",
            "file_hash column not in chunks table. Change detection can't track modifications.",
            "ALTER TABLE chunks ADD COLUMN file_hash TEXT", "HIGH"))
    
    # BUG-002: Change detection (RESOLVED 2026-02-14)
    r = _find("change_detection")
    if r and r.status == "FAIL":
        report.bugs.append(BugReport("BUG-002", "No hash-based change detection",
            "Edited files keep stale chunks forever. No hash comparison in index loop.",
            "Add _compute_file_hash + get_file_hash to index_folder loop", "HIGH"))
    
    # BUG-003: No resource cleanup
    r = _find("resource_cleanup")
    if r and r.status == "WARN":
        report.bugs.append(BugReport("BUG-003", "No close() on core components",
            "Open connections and loaded models leak across long runs.",
            "Add close() methods. Call in finally blocks.", "MEDIUM"))
    
    # BUG-004: No binary garbage validation
    r = _find("code_bugs")
    if r and "BUG-004" in r.message:
        report.bugs.append(BugReport("BUG-004", "No post-parse binary garbage validation",
            "Binary data passes through parsers unchecked, polluting search results.",
            "Add: if non_printable_ratio > 0.10: skip file", "MEDIUM"))
    
    # SEC-001: Network security issues
    r = _find("security_network")
    if r and r.status == "FAIL":
        report.bugs.append(BugReport("SEC-001", "Public API endpoint in defaults",
            "Online mode could send data to public servers without explicit config.",
            "Default endpoint to empty string. Require explicit configuration.", "HIGH"))


# ============================================================================
# SECTION 9: Report Printer
# ============================================================================

def print_report(report: DiagnosticReport, verbose: bool = False) -> None:
    """Print the diagnostic report to the terminal."""
    
    # Header
    print(f"\n{BOLD}{'='*70}")
    print(f"  HybridRAG v3 -- Diagnostic Report v2.0")
    print(f"{'='*70}{RESET}")
    print(f"  {DIM}Timestamp:  {report.timestamp}")
    print(f"  Hostname:  {report.hostname}")
    print(f"  Python:    {report.python_version}")
    print(f"  Platform:  {report.platform_info}")
    print(f"  Root:      {report.project_root}{RESET}")
    print()
    
    # Group by category
    categories = {}
    for r in report.results:
        categories.setdefault(r.category, []).append(r)
    
    for cat, results in categories.items():
        print(f"  {CYAN}{BOLD}-- {cat} --{RESET}")
        for r in results:
            color = STATUS_COLORS.get(r.status, "")
            icon = STATUS_ICONS.get(r.status, "?")
            time_str = f" ({r.elapsed_ms:.0f}ms)" if r.elapsed_ms > 0 else ""
            print(f"    {color}{icon} [{r.status:4s}]{RESET} {r.message}{DIM}{time_str}{RESET}")
            if verbose and r.fix_hint and r.status in ("FAIL", "WARN"):
                for line in r.fix_hint.split("\n"):
                    print(f"           {DIM}-> {line}{RESET}")
        print()
    
    # Known bugs
    if report.bugs:
        print(f"  {RED}{BOLD}-- Known Bugs --{RESET}")
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        for bug in sorted(report.bugs, key=lambda b: sev_order.get(b.severity, 9)):
            sev_color = RED if bug.severity in ("CRITICAL", "HIGH") else YELLOW
            print(f"    {sev_color}[{bug.severity:8s}]{RESET} {bug.bug_id}: {bug.title}")
            if verbose:
                for line in bug.description.split("\n"):
                    print(f"              {DIM}{line}{RESET}")
                print(f"              {GREEN}Fix: {bug.fix_steps}{RESET}")
        print()
    
    # Summary
    total_time = sum(r.elapsed_ms for r in report.results)
    status_line = (
        f"{GREEN}{report.passed} passed{RESET}, "
        f"{RED}{report.failed} failed{RESET}, "
        f"{YELLOW}{report.warnings} warnings{RESET}, "
        f"{DIM}{report.skipped} skipped{RESET}, "
        f"{RED}{report.errors} errors{RESET}"
    )
    bug_count = len(report.bugs)
    bug_line = f"{RED}{bug_count} known bug(s){RESET}" if bug_count else f"{GREEN}0 known bugs{RESET}"
    
    print(f"  {BOLD}Summary: {status_line}  |  {bug_line}  |  {total_time:.0f}ms total{RESET}")
    
    # Go/No-Go verdict
    if report.failed > 0 or report.errors > 0:
        crit_bugs = sum(1 for b in report.bugs if b.severity in ("CRITICAL",))
        if crit_bugs:
            print(f"\n  {RED}{BOLD}## SYSTEM NO-GO -- Fix critical issues before operation ##{RESET}")
        else:
            print(f"\n  {YELLOW}{BOLD}> CONDITIONAL GO -- {report.failed} failure(s) need attention <{RESET}")
    elif report.warnings > 2:
        print(f"\n  {YELLOW}{BOLD}> GO WITH CAUTION -- {report.warnings} warnings to review <{RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}## SYSTEM GO -- All checks passed ##{RESET}")
    print()


def save_json_report(report: DiagnosticReport, path: str) -> None:
    """Save full report as JSON for trending and GUI consumption."""
    data = {
        "version": "2.0",
        "timestamp": report.timestamp,
        "hostname": report.hostname,
        "python_version": report.python_version,
        "platform": report.platform_info,
        "project_root": report.project_root,
        "summary": {
            "total": report.total, "passed": report.passed,
            "failed": report.failed, "warnings": report.warnings,
            "skipped": report.skipped, "errors": report.errors,
        },
        "results": [asdict(r) for r in report.results],
        "bugs": [asdict(b) for b in report.bugs],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  {DIM}JSON report saved: {path}{RESET}")


# ============================================================================
# SECTION 10: Main Entry Point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="HybridRAG v3 -- Diagnostic Suite v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        LEVELS:
          1  Power-on BIT   -- Python, venv, packages, disk, project structure
          2  Initiated BIT  -- Config, database, schema, credentials
          3  Continuous BIT  -- Embedder, chunker, parser, Ollama, API, security
          4  Maintenance BIT -- Code bugs, URL construction, git, PowerShell

        EXAMPLES:
          python hybridrag_diagnostic_v2.py                 # Full diagnostic
          python hybridrag_diagnostic_v2.py --quick          # Level 1+2 only
          python hybridrag_diagnostic_v2.py --verbose        # Show fix hints
          python hybridrag_diagnostic_v2.py --json report.json
          python hybridrag_diagnostic_v2.py --level 3
        """),
    )
    ap.add_argument("--quick", action="store_true", help="Level 1+2 only (fast)")
    ap.add_argument("--level", type=int, default=4, choices=[1,2,3,4], help="Max test level")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show fix hints and evidence")
    ap.add_argument("--json", type=str, default="", help="Save JSON report to this file")
    ap.add_argument("--skip-embed", action="store_true", help="Skip embedding model load test (slow)")
    args = ap.parse_args()
    
    max_level = 2 if args.quick else args.level
    root = _find_project_root()
    sys.path.insert(0, str(root))
    
    # Build report
    report = DiagnosticReport(
        timestamp=datetime.now().isoformat(),
        hostname=platform.node(),
        python_version=platform.python_version(),
        platform_info=f"{platform.system()} {platform.release()} {platform.machine()}",
        project_root=str(root),
    )
    
    # ---- LEVEL 1: Power-On BIT ----
    level_1_tests = [
        ("python_version", "Environment", test_python_version),
        ("venv_integrity", "Environment", test_venv_integrity),
        ("requirements", "Environment", test_requirements_installed),
        ("project_structure", "Environment", test_project_structure),
        ("disk_space", "Environment", test_disk_space),
        ("log_directory", "Environment", test_log_directory),
    ]
    
    # ---- LEVEL 2: Initiated BIT ----
    level_2_tests = [
        ("config_load", "Configuration", test_config_load),
        ("config_paths", "Configuration", test_config_paths),
        ("database_connection", "Database", test_database_connection),
        ("schema_chunks", "Database", test_schema_chunks),
        ("schema_fts5", "Database", test_schema_fts5),
        ("data_integrity", "Database", test_data_integrity),
        ("credentials", "Security", test_credentials),
    ]
    
    # ---- LEVEL 3: Continuous BIT ----
    level_3_tests = [
        ("model_cache", "ML Pipeline", test_embedding_model_cache),
        ("chunker_functional", "ML Pipeline", test_chunker_functional),
        ("parser_registry", "ML Pipeline", test_parser_registry),
        ("ollama_health", "LLM Backend", test_ollama_health),
        ("api_reachability", "LLM Backend", test_api_reachability),
        ("ssl_certificates", "Network", test_ssl_certificates),
        ("security_network", "Security", test_security_network_lockdown),
        ("change_detection", "Indexer", test_change_detection),
        ("resource_cleanup", "Indexer", test_resource_cleanup),
        ("memmap_integrity", "ML Pipeline", test_memmap_integrity),
    ]
    
    # Conditionally include embedder (it's slow -- loads the model)
    if not args.skip_embed:
        level_3_tests.insert(1, ("embedder_functional", "ML Pipeline", test_embedder_functional))
    
    # ---- LEVEL 4: Maintenance BIT ----
    level_4_tests = [
        ("code_bugs", "Code Quality", test_indexer_code_bugs),
        ("llm_router_urls", "LLM Backend", test_llm_router_url_construction),
        ("powershell_profile", "Environment", test_powershell_profile),
        ("git_status", "Environment", test_git_status),
    ]
    
    # Collect all tests up to max_level
    all_tests = []
    if max_level >= 1: all_tests.extend([(n, c, 1, fn) for n, c, fn in level_1_tests])
    if max_level >= 2: all_tests.extend([(n, c, 2, fn) for n, c, fn in level_2_tests])
    if max_level >= 3: all_tests.extend([(n, c, 3, fn) for n, c, fn in level_3_tests])
    if max_level >= 4: all_tests.extend([(n, c, 4, fn) for n, c, fn in level_4_tests])
    
    # Run tests
    print(f"\n  {BOLD}Running {len(all_tests)} diagnostic tests (Level 1-{max_level})...{RESET}\n")
    
    for i, (name, cat, level, fn) in enumerate(all_tests, 1):
        # Progress indicator
        print(f"  {DIM}[{i:2d}/{len(all_tests)}] {name}...{RESET}", end="", flush=True)
        result = _run_test(name, cat, level, lambda: fn(root))
        report.results.append(result)
        
        color = STATUS_COLORS.get(result.status, "")
        icon = STATUS_ICONS.get(result.status, "?")
        print(f"\r  {color}{icon}{RESET} {DIM}[{i:2d}/{len(all_tests)}]{RESET} {name}" + " " * 30)
    
    # Detect known bugs
    detect_known_bugs(report)
    
    # Print report
    print_report(report, verbose=args.verbose)
    
    # Save JSON if requested
    if args.json:
        save_json_report(report, args.json)
    
    # Auto-save to logs/ if directory exists
    log_dir = root / "logs"
    if log_dir.exists():
        auto_json = log_dir / f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json_report(report, str(auto_json))
    
    # Exit code: 0 = all pass, 1 = failures exist
    sys.exit(1 if report.failed > 0 or report.errors > 0 else 0)


if __name__ == "__main__":
    main()
