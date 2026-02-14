# ============================================================================
# HybridRAG v3 -- Diagnostic Package: Core Data Structures & Helpers
# ============================================================================
# FILE: src/diagnostic/__init__.py
#
# WHAT THIS FILE DOES:
#   Provides the shared building blocks that every diagnostic module uses:
#   - TestResult: outcome of one health check
#   - PerfMetric: one performance measurement with statistics
#   - DiagnosticReport: collects all results, metrics, and bugs
#   - Terminal color helpers for pretty output
#   - Memory measurement utility
#   - Benchmark runner (times a function N times, computes stats)
#   - Safe test wrapper (catches crashes so other tests keep running)
#
# WHY IT'S SEPARATE:
#   Every test module (health_tests.py, perf_benchmarks.py, etc.) needs
#   these same structures. Putting them here avoids circular imports and
#   keeps each file focused on one job.
# ============================================================================

from __future__ import annotations

import os
import sys
import time
import json
import traceback
import statistics
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable

# Project root = two levels up from src/diagnostic/
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TestResult:
    """
    One diagnostic test outcome.

    name:      Short ID like "config_load" (used as JSON key)
    category:  Group like "Config", "Database", "Security"
    status:    PASS | FAIL | WARN | SKIP | ERROR
    message:   One-line human-readable summary
    details:   Extra info (shown in --verbose mode)
    elapsed_ms: How long the test took
    fix_hint:  If FAIL/WARN, suggestion for how to fix
    """
    name: str
    category: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    fix_hint: str = ""


@dataclass
class PerfMetric:
    """
    One performance measurement with timing statistics.

    value_extractor in _benchmark() computes the value from elapsed time.
    min/max/avg/std track variation across iterations.
    memory_before/after detect leaks.
    """
    name: str
    category: str
    value: float
    unit: str
    iterations: int = 1
    min_val: float = 0.0
    max_val: float = 0.0
    avg_val: float = 0.0
    std_dev: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Collects all test results, perf metrics, and known bugs."""
    timestamp: str = ""
    python_version: str = ""
    platform: str = ""
    cwd: str = ""
    results: List[TestResult] = field(default_factory=list)
    perf_metrics: List[PerfMetric] = field(default_factory=list)
    known_bugs: List[Dict[str, str]] = field(default_factory=list)
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    skipped: int = 0
    errors: int = 0
    total_elapsed_ms: float = 0.0

    def add_result(self, r: TestResult):
        self.results.append(r)
        self.total_tests += 1
        _map = {"PASS": "passed", "FAIL": "failed", "WARN": "warnings",
                "SKIP": "skipped", "ERROR": "errors"}
        attr = _map.get(r.status)
        if attr:
            setattr(self, attr, getattr(self, attr) + 1)
        self.total_elapsed_ms += r.elapsed_ms

    def add_perf(self, m: PerfMetric):
        self.perf_metrics.append(m)

    def add_bug(self, bug_id: str, title: str, description: str,
                fix: str, severity: str = "HIGH"):
        self.known_bugs.append({
            "id": bug_id, "title": title, "description": description,
            "fix": fix, "severity": severity,
        })


# ============================================================================
# Terminal color helpers
# ============================================================================

def _supports_color() -> bool:
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True

USE_COLOR = _supports_color()
GREEN  = "\033[92m" if USE_COLOR else ""
RED    = "\033[91m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
CYAN   = "\033[96m" if USE_COLOR else ""
BOLD   = "\033[1m"  if USE_COLOR else ""
DIM    = "\033[2m"  if USE_COLOR else ""
RESET  = "\033[0m"  if USE_COLOR else ""

def status_color(s: str) -> str:
    return {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW,
            "SKIP": DIM, "ERROR": RED}.get(s, "")

def status_icon(s: str) -> str:
    return {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]",
            "SKIP": "[SKIP]", "ERROR": "[ERR ]"}.get(s, "[????]")


# ============================================================================
# Memory measurement
# ============================================================================

def get_memory_mb() -> float:
    """
    Get current process RSS (Resident Set Size) in MB.

    RSS = how much physical RAM this process actually uses right now.
    Falls back to 0.0 if psutil isn't installed.
    """
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


# ============================================================================
# Benchmark runner
# ============================================================================

def benchmark(func: Callable, iterations: int = 3, label: str = "",
              category: str = "", unit: str = "ms",
              value_extractor: Callable = None) -> PerfMetric:
    """
    Run a function N times and collect timing statistics.

    value_extractor(elapsed_seconds, func_return_value) -> metric_value
    If None, reports elapsed milliseconds.
    """
    values = []
    mem_before = get_memory_mb()

    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start
        if value_extractor:
            values.append(value_extractor(elapsed, result))
        else:
            values.append(elapsed * 1000)

    mem_after = get_memory_mb()
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0

    return PerfMetric(
        name=label, category=category, value=avg, unit=unit,
        iterations=iterations, min_val=min(values), max_val=max(values),
        avg_val=avg, std_dev=std,
        memory_before_mb=mem_before, memory_after_mb=mem_after,
    )


# ============================================================================
# Safe test wrapper
# ============================================================================

def run_test(test_func, *args, **kwargs) -> TestResult:
    """
    Run a test safely -- if it crashes, return an ERROR result
    instead of stopping the entire diagnostic.
    """
    start = time.perf_counter()
    try:
        result = test_func(*args, **kwargs)
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result
    except Exception as e:
        return TestResult(
            name=getattr(test_func, "__name__", "unknown"),
            category="Error", status="ERROR",
            message=f"Test crashed: {type(e).__name__}: {e}",
            details={"traceback": traceback.format_exc()},
            elapsed_ms=(time.perf_counter() - start) * 1000,
            fix_hint="Run with --verbose for full traceback.",
        )
