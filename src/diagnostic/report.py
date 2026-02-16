# ============================================================================
# HybridRAG v3 -- Diagnostic: Bug Detection & Report Printer
# ============================================================================
# FILE: src/diagnostic/report.py
#
# WHAT THIS FILE DOES:
#   1. detect_known_bugs() -- scans test results and flags known issues
#   2. print_report()      -- color-coded terminal output
#   3. save_json_report()  -- JSON file for automated trend tracking
#
# WHY IT'S SEPARATE:
#   Presentation logic shouldn't be mixed with test logic. This file
#   only reads from DiagnosticReport -- it never modifies test behavior.
# ============================================================================

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict
from typing import Optional

from . import (
    DiagnosticReport, TestResult, PerfMetric,
    GREEN, RED, YELLOW, CYAN, BOLD, DIM, RESET,
    status_color, status_icon, get_memory_mb,
)


# ============================================================================
# Known bug detection
# ============================================================================

def detect_known_bugs(report: DiagnosticReport) -> None:
    """Scan test results and register all detected known bugs."""

    def _find(name: str) -> Optional[TestResult]:
        return next((r for r in report.results if r.name == name), None)

    # BUG-001: Missing file_hash column
    r = _find("schema_chunks")
    if r and not r.details.get("has_file_hash", True):
        report.add_bug("BUG-001", "Missing file_hash column in chunks table",
            "_file_changed() queries file_hash but column doesn't exist.\n"
            "Bare except swallows the error -- method always returns True.",
            "1. ALTER TABLE chunks ADD COLUMN file_hash TEXT;\n"
            "2. Store hash during indexing\n"
            "3. Wire _file_changed() into skip logic", "HIGH")

    # BUG-002: _file_changed() never called
    r = _find("change_detection")
    if r and r.status in ("FAIL", "WARN"):
        report.add_bug("BUG-002", "_file_changed() exists but never called",
            "index_folder() only checks chunk existence, not file modification.\n"
            "Edited files keep stale chunks forever.",
            "if already_indexed AND NOT file_changed: skip\n"
            "elif already_indexed AND file_changed: delete old, re-index", "HIGH")

    # BUG-003: No resource cleanup
    r = _find("resource_cleanup")
    if r and r.status == "WARN":
        report.add_bug("BUG-003", "No close() on VectorStore / Embedder",
            "Open connections and loaded models leak across long runs.",
            "Add close() methods. Call in finally block.", "MEDIUM")

    # BUG-004: Binary garbage detection
    # The preflight gate (_preflight_check) and _validate_text() together
    # catch corrupt files. Only flag BUG-004 if the preflight gate is missing.
    r = _find("change_detection")
    has_preflight = (r and r.details.get("has_preflight_check", False))
    if not has_preflight:
        report.add_bug("BUG-004", "No pre-parse binary garbage validation",
            "If a parser returns binary data, it gets chunked and embedded\n"
            "without validation, polluting search results.",
            "Update indexer.py to include _preflight_check()", "MEDIUM")

    # SEC-001: Public API default
    r = _find("security_endpoint")
    if r and r.status in ("FAIL", "WARN"):
        report.add_bug("SEC-001", "API endpoint defaults to public OpenAI",
            "Online mode without explicit config sends data externally.\n"
            "Data exfiltration risk in restricted environments.",
            "1. Default endpoint to empty string\n"
            "2. Add URL allowlist\n"
            "3. Validate before HTTP calls\n"
            "4. Add network kill switch", "CRITICAL")


# ============================================================================
# Terminal report printer
# ============================================================================

def print_report(report: DiagnosticReport, verbose: bool = False) -> None:
    """Print formatted diagnostic report to terminal."""
    W = 72
    sep = "-" * W

    print(f"\n  {BOLD}{'=' * W}{RESET}")
    print(f"  {BOLD}  HybridRAG Diagnostic & Performance Report{RESET}")
    print(f"  {DIM}  {report.timestamp} | Python {report.python_version} | {report.platform}{RESET}")
    print(f"  {BOLD}{'=' * W}{RESET}")

    # --- Health checks ---
    if report.results:
        print(f"\n  {BOLD}{CYAN}PIPELINE HEALTH{RESET}")
        print(f"  {sep}")
        cat = ""
        for r in report.results:
            if r.category != cat:
                cat = r.category
                print(f"\n  {DIM}-- {cat} --{RESET}")
            c = status_color(r.status)
            ic = status_icon(r.status)
            t = f" {DIM}({r.elapsed_ms:.0f}ms){RESET}" if r.elapsed_ms > 0 else ""
            print(f"  {c}{ic}{RESET} {r.message}{t}")
            if verbose and r.details:
                for k, v in r.details.items():
                    vs = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
                    print(f"  {DIM}       {k}: {vs[:100]}{RESET}")
            if r.fix_hint and r.status in ("FAIL", "WARN", "ERROR"):
                for ln in textwrap.wrap(r.fix_hint, width=W - 10)[:3]:
                    print(f"  {YELLOW}       -> {ln}{RESET}")

    # --- Performance ---
    if report.perf_metrics:
        print(f"\n  {BOLD}{CYAN}PERFORMANCE BENCHMARKS{RESET}")
        print(f"  {sep}")
        for m in report.perf_metrics:
            if m.value == 0 and "skip" in m.details:
                print(f"  {DIM}  {m.name}: skipped ({m.details.get('skip', '')}){RESET}")
                continue
            if "error" in m.details:
                print(f"  {RED}  {m.name}: ERROR ({m.details['error'][:60]}){RESET}")
                continue
            if m.unit in ("ms", "ms/query"):
                vs = f"{m.avg_val:.1f} ms"
            elif "/sec" in m.unit:
                vs = f"{m.avg_val:,.0f} {m.unit}"
            else:
                vs = f"{m.avg_val:.2f} {m.unit}"
            spread = ""
            if m.iterations > 1 and m.min_val != m.max_val:
                if m.unit in ("ms", "ms/query"):
                    spread = f" {DIM}(min={m.min_val:.1f} max={m.max_val:.1f} n={m.iterations}){RESET}"
                else:
                    spread = f" {DIM}(min={m.min_val:,.0f} max={m.max_val:,.0f} n={m.iterations}){RESET}"
            mem = ""
            if m.memory_before_mb > 0 and m.memory_after_mb > 0:
                delta = m.memory_after_mb - m.memory_before_mb
                if abs(delta) > 1.0:
                    mem = f" {DIM}[RAM: {'+' if delta > 0 else ''}{delta:.0f}MB]{RESET}"
            label = m.name.replace("_", " ").title()
            dots = "." * max(1, 40 - len(label))
            print(f"  {GREEN}  {label} {dots} {vs}{spread}{mem}{RESET}")

    # --- Known bugs ---
    if report.known_bugs:
        print(f"\n  {BOLD}{CYAN}KNOWN BUGS DETECTED{RESET}")
        print(f"  {sep}")
        for b in report.known_bugs:
            sc = RED if b["severity"] in ("CRITICAL", "HIGH") else YELLOW
            print(f"  {sc}[{b['id']}] {b['title']} ({b['severity']}){RESET}")
            if verbose:
                for ln in b["description"].split("\n"):
                    print(f"  {DIM}    {ln}{RESET}")
                print(f"  {YELLOW}    FIX:{RESET}")
                for ln in b["fix"].split("\n"):
                    print(f"  {YELLOW}      {ln}{RESET}")
                print()

    # --- Summary ---
    print(f"\n  {BOLD}{'=' * W}{RESET}")
    pc = GREEN if report.passed == report.total_tests else ""
    fc = RED if report.failed > 0 else GREEN
    wc = YELLOW if report.warnings > 0 else GREEN
    print(f"  {BOLD}SUMMARY:{RESET} "
          f"{pc}{report.passed} passed{RESET}, {fc}{report.failed} failed{RESET}, "
          f"{wc}{report.warnings} warnings{RESET}, {report.skipped} skipped, "
          f"{report.errors} errors {DIM}({report.total_elapsed_ms:.0f}ms){RESET}")
    if report.known_bugs:
        cr = sum(1 for b in report.known_bugs if b["severity"] == "CRITICAL")
        hi = sum(1 for b in report.known_bugs if b["severity"] == "HIGH")
        print(f"  {BOLD}BUGS:{RESET} {RED}{cr} critical{RESET}, "
              f"{RED}{hi} high{RESET}, {len(report.known_bugs)-cr-hi} medium/low")
    mem = get_memory_mb()
    if mem > 0:
        print(f"  {BOLD}MEMORY:{RESET} {mem:.0f} MB RSS at end of diagnostic")

    if report.failed == 0 and not any(b["severity"] == "CRITICAL" for b in report.known_bugs):
        print(f"\n  {GREEN}{BOLD}  [OK] SYSTEM GO -- safe to proceed{RESET}")
    else:
        print(f"\n  {RED}{BOLD}  [FAIL] SYSTEM NO-GO -- fix critical issues first{RESET}")
    print(f"  {BOLD}{'=' * W}{RESET}\n")


# ============================================================================
# JSON export
# ============================================================================

def save_json_report(report: DiagnosticReport, path: str) -> None:
    """Save report as JSON for automated trend tracking."""
    data = {
        "timestamp": report.timestamp,
        "python_version": report.python_version,
        "platform": report.platform,
        "summary": {
            "total": report.total_tests, "passed": report.passed,
            "failed": report.failed, "warnings": report.warnings,
            "skipped": report.skipped, "errors": report.errors,
            "elapsed_ms": report.total_elapsed_ms,
        },
        "results": [asdict(r) for r in report.results],
        "performance": [asdict(m) for m in report.perf_metrics],
        "known_bugs": report.known_bugs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  {DIM}JSON saved: {path}{RESET}")
