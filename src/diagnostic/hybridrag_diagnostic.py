#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Diagnostic Tool (Entry Point)
# ============================================================================
# FILE: src/diagnostic/hybridrag_diagnostic.py
#
# HOW TO RUN:
#   . .\start_hybridrag.ps1
#   rag-diag                                              # Quick check + fault analysis
#   rag-diag --verbose                                    # Full details + evidence trail
#   rag-diag --test-embed                                 # Embedding benchmark
#   rag-diag --test-query "freq range"
#   rag-diag --test-parse "C:\docs\manual.pdf"
#   rag-diag --json-file report.json                      # Trend tracking
#   rag-diag --fix-preview --verbose                      # Bug fix details
#   rag-diag --perf-only                                  # Benchmarks only
#   rag-diag --no-fault-analysis                          # Skip fault analysis
#
# WHAT'S NEW (2026-02-09):
#   - FAULT ANALYSIS ENGINE: After all tests run, the system examines
#     combined results and produces a ranked list of the 3 most likely
#     root causes, each with a recommended next diagnostic step.
#   - Parser smoke tests added to catch missing dependencies early.
#   - Fault analysis logs written to logs/fault_analysis_*.json for
#     future GUI admin panel consumption.
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import textwrap
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.diagnostic import DiagnosticReport, run_test, CYAN, DIM, RESET, GREEN, RED, BOLD
from src.diagnostic.health_tests import (
    test_config_load, test_config_paths,
    test_sqlite_connection, test_schema_chunks_table,
    test_schema_fts5, test_data_integrity,
    test_indexer_change_detection, test_resource_cleanup,
)
from src.diagnostic.component_tests import (
    test_parser_registry, test_parse_file,
    test_chunker, test_embedder, test_memmap_store,
    test_security_endpoint, test_security_network,
    test_parser_smoke,
)
from src.diagnostic.perf_benchmarks import (
    perf_config_load, perf_sqlite_query, perf_chunker,
    perf_embedder, perf_vector_search, perf_fts5_search,
    perf_hybrid_search,
)
from src.diagnostic.report import detect_known_bugs, print_report, save_json_report
from src.diagnostic.fault_analysis import (
    run_fault_analysis, print_fault_analysis, log_fault_analysis,
)


def main():
    ap = argparse.ArgumentParser(
        description="HybridRAG v3 -- Diagnostic, Pipeline Test & Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        EXAMPLES:
          rag-diag                                    # Quick check + fault analysis
          rag-diag --verbose                          # Full details + evidence trail
          rag-diag --test-embed                       # With embedding benchmark
          rag-diag --test-query "freq range"          # End-to-end query test
          rag-diag --test-parse "C:\\docs\\m.pdf"      # Test specific file
          rag-diag --json-file report.json            # Save for trend tracking
          rag-diag --fix-preview --verbose            # Bug fix details
          rag-diag --perf-only                        # Benchmarks only (no health)
          rag-diag --no-fault-analysis                # Skip fault analysis
        """))
    ap.add_argument("--verbose", "-v", action="store_true", help="Detailed output + evidence trail")
    ap.add_argument("--test-embed", action="store_true",
                    help="Live embedding benchmark (loads model, ~30s)")
    ap.add_argument("--test-query", type=str, default="",
                    help="End-to-end query test")
    ap.add_argument("--test-parse", type=str, default="",
                    help="Test parsing a specific file")
    ap.add_argument("--json", action="store_true", help="JSON to stdout")
    ap.add_argument("--json-file", type=str, default="",
                    help="Save JSON for trend tracking")
    ap.add_argument("--perf-only", action="store_true", help="Skip health checks")
    ap.add_argument("--fix-preview", action="store_true",
                    help="Show known bug details and fixes")
    ap.add_argument("--benchmark-iters", type=int, default=3,
                    help="Benchmark iterations (default: 3)")
    ap.add_argument("--no-fault-analysis", action="store_true",
                    help="Skip fault analysis (faster)")
    args = ap.parse_args()

    report = DiagnosticReport()
    report.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.python_version = sys.version.split()[0]
    report.platform = sys.platform
    report.cwd = str(Path.cwd())
    iters = args.benchmark_iters

    # -- HEALTH CHECKS --
    if not args.perf_only:
        # Tests run in dependency order: config -> database -> indexer ->
        # parsers -> embedding -> storage -> security
        # This order matters because later tests depend on earlier ones.
        health_tests = [
            ("Config Load",        test_config_load),
            ("Config Paths",       test_config_paths),
            ("SQLite Connection",  test_sqlite_connection),
            ("Schema: chunks",     test_schema_chunks_table),
            ("Schema: FTS5",       test_schema_fts5),
            ("Data Integrity",     test_data_integrity),
            ("Change Detection",   test_indexer_change_detection),
            ("Resource Cleanup",   test_resource_cleanup),
            ("Parser Registry",    test_parser_registry),
            ("Parser Smoke Test",  test_parser_smoke),
            ("Chunker",            test_chunker),
            ("Embedder",           lambda: test_embedder(args.test_embed)),
            ("Memmap Store",       test_memmap_store),
            ("Security: Endpoint", test_security_endpoint),
            ("Security: Network",  test_security_network),
        ]
        if args.test_parse:
            health_tests.insert(10, ("Parse File", lambda: test_parse_file(args.test_parse)))

        total_h = len(health_tests)
        print(f"\n  {CYAN}Running {total_h} pipeline health checks...{RESET}")
        h_start = time.time()

        for i, (label, func) in enumerate(health_tests, 1):
            elapsed = time.time() - h_start
            avg_per = elapsed / i if i > 1 else 0
            remaining = avg_per * (total_h - i)
            eta = f"~{remaining:.0f}s left" if i > 1 else "estimating..."
            progress = f"  [{i}/{total_h}] {label}... ({eta})"
            print(f"\r{progress:<70}", end="", flush=True)
            report.add_result(run_test(func))

        elapsed_h = time.time() - h_start
        print(f"\r  {GREEN}[OK] {total_h} health checks done in {elapsed_h:.1f}s{' ' * 30}{RESET}")

    # -- PERFORMANCE BENCHMARKS --
    perf_tests = [
        ("Config Load",    perf_config_load),
        ("SQLite Query",   perf_sqlite_query),
        ("Chunker",        perf_chunker),
    ]
    if args.test_embed:
        perf_tests.append(("Embedder", perf_embedder))
    perf_tests.append(("Vector Search", perf_vector_search))
    perf_tests.append(("FTS5 Search",   perf_fts5_search))
    if args.test_embed:
        perf_tests.append(("Hybrid Search", perf_hybrid_search))

    total_p = len(perf_tests)
    print(f"\n  {CYAN}Running {total_p} performance benchmarks ({iters} iterations each)...{RESET}")
    p_start = time.time()

    for i, (label, func) in enumerate(perf_tests, 1):
        elapsed = time.time() - p_start
        avg_per = elapsed / i if i > 1 else 0
        remaining = avg_per * (total_p - i)
        eta = f"~{remaining:.0f}s left" if i > 1 else "estimating..."
        print(f"\r  [{i}/{total_p}] {label}... ({eta}){' ' * 20}", end="", flush=True)
        _safe_perf(report, func, iters)

    elapsed_p = time.time() - p_start
    print(f"\r  {GREEN}[OK] {total_p} benchmarks done in {elapsed_p:.1f}s{' ' * 30}{RESET}")

    # -- END-TO-END QUERY TEST --
    if args.test_query:
        _run_e2e_query(report, args.test_query)

    # -- KNOWN BUG DETECTION --
    if not args.perf_only:
        detect_known_bugs(report)

    # -- OUTPUT: STANDARD REPORT --
    if args.json:
        data = {"timestamp": report.timestamp,
                "summary": {"total": report.total_tests, "passed": report.passed,
                             "failed": report.failed, "warnings": report.warnings},
                "results": [asdict(r) for r in report.results],
                "performance": [asdict(m) for m in report.perf_metrics],
                "bugs": report.known_bugs}
        print(json.dumps(data, indent=2, default=str))
    else:
        print_report(report, verbose=args.verbose or args.fix_preview)

    # -- FAULT ANALYSIS ENGINE --
    # The "smart mechanic": examines ALL results together, correlates
    # failures across subsystems, ranks the 3 most likely root causes,
    # and tells you exactly what to check next for each one.
    if not args.no_fault_analysis and not args.perf_only:
        fault_result = run_fault_analysis(report)
        print_fault_analysis(fault_result, verbose=args.verbose)

        # Log to JSON for future GUI admin panel
        try:
            log_path = log_fault_analysis(fault_result)
            if args.verbose:
                print(f"  {DIM}Fault analysis logged: {log_path}{RESET}")
        except Exception as e:
            print(f"  {DIM}Fault log skipped: {e}{RESET}")

    if args.json_file:
        save_json_report(report, args.json_file)

    has_crit = any(b["severity"] == "CRITICAL" for b in report.known_bugs)
    sys.exit(1 if report.failed > 0 or has_crit else 0)


def _safe_perf(report, func, iters):
    """Run a perf benchmark, catching errors gracefully."""
    try:
        m = func(iters)
        if m:
            report.add_perf(m)
    except Exception as e:
        print(f"  {DIM}  {func.__name__} skipped: {e}{RESET}")


def _run_e2e_query(report, query_text):
    """Run a full end-to-end query test."""
    from src.diagnostic import TestResult, PerfMetric, PROJ_ROOT
    print(f"\n  {CYAN}Running end-to-end query test...{RESET}")
    try:
        from src.core.config import load_config, ensure_directories
        from src.core.vector_store import VectorStore
        from src.core.embedder import Embedder
        from src.core.llm_router import LLMRouter
        from src.core.query_engine import QueryEngine

        cfg = load_config(str(PROJ_ROOT))
        ensure_directories(cfg)
        vs = VectorStore(db_path=cfg.paths.database, embedding_dim=cfg.embedding.dimension)
        vs.connect()
        embedder = Embedder(cfg.embedding.model_name)
        llm = LLMRouter(cfg, api_key=os.getenv("OPENAI_API_KEY"))
        qe = QueryEngine(cfg, vs, embedder, llm)

        t0 = time.perf_counter()
        result = qe.query(query_text)
        ms = (time.perf_counter() - t0) * 1000

        report.add_result(TestResult(
            "e2e_query", "Query", "PASS" if not result.error else "FAIL",
            f"Query in {ms:.0f}ms ({result.chunks_used} chunks, mode={result.mode})",
            {"query": query_text, "answer": result.answer[:200],
             "chunks": result.chunks_used, "ms": ms, "error": result.error},
            elapsed_ms=ms))
        report.add_perf(PerfMetric("e2e_query", "Query", ms, "ms/query",
                                   details={"query": query_text}))
    except Exception as e:
        report.add_result(TestResult("e2e_query", "Query", "ERROR", f"{e}"))


if __name__ == "__main__":
    main()
