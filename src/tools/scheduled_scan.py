#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Scheduled Source Integrity Scan
# ============================================================================
# FILE: src/tools/scheduled_scan.py
#
# WHAT THIS DOES (plain English):
#   Runs a deep source file integrity scan and writes the results to a
#   log file. Designed to be called from:
#     - Windows Task Scheduler (weekly)
#     - After rag-index completes
#     - Manually: python src\tools\scheduled_scan.py
#
#   The log file goes to: logs/scan_report_YYYY-MM-DD.txt
#   If critical issues are found, the exit code is 1 (for Task Scheduler
#   alerting). If everything is clean, exit code is 0.
#
# HOW TO SET UP WEEKLY SCHEDULING:
#   1. Open Windows Task Scheduler (taskschd.msc)
#   2. Create Basic Task -> Name: "HybridRAG Source Scan"
#   3. Trigger: Weekly (pick a day)
#   4. Action: Start a program
#      Program: {PROJECT_ROOT}\.venv\Scripts\python.exe
#      Arguments: {PROJECT_ROOT}\src\tools\scheduled_scan.py
#      Start in: {PROJECT_ROOT}
#   5. Check "Run whether user is logged on or not"
#
# USAGE:
#   python src\tools\scheduled_scan.py           # Deep scan + log
#   python src\tools\scheduled_scan.py --fast    # Fast scan only (no parse)
#   python src\tools\scheduled_scan.py --notify  # Print summary to console
#
# PowerShell (after sourcing start_hybridrag.ps1):
#   rag-scan-log                     # Deep scan + log
#   rag-scan-log --fast              # Fast scan only
#
# INTERNET ACCESS: NONE -- all checks are local file operations
# ============================================================================

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Scheduled integrity scan with logging"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast scan only (structure checks, no deep parse)"
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Print summary to console after scan"
    )
    parser.add_argument(
        "--source", "-s",
        help="Source directory to scan (default: HYBRIDRAG_INDEX_FOLDER)"
    )
    args = parser.parse_args()

    # Import the scanner
    try:
        from src.tools.scan_source_files import (
            discover_files, scan_files, Severity
        )
    except ImportError:
        # Direct path fallback
        scan_path = PROJECT_ROOT / "src" / "tools" / "scan_source_files.py"
        if scan_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "scan_source_files", str(scan_path)
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            discover_files = mod.discover_files
            scan_files = mod.scan_files
            Severity = mod.Severity
        else:
            print("[ERROR] scan_source_files.py not found")
            sys.exit(1)

    # Determine source directory
    source_dir = None
    if args.source:
        source_dir = Path(args.source)
    else:
        env_dir = os.environ.get("HYBRIDRAG_INDEX_FOLDER")
        if env_dir:
            source_dir = Path(env_dir)

    if not source_dir or not source_dir.exists():
        msg = f"[ERROR] Source dir not found: {source_dir}"
        print(msg)
        sys.exit(1)

    # Create logs directory
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"scan_report_{timestamp}.txt"

    # Run scan
    deep = not args.fast
    scan_label = "Deep (structure + parse)" if deep else "Fast (structure only)"

    start_time = time.time()
    files = discover_files(source_dir)
    findings = scan_files(files, deep=deep, progress=False)
    elapsed = time.time() - start_time

    # Count by severity
    critical = [f for f in findings if f.severity == Severity.CRITICAL]
    warnings = [f for f in findings if f.severity == Severity.WARNING]

    # Build report
    lines = []
    lines.append("=" * 72)
    lines.append("  HYBRIDRAG3 SOURCE FILE INTEGRITY SCAN REPORT")
    lines.append("=" * 72)
    lines.append(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Source:     {source_dir}")
    lines.append(f"  Scan mode: {scan_label}")
    lines.append(f"  Duration:  {elapsed:.1f} seconds")
    lines.append(f"  Files:     {len(files)} scanned")
    lines.append("")
    lines.append(f"  Problems:  {len(findings)} total")
    lines.append(f"    CRITICAL: {len(critical)}")
    lines.append(f"    WARNING:  {len(warnings)}")
    lines.append("")

    if findings:
        lines.append("-" * 72)
        lines.append("  FINDINGS")
        lines.append("-" * 72)
        for i, f in enumerate(findings, 1):
            lines.append(f"  #{i}")
            lines.append(f"  {f}")
            lines.append("")
        lines.append("-" * 72)
        lines.append("")
        lines.append("  ACTION REQUIRED:")
        lines.append("    Run: rag-scan            (interactive cleanup)")
        lines.append("    Or:  rag-scan --auto-quarantine  (automatic)")
    else:
        lines.append("  All files look healthy. No action needed.")

    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)

    # Write log file
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(report)

    # Console output
    if args.notify or findings:
        print(report)

    if not args.notify and not findings:
        print(f"  [OK] Scan complete: {len(files)} files, "
              f"0 problems. Log: {log_file}")

    if findings:
        print(f"  Log saved: {log_file}")

    # Exit code: 1 if critical issues found (for Task Scheduler alerting)
    sys.exit(1 if critical else 0)


if __name__ == "__main__":
    main()
