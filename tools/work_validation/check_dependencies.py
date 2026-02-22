#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Dependency Check (check_dependencies.py)
# ============================================================================
# PURPOSE:
#   Pre-flight check for the work laptop. Run this FIRST before any other
#   validation script. Checks:
#     1. PyPI reachability through enterprise proxy
#     2. Ollama installed and reachable
#     3. Git installed (informational only)
#     4. Python version
#     5. Key Python packages
#
# USAGE:
#   python check_dependencies.py
#   python check_dependencies.py --install     # Also install from requirements.txt
#   python check_dependencies.py --wheels      # Install from wheels/ bundle instead
#
# INTERNET ACCESS: Tested (PyPI check), not required (wheels fallback)
# ============================================================================

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def log(tag: str, message: str) -> None:
    """Print tagged log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{tag}] {message}")


# ---------------------------------------------------------------------------
# Check 1: Python version
# ---------------------------------------------------------------------------

def check_python() -> bool:
    """Check Python version is 3.10+."""
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"
    if major >= 3 and minor >= 10:
        log("OK", f"Python {version_str} ({sys.executable})")
        return True
    else:
        log("FAIL", f"Python {version_str} -- requires 3.10 or newer")
        log("FAIL", "Install Python 3.10+ from python.org or your software store")
        return False


# ---------------------------------------------------------------------------
# Check 2: PyPI reachability
# ---------------------------------------------------------------------------

def check_pypi_reachable() -> bool:
    """
    Test whether pip can reach PyPI through any enterprise proxy.
    Uses a lightweight HEAD request to pypi.org to test connectivity.
    """
    log("INFO", "Testing PyPI reachability...")

    # Method 1: Try urllib to pypi.org
    try:
        req = urllib.request.Request(
            "https://pypi.org/simple/pip/",
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                log("OK", "PyPI is reachable (direct HTTPS)")
                return True
    except Exception:
        pass

    # Method 2: Try pip index (works through proxy configured in pip.conf)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "pip"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            log("OK", "PyPI is reachable (via pip)")
            return True
    except Exception:
        pass

    # Method 3: Try pip install --dry-run
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "pip"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            log("OK", "PyPI is reachable (via pip dry-run)")
            return True
    except Exception:
        pass

    log("WARN", "PyPI is NOT reachable from this machine")
    log("WARN", "This may be due to enterprise firewall or proxy settings")
    log("WARN", "Workaround: install from the included wheels/ bundle instead")
    log("WARN", "  python check_dependencies.py --wheels")
    return False


# ---------------------------------------------------------------------------
# Check 3: Install from PyPI or wheels
# ---------------------------------------------------------------------------

def install_from_pypi(requirements_path: Path) -> bool:
    """Install dependencies from requirements.txt via PyPI."""
    if not requirements_path.exists():
        log("WARN", f"requirements.txt not found at: {requirements_path}")
        return False

    log("INFO", f"Installing from {requirements_path}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path),
         "--quiet"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        log("OK", "All dependencies installed from PyPI")
        return True
    else:
        log("FAIL", "pip install failed:")
        for line in result.stderr.strip().split("\n")[-5:]:
            log("FAIL", f"  {line}")
        return False


def install_from_wheels(wheels_dir: Path) -> bool:
    """Install dependencies from pre-downloaded wheels bundle."""
    if not wheels_dir.exists() or not wheels_dir.is_dir():
        log("FAIL", f"Wheels directory not found: {wheels_dir}")
        log("FAIL", "Run build_wheels_bundle.py on the home PC first,")
        log("FAIL", "then copy the wheels/ folder into this directory.")
        return False

    whl_files = list(wheels_dir.glob("*.whl"))
    if not whl_files:
        log("FAIL", f"No .whl files found in {wheels_dir}")
        return False

    log("INFO", f"Installing from {len(whl_files)} wheel files...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         "--find-links", str(wheels_dir), "-r",
         str(wheels_dir.parent / "requirements.txt"),
         "--quiet"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        log("OK", "All dependencies installed from wheels bundle")
        return True
    else:
        log("FAIL", "Wheel install failed:")
        for line in result.stderr.strip().split("\n")[-5:]:
            log("FAIL", f"  {line}")
        return False


# ---------------------------------------------------------------------------
# Check 4: Ollama
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    """Check if Ollama is installed and reachable."""
    # Check binary exists
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            log("OK", f"Ollama installed: {version}")
        else:
            log("FAIL", "Ollama binary found but returned error")
            _print_ollama_instructions()
            return False
    except FileNotFoundError:
        log("FAIL", "Ollama is NOT installed")
        _print_ollama_instructions()
        return False
    except Exception as e:
        log("FAIL", f"Ollama check failed: {e}")
        _print_ollama_instructions()
        return False

    # Check API is reachable
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags", method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                log("OK", "Ollama API reachable at localhost:11434")
                return True
    except Exception:
        log("WARN", "Ollama is installed but API is not responding")
        log("WARN", "Start it with: ollama serve")
        log("WARN", "Or check if Ollama app is running in system tray")
        return False

    return False


def _print_ollama_instructions():
    """Print instructions for installing Ollama on work laptop."""
    log("FAIL", "")
    log("FAIL", "Ollama must be installed before running model validation.")
    log("FAIL", "Installation options (in order of preference):")
    log("FAIL", "")
    log("FAIL", "  1. Check your work software store (Software Center,")
    log("FAIL", "     Intune Company Portal, or similar) for 'Ollama'")
    log("FAIL", "")
    log("FAIL", "  2. If not in software store, request approval from IT")
    log("FAIL", "     to install from the official Ollama website:")
    log("FAIL", "     https://ollama.com/download")
    log("FAIL", "")
    log("FAIL", "  DO NOT install Ollama from any other source.")
    log("FAIL", "  DO NOT use unofficial mirrors or third-party installers.")
    log("FAIL", "")


# ---------------------------------------------------------------------------
# Check 5: Git (informational)
# ---------------------------------------------------------------------------

def check_git() -> bool:
    """Check if git is installed. Informational only."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log("OK", f"Git installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        log("WARN", "Git is not installed (optional -- not required for validation)")
        return False
    except Exception:
        log("WARN", "Git check failed (optional -- not required for validation)")
        return False
    return False


# ---------------------------------------------------------------------------
# Check 6: Key Python packages
# ---------------------------------------------------------------------------

KEY_PACKAGES = [
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers"),
    ("yaml", "PyYAML"),
    ("torch", "torch (PyTorch)"),
    ("httpx", "httpx"),
    ("openai", "openai"),
    ("keyring", "keyring"),
]


def check_key_packages() -> tuple:
    """Check if key Python packages are importable. Returns (ok, missing)."""
    ok_list = []
    missing_list = []

    for module_name, display_name in KEY_PACKAGES:
        try:
            importlib.import_module(module_name)
            ok_list.append(display_name)
        except ImportError:
            missing_list.append(display_name)

    if not missing_list:
        log("OK", f"All {len(ok_list)} key packages are installed")
    else:
        log("WARN", f"{len(missing_list)} key packages missing: "
            + ", ".join(missing_list))
        log("WARN", "Run with --install to install from PyPI,")
        log("WARN", "or with --wheels to install from offline bundle")

    return ok_list, missing_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="HybridRAG v3 -- Dependency Check"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Install dependencies from requirements.txt (needs PyPI access)",
    )
    parser.add_argument(
        "--wheels", action="store_true",
        help="Install dependencies from wheels/ bundle (offline fallback)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    print("")
    log("INFO", "=" * 55)
    log("INFO", "HybridRAG v3 -- Dependency Check")
    log("INFO", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("INFO", "=" * 55)
    print("")

    results = {}

    # 1. Python version
    results["python"] = check_python()
    print("")

    # 2. PyPI reachability
    results["pypi"] = check_pypi_reachable()
    print("")

    # 3. Install if requested
    if args.install:
        req_path = script_dir / "requirements.txt"
        if not req_path.exists():
            # Try project root
            req_path = script_dir.parents[1] / "requirements.txt"
        if results["pypi"]:
            results["install"] = install_from_pypi(req_path)
        else:
            log("FAIL", "Cannot install from PyPI -- not reachable")
            log("FAIL", "Use --wheels instead for offline install")
            results["install"] = False
        print("")

    if args.wheels:
        wheels_dir = script_dir / "wheels"
        results["install"] = install_from_wheels(wheels_dir)
        print("")

    # 4. Ollama
    results["ollama"] = check_ollama()
    print("")

    # 5. Git (informational)
    results["git"] = check_git()
    print("")

    # 6. Key packages
    ok_pkgs, missing_pkgs = check_key_packages()
    results["packages"] = len(missing_pkgs) == 0
    print("")

    # Summary
    log("INFO", "=" * 55)
    log("INFO", "DEPENDENCY CHECK SUMMARY")
    log("INFO", "=" * 55)

    critical_ok = results["python"] and results["ollama"]
    all_ok = critical_ok and results["packages"]

    for name, status in results.items():
        tag = "OK" if status else ("FAIL" if name in ("python", "ollama") else "WARN")
        log(tag, f"  {name}: {'passed' if status else 'NEEDS ATTENTION'}")

    print("")
    if all_ok:
        log("OK", "All checks passed. Ready for validation.")
    elif critical_ok:
        log("WARN", "Some optional checks need attention (see above).")
        log("WARN", "You can proceed but may need to install packages first.")
    else:
        log("FAIL", "Critical checks failed. Fix issues above before continuing.")

    print("")
    sys.exit(0 if critical_ok else 1)


if __name__ == "__main__":
    main()
