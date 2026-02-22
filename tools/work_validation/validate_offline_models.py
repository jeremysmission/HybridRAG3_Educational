#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Offline Model Validation (validate_offline_models.py)
# ============================================================================
# PURPOSE:
#   Test each WORK_ONLY Ollama model against each profile with a test query.
#   Logs [OK]/[FAIL]/[WARN] per model-profile combination.
#
# PREREQUISITES:
#   - Ollama installed and running (ollama serve)
#   - Models pulled via setup_work_models.ps1
#
# USAGE:
#   python validate_offline_models.py
#   python validate_offline_models.py --log validation_results.log
#
# INTERNET ACCESS: NONE (all queries go to local Ollama)
# ============================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration: Models and profiles (from _model_meta.py)
# ---------------------------------------------------------------------------

WORK_MODELS = {
    "phi4-mini":           {"size_gb": 2.3, "note": "Primary for 7/9 profiles (MIT, Microsoft/USA)"},
    "mistral:7b":          {"size_gb": 4.1, "note": "Alt for eng/sys/fe/cyber (Apache 2.0, Mistral/France)"},
    "phi4:14b-q4_K_M":    {"size_gb": 9.1, "note": "Logistics primary, CAD alt (MIT, Microsoft/USA)"},
    "gemma3:4b":           {"size_gb": 3.3, "note": "PM fast summarization (Apache 2.0, Google/USA)"},
    "mistral-nemo:12b":    {"size_gb": 7.1, "note": "Upgrade for sw/eng/sys/cyber/gen (Apache 2.0, Mistral+NVIDIA, 128K ctx)"},
}

PROFILES = {
    "sw": {
        "label": "Software Engineering",
        "primary": "phi4-mini",
        "alt": "mistral:7b",
        "upgrade": "mistral-nemo:12b",
        "temperature": 0.1,
        "test_query": "Explain the difference between a stack and a heap in memory management.",
        "expected_keywords": ["stack", "heap", "memory", "allocat"],
    },
    "eng": {
        "label": "Engineer",
        "primary": "phi4-mini",
        "alt": "mistral:7b",
        "secondary_test": "phi4:14b-q4_K_M",
        "upgrade": "mistral-nemo:12b",
        "temperature": 0.1,
        "test_query": "What is the operating frequency of a standard GPS L1 signal?",
        "expected_keywords": ["1575", "mhz", "l1", "gps"],
    },
    "pm": {
        "label": "Program Manager",
        "primary": "phi4-mini",
        "alt": "gemma3:4b",
        "temperature": 0.25,
        "test_query": "Summarize the key risks in a project that is 3 weeks behind schedule.",
        "expected_keywords": ["risk", "schedule", "delay", "resource"],
    },
    "log": {
        "label": "Logistics",
        "primary": "phi4:14b-q4_K_M",
        "alt": "phi4-mini",
        "temperature": 0.0,
        "test_query": "What is the lead time for a standard M8 hex bolt from a domestic supplier?",
        "expected_keywords": ["lead", "time", "week", "day", "bolt"],
    },
    "draft": {
        "label": "CAD/Drafting",
        "primary": "phi4-mini",
        "alt": "phi4:14b-q4_K_M",
        "temperature": 0.05,
        "test_query": "What does GD&T symbol MMC mean on an engineering drawing?",
        "expected_keywords": ["maximum", "material", "condition", "tolerance"],
    },
    "sys": {
        "label": "SysAdmin",
        "primary": "phi4-mini",
        "alt": "mistral:7b",
        "upgrade": "mistral-nemo:12b",
        "temperature": 0.1,
        "test_query": "How do you check which ports are listening on a Windows server?",
        "expected_keywords": ["netstat", "port", "listen", "tcp"],
    },
    "fe": {
        "label": "Field Engineer",
        "primary": "phi4-mini",
        "alt": "mistral:7b",
        "temperature": 0.1,
        "test_query": "What safety precautions are required before performing high-voltage equipment inspection in the field?",
        "expected_keywords": ["lockout", "ppe", "voltage", "safety"],
    },
    "cyber": {
        "label": "Cybersecurity Analyst",
        "primary": "phi4-mini",
        "alt": "mistral:7b",
        "upgrade": "mistral-nemo:12b",
        "temperature": 0.1,
        "test_query": "What are the key steps in responding to a suspected ransomware incident?",
        "expected_keywords": ["isolate", "contain", "backup", "incident"],
    },
    "gen": {
        "label": "General AI",
        "primary": "mistral:7b",
        "alt": "phi4-mini",
        "upgrade": "mistral-nemo:12b",
        "temperature": 0.3,
        "test_query": "What are the main causes of inflation in a modern economy?",
        "expected_keywords": ["demand", "supply", "money", "price"],
    },
}

OLLAMA_BASE = "http://127.0.0.1:11434"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.results: list = []
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, tag: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{tag}] {message}"
        print(line)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def record(self, model: str, profile: str, tag: str, detail: str,
               latency_ms: float = 0) -> None:
        self.results.append({
            "model": model, "profile": profile, "tag": tag,
            "detail": detail, "latency_ms": round(latency_ms),
        })
        self.log(tag, f"{model} x {profile}: {detail}"
                 + (f" ({latency_ms:.0f}ms)" if latency_ms else ""))


# ---------------------------------------------------------------------------
# Ollama client (stdlib only, no httpx/requests dependency)
# ---------------------------------------------------------------------------

def check_ollama_running() -> bool:
    """Check if Ollama API is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_installed_models() -> list:
    """Get list of installed model names from Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def query_ollama(model: str, prompt: str, temperature: float = 0.1,
                 timeout_sec: int = 120) -> dict:
    """
    Send a generate request to Ollama and return the result.

    Returns dict with keys: success, response, latency_ms, error
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 200,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            elapsed_ms = (time.monotonic() - start) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "")
            return {
                "success": True,
                "response": response_text,
                "latency_ms": elapsed_ms,
                "error": None,
            }
    except urllib.error.URLError as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "success": False,
            "response": "",
            "latency_ms": elapsed_ms,
            "error": f"URLError: {e.reason}",
        }
    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "success": False,
            "response": "",
            "latency_ms": elapsed_ms,
            "error": f"{type(e).__name__}: {e}",
        }


def check_keywords(response: str, keywords: list) -> tuple:
    """Check if response contains expected keywords. Returns (found, missing)."""
    response_lower = response.lower()
    found = [k for k in keywords if k.lower() in response_lower]
    missing = [k for k in keywords if k.lower() not in response_lower]
    return found, missing


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation(log: Logger) -> int:
    """Run full offline validation. Returns exit code (0=all OK, 1=failures)."""

    log.log("INFO", "=" * 60)
    log.log("INFO", "HybridRAG v3 -- Offline Model Validation")
    log.log("INFO", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log("INFO", "=" * 60)

    # --- Pre-flight: Check Ollama ---
    log.log("INFO", "Checking Ollama connectivity...")
    if not check_ollama_running():
        log.log("FAIL", "Ollama is not running at " + OLLAMA_BASE)
        log.log("FAIL", "Start Ollama with: ollama serve")
        return 1
    log.log("OK", "Ollama is running")

    # --- Pre-flight: Check installed models ---
    installed = list_installed_models()
    log.log("INFO", f"Installed models: {len(installed)}")
    for m in installed:
        log.log("INFO", f"  - {m}")

    missing_models = []
    for model_tag in WORK_MODELS:
        # Check by prefix (qwen3:8b matches qwen3:8b-q4_K_M etc)
        base = model_tag.split(":")[0]
        if not any(base in inst for inst in installed):
            missing_models.append(model_tag)
            log.log("FAIL", f"Model NOT installed: {model_tag}")
        else:
            log.log("OK", f"Model installed: {model_tag}")

    if missing_models:
        log.log("WARN", f"{len(missing_models)} models missing. "
                "Run setup_work_models.ps1 first.")
        log.log("WARN", "Continuing with available models...")

    # --- Test each profile ---
    log.log("INFO", "")
    log.log("INFO", "=" * 60)
    log.log("INFO", "Starting per-profile validation...")
    log.log("INFO", "=" * 60)

    fail_count = 0
    warn_count = 0
    ok_count = 0

    for uc_key, profile in PROFILES.items():
        log.log("INFO", "")
        log.log("INFO", f"--- Profile: {profile['label']} ({uc_key}) ---")

        # Determine which models to test for this profile
        models_to_test = [
            ("primary", profile["primary"]),
            ("alt", profile.get("alt", "")),
        ]
        if profile.get("secondary_test"):
            models_to_test.append(("secondary", profile["secondary_test"]))
        if profile.get("upgrade"):
            models_to_test.append(("upgrade", profile["upgrade"]))

        for role, model_tag in models_to_test:
            if not model_tag:
                continue

            # Skip if model not installed
            base = model_tag.split(":")[0]
            if not any(base in inst for inst in installed):
                log.record(model_tag, uc_key, "WARN",
                           f"Skipped ({role}) -- model not installed")
                warn_count += 1
                continue

            # Run test query
            log.log("INFO", f"Testing {model_tag} as {role}...")
            result = query_ollama(
                model=model_tag,
                prompt=profile["test_query"],
                temperature=profile["temperature"],
            )

            if not result["success"]:
                log.record(model_tag, uc_key, "FAIL",
                           f"Query failed ({role}): {result['error']}",
                           result["latency_ms"])
                fail_count += 1
                continue

            # Check response quality
            response = result["response"]
            if len(response.strip()) < 10:
                log.record(model_tag, uc_key, "FAIL",
                           f"Empty/tiny response ({role}): {len(response)} chars",
                           result["latency_ms"])
                fail_count += 1
                continue

            # Check for expected keywords
            found, missing = check_keywords(response, profile["expected_keywords"])
            keyword_ratio = len(found) / len(profile["expected_keywords"])

            if keyword_ratio >= 0.5:
                log.record(model_tag, uc_key, "OK",
                           f"Response valid ({role}), "
                           f"keywords {len(found)}/{len(profile['expected_keywords'])}",
                           result["latency_ms"])
                ok_count += 1
            else:
                log.record(model_tag, uc_key, "WARN",
                           f"Low keyword match ({role}): "
                           f"{len(found)}/{len(profile['expected_keywords'])} "
                           f"(missing: {missing})",
                           result["latency_ms"])
                warn_count += 1

            # Print first 200 chars of response for manual review
            preview = response.strip()[:200].replace("\n", " ")
            log.log("INFO", f"  Response preview: {preview}...")

    # --- Summary ---
    log.log("INFO", "")
    log.log("INFO", "=" * 60)
    log.log("INFO", "VALIDATION SUMMARY")
    log.log("INFO", "=" * 60)
    log.log("INFO", f"  OK:   {ok_count}")
    log.log("INFO", f"  WARN: {warn_count}")
    log.log("INFO", f"  FAIL: {fail_count}")
    log.log("INFO", f"  Total: {ok_count + warn_count + fail_count}")
    log.log("INFO", "=" * 60)

    if fail_count > 0:
        log.log("FAIL", "Validation completed with failures.")
        return 1
    elif warn_count > 0:
        log.log("WARN", "Validation completed with warnings.")
        return 0
    else:
        log.log("OK", "All validations passed.")
        return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HybridRAG v3 -- Offline Model Validation"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to log file (default: print to console only)"
    )
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    log = Logger(log_path)

    exit_code = run_validation(log)

    if log_path:
        log.log("INFO", f"Results written to: {log_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
