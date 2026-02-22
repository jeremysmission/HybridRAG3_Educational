#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Online API Validation (validate_online_api.py)
# ============================================================================
# PURPOSE:
#   Test Azure OpenAI API connectivity and available models on the WORK
#   laptop. This script is completely separate from offline validation
#   and from any personal API credentials.
#
# IMPORTANT CONSTRAINTS:
#   - Work laptop uses Azure OpenAI endpoint (NOT personal OpenAI)
#   - Only GPT-3.5 Turbo and GPT-4 are CONFIRMED available
#   - Only Azure-hosted models are available on this endpoint
#   - This script PROBES the endpoint to discover models, does not assume
#   - Handles 401/connection errors with full diagnostic output
#   - NEVER mixes personal and work credentials
#
# CREDENTIAL RESOLUTION:
#   This script uses the HybridRAG credential system:
#     1. Windows Credential Manager (keyring) -- preferred
#     2. Environment variables (AZURE_OPENAI_API_KEY, etc.)
#     3. Config file
#   It does NOT accept command-line API keys (security policy).
#
# USAGE:
#   python validate_online_api.py
#   python validate_online_api.py --log online_validation.log
#   python validate_online_api.py --endpoint https://your.azure.endpoint.com
#
# INTERNET ACCESS: YES (Azure OpenAI API calls)
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Known Azure deployment names to probe
# ---------------------------------------------------------------------------
# These are common Azure OpenAI deployment names. The script will try each
# and report which ones respond. Only GPT-3.5 Turbo and GPT-4 are confirmed
# on the work endpoint -- others are probed on a best-effort basis.

AZURE_DEPLOYMENTS_TO_PROBE = [
    # Confirmed available
    {"name": "gpt-35-turbo",     "display": "GPT-3.5 Turbo",  "confirmed": True},
    {"name": "gpt-4",            "display": "GPT-4",           "confirmed": True},
    # Probe on best-effort basis (may or may not be deployed)
    {"name": "gpt-4o",           "display": "GPT-4o",          "confirmed": False},
    {"name": "gpt-4o-mini",      "display": "GPT-4o Mini",     "confirmed": False},
    {"name": "gpt-4-turbo",      "display": "GPT-4 Turbo",     "confirmed": False},
    {"name": "gpt-4.1",          "display": "GPT-4.1",         "confirmed": False},
    {"name": "gpt-4.1-mini",     "display": "GPT-4.1 Mini",    "confirmed": False},
]

# Default Azure API version
DEFAULT_API_VERSION = "2024-02-02"

# Test query for model validation
TEST_QUERY = "Say hello in exactly 5 words."


# ---------------------------------------------------------------------------
# Logger (same pattern as offline validator)
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

    def record(self, deployment: str, tag: str, detail: str,
               latency_ms: float = 0) -> None:
        self.results.append({
            "deployment": deployment, "tag": tag,
            "detail": detail, "latency_ms": round(latency_ms),
        })
        self.log(tag, f"{deployment}: {detail}"
                 + (f" ({latency_ms:.0f}ms)" if latency_ms else ""))


# ---------------------------------------------------------------------------
# Credential resolution (uses HybridRAG credential system if available,
# falls back to direct env var reading)
# ---------------------------------------------------------------------------

def resolve_work_credentials(
    override_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve work API credentials. Returns dict with:
      endpoint, api_key, deployment, api_version, source_key, source_endpoint
    """
    result = {
        "endpoint": None,
        "api_key": None,
        "deployment": None,
        "api_version": None,
        "source_key": "not found",
        "source_endpoint": "not found",
        "is_azure": False,
    }

    # Try HybridRAG credential module first
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.security.credentials import resolve_credentials
        creds = resolve_credentials()
        if creds.has_key:
            result["api_key"] = creds.api_key
            result["source_key"] = creds.source_key or "hybridrag"
        if creds.has_endpoint:
            result["endpoint"] = creds.endpoint
            result["source_endpoint"] = creds.source_endpoint or "hybridrag"
        if creds.deployment:
            result["deployment"] = creds.deployment
        if creds.api_version:
            result["api_version"] = creds.api_version
    except Exception:
        # Fall back to direct env var reading
        pass

    # Override endpoint if specified on command line
    if override_endpoint:
        result["endpoint"] = override_endpoint.rstrip("/")
        result["source_endpoint"] = "command-line"

    # Direct env var fallback for key
    if not result["api_key"]:
        for var in ["HYBRIDRAG_API_KEY", "AZURE_OPENAI_API_KEY",
                     "AZURE_OPEN_AI_KEY", "OPENAI_API_KEY"]:
            val = os.environ.get(var, "").strip()
            if val:
                result["api_key"] = val
                result["source_key"] = f"env:{var}"
                break

    # Direct env var fallback for endpoint
    if not result["endpoint"]:
        for var in ["HYBRIDRAG_API_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
                     "OPENAI_API_ENDPOINT", "AZURE_OPENAI_BASE_URL"]:
            val = os.environ.get(var, "").strip()
            if val:
                result["endpoint"] = val.rstrip("/")
                result["source_endpoint"] = f"env:{var}"
                break

    # Direct env var fallback for deployment
    if not result["deployment"]:
        for var in ["AZURE_OPENAI_DEPLOYMENT", "AZURE_DEPLOYMENT",
                     "OPENAI_DEPLOYMENT"]:
            val = os.environ.get(var, "").strip()
            if val:
                result["deployment"] = val
                break

    # Direct env var fallback for api version
    if not result["api_version"]:
        for var in ["AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION"]:
            val = os.environ.get(var, "").strip()
            if val:
                result["api_version"] = val
                break
        if not result["api_version"]:
            result["api_version"] = DEFAULT_API_VERSION

    # Detect Azure from URL patterns
    if result["endpoint"]:
        ep_lower = result["endpoint"].lower()
        azure_patterns = ["azure", "aoai", "cognitiveservices", ".openai.azure.com"]
        result["is_azure"] = any(p in ep_lower for p in azure_patterns)

    return result


def mask_key(key: Optional[str]) -> str:
    """Mask API key for safe display."""
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "****"
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# Azure API calls (stdlib only)
# ---------------------------------------------------------------------------

def build_azure_chat_url(
    base_endpoint: str,
    deployment: str,
    api_version: str,
) -> str:
    """Build Azure OpenAI chat completions URL."""
    base = base_endpoint.rstrip("/")
    # Strip any existing path components that would cause doubling
    for suffix in ["/openai", "/deployments"]:
        if base.endswith(suffix):
            base = base[:len(base) - len(suffix)]
    return (
        f"{base}/openai/deployments/{deployment}"
        f"/chat/completions?api-version={api_version}"
    )


def call_azure_chat(
    url: str,
    api_key: str,
    prompt: str,
    is_azure: bool = True,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    """
    Send a chat completion request to Azure OpenAI.
    Returns dict with: success, response, model, latency_ms, status_code, error
    """
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.1,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if is_azure:
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    # Create SSL context that works in enterprise environments
    ctx = ssl.create_default_context()

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
            elapsed_ms = (time.monotonic() - start) * 1000
            data = json.loads(resp.read().decode("utf-8"))
            answer = ""
            if "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                answer = msg.get("content", "")
            return {
                "success": True,
                "response": answer,
                "model": data.get("model", "unknown"),
                "latency_ms": elapsed_ms,
                "status_code": resp.status,
                "error": None,
                "usage": data.get("usage", {}),
            }

    except urllib.error.HTTPError as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "success": False,
            "response": "",
            "model": "",
            "latency_ms": elapsed_ms,
            "status_code": e.code,
            "error": f"HTTP {e.code}: {e.reason}",
            "error_body": error_body,
        }

    except urllib.error.URLError as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "success": False,
            "response": "",
            "model": "",
            "latency_ms": elapsed_ms,
            "status_code": 0,
            "error": f"Connection failed: {e.reason}",
            "error_body": "",
        }

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "success": False,
            "response": "",
            "model": "",
            "latency_ms": elapsed_ms,
            "status_code": 0,
            "error": f"{type(e).__name__}: {e}",
            "error_body": "",
        }


def try_list_models(
    endpoint: str,
    api_key: str,
    api_version: str,
    is_azure: bool = True,
) -> Dict[str, Any]:
    """
    Try to list available models/deployments on the endpoint.
    Azure: GET /openai/models?api-version=...
    OpenAI: GET /v1/models
    """
    base = endpoint.rstrip("/")
    if is_azure:
        url = f"{base}/openai/models?api-version={api_version}"
        headers = {"api-key": api_key}
    else:
        url = f"{base}/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}

    req = urllib.request.Request(url, headers=headers, method="GET")
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = []
            if "data" in data and isinstance(data["data"], list):
                for m in data["data"]:
                    mid = m.get("id", "") or m.get("model", "")
                    if mid:
                        models.append(mid)
            return {"success": True, "models": sorted(models),
                    "status_code": resp.status, "error": None}

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {"success": False, "models": [],
                "status_code": e.code,
                "error": f"HTTP {e.code}: {e.reason}",
                "error_body": error_body}

    except Exception as e:
        return {"success": False, "models": [],
                "status_code": 0,
                "error": f"{type(e).__name__}: {e}",
                "error_body": ""}


# ---------------------------------------------------------------------------
# Diagnostic helpers for common error codes
# ---------------------------------------------------------------------------

def diagnose_error(status_code: int, error: str, endpoint: str,
                   error_body: str = "") -> List[str]:
    """Return list of diagnostic suggestions for a given error."""
    suggestions = []

    if status_code == 401:
        suggestions.append("401 Unauthorized -- the API key was rejected")
        suggestions.append("Possible causes:")
        suggestions.append("  1. API key is expired or revoked")
        suggestions.append("  2. API key does not have access to this endpoint")
        suggestions.append("  3. Wrong auth scheme (api-key vs Bearer)")
        suggestions.append("  4. Key was stored for a different endpoint")
        suggestions.append("Actions:")
        suggestions.append("  - Verify key in Windows Credential Manager")
        suggestions.append("  - Ask IT to confirm key is active for this endpoint")
        suggestions.append(f"  - Endpoint used: {endpoint}")
        if error_body:
            suggestions.append(f"  - Server response: {error_body[:500]}")

    elif status_code == 403:
        suggestions.append("403 Forbidden -- access denied by policy")
        suggestions.append("Possible causes:")
        suggestions.append("  1. IP allowlist or VPN requirement")
        suggestions.append("  2. Resource group access policy")
        suggestions.append("  3. Azure RBAC role not assigned")
        suggestions.append("Actions:")
        suggestions.append("  - Ensure you are on enterprise VPN")
        suggestions.append("  - Ask IT to verify RBAC permissions")

    elif status_code == 404:
        suggestions.append("404 Not Found -- deployment or endpoint path is wrong")
        suggestions.append("Possible causes:")
        suggestions.append("  1. Deployment name does not exist on this endpoint")
        suggestions.append("  2. Endpoint URL has wrong path")
        suggestions.append("  3. API version is not supported")
        suggestions.append("Actions:")
        suggestions.append("  - Verify deployment name in Azure Portal")
        suggestions.append(f"  - Endpoint used: {endpoint}")

    elif status_code == 429:
        suggestions.append("429 Rate Limited -- too many requests")
        suggestions.append("Actions:")
        suggestions.append("  - Wait 60 seconds and retry")
        suggestions.append("  - Check quota in Azure Portal")

    elif status_code >= 500:
        suggestions.append(f"{status_code} Server Error -- Azure service issue")
        suggestions.append("Actions:")
        suggestions.append("  - Wait and retry in a few minutes")
        suggestions.append("  - Check Azure status page")

    elif status_code == 0:
        suggestions.append("Connection failed -- could not reach endpoint")
        suggestions.append("Possible causes:")
        suggestions.append("  1. No network/internet access")
        suggestions.append("  2. Enterprise firewall blocking the connection")
        suggestions.append("  3. VPN not connected")
        suggestions.append("  4. Endpoint URL is wrong")
        suggestions.append("  5. DNS resolution failed")
        suggestions.append("Actions:")
        suggestions.append(f"  - Verify endpoint URL: {endpoint}")
        suggestions.append("  - Check VPN connection")
        suggestions.append("  - Try: curl -v " + endpoint + " (from cmd)")
        if "ssl" in error.lower() or "certificate" in error.lower():
            suggestions.append("  - SSL/certificate issue detected -- "
                             "enterprise proxy may need custom CA cert")

    return suggestions


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation(log: Logger, override_endpoint: Optional[str] = None) -> int:
    """Run online API validation. Returns exit code (0=OK, 1=failures)."""

    log.log("INFO", "=" * 60)
    log.log("INFO", "HybridRAG v3 -- Online API Validation (Work Endpoint)")
    log.log("INFO", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.log("INFO", "NOTE: Only Azure-hosted models tested on this endpoint")
    log.log("INFO", "=" * 60)

    # --- Step 1: Resolve credentials ---
    log.log("INFO", "Resolving work API credentials...")
    creds = resolve_work_credentials(override_endpoint)

    log.log("INFO", f"  Endpoint:      {creds['endpoint'] or '(NOT SET)'}")
    log.log("INFO", f"  Source:        {creds['source_endpoint']}")
    log.log("INFO", f"  API Key:       {mask_key(creds['api_key'])}")
    log.log("INFO", f"  Key Source:    {creds['source_key']}")
    log.log("INFO", f"  Deployment:    {creds['deployment'] or '(auto-probe)'}")
    log.log("INFO", f"  API Version:   {creds['api_version']}")
    log.log("INFO", f"  Detected Azure: {creds['is_azure']}")

    if not creds["endpoint"]:
        log.log("FAIL", "No API endpoint configured.")
        log.log("FAIL", "Set one of: HYBRIDRAG_API_ENDPOINT, AZURE_OPENAI_ENDPOINT")
        log.log("FAIL", "Or use: python validate_online_api.py --endpoint <URL>")
        return 1

    if not creds["api_key"]:
        log.log("FAIL", "No API key configured.")
        log.log("FAIL", "Set one of: HYBRIDRAG_API_KEY, AZURE_OPENAI_API_KEY")
        log.log("FAIL", "Or store via: python -m src.security.credentials store")
        return 1

    # --- Step 2: Probe model list endpoint ---
    log.log("INFO", "")
    log.log("INFO", "--- Probing model list endpoint ---")

    model_result = try_list_models(
        creds["endpoint"], creds["api_key"],
        creds["api_version"], creds["is_azure"],
    )

    discovered_models = []
    if model_result["success"]:
        log.log("OK", f"Model list endpoint responded: {len(model_result['models'])} models")
        for m in model_result["models"]:
            log.log("INFO", f"  - {m}")
            discovered_models.append(m)
    else:
        log.log("WARN", f"Model list probe failed: {model_result['error']}")
        if model_result.get("status_code"):
            for suggestion in diagnose_error(
                model_result["status_code"],
                model_result["error"],
                creds["endpoint"],
                model_result.get("error_body", ""),
            ):
                log.log("INFO", f"  {suggestion}")
        log.log("INFO", "Continuing with deployment probing...")

    # --- Step 3: Probe each known deployment ---
    log.log("INFO", "")
    log.log("INFO", "--- Probing Azure deployments ---")

    ok_count = 0
    fail_count = 0
    warn_count = 0
    available_deployments = []

    for dep_info in AZURE_DEPLOYMENTS_TO_PROBE:
        dep_name = dep_info["name"]
        dep_display = dep_info["display"]
        confirmed = dep_info["confirmed"]

        # Skip non-confirmed if not discovered
        if not confirmed and discovered_models:
            if dep_name not in discovered_models:
                log.log("INFO", f"  {dep_display} ({dep_name}): "
                        "not in model list, skipping probe")
                continue

        log.log("INFO", f"  Probing {dep_display} ({dep_name})...")

        url = build_azure_chat_url(
            creds["endpoint"], dep_name, creds["api_version"],
        )

        result = call_azure_chat(
            url=url,
            api_key=creds["api_key"],
            prompt=TEST_QUERY,
            is_azure=creds["is_azure"],
        )

        if result["success"]:
            log.record(dep_name, "OK",
                       f"Responded: model={result['model']}, "
                       f"response='{result['response'][:80]}...'",
                       result["latency_ms"])
            available_deployments.append(dep_name)
            ok_count += 1
        else:
            status = result.get("status_code", 0)

            if status == 404:
                # Deployment not found -- expected for non-confirmed
                if confirmed:
                    log.record(dep_name, "FAIL",
                               f"404 -- expected to be available but not found",
                               result["latency_ms"])
                    fail_count += 1
                else:
                    log.record(dep_name, "INFO",
                               f"404 -- not deployed (expected)",
                               result["latency_ms"])
            elif status == 401:
                log.record(dep_name, "FAIL",
                           f"401 Unauthorized",
                           result["latency_ms"])
                fail_count += 1
                # Print full diagnostics for 401
                log.log("INFO", "")
                log.log("INFO", "  === 401 DIAGNOSTIC DETAILS ===")
                log.log("INFO", f"  URL attempted: {url}")
                log.log("INFO", f"  Auth header: api-key" if creds["is_azure"]
                        else f"  Auth header: Bearer")
                log.log("INFO", f"  Key preview: {mask_key(creds['api_key'])}")
                for s in diagnose_error(401, result["error"],
                                       creds["endpoint"],
                                       result.get("error_body", "")):
                    log.log("INFO", f"  {s}")
                log.log("INFO", "  === END DIAGNOSTIC ===")
                log.log("INFO", "")
            else:
                log.record(dep_name, "FAIL",
                           f"Error: {result['error']}",
                           result["latency_ms"])
                fail_count += 1
                for s in diagnose_error(
                    status, result["error"],
                    creds["endpoint"],
                    result.get("error_body", ""),
                ):
                    log.log("INFO", f"  {s}")

    # --- Step 4: Test online/offline mode switching ---
    log.log("INFO", "")
    log.log("INFO", "--- Mode Switching Test ---")

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.core.network_gate import configure_gate, get_gate, NetworkMode

        # Test offline mode
        configure_gate(mode="offline")
        gate = get_gate()
        offline_ok = not gate.is_allowed("https://api.example.com")
        if offline_ok:
            log.log("OK", "Offline mode correctly blocks external URLs")
        else:
            log.log("FAIL", "Offline mode did NOT block external URLs")
            fail_count += 1

        # Test online mode
        if creds["endpoint"]:
            configure_gate(
                mode="online",
                api_endpoint=creds["endpoint"],
            )
            gate = get_gate()
            online_ok = gate.is_allowed(creds["endpoint"])
            if online_ok:
                log.log("OK", "Online mode correctly allows configured endpoint")
            else:
                log.log("FAIL", "Online mode blocked the configured endpoint")
                fail_count += 1

            random_blocked = not gate.is_allowed("https://random-site.com")
            if random_blocked:
                log.log("OK", "Online mode correctly blocks non-configured URLs")
            else:
                log.log("WARN", "Online mode did not block random URLs")
                warn_count += 1

        # Reset to offline
        configure_gate(mode="offline")
        log.log("OK", "Mode switching works correctly")

    except ImportError:
        log.log("WARN", "NetworkGate module not available -- skipping mode test")
        log.log("INFO", "  (This is expected if running outside the project directory)")
        warn_count += 1
    except Exception as e:
        log.log("WARN", f"Mode switching test error: {type(e).__name__}: {e}")
        warn_count += 1

    # --- Summary ---
    log.log("INFO", "")
    log.log("INFO", "=" * 60)
    log.log("INFO", "ONLINE VALIDATION SUMMARY")
    log.log("INFO", "=" * 60)
    log.log("INFO", f"  Endpoint:     {creds['endpoint']}")
    log.log("INFO", f"  Key Source:   {creds['source_key']}")
    log.log("INFO", f"  Available deployments: {available_deployments or 'NONE'}")
    log.log("INFO", f"  OK:   {ok_count}")
    log.log("INFO", f"  WARN: {warn_count}")
    log.log("INFO", f"  FAIL: {fail_count}")
    log.log("INFO", "=" * 60)

    if not available_deployments:
        log.log("WARN", "No deployments responded successfully.")
        log.log("INFO", "Check endpoint URL and API key configuration.")

    if fail_count > 0:
        log.log("FAIL", "Online validation completed with failures.")
        return 1
    elif warn_count > 0:
        log.log("WARN", "Online validation completed with warnings.")
        return 0
    else:
        log.log("OK", "Online validation passed.")
        return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HybridRAG v3 -- Online API Validation (Work Endpoint)"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to log file (default: print to console only)"
    )
    parser.add_argument(
        "--endpoint", type=str, default=None,
        help="Override endpoint URL (for testing different endpoints)"
    )
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else None
    log = Logger(log_path)

    exit_code = run_validation(log, args.endpoint)

    if log_path:
        log.log("INFO", f"Results written to: {log_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
