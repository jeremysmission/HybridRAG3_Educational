#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- List Available AI Models (scripts/_list_models.py)
# ============================================================================
#
# WHAT THIS DOES (plain English):
#   Shows ALL AI models available to HybridRAG in a single view:
#     - Offline models: auto-detected from Ollama (localhost only)
#     - Online models: queried from your API provider (one GET request)
#   Highlights which model is currently active.
#
# HOW TO RUN:
#   rag-models                (after sourcing start_hybridrag.ps1)
#   python scripts/_list_models.py   (direct)
#
# HOW TO ADD/REMOVE MODELS:
#   Offline:
#     ollama pull mistral      (add a new offline model)
#     ollama rm phi4-mini      (remove an offline model)
#
#   Online:
#     Models are auto-detected from your API provider.
#     Use rag-mode-online to switch between them.
#
# INTERNET ACCESS:
#   Offline section: NONE (Ollama query is localhost only)
#   Online section: YES if API key is configured (one GET /models request)
#                   Skipped gracefully if no key or no connectivity
# ============================================================================

import os
import subprocess
import sys
import yaml

sys.path.insert(0, os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."))

from src.core.llm_router import get_available_deployments
from src.security.credentials import resolve_credentials as canonical_resolve


def get_ollama_models():
    """Auto-detect installed Ollama models. Returns list of dicts."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []

        models = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                size_str = "unknown"
                for i, p in enumerate(parts):
                    if p in ("GB", "MB", "KB") and i > 0:
                        size_str = parts[i - 1] + " " + p
                        break
                models.append({"name": name, "size": size_str})
        return models

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def get_online_models(max_display=25):
    """
    Fetch available models from the API provider via centralized discovery.

    Delegates to get_available_deployments() in llm_router.py, which
    auto-detects Azure vs OpenAI and uses the correct API path.

    Returns (model_id_list, total_count) for backward-compatible display.
    """
    all_models = get_available_deployments()
    if not all_models:
        return [], 0
    total = len(all_models)
    return all_models[:max_display], total


def main():
    # ---- Load config ----
    try:
        with open("config/default_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("  [FAIL] config/default_config.yaml not found")
        sys.exit(1)

    current_mode = cfg.get("mode", "offline")
    offline_model = cfg.get("ollama", {}).get("model", "phi4-mini")
    online_model = cfg.get("api", {}).get("model", "(not set)")
    online_endpoint = cfg.get("api", {}).get("endpoint", "(not set)")
    online_deployment = cfg.get("api", {}).get("deployment", "")

    # ---- Header ----
    print()
    print("  " + "=" * 60)
    print("  AI MODELS AVAILABLE")
    print("  " + "=" * 60)
    print()
    mode_display = "OFFLINE (local)" if current_mode == "offline" else "ONLINE (API)"
    print(f"  Current mode: {mode_display}")
    print()

    # ---- Offline Models ----
    print("  OFFLINE AI MODELS (Local)")
    print("  " + "-" * 55)

    ollama_models = get_ollama_models()
    if ollama_models:
        for m in ollama_models:
            if current_mode == "offline" and m["name"] == offline_model:
                marker = "  << ACTIVE"
            elif m["name"] == offline_model:
                marker = "  (configured)"
            else:
                marker = ""
            name_padded = m["name"].ljust(38)
            print(f"    {name_padded} ({m['size']}){marker}")
    else:
        print("    (none detected -- is Ollama installed?)")

    print()
    print("    Add:    ollama pull <model_name>")
    print("    Remove: ollama rm <model_name>")
    print("    Switch: rag-mode-offline")

    # ---- Online Models ----
    print()
    print("  ONLINE API MODELS")
    print("  " + "-" * 55)

    # Resolve credentials via canonical resolver
    creds = canonical_resolve()
    resolved_endpoint = creds.endpoint or online_endpoint
    has_creds = creds.has_key and bool(resolved_endpoint)

    if has_creds:
        print(f"    Endpoint: {resolved_endpoint}")
        print(f"    Fetching available models...")
        print()

        models, total = get_online_models()

        if models:
            for m in models:
                if m == online_model:
                    if current_mode == "online":
                        marker = "  << ACTIVE"
                    else:
                        marker = "  (configured)"
                else:
                    marker = ""
                # Truncate very long model names
                name_display = m[:50]
                print(f"    {name_display}{marker}")

            if total > len(models):
                print(f"    ... and {total - len(models)} more")
                print(f"    (use rag-mode-online to see all and select)")
        else:
            print(f"    Current model: {online_model}")
            print(f"    (could not fetch model list from provider)")
    else:
        print(f"    Current model: {online_model}")
        print(f"    Endpoint: {online_endpoint}")
        if not creds.has_key:
            print(f"    (no API key stored -- run rag-store-key to enable model listing)")

    # Show deployment from credentials (keyring/env) or config fallback
    resolved_deployment = creds.deployment or online_deployment
    if resolved_deployment:
        dep_source = creds.source_deployment or "config"
        print(f"    Deployment: {resolved_deployment} ({dep_source})")

    print()
    print("    Change model: rag-mode-online")
    print("    Change key:   rag-store-key")
    print("    Change URL:   rag-store-endpoint")

    print()
    print("  " + "=" * 60)
    print()


if __name__ == "__main__":
    main()
