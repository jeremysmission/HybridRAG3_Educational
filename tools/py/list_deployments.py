# ============================================================================
# list_deployments.py -- Show available deployments and routing table
# ============================================================================
# Diagnostic tool that combines deployment discovery with model routing.
#
# WHAT IT SHOWS:
#   1. Resolved credentials summary (masked key, endpoint, deployment, version)
#   2. Available deployments from the configured API endpoint
#   3. Routing table: best model per use case based on live deployments
#
# HOW TO RUN:
#   python tools/py/list_deployments.py        (direct)
#   rag-list-deployments                       (after sourcing start_hybridrag.ps1)
#
# INTERNET ACCESS: YES (one GET request for deployment discovery)
#                  Skipped gracefully if no credentials or no connectivity
# ============================================================================

import sys
import os

sys.path.insert(0, os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."))

from src.security.credentials import resolve_credentials, credential_status
from src.core.llm_router import get_available_deployments, refresh_deployments


def main():
    print()
    print("  " + "=" * 60)
    print("  DEPLOYMENT DISCOVERY + ROUTING TABLE")
    print("  " + "=" * 60)
    print()

    # ---- Section 1: Credential summary ----
    print("  CREDENTIALS")
    print("  " + "-" * 55)

    creds = resolve_credentials()
    status = credential_status()

    key_display = creds.key_preview if creds.has_key else "NOT SET"
    ep_display = creds.endpoint or "NOT SET"
    dep_display = creds.deployment or "NOT SET"
    ver_display = creds.api_version or "NOT SET"

    print(f"    API Key:     {key_display}  (source: {status['api_key_source']})")
    print(f"    Endpoint:    {ep_display}  (source: {status['api_endpoint_source']})")
    print(f"    Deployment:  {dep_display}  (source: {status['deployment_source']})")
    print(f"    API Version: {ver_display}  (source: {status['api_version_source']})")
    print()

    if not creds.is_online_ready:
        print("  [FAIL] Credentials incomplete -- need both API key and endpoint")
        print("         Run: rag-store-key and rag-store-endpoint")
        print()
        print("  " + "=" * 60)
        print()
        sys.exit(1)

    print(f"  [OK] Credentials ready for online mode")
    print()

    # ---- Section 2: Available deployments ----
    print("  AVAILABLE DEPLOYMENTS")
    print("  " + "-" * 55)

    # Use --refresh flag to force cache clear
    if "--refresh" in sys.argv:
        deployments = refresh_deployments()
    else:
        deployments = get_available_deployments()

    if not deployments:
        print("    [WARN] No deployments found (check endpoint and connectivity)")
        print()
        print("  " + "=" * 60)
        print()
        sys.exit(0)

    for dep in sorted(deployments):
        print(f"    {dep}")

    print()
    print(f"  [OK] {len(deployments)} deployments available")
    print()

    # ---- Section 3: Routing table ----
    try:
        # Import here to avoid circular import at module level
        scripts_dir = os.path.join(
            os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."), "scripts"
        )
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from _model_meta import (
            get_routing_table, USE_CASES, lookup_known_model, use_case_score,
        )
    except ImportError as e:
        print(f"  [WARN] Could not load model intelligence: {e}")
        print()
        print("  " + "=" * 60)
        print()
        sys.exit(0)

    print("  ROUTING TABLE (best model per use case)")
    print("  " + "-" * 55)

    table = get_routing_table(deployments)

    # Header
    print(f"    {'Use Case':<25} {'Model':<25} {'Score':>5}")
    print(f"    {'-' * 25} {'-' * 25} {'-' * 5}")

    for uc_key, uc_info in USE_CASES.items():
        model_id = table.get(uc_key)
        if model_id:
            # Compute score for display
            kb = lookup_known_model(model_id)
            if kb:
                score = use_case_score(kb["tier_eng"], kb["tier_gen"], uc_key)
            else:
                score = use_case_score(45, 45, uc_key)
            score_str = str(score)
        else:
            model_id = "(none eligible)"
            score_str = "---"

        label = uc_info["label"]
        print(f"    {label:<25} {model_id:<25} {score_str:>5}")

    print()
    print("  " + "=" * 60)
    print()


if __name__ == "__main__":
    main()
