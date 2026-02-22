# ===========================================================================
# HybridRAG v3 -- Network Gate Status Tool
# ===========================================================================
# FILE: src/tools/net_status.py
#
# WHAT THIS IS:
#   Shows the current network access control status, allowed hosts,
#   and recent connection audit log. Run this to verify the gate is
#   configured correctly before doing anything sensitive.
#
# USAGE:
#   python -m src.tools.net_status
#   python -m src.tools.net_status --audit    (show recent audit log)
#   python -m src.tools.net_status --test URL (test if a URL is allowed)
#
# INTERNET ACCESS: NONE (this tool only reads gate state, no connections)
# ===========================================================================

import os
import sys

# Make sure we can import from the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def show_status():
    """Show current network gate configuration."""
    from src.core.config import load_config
    from src.core.network_gate import get_gate

    # Loading config auto-configures the gate
    config = load_config(project_root)
    gate = get_gate()

    print("=" * 60)
    print("  NETWORK GATE STATUS")
    print("=" * 60)
    print(f"  Mode:           {gate.mode_name.upper()}")
    print(f"  Config mode:    {config.mode}")
    print(f"  API endpoint:   {config.api.endpoint or '(not set)'}")
    print(f"  Allowed hosts:  {', '.join(gate._allowed_hosts) or '(localhost only)'}")

    if gate._allowed_prefixes:
        print(f"  Allowed paths:  {', '.join(gate._allowed_prefixes)}")

    admin_mode = os.environ.get("HYBRIDRAG_ADMIN_MODE", "").strip()
    if admin_mode:
        print(f"  Admin override: HYBRIDRAG_ADMIN_MODE={admin_mode}")

    print()
    print("  WHAT THIS MEANS:")
    if gate.mode_name == "offline":
        print("    - Only localhost connections allowed (Ollama)")
        print("    - ALL internet access is blocked")
        print("    - Switch to online: set mode=online in config/default_config.yaml")
    elif gate.mode_name == "online":
        print("    - Localhost allowed (Ollama)")
        print(f"    - API endpoint allowed: {', '.join(gate._allowed_hosts)}")
        print("    - ALL other internet access is blocked")
        print("    - To add more hosts: use api.allowed_endpoint_prefixes in YAML")
    elif gate.mode_name == "admin":
        print("    - WARNING: All outbound connections are allowed")
        print("    - Use for maintenance only (pip install, model download)")
        print("    - Remove HYBRIDRAG_ADMIN_MODE env var when done")

    # Show audit summary if available
    summary = gate.get_audit_summary()
    if summary["total_checks"] > 0:
        print()
        print(f"  AUDIT: {summary['allowed']} allowed, {summary['denied']} denied")
        if summary["unique_hosts_contacted"]:
            print(f"  Hosts contacted: {', '.join(summary['unique_hosts_contacted'])}")

    print("=" * 60)


def show_audit(n=20):
    """Show recent audit log entries."""
    from src.core.config import load_config
    from src.core.network_gate import get_gate

    load_config(project_root)
    gate = get_gate()
    log = gate.get_audit_log(last_n=n)

    print("=" * 60)
    print(f"  NETWORK AUDIT LOG (last {n} entries)")
    print("=" * 60)

    if not log:
        print("  (no entries yet -- connections haven't been made)")
    else:
        for entry in log:
            print(f"  {entry.to_log_line()}")

    print("=" * 60)


def test_url(url):
    """Test if a specific URL would be allowed."""
    from src.core.config import load_config
    from src.core.network_gate import get_gate, NetworkBlockedError

    load_config(project_root)
    gate = get_gate()

    print(f"  Testing: {url}")
    print(f"  Mode:    {gate.mode_name.upper()}")

    try:
        gate.check_allowed(url, "manual_test", "net_status_tool")
        print(f"  Result:  [ALLOW] -- this URL is permitted")
    except NetworkBlockedError as e:
        print(f"  Result:  [DENY] -- {e.reason}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--audit" in args:
        show_audit()
    elif "--test" in args:
        idx = args.index("--test")
        if idx + 1 < len(args):
            test_url(args[idx + 1])
        else:
            print("Usage: python -m src.tools.net_status --test URL")
    else:
        show_status()
