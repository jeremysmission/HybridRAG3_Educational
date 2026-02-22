# ============================================================================
# HybridRAG -- System Diagnostic Launcher (src/diagnostic/system_diagnostic.py)
# ============================================================================
#
# WHAT THIS FILE IS:
#   A convenience stub that opens the index_status tool in Notepad for
#   quick review. This is a legacy helper from early development.
#
# NOTE:
#   The real diagnostic suite lives in src/diagnostic/hybridrag_diagnostic.py
#   For full system diagnostics, use: python -m src.diagnostic
# ============================================================================
import subprocess
import sys
import os

def main():
    """Open the index status tool for quick review."""
    tool_path = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "index_status.py")
    tool_path = os.path.abspath(tool_path)
    if os.path.exists(tool_path):
        if sys.platform == "win32":
            subprocess.Popen(["notepad", tool_path])
        else:
            print(f"  Open this file: {tool_path}")
    else:
        print(f"  [WARN] File not found: {tool_path}")
        print("  Use 'python -m src.diagnostic' for full diagnostics instead.")

if __name__ == "__main__":
    main()
