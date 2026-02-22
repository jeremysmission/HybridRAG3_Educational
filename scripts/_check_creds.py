#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 - Check Credential Status (scripts/_check_creds.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Checks whether an API key and endpoint URL are available via the
#   3-layer credential system (keyring -> env vars -> config).
#   Prints results so rag-mode-online in PowerShell can decide whether
#   to proceed or show an error.
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-mode-online function
#   You never need to run this file directly.
#
# WHAT IT PRINTS:
#   KEY:True        (or KEY:False if no API key found anywhere)
#   ENDPOINT:True   (or ENDPOINT:False if no endpoint found anywhere)
#   KEY_SRC:keyring (or KEY_SRC:env:VAR_NAME or KEY_SRC:config or KEY_SRC:none)
#
# BUG FIX (2026-02-16):
#   Previous version imported 'credential_status' which did not exist
#   in credentials.py. This caused a silent ImportError (stderr was
#   redirected to $null in PowerShell), making rag-mode-online always
#   report "No API key found!" even when a key was stored.
#   Now uses resolve_credentials() which DOES exist and returns the
#   ApiCredentials dataclass with .has_key, .has_endpoint, .source_key.
#
# INTERNET ACCESS: NONE. Only reads from local credential stores.
# ============================================================================

import sys
import os

# Add the project root folder to Python's search path so we can import
# our own modules (like src.security.credentials). The project root is
# stored in an environment variable by start_hybridrag.ps1.
sys.path.insert(0, os.environ.get('HYBRIDRAG_PROJECT_ROOT', '.'))

# Import the credential resolver -- this is the canonical function that
# checks keyring, env vars, and config in priority order.
# It returns an ApiCredentials dataclass (not a dict).
from src.security.credentials import resolve_credentials

# Load config dict so resolve_credentials can check config file as fallback
import yaml
config_dict = None
try:
    with open('config/default_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
except Exception:
    pass

# Resolve credentials using the 3-layer system
creds = resolve_credentials(config_dict)

# Print results in the KEY:VALUE format that PowerShell parses.
# PowerShell looks for "KEY:True" in this output to decide what to do.
print('KEY:' + str(creds.has_key))
print('ENDPOINT:' + str(creds.has_endpoint))
print('KEY_SRC:' + str(creds.source_key or 'none'))
