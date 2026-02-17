# ============================================================================
# HybridRAG v3 - Set Mode to Offline (scripts/_set_offline.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Opens the config file (config/default_config.yaml), changes the
#   "mode" setting from "online" to "offline", and saves it.
#
#   After this runs, the next time you use rag-query, HybridRAG will
#   use local Ollama instead of the cloud API.
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-mode-offline function
#   You never need to run this file directly.
#
# PORTABILITY:
#   Uses HYBRIDRAG_PROJECT_ROOT env var to find config on any machine.
#   Falls back to current directory if the env var is not set.
#
# INTERNET ACCESS: NONE. Only modifies a local file.
# ============================================================================

import os
import yaml


def _config_path():
    """Build the full path to default_config.yaml using the project root.

    WHY THIS EXISTS:
      If PowerShell's working directory is not the repo root, a bare
      relative path like 'config/default_config.yaml' would fail.
      HYBRIDRAG_PROJECT_ROOT (set by start_hybridrag.ps1) ensures we
      always find the config regardless of the current directory.
    """
    root = os.environ.get('HYBRIDRAG_PROJECT_ROOT', '.')
    return os.path.join(root, 'config', 'default_config.yaml')


# Step 1: Read the current config
cfg_file = _config_path()
with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

# Step 2: Change mode to offline
cfg['mode'] = 'offline'

# Step 3: Write back (separate from read -- never read+write same file
# in one expression)
with open(cfg_file, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# Step 4: Confirm
print('Mode set to: offline')
