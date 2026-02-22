# ============================================================================
# HybridRAG v3 - Set Mode to Online (scripts/_set_online.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Opens the config file (config/default_config.yaml), changes the
#   "mode" setting from "offline" to "online", and saves it.
#
#   After this runs, the next time you use rag-query, HybridRAG will
#   send your question to the company GPT API instead of local Ollama.
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-mode-online function
#   You never need to run this file directly.
#
# WHAT IT CHANGES:
#   config/default_config.yaml -> mode: online
#
# WHAT IT DOES NOT CHANGE:
#   HuggingFace lockdown stays active (HF_HUB_OFFLINE=1)
#   Model caches stay local
#   All security layers stay active
#
# PORTABILITY:
#   Uses HYBRIDRAG_PROJECT_ROOT env var to find config on any machine.
#   Falls back to current directory if the env var is not set.
#
# INTERNET ACCESS: NONE. Only modifies a local file.
# ============================================================================

# os gives us access to environment variables and path building.
# yaml reads and writes YAML files (the .yaml config format).
import os
import yaml


def _config_path():
    """Build the full path to default_config.yaml using the project root.

    WHY THIS EXISTS:
      If PowerShell's working directory is not the repo root (for example,
      if you cd somewhere else before running a script), a bare relative
      path like 'config/default_config.yaml' would fail with FileNotFoundError.
      By reading HYBRIDRAG_PROJECT_ROOT (set by start_hybridrag.ps1), we
      always find the config regardless of the current directory.
    """
    root = os.environ.get('HYBRIDRAG_PROJECT_ROOT', '.')
    return os.path.join(root, 'config', 'default_config.yaml')


# Step 1: Open the config file and read its contents into a Python dictionary.
# _config_path() builds the full portable path instead of a bare relative one.
cfg_file = _config_path()
with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

# Step 2: Change the mode setting to "online".
# This is like editing the file by hand and changing "mode: offline"
# to "mode: online", but done programmatically so there are no typos.
cfg['mode'] = 'online'

# Step 3: Write the updated config back to the file.
# IMPORTANT: We read FIRST (above), then write SECOND (here) -- never
# read and write the same file in a single expression.
# default_flow_style=False keeps the YAML in the readable multi-line format
# instead of compressing it into one long line.
with open(cfg_file, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# Step 4: Confirm the change
print('Mode set to: online')
