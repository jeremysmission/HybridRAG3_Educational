# ============================================================================
# HybridRAG v3 - Switch Performance Profile (scripts/_profile_switch.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Changes the performance settings in config/default_config.yaml to
#   match one of three predefined profiles. This controls how fast
#   indexing runs and how many search results are returned.
#
# USAGE (called by PowerShell, not directly):
#   python scripts/_profile_switch.py laptop_safe
#   python scripts/_profile_switch.py desktop_power
#   python scripts/_profile_switch.py server_max
#
# PROFILE DETAILS:
#
#   laptop_safe (8-16GB RAM machines):
#     - Embedding batch_size: 16 (process 16 chunks at a time)
#     - Search top_k: 5 (return top 5 results per query)
#     - Indexing block: 200K chars (small blocks, less RAM)
#     - Concurrent files: 1 (one file at a time, safest)
#     - Best for: Your current work laptop (16GB RAM)
#
#   desktop_power (32-64GB RAM machines):
#     - Embedding batch_size: 64 (4x faster indexing)
#     - Search top_k: 10 (more results, better coverage)
#     - Indexing block: 500K chars (bigger blocks, faster)
#     - Concurrent files: 2 (two files at once)
#     - Best for: Your new work laptop (64GB RAM)
#
#   server_max (64GB+ RAM machines):
#     - Embedding batch_size: 128 (8x faster indexing)
#     - Search top_k: 15 (maximum coverage)
#     - Indexing block: 1M chars (large blocks, fastest)
#     - Concurrent files: 4 (four files at once)
#     - Best for: Overnight indexing runs on powerful hardware
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-profile [profile_name]
#   You never need to run this file directly.
#
# WHAT IT CHANGES:
#   config/default_config.yaml (embedding, vector_search, indexing sections)
#
# IMPORTANT:
#   After switching profiles, you need to RE-INDEX your documents for the
#   new batch size to take effect. Existing indexed data still works
#   fine with any profile - only indexing speed changes.
#
# PORTABILITY:
#   Uses HYBRIDRAG_PROJECT_ROOT env var to find config on any machine.
#   Falls back to current directory if the env var is not set.
#
# INTERNET ACCESS: NONE. Only modifies a local file.
# ============================================================================

import os
import sys
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


# -- Define the three profiles --
# Each profile is a dictionary of settings that get written into the
# config YAML file. The keys (like 'embedding', 'vector_search') match
# the section names in default_config.yaml.
profiles = {
    'laptop_safe': {
        'embedding': {'batch_size': 16},
        'vector_search': {'top_k': 5},
        'indexing': {'block_chars': 200000, 'max_concurrent_files': 1},
    },
    'desktop_power': {
        'embedding': {'batch_size': 64},
        'vector_search': {'top_k': 10},
        'indexing': {'block_chars': 500000, 'max_concurrent_files': 2},
    },
    'server_max': {
        'embedding': {'batch_size': 128},
        'vector_search': {'top_k': 15},
        'indexing': {'block_chars': 1000000, 'max_concurrent_files': 4},
    },
}

# -- Read the command-line argument --
# sys.argv is a list of arguments passed to this script.
# sys.argv[0] is the script name itself (_profile_switch.py)
# sys.argv[1] is the profile name (laptop_safe, desktop_power, or server_max)
if len(sys.argv) < 2 or sys.argv[1] not in profiles:
    print('Usage: python _profile_switch.py [laptop_safe|desktop_power|server_max]')
    sys.exit(1)

profile = sys.argv[1]
settings = profiles[profile]

# -- Read the current config --
cfg_file = _config_path()
with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

# -- Apply the profile settings (deep merge) --
# "Deep merge" means we only change the specific keys in each section,
# leaving all other settings untouched. For example, if the 'embedding'
# section also has a 'model_name' setting, we don't touch it - we only
# change 'batch_size'.
for section_name, values in settings.items():
    # If the section doesn't exist in the config yet, create it
    if section_name not in cfg:
        cfg[section_name] = {}
    # Update each individual setting within the section
    for key, val in values.items():
        cfg[section_name][key] = val

# -- Save the updated config --
# IMPORTANT: read and write are separate operations (never in one expression)
with open(cfg_file, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# -- Print confirmation --
desc = {
    'laptop_safe': 'Conservative - stability on 8-16GB RAM (batch=16)',
    'desktop_power': 'Aggressive - throughput on 32-64GB RAM (batch=64)',
    'server_max': 'Maximum - for 64GB+ workstations (batch=128)',
}
print('  Applied: ' + profile)
print('  ' + desc[profile])
