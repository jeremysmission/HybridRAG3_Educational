# ============================================================================
# HybridRAG v3 - Switch Performance Profile (scripts/_profile_switch.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Changes ALL hardware-dependent settings in config/default_config.yaml
#   to match one of three predefined profiles. This controls:
#     - Embedding model (which AI embeds your documents)
#     - Embedding dimension (384 / 768 / 1024)
#     - Embedding device (cpu / cuda)
#     - LLM model (which AI answers your questions)
#     - LLM context window (how much text the AI can read)
#     - Batch size, top_k, block_chars (performance tuning)
#
# USAGE (called by PowerShell, not directly):
#   python scripts/_profile_switch.py laptop_safe
#   python scripts/_profile_switch.py desktop_power
#   python scripts/_profile_switch.py server_max
#
# PROFILE DETAILS:
#
#   laptop_safe (8-16GB RAM, no GPU):
#     - Embedder: all-MiniLM-L6-v2 (384d, CPU)
#     - LLM: phi4-mini (3.8B, 8K context)
#     - batch_size=16, top_k=5, block=200K
#
#   desktop_power (64GB RAM, 12GB VRAM):
#     - Embedder: nomic-embed-text-v1.5 (768d, CUDA)
#     - LLM: mistral-nemo:12b (12B, 128K context)
#     - batch_size=64, top_k=10, block=500K
#
#   server_max (64GB+ RAM, 24GB+ VRAM):
#     - Embedder: snowflake-arctic-embed-l-v2.0 (1024d, CUDA)
#     - LLM: phi4:14b-q4_K_M (14B, 16K context)
#     - batch_size=128, top_k=15, block=1M
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-profile [profile_name]
#
# WHAT IT CHANGES:
#   config/default_config.yaml (embedding, ollama, retrieval, indexing)
#
# IMPORTANT:
#   If the embedding model changes, ALL documents must be RE-INDEXED.
#   Existing 384-dim vectors are incompatible with a 768-dim model.
#   This script detects model changes and prints a clear warning.
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

sys.path.insert(0, os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."))
sys.path.insert(0, os.path.join(
    os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."), "scripts"
))


def _config_path():
    """Build the full path to default_config.yaml using the project root."""
    root = os.environ.get('HYBRIDRAG_PROJECT_ROOT', '.')
    return os.path.join(root, 'config', 'default_config.yaml')


# -- Define the three profiles --
# Each profile is a dictionary of settings that get deep-merged into
# default_config.yaml. Keys match the YAML section names.
profiles = {
    'laptop_safe': {
        'embedding': {
            'model_name': 'all-MiniLM-L6-v2',
            'dimension': 384,
            'batch_size': 16,
            'device': 'cpu',
        },
        'ollama': {
            'model': 'phi4-mini',
            'context_window': 8192,
        },
        'retrieval': {'top_k': 5},
        'indexing': {'block_chars': 200000},
    },
    'desktop_power': {
        'embedding': {
            'model_name': 'nomic-ai/nomic-embed-text-v1.5',
            'dimension': 768,
            'batch_size': 64,
            'device': 'cuda',
        },
        'ollama': {
            'model': 'mistral-nemo:12b',
            'context_window': 16384,
        },
        'retrieval': {'top_k': 10},
        'indexing': {'block_chars': 500000},
    },
    'server_max': {
        'embedding': {
            'model_name': 'snowflake-arctic-embed-l-v2.0',
            'dimension': 1024,
            'batch_size': 128,
            'device': 'cuda',
        },
        'ollama': {
            'model': 'phi4:14b-q4_K_M',
            'context_window': 16384,
        },
        'retrieval': {'top_k': 15},
        'indexing': {'block_chars': 1000000},
    },
}


# -- Read the command-line argument --
if len(sys.argv) < 2 or sys.argv[1] not in profiles:
    print('Usage: python _profile_switch.py [laptop_safe|desktop_power|server_max]')
    sys.exit(1)

profile = sys.argv[1]
settings = profiles[profile]

# -- Read the current config --
cfg_file = _config_path()
with open(cfg_file, 'r') as f:
    cfg = yaml.safe_load(f)

# -- Detect embedding model change (requires re-index) --
old_model = cfg.get('embedding', {}).get('model_name', '')
new_model = settings.get('embedding', {}).get('model_name', '')
model_changed = old_model and new_model and old_model != new_model

# -- Apply the profile settings (deep merge) --
# Only changes the specific keys in each section, leaving all other
# settings untouched.
for section_name, values in settings.items():
    if section_name not in cfg:
        cfg[section_name] = {}
    for key, val in values.items():
        cfg[section_name][key] = val

# -- Save the updated config --
with open(cfg_file, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# -- Print confirmation --
desc = {
    'laptop_safe': (
        'Laptop (8-16GB RAM, CPU)\n'
        '  Embedder: all-MiniLM-L6-v2 (384d, cpu)\n'
        '  Default LLM: phi4-mini (3.8B, 8K ctx)'
    ),
    'desktop_power': (
        'Desktop (64GB RAM, 12GB VRAM)\n'
        '  Embedder: nomic-embed-text-v1.5 (768d, cuda)\n'
        '  Default LLM: mistral-nemo:12b (12B, 16K ctx)'
    ),
    'server_max': (
        'Server (64GB+ RAM, 24GB+ VRAM)\n'
        '  Embedder: snowflake-arctic-embed-l-v2.0 (1024d, cuda)\n'
        '  Default LLM: phi4:14b-q4_K_M (14B, 16K ctx)'
    ),
}
print('[OK]  Profile applied: ' + profile)
print('  ' + desc[profile])

# -- Show ranked model table per use case --
try:
    from _model_meta import get_profile_ranking_table, USE_CASES

    table = get_profile_ranking_table(profile)
    if table:
        print('')
        print('  Best model per use case on this hardware:')
        print('  %-22s %-22s %s' % ('Use Case', '#1 (default)', '#2 (fallback)'))
        print('  %-22s %-22s %s' % ('-' * 22, '-' * 22, '-' * 22))

        display_order = ['sw', 'eng', 'sys', 'draft', 'log', 'pm', 'fe', 'cyber', 'gen']
        for uc_key in display_order:
            if uc_key not in table:
                continue
            ranked = table[uc_key]
            label = USE_CASES[uc_key]['label']
            col1 = ranked[0]['model'] if len(ranked) > 0 else '---'
            col2 = ranked[1]['model'] if len(ranked) > 1 else '---'
            print('  %-22s %-22s %s' % (label, col1, col2))
except Exception:
    pass  # Model ranking is informational; never block profile switch

# -- Warn about re-index if embedding model changed --
if model_changed:
    print('')
    print('[WARN] Embedding model changed: %s -> %s' % (old_model, new_model))
    print('       Existing vectors are INCOMPATIBLE with the new model.')
    print('       You MUST re-index all documents before querying.')
    print('       Run: rag-index "D:\\RAG Source Data"')
