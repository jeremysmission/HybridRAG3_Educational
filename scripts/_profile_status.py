# ============================================================================
# HybridRAG v3 - Show Profile Status (scripts/_profile_status.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Reads config/default_config.yaml and shows the current performance
#   profile settings: batch size, chunk size, and search depth.
#   Then infers which profile is active based on the batch size:
#     batch_size 16  = laptop_safe    (conservative, 8-16GB RAM)
#     batch_size 64  = desktop_power  (aggressive, 32-64GB RAM)
#     batch_size 128 = server_max     (maximum, 64GB+ RAM)
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-profile status
#   You never need to run this file directly.
#
# WHY PROFILES MATTER:
#   Higher batch sizes mean the embedder processes more chunks at once
#   during indexing, which is faster but uses more RAM. On a laptop
#   with 16GB, a batch of 128 could cause the system to slow down or
#   crash from running out of memory. On a 64GB machine, batch 128
#   is comfortable and indexes 8x faster than batch 16.
#
# INTERNET ACCESS: NONE. Only reads a local file.
# ============================================================================

import yaml

# Read the config file
with open('config/default_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Extract the key settings that define a profile.
# .get() with a default of '?' means "if the setting doesn't exist,
# show a question mark instead of crashing"
eb = cfg.get('embedding', {}).get('batch_size', '?')    # Embedding batch size
ck = cfg.get('chunking', {}).get('max_tokens', '?')     # Chunk size in tokens
tk = cfg.get('vector_search', {}).get('top_k', '?')     # How many results to return

# Print the current values
print('  Embedding batch_size: ' + str(eb))
print('  Chunk max_tokens:     ' + str(ck))
print('  Search top_k:         ' + str(tk))

# Infer which profile is active based on the batch size.
# If someone manually edited the config to a non-standard batch size,
# we just call it "custom".
if eb == 16:
    print('  Profile:              laptop_safe')
elif eb == 64:
    print('  Profile:              desktop_power')
elif eb == 128:
    print('  Profile:              server_max')
else:
    print('  Profile:              custom')
