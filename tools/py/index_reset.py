# ============================================================================
# HybridRAG -- Index Reset (tools/py/index_reset.py)
# ============================================================================
#
# *** WARNING: THIS IS DESTRUCTIVE -- IT DELETES YOUR ENTIRE INDEX ***
#
# WHAT THIS DOES:
#   Deletes all database files (.sqlite3, .db) and embedding cache files
#   (.npy) from the project, forcing a complete re-index from scratch.
#
# WHEN TO USE:
#   - Your index is corrupted and queries return garbage
#   - You changed the embedding model and need fresh embeddings
#   - You want to start clean after major config changes
#
# WHAT YOU LOSE:
#   - All indexed document chunks (the searchable database)
#   - All pre-computed embeddings (the math that makes search fast)
#   - You will need to re-run "rag-index" which can take hours
#
# WHAT YOU KEEP:
#   - Your original source documents (untouched)
#   - Your config files and API keys
#   - All code and scripts
#
# ANALOGY:
#   Like erasing a book's index pages -- the book content is still there,
#   but you have to rebuild the index before you can look things up again.
#
# HOW TO USE:
#   python tools/py/index_reset.py
#   Then run: rag-index to rebuild
# ============================================================================
import sys, os, glob
sys.path.insert(0, os.getcwd())

candidates = glob.glob("**/*.sqlite3", recursive=True) + glob.glob("**/*.db", recursive=True)
deleted = []

for db in candidates:
    if os.path.exists(db):
        size = os.path.getsize(db) / (1024*1024)
        os.remove(db)
        deleted.append(f"{db} ({size:.1f} MB)")
        print(f"  [DELETED] {db} ({size:.1f} MB)")

# Also clean numpy memmap files
for npy in glob.glob("**/*.npy", recursive=True):
    if os.path.exists(npy):
        os.remove(npy)
        print(f"  [DELETED] {npy}")

if not deleted:
    print("  No database files found to delete.")
else:
    print()
    print("  Index cleared. Run rag-index to rebuild.")