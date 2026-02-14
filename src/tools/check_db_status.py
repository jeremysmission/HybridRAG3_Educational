# ============================================================================
# HybridRAG — Quick Database Status Check (src/tools/check_db_status.py)
# ============================================================================
#
# WHAT THIS DOES:
#   A quick 3-line check to see if your database exists and has data.
#   Run this when you want to know "did indexing work?"
#
# USAGE:
#   python src/tools/check_db_status.py
#
# OUTPUT EXAMPLE:
#   DB: D:\HybridRAG_Data\hybridrag.sqlite3
#   Chunks: 12,847
#   Source files: 342
#
# INTERNET ACCESS: None
# ============================================================================

import os, sqlite3

# The database location comes from the HYBRIDRAG_DATA_DIR environment variable
# This is set by start_hybridrag.ps1 when you activate the environment
db = os.path.join(os.getenv('HYBRIDRAG_DATA_DIR', ''), 'hybridrag.sqlite3')

if not os.path.exists(db):
    print('DB not found at:', db)
    print('Run rag-index first.')
else:
    # Open the database and count what's in it
    con = sqlite3.connect(db)
    try:
        count = con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        sources = con.execute('SELECT COUNT(DISTINCT source_path) FROM chunks').fetchone()[0]
        print(f'DB: {db}')
        print(f'Chunks: {count}')
        print(f'Source files: {sources}')
    except Exception as e:
        # Table might not exist if indexing never ran
        print('DB exists but chunks table missing:', e)
    con.close()
