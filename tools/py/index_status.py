# ============================================================================
# HybridRAG -- Index Status (tools/py/index_status.py)
# ============================================================================
#
# WHAT THIS DOES:
#   Opens the SQLite database where HybridRAG stores its indexed document
#   chunks and shows you a summary: how many chunks, how many source
#   files, database size, and when the last indexing run happened.
#
# ANALOGY:
#   Like checking the table of contents of a book -- it tells you how
#   many entries are in the index without showing you all the content.
#
# HOW TO USE:
#   python tools/py/index_status.py
#
# TYPICAL OUTPUT:
#   Database: data/hybridrag.sqlite3
#   Size:     45.2 MB
#   Chunks:   39,602
#   Files:    1,345
#   Last indexed: 2026-02-15 14:30:00
# ============================================================================
import sys, os, sqlite3, glob
sys.path.insert(0, os.getcwd())

try:
    from src.core.config import Config
    cfg = Config()
    db_path = cfg.database.path if hasattr(cfg, "database") else None
except Exception:
    db_path = None

# Try common database locations
candidates = [db_path] if db_path else []
candidates += glob.glob("**/*.sqlite3", recursive=True)
candidates += glob.glob("**/*.db", recursive=True)

if not candidates:
    print("  No database found.")
    sys.exit(0)

for db in candidates:
    if not db or not os.path.exists(db):
        continue
    
    print(f"  Database: {db}")
    size_mb = os.path.getsize(db) / (1024 * 1024)
    print(f"  Size:     {size_mb:.1f} MB")
    
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        
        # Count chunks
        try:
            cur.execute("SELECT COUNT(*) FROM chunks")
            chunks = cur.fetchone()[0]
            print(f"  Chunks:   {chunks}")
        except Exception: pass
        
        # Count unique files
        try:
            cur.execute("SELECT COUNT(DISTINCT source_file) FROM chunks")
            files = cur.fetchone()[0]
            print(f"  Files:    {files}")
        except Exception: pass
        
        # Last indexed
        try:
            cur.execute("SELECT MAX(indexed_at) FROM chunks")
            last = cur.fetchone()[0]
            if last:
                print(f"  Last indexed: {last}")
        except Exception: pass
        
        # Index runs
        try:
            cur.execute("SELECT COUNT(*) FROM index_runs")
            runs = cur.fetchone()[0]
            print(f"  Total runs:   {runs}")
        except Exception: pass
        
        conn.close()
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print()
    break