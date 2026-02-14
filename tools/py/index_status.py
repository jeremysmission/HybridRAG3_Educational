import sys, os, sqlite3, glob
sys.path.insert(0, os.getcwd())

try:
    from src.core.config import Config
    cfg = Config()
    db_path = cfg.database.path if hasattr(cfg, "database") else None
except:
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
        except: pass
        
        # Count unique files
        try:
            cur.execute("SELECT COUNT(DISTINCT source_file) FROM chunks")
            files = cur.fetchone()[0]
            print(f"  Files:    {files}")
        except: pass
        
        # Last indexed
        try:
            cur.execute("SELECT MAX(indexed_at) FROM chunks")
            last = cur.fetchone()[0]
            if last:
                print(f"  Last indexed: {last}")
        except: pass
        
        # Index runs
        try:
            cur.execute("SELECT COUNT(*) FROM index_runs")
            runs = cur.fetchone()[0]
            print(f"  Total runs:   {runs}")
        except: pass
        
        conn.close()
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    print()
    break