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