# HybridRAG3 -- Troubleshooting Cheat Sheet
# ============================================================================
# Quick reference for setup, daily use, maintenance, and scaling.
# Designed to be read on your phone via the GitHub mobile app.
# ============================================================================


## 1. SETUP (First-Time Installation)

### Prerequisites Checklist
```
[ ] Windows 10/11
[ ] Python 3.11 (py -3.11 --version)
[ ] Git installed (git --version)
[ ] ~3 GB free disk space
[ ] Ollama installed (optional, for offline LLM mode)
[ ] Tesseract installed (optional, for OCR on scanned PDFs)
```

### Fresh Install (10 Minutes)
```powershell
# 1. Clone or extract the project
cd {PROJECT_ROOT}

# 2. Create virtual environment (MUST use Python 3.11)
py -3.11 -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies (~800 MB download, includes PyTorch)
pip install -r requirements.txt

# 5. Edit machine-specific paths
notepad start_hybridrag.ps1
#   Set $DATA_DIR and $SOURCE_DIR for YOUR machine

# 6. Launch environment
. .\start_hybridrag.ps1

# 7. Verify everything works
rag-diag
```

### Common Setup Problems

| Problem | Cause | Fix |
|---------|-------|-----|
| `py -3.11` not found | Python 3.11 not installed or not in PATH | Install from python.org, check "Add to PATH" |
| `pip install` hangs on torch | PyTorch is ~280 MB | Be patient, or use `--timeout 300` |
| `rag-diag` command not found | start_hybridrag.ps1 not sourced | Run `. .\start_hybridrag.ps1` (note the dot-space) |
| "Execution policy" error | PowerShell blocks scripts | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Wrong Python version in venv | Created venv with wrong Python | Delete .venv/, recreate with `py -3.11 -m venv .venv` |
| Model download fails | No internet / firewall blocks HuggingFace | First run needs internet. Model caches locally after. |


## 2. DAILY USE

### Opening PowerShell (If You Have Never Used It)

1. Press **Win+X** on your keyboard (hold the Windows key, tap X).
   A menu appears in the bottom-left corner.
2. Click **Terminal** (or **Windows PowerShell**).
   A dark window with a blinking cursor appears. This is PowerShell.
3. Type the following and press **Enter**:
   ```
   cd "{PROJECT_ROOT}"
   ```
4. You are now in the project folder. Continue with "Starting Up" below.

### Starting Up
```powershell
# Always run this first (sets paths, security, aliases)
# Note the dot-space-dot at the start -- this is required
. .\start_hybridrag.ps1

# Or double-click start_rag.bat from File Explorer (does the same thing)
```

### Launching the GUI
```powershell
# Option A: PowerShell script
.\tools\launch_gui.ps1

# Option B: Python directly
python src/gui/launch_gui.py
```
Wait for `[OK] Backends attached to GUI` in the terminal before querying.

### Core Commands (CLI)
```
rag-paths              Show configured paths + network status
rag-index              Index all documents in source folder
rag-query "question"   Ask a question about your documents
rag-diag               Run full diagnostic suite
rag-status             Quick health check
rag-profile            Show current performance profile
rag-server             Start FastAPI REST API server (http://127.0.0.1:8000)
```

### API Mode (Online Queries)
```
rag-store-key          Store API key (encrypted in Windows Credential Manager)
rag-store-endpoint     Store API endpoint URL
rag-cred-status        Check what credentials are stored
rag-mode-online        Switch to API mode
rag-mode-offline       Switch to Ollama mode
rag-test-api           Test API connectivity
```

### Performance Profiles
```
rag-profile laptop_safe       8-16 GB RAM, conservative
rag-profile desktop_power     32-64 GB RAM, aggressive batching
rag-profile server_max        64+ GB RAM, maximum throughput
```

### REST API Server (FastAPI)
```powershell
# Start the API server (default: localhost:8000)
rag-server

# Custom port
rag-server -Port 9000

# Open Swagger docs in browser
start http://localhost:8000/docs
```

**API Endpoints:**
```
GET  /health         Health check (always returns 200 if running)
GET  /status         Database stats, mode, chunk/source counts
GET  /config         Current configuration (read-only)
POST /query          Ask a question (body: {"question": "..."})
POST /index          Start indexing (background, non-blocking)
GET  /index/status   Check indexing progress
PUT  /mode           Switch offline/online (body: {"mode": "offline"})
```

**Security:** The server binds to localhost only (127.0.0.1).
It does NOT open any ports to the network by default.


## 3. TROUBLESHOOTING BY ERROR CODE

### Configuration Errors (CONF-xxx)

| Code | Error | What It Means | Fix |
|------|-------|---------------|-----|
| CONF-001 | EndpointNotConfigured | API endpoint URL is empty | `rag-store-endpoint` and enter your URL |
| CONF-002 | ApiKeyNotConfigured | API key is missing | `rag-store-key` and enter your key |
| CONF-003 | InvalidEndpoint | URL is malformed | Check for typos, must start with https:// |
| CONF-004 | DeploymentNotConfigured | Azure deployment name missing | Set in default_config.yaml under api.deployment |
| CONF-005 | ProviderConfigError | Invalid provider setting | Check api section in default_config.yaml |
| CONF-006 | ApiVersionNotConfigured | Azure API version missing | Set in default_config.yaml under api.api_version |

### Network Errors (NET-xxx)

| Code | Error | What It Means | Fix |
|------|-------|---------------|-----|
| NET-001 | ConnectionFailed | Cannot reach server | Check network, VPN, or if Ollama is running |
| NET-002 | TLSValidation | SSL certificate error | Often a corporate proxy issue. Check proxy settings. |
| NET-003 | ProxyError | Proxy connection failed | Verify proxy URL in environment variables |
| NET-004 | Timeout | Request took too long | Increase timeout in config, or check server load |

### API/Auth Errors (API-xxx)

| Code | Error | What It Means | Fix |
|------|-------|---------------|-----|
| API-001 | AuthRejected (401) | Invalid API key | Re-enter key with `rag-store-key` |
| API-002 | Forbidden (403) | Key lacks permissions | Check API key scopes/permissions |
| API-003 | NotFound (404) | Wrong endpoint or deployment | Verify endpoint URL and model/deployment name |
| API-004 | RateLimited (429) | Too many requests | Wait and retry, or reduce query rate |
| API-005 | ServerError (5xx) | Server-side issue | Wait and retry; not your fault |

### Ollama Errors (OLL-xxx)

| Code | Error | What It Means | Fix |
|------|-------|---------------|-----|
| OLL-001 | OllamaNotRunning | Ollama service not started | Start Ollama: `ollama serve` in a new terminal |
| OLL-002 | ModelNotFound | Requested model not downloaded | `ollama pull phi4-mini` |

### Index Errors (IDX-xxx)

| Code | Error | What It Means | Fix |
|------|-------|---------------|-----|
| IDX-001 | IndexNotFound | Database file missing | Run `rag-index` to create it |
| IDX-002 | IndexCorrupted | Database or memmap damaged | See "Database Recovery" below |


## 4. MAINTENANCE

### Regular Maintenance Tasks

**Weekly:**
- Run `rag-diag` to check system health
- Review logs/ folder for errors
- Check `rag-status` for stale index warnings

**After Adding New Documents:**
- Run `rag-index` to index new files
- The indexer skips already-indexed files (incremental)
- New files are detected by size + modification time

**After Updating Code (git pull):**
- Activate venv: `.\.venv\Scripts\Activate.ps1`
- Check for new dependencies: `pip install -r requirements.txt`
- Source environment: `. .\start_hybridrag.ps1`
- Run diagnostics: `rag-diag`

### Database Recovery

If the SQLite database or memmap files are corrupted:

```powershell
# Option 1: Rebuild memmap from SQLite (keeps all data)
python -m src.tools.rebuild_memmap_from_sqlite

# Option 2: Full re-index (nuclear option -- deletes and rebuilds everything)
# Delete the database files, then re-index:
#   Remove hybridrag.sqlite3, embeddings.f16.dat, embeddings_meta.json
#   Then run rag-index
```

### Log File Locations
```
logs/hybridrag.log           Main application log
logs/indexing_run_*.log       Per-run indexing logs
logs/diagnostic_*.log         Diagnostic run logs
```

### Cleaning Up Disk Space
```powershell
# These are safe to delete (will be recreated as needed):
logs/                         Log files (re-created on next run)
__pycache__/                  Python bytecode cache

# These are safe to delete but require re-download:
.model_cache/                 Embedding model (87 MB, re-downloads)
.hf_cache/                    HuggingFace cache (re-downloads)

# NEVER delete these (data loss):
# hybridrag.sqlite3            Your indexed data
# embeddings.f16.dat           Your embedding vectors
# .venv/                       Your installed packages
```


## 5. BULK FILE TRANSFER (New in V2)

### When to Use
When you need to copy thousands of files from a network drive into
HybridRAG's source folder for indexing. This replaces manual robocopy.

### Quick Start
```powershell
python -m src.tools.bulk_transfer_v2 `
    --sources "\\Server\Share\Engineering" "\\Server\Share\Reports" `
    --dest "D:\RAG_Staging"
```

### What It Does (vs. Robocopy)
```
Robocopy: Copies everything. Hope for the best.
V2:       Smart filter -> parallel copy -> hash verify -> atomic staging
```

### Key Options
```
--workers 8             Parallel threads (default 8, reduce if network is slow)
--no-dedup              Disable deduplication (copy duplicates too)
--no-verify             Skip SHA-256 verification (faster but risky)
--no-resume             Start fresh, ignore previous runs
--include-hidden        Include hidden/system files
--bandwidth-limit 50    Limit to 50 bytes/sec (use for throttling)
```

### Three-Stage Staging Explained
```
incoming/    Files land here as .tmp during copy (may be partial)
verified/    Files move here AFTER hash verification (safe to index)
quarantine/  Files with problems go here (check after transfer)
```

**Point HybridRAG's source_folder at the verified/ directory.**

### After a Transfer
1. Check `quarantine/` for failed files
2. Read `.reason` files next to quarantined files
3. Check the manifest database: `_transfer_manifest.db`
4. Point `rag-index` at the `verified/` folder
5. Run `rag-index` to index the transferred files

### Transfer Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Many "locked" files | Files open in Outlook/Word/Excel | Close the applications, or run transfer after hours |
| "Permission denied" errors | No read access to source | Check your network permissions with IT |
| Slow transfer speed | Network congestion | Reduce --workers to 4, or run during off-hours |
| "Hash mismatch" quarantines | Network errors during copy | Re-run transfer (resume skips already-copied files) |
| "Encoding issue" skips | Non-UTF-8 characters in filenames | Rename the file on the source drive |
| "Path too long" skips | Path > 260 characters | Shorten folder names on source, or move files up |


## 6. SCALING CONSIDERATIONS

### Know Your Bottlenecks

| Document Count | Bottleneck | What to Watch |
|---------------|-----------|---------------|
| < 1,000 | Nothing | Everything is fast |
| 1,000 - 10,000 | Embedding speed | Use GPU, increase batch_size |
| 10,000 - 50,000 | Disk I/O (memmap) | Use SSD, not HDD |
| 50,000 - 100,000 | SQLite write speed | WAL mode (already on by default) |
| > 100,000 | Memory for vector search | Monitor RAM during queries |

### Performance Tuning

**Speed Up Indexing:**
```yaml
# In config/default_config.yaml:
embedding:
  batch_size: 64          # Increase if you have 32+ GB RAM
  device: cuda            # Use GPU if available (10-50x faster)

indexing:
  block_chars: 500000     # Increase if RAM allows
```

**Speed Up Queries:**
```yaml
retrieval:
  top_k: 3               # Fewer results = faster search
  hybrid_search: true     # Keep ON (vector + keyword is better)
  reranker_enabled: false # Keep OFF unless you need precision
  min_score: 0.35         # Raise threshold to skip weak results
```

### RAM Usage Estimates

| Component | RAM per 1,000 Docs | Notes |
|-----------|-------------------|-------|
| Embedding model | ~400 MB (fixed) | Loaded once, shared |
| SQLite + FTS5 | ~50 MB | Grows with document count |
| Memmap vectors | ~1.5 MB | 384 dims x 4 bytes x chunks |
| Query-time vectors | ~20 MB | Loaded on demand |

**Rule of thumb: 1 GB base + 2 MB per 1,000 documents**

### When to Consider Upgrading

| Signal | What It Means | Upgrade Path |
|--------|--------------|--------------|
| Indexing takes > 1 hour | CPU embedding bottleneck | Add GPU or use CUDA |
| Queries take > 5 seconds | Vector search is slow | Switch to FAISS or upgrade RAM |
| SQLite DB > 5 GB | Large corpus | Consider PostgreSQL + pgvector |
| > 500K chunks in memmap | Memmap reads slow down | Switch to FAISS HNSW index |
| Multiple users querying | SQLite single-writer lock | Move to client-server architecture |

### What NOT to Scale (Diminishing Returns)
- **Workers > 16**: More threads does not help on a single network link
- **Chunk overlap > 300**: More overlap = more chunks = slower, not better
- **top_k > 10**: LLM context window fills up, quality drops
- **batch_size > 128**: GPU memory limit; crashes, not faster


## 7. ARCHITECTURE QUICK REFERENCE

### Data Flow
```
Documents -> Parser -> Raw Text -> Chunker -> Chunks -> Embedder -> Vectors
                                                                       |
                                                                       v
                                                              SQLite + Memmap
                                                                       |
Query -> Embedder -> Vector Search -+-> RRF Fusion -> Top-K -> LLM -> Answer
                    BM25 Search ----+

Access methods:
  CLI:      rag-query "question"         (PowerShell command)
  REST API: POST http://localhost:8000/query  (FastAPI server via rag-server)
  Swagger:  http://localhost:8000/docs   (interactive API docs)
```

### Key Config Values
```yaml
Chunk size:        1200 characters (with 200 overlap)
Embedding model:   all-MiniLM-L6-v2 (384 dimensions)
Vector store:      SQLite + memory-mapped float16 file
Search method:     Hybrid (vector cosine + BM25 keyword + RRF fusion)
RRF k-value:       60 (balances vector vs keyword ranking)
Top-K results:     5 (number of chunks sent to LLM)
Min score:         0.30 (below this, chunks are discarded)
Security:          3-layer network lockdown (PS + Python + Config)
```

### File Locations That Matter
```
config/default_config.yaml    All settings (edit this)
config/profiles.yaml          Hardware profiles
start_hybridrag.ps1           Machine-specific paths (edit per machine)
src/core/config.py            Config loader and validation
src/core/indexer.py            Main indexing engine
src/core/retriever.py          Query/search engine
src/core/boot.py               Startup validation pipeline
src/api/server.py              FastAPI REST API server
src/api/routes.py              API endpoint definitions
src/api/models.py              Request/response schemas
```


## 8. EMERGENCY PROCEDURES

### "Everything is Broken"
```powershell
# 1. Check Python version
py -3.11 --version

# 2. Recreate virtual environment
Remove-Item -Recurse .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Re-source environment
. .\start_hybridrag.ps1

# 4. Run diagnostics
rag-diag --verbose
```

### "Queries Return Garbage"
1. Check `rag-status` -- is the index populated?
2. Check `rag-diag` -- are there embedding errors?
3. Try a simple query: `rag-query "test"`
4. Check min_score threshold (lower = more results, possibly worse)
5. Check if source documents are actually parseable (not scanned images without OCR)

### "Indexing Crashes Midway"
1. Just re-run `rag-index` -- it resumes from where it left off
2. Crashed chunks are skipped via INSERT OR IGNORE
3. Check logs/ for the specific error
4. If it keeps crashing on the same file, that file may be corrupt

### "Out of Memory During Indexing"
1. Switch to laptop_safe profile: `rag-profile laptop_safe`
2. Reduce batch_size in config: `embedding.batch_size: 8`
3. Reduce block_chars: `indexing.block_chars: 100000`
4. Close other applications during indexing

### "Database Locked" Errors
1. Check if another rag-index process is running
2. Close any SQLite browser tools (DB Browser, etc.)
3. Delete any .sqlite3-wal and .sqlite3-shm files (safe to delete)
4. If the above fails, the database may be corrupted -- see "Database Recovery"
