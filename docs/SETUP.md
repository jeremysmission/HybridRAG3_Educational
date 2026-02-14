# HybridRAG v3 -- Setup Guide

Last Updated: 2026-02-10

## Prerequisites

- Windows 10/11
- Python 3.11 (tested on 3.11.9, must match across all machines)
- Git (for home machine version control)
- ~3GB disk space (venv + model cache)
- Ollama (optional, for offline LLM mode)
- Tesseract (optional, for OCR on scanned documents)

## Fresh Install (New Machine)

### Step 1: Get the project files

**Option A -- From GitHub (home machine):**
```powershell
cd "D:\"
git clone https://github.com/the developersmission/HybridRAG3.git
cd HybridRAG3
```

**Option B -- From zip file (work machine):**
Download the repo zip from GitHub (Code button, Download ZIP).
Extract to your project folder (e.g., {PROJECT_ROOT}).

### Step 2: Create virtual environment

Use the exact Python version to match other machines:
```powershell
cd "{PROJECT_ROOT}"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If py -3.11 doesn't work, check installed versions with:
```powershell
py --list
```

### Step 3: Upgrade pip

```powershell
python -m pip install --upgrade pip
```

If you get an access denied error:
```powershell
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Step 4: Install dependencies

```powershell
pip install -r requirements.txt
```

Total download is ~800MB. The largest package is PyTorch (~280MB).

**If torch fails to install:** Make sure requirements.txt has:
```
torch==2.10.0
```
Not the old torch==2.5.1+cpu line. The 2.10.0 version includes both
CPU and GPU support automatically.

**If scikit-learn fails:** It must match your Python version.
Python 3.11 uses scikit-learn==1.8.0. Python 3.10 uses scikit-learn==1.7.2.

**If sympy conflicts:** Make sure requirements.txt has:
```
sympy>=1.13.3
```
Torch 2.10 requires sympy 1.13.3 or newer.

### Step 5: Configure machine-specific paths

Edit start_hybridrag.ps1 and set the paths for THIS machine:
```powershell
notepad start_hybridrag.ps1
```

Find and update these lines:
```powershell
$DATA_DIR   = "{DATA_DIR}"     # Where SQLite + embeddings go
$SOURCE_DIR = "{SOURCE_DIR}"       # Where your documents are
```

These paths are different on each machine. Do NOT copy start_hybridrag.ps1
between machines without updating the paths.

### Step 6: Launch the environment

```powershell
. .\start_hybridrag.ps1
```

Or double-click start_rag.bat from File Explorer.

You should see:
- Python version confirmation
- Network lockdown status (all blocked)
- Configured paths
- Available commands list
- "API + Profile Commands loaded" with 8 commands

### Step 7: Run diagnostics

```powershell
rag-diag
```

Expected results on a fresh install:
- Most tests PASS
- Embedder test may show "use --test-embed for live" (normal)
- Ollama test may show not running (normal if not installed yet)
- BUG-002 and BUG-004 are known, documented, low priority

### Step 8: Download the embedding model (first time only)

The model downloads automatically on first rag-index or rag-diag --test-embed.
It's ~87MB and gets cached in .model_cache/ inside the project folder.

If you're on an offline machine, copy the .model_cache/ folder from a
connected machine.

### Step 9: Install Ollama (optional, for offline mode)

Download from https://ollama.com and install. Then pull the model:
```powershell
ollama pull llama3
```

Start the Ollama server (needs its own terminal window):
```powershell
ollama serve
```

Verify it's running:
```powershell
curl http://localhost:11434/api/tags
```

**Corporate proxy note:** If you get a 301 redirect error when querying
Ollama, the corporate proxy is intercepting localhost requests. The
NO_PROXY=localhost,127.0.0.1 line in start_hybridrag.ps1 fixes this.

### Step 10: Set up API mode (optional, for online mode)

Store your API key securely:
```powershell
rag-store-key
```
This stores the key in Windows Credential Manager (DPAPI encrypted).
The key never appears in any file, log, or config.

Store your company API endpoint:
```powershell
rag-store-endpoint
```

Verify credentials:
```powershell
rag-cred-status
```

Test the connection:
```powershell
rag-test-api
```

Switch to online mode:
```powershell
rag-mode-online
```

### Step 11: Set performance profile

Check your current profile:
```powershell
rag-profile
```

Switch based on your hardware:
```powershell
rag-profile laptop_safe       # 8-16GB RAM (work laptop)
rag-profile desktop_power     # 32-64GB RAM (power desktop)
rag-profile server_max        # 64GB+ RAM (server/workstation)
```

### Step 12: Index your documents

```powershell
rag-index
```

First run will:
1. Load the embedding model (downloads if not cached)
2. Parse all supported files in source folder
3. Chunk text, compute embeddings, store in SQLite + memmap
4. Auto-rebuild FTS5 keyword index

### Step 13: Query

```powershell
rag-query "What is the operating frequency range?"
```

### Step 14: Add to PowerShell profile (optional)

To auto-load HybridRAG in every new terminal:
```powershell
notepad $PROFILE
```

Add these lines:
```powershell
cd "{PROJECT_ROOT}"
. .\start_hybridrag.ps1
Write-Host "HybridRAG environment ready." -ForegroundColor Green
```

## Updating an Existing Machine

### From GitHub (home machine):
```powershell
cd "{PROJECT_ROOT}"
git pull origin main
. .\start_hybridrag.ps1
```

### From zip file (work machine):

1. Download the repo zip from GitHub (no login needed for public repos)
2. Before extracting, set aside machine-specific files:
```powershell
Copy-Item start_hybridrag.ps1 start_hybridrag_KEEP.ps1
Copy-Item .gitignore gitignore_KEEP.txt
```
3. Extract zip contents over the project folder ("Replace all")
4. Restore machine-specific files:
```powershell
Copy-Item start_hybridrag_KEEP.ps1 start_hybridrag.ps1 -Force
Copy-Item gitignore_KEEP.txt .gitignore -Force
Remove-Item start_hybridrag_KEEP.ps1
Remove-Item gitignore_KEEP.txt
```
5. Test:
```powershell
. .\start_hybridrag.ps1
rag-cred-status
rag-profile
rag-diag
```

## Machine-Specific Files

These files are different on each machine and should NOT be overwritten
when syncing code between machines:

| File | Why it's different |
|------|--------------------|
| start_hybridrag.ps1 | Different project/data/source paths per machine |
| .gitignore | May have machine-specific entries |
| .venv/ | Different Python builds, local packages |
| .model_cache/ | Downloaded locally, 87MB |
| .hf_cache/ | HuggingFace cache, local |
| .torch_cache/ | PyTorch cache, local |
| config/system_profile.json | Auto-detected hardware fingerprint |

These files sync normally between machines:
- All source code (src/, scripts/, tests/)
- Config templates (config/default_config.yaml, config/profiles.yaml)
- Documentation (docs/, README.md)
- Requirements files
- api_mode_commands.ps1, start_rag.bat

## Directory Structure

```
HybridRAG3/                         Project root
|
|-- .venv/                          Python virtual environment (local, not in git)
|-- .model_cache/                   Embedding model cache (local, not in git)
|
|-- config/
|   |-- default_config.yaml         All settings (paths, models, thresholds)
|   |-- profiles.yaml               Hardware profile definitions
|   +-- system_profile.json         Auto-detected hardware (local, not in git)
|
|-- src/
|   |-- core/                       Core RAG pipeline
|   |-- parsers/                    Document parsers (PDF, DOCX, PPTX, etc.)
|   |-- security/                   Credential management (keyring/DPAPI)
|   |-- diagnostic/                 Testing, benchmarks, fault analysis
|   |-- tools/                      Utility scripts and entry points
|   |-- monitoring/                 Structured logging and run tracking
|   +-- gui/                        GUI (placeholder for future development)
|
|-- scripts/                        Helper Python scripts for PowerShell commands
|   |-- _check_creds.py            Check credential status
|   |-- _set_online.py             Set config mode to online
|   |-- _set_offline.py            Set config mode to offline
|   |-- _test_api.py               API connectivity test
|   |-- _profile_status.py         Show current performance profile
|   +-- _profile_switch.py         Switch performance profile
|
|-- tests/
|   +-- cli_test_phase1.py         rag-query entry point
|
|-- docs/                           Technical documentation
|-- logs/                           Diagnostic and run logs (not in git)
|
|-- api_mode_commands.ps1           API mode + profile PowerShell commands
|-- start_hybridrag.ps1             Environment setup (machine-specific paths)
|-- start_rag.bat                   Double-click launcher
|-- requirements.txt                Python dependencies
|-- requirements-lock.txt           Exact versions installed
+-- README.md                       Project overview
```

## Troubleshooting

**"Model not found" on first run:**
The embedding model downloads automatically (~87MB). If on an offline
machine, copy the .model_cache/ folder from a connected machine.

**Ollama timeout:**
First query after model load takes longer. Default timeout is 180s.
If still timing out, increase timeout_seconds in config.

**Corporate proxy blocks Ollama:**
The NO_PROXY=localhost,127.0.0.1 line in start_hybridrag.ps1 prevents
the corporate proxy from intercepting localhost connections to Ollama.
If you still have issues, check that the line is present and that you
ran . .\start_hybridrag.ps1 (not just .\start_hybridrag.ps1).

**Permission errors on network drives:**
Run rag-index as the same user who has read access to the network share.
The indexer has retry logic (3 attempts with exponential backoff).

**"No relevant information found":**
Try lowering min_score in config (e.g., 0.10) or check that FTS rebuilt
successfully (look for fts_rebuilt in indexing output).

**PowerShell parse errors in api_mode_commands.ps1:**
This was caused by inline Python code inside PowerShell. Fixed in the
current version by moving all Python to separate .py files in scripts/.
If you see these errors, you have an old version of api_mode_commands.ps1.

**pip install fails with access denied:**
Use python -m pip instead of bare pip:
```powershell
python -m pip install -r requirements.txt
```

**sympy/torch version conflict:**
Make sure requirements.txt has sympy>=1.13.3 (not sympy==1.13.1).
Torch 2.10 requires the newer version.

**scikit-learn version mismatch:**
Python 3.11 needs scikit-learn==1.8.0. Python 3.10 needs scikit-learn==1.7.2.
This is why all machines should use the same Python version.

## Configuration Reference

All settings in config/default_config.yaml. Key settings:

| Setting | Default | What it does |
|---------|---------|-------------|
| mode | offline | "offline" (Ollama) or "online" (GPT API) |
| source_folder | (env var) | Path to documents to index |
| chunk_size | 1200 | Characters per chunk |
| overlap | 200 | Overlap between chunks |
| top_k | 5 | Number of chunks returned per query |
| min_score | 0.20 | Minimum relevance threshold |
| hybrid_search | true | Enable vector + BM25 fusion |
| rrf_k | 60 | RRF smoothing parameter |
| timeout_seconds | 180 | LLM response timeout |
| embedding.batch_size | 16 | Chunks processed per embedding batch |
| embedding.model_name | all-MiniLM-L6-v2 | Embedding model |

## Environment Variables (set by start_hybridrag.ps1)

| Variable | Purpose |
|----------|---------|
| HYBRIDRAG_PROJECT_ROOT | Project folder path |
| HYBRIDRAG_INDEX_FOLDER | Source document folder path |
| HYBRIDRAG_DATA_DIR | Database/embeddings storage path |
| HYBRIDRAG_EMBED_BATCH | Override embedding batch size |
| HYBRIDRAG_RETRIEVAL_BLOCK_ROWS | Override vector search block size |
| HYBRIDRAG_NETWORK_KILL_SWITCH | Master network kill switch |
| HF_HUB_OFFLINE | Block HuggingFace downloads |
| TRANSFORMERS_OFFLINE | Block transformers downloads |
| HF_HUB_DISABLE_TELEMETRY | Block HuggingFace telemetry |
| NO_PROXY | Prevent proxy from intercepting localhost |
| PYTHONPATH | Python module search path |
