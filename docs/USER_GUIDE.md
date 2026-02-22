# HybridRAG3 -- User Guide

Last Updated: 2026-02-21

This guide covers daily use of HybridRAG after installation is
complete. For first-time setup, see [INSTALL_AND_SETUP.md](INSTALL_AND_SETUP.md).

---

## Table of Contents

1. [Starting Up](#1-starting-up)
2. [Indexing Documents](#2-indexing-documents)
3. [Asking Questions (CLI)](#3-asking-questions-cli)
4. [Using the GUI](#4-using-the-gui)
5. [Online vs Offline Mode](#5-online-vs-offline-mode)
6. [Managing Credentials](#6-managing-credentials)
7. [Performance Profiles](#7-performance-profiles)
8. [Engineering Tuning](#8-engineering-tuning)
9. [REST API](#9-rest-api)
10. [Supported File Formats](#10-supported-file-formats)
11. [Tips for Better Answers](#11-tips-for-better-answers)
12. [Common Problems and Fixes](#12-common-problems-and-fixes)

---

## 1. Starting Up

### Opening PowerShell (If You Have Never Used It)

1. Press **Win+X** on your keyboard (hold the Windows key, tap X).
2. Click **Terminal** (or **Windows PowerShell**).
3. Type `cd "{PROJECT_ROOT}"` and press **Enter**.

### Daily Startup (Command Line)

From your PowerShell terminal, run:

```powershell
. .\start_hybridrag.ps1
```

This loads all commands, sets security lockdown, and configures paths.
You only need to do this once per terminal session.

**Shortcut**: Add it to your PowerShell profile for automatic loading:

```powershell
notepad $PROFILE
```

Add:

```powershell
cd "{PROJECT_ROOT}"
. .\start_hybridrag.ps1
```

### Launching the GUI

```powershell
python src/gui/launch_gui.py
```

What happens:
1. Boot pipeline runs (2-4 seconds) -- loads config, checks
   credentials, sets network gate
2. Window opens immediately
3. Backends load in background (30-60 seconds on 8 GB laptop,
   nearly instant on 64 GB workstation)
4. Status bar updates when system is ready

You can look around the GUI while backends load, but queries and
indexing are not available until you see `[OK] Backends attached to GUI`
in the terminal.

---

## 2. Indexing Documents

Indexing reads your documents and builds the searchable database.
It only needs to run once per document set, or when documents change.

### From the Command Line

```powershell
rag-index
```

The system:
1. Scans the source folder recursively for supported file types
2. Skips files that have not changed since the last run
3. Extracts text from each new/changed file
4. Splits text into chunks (~1,200 characters each)
5. Computes embedding vectors for each chunk
6. Stores everything in the local database

**First run**: Takes a few hours for ~1,345 files. Subsequent runs
process only changed files and finish in seconds.

### From the GUI

1. In the **Index Panel**, click **Browse** and select your
   document folder
2. Click **Start Indexing**
3. Watch the progress bar and file counter
4. "Last run" label updates when complete

**Stopping early**: Click **Stop**. The system finishes the current
file, then halts. Click **Start Indexing** again later to resume
where it left off.

### What Gets Indexed

HybridRAG supports 49+ file formats. The most common:

| Format | Extensions |
|--------|-----------|
| PDF | .pdf |
| Word | .docx, .doc |
| PowerPoint | .pptx |
| Excel | .xlsx |
| Email | .eml, .msg, .mbox |
| Images (OCR) | .png, .jpg, .tiff, .bmp, .gif |
| Plain text | .txt, .md, .csv, .json, .xml, .log, .yaml |
| Web | .html, .htm |
| CAD | .dxf, .stp, .step, .stl |
| Diagrams | .vsdx |
| Security | .evtx, .pcap, .cer, .pem |
| Database | .accdb, .mdb |

For the full list, see [FORMAT_SUPPORT.md](FORMAT_SUPPORT.md).

### Re-Indexing

When documents change, run `rag-index` again. Only new and modified
files are processed (detected by file size and modification time).
Unchanged files are skipped automatically.

If a source file is deleted, its chunks are cleaned up on the next
indexing run.

---

## 3. Asking Questions (CLI)

```powershell
rag-query "What is the operating frequency of the XR-7?"
```

The system searches your indexed documents, retrieves the most
relevant passages, and sends them to an AI model that writes a
direct answer with source citations.

**Example output:**

```
Answer: The XR-7 transceiver operates at 2.4 GHz in the ISM band.

Sources:
  - Technical_Specification.pdf (2 chunks)
  - Installation_Guide.pdf (1 chunk)

Latency: 3421 ms | Tokens in: 450 | Tokens out: 45
```

### Tips for Good Questions

- **Be specific**: "What is the calibration procedure for the XR-7?"
  works better than "Tell me about the manual."
- **Use document terminology**: If your documents say "RF operating
  band," use that phrase in your question.
- **One question at a time**: Three separate queries get better answers
  than one compound question.
- **Include context when helpful**: "According to the installation
  guide, what access level is required?" helps the system find the
  right document.

---

## 4. Using the GUI

For the complete GUI reference with every button and slider explained,
see [GUI_GUIDE.md](GUI_GUIDE.md).

### Window Layout

```
+----------------------------------------------------------+
|  TITLE BAR  --  "HybridRAG v3"   [OFFLINE] [ONLINE] [Theme] |
+----------------------------------------------------------+
|  QUERY PANEL                                             |
|  Use case dropdown, model display, question box,         |
|  answer area, sources, metrics                           |
+----------------------------------------------------------+
|  INDEX PANEL                                             |
|  Folder picker, Start/Stop, progress bar, last run info  |
+----------------------------------------------------------+
|  STATUS BAR                                              |
|  LLM | Ollama | Gate        (auto-refreshes every 5 sec) |
+----------------------------------------------------------+
```

Menu bar: **File** | **Engineering** | **Help**

### Query Panel

1. **Use case dropdown** -- Select your role. This auto-selects the
   best AI model for that type of question.

   | Use Case | Best For |
   |----------|----------|
   | Software Engineering | Code, debugging, algorithms |
   | Engineering / STEM | Specs, math, technical analysis |
   | Systems Administration | Scripts, configs, networking |
   | Drafting / AutoCAD | Technical specs, drawings |
   | Logistics Analyst | Data analysis, part numbers |
   | Program Management | Reports, schedules, docs |
   | Field Engineer | Site surveys, equipment, safety |
   | Cybersecurity Analyst | Incidents, threats, logs |
   | General AI | Broad questions, creative tasks |

2. **Question box** -- Type your question and press **Enter** or click
   **Ask**.

3. **Answer area** -- Shows the AI-generated answer. Text is
   selectable (Ctrl+C to copy).

4. **Sources line** -- Shows which documents contributed:
   `Sources: Manual.pdf (3 chunks), Datasheet.xlsx (2 chunks)`

5. **Metrics line** -- Shows performance:
   `Latency: 1,234 ms | Tokens in: 450 | Tokens out: 120`

### Mode Toggle

The title bar has **OFFLINE** and **ONLINE** buttons. The active mode
is green.

- **OFFLINE**: Queries go to Ollama on your local machine. No data
  leaves the computer.
- **ONLINE**: Queries go to the configured cloud API. Requires stored
  credentials. Only the question and top document chunks are sent --
  never full documents.

Click either button to switch modes. Switching to offline is always
immediate. Switching to online checks for stored credentials first.

### Theme Toggle

Click the **Theme** button (labeled "Light" or "Dark") in the top-right
of the title bar to switch between dark mode (default) and light mode.

### Status Bar

Auto-refreshes every 5 seconds:

- **LLM**: Which model is active (e.g., `phi4-mini (Ollama)`)
- **Ollama**: Ready (green) or Offline (gray)
- **Gate**: OFFLINE (gray) or ONLINE (green)

If Ollama shows "Offline" but you believe it is running, check that
your corporate proxy is not intercepting localhost (the `NO_PROXY`
environment variable in `start_hybridrag.ps1` handles this).

---

## 5. Online vs Offline Mode

| | Offline | Online |
|-|---------|--------|
| **LLM runs** | On your computer (Ollama) | In the cloud (API) |
| **Internet needed** | No | Yes (configured endpoint only) |
| **Answer speed** | 5-30 sec (GPU), 1-3 min (CPU) | 2-5 seconds |
| **Data sent** | Nothing leaves machine | Question + top chunks only |
| **Cost** | Free (your hardware) | Per-token API pricing |
| **Requires** | Ollama running | API key + endpoint stored |

### When to Use Offline

- On a restricted or offline network
- Working with sensitive documents
- Cost is a concern
- Latency is acceptable

### When to Use Online

- Need faster answers
- Need higher-quality responses
- Have API credits/budget
- On an unrestricted network

### Switching Modes

**Command line:**

```powershell
rag-mode-online
rag-mode-offline
```

**GUI:** Click the **OFFLINE** or **ONLINE** button in the title bar.

You can switch between modes at any time. The indexed documents and
search database are the same regardless of mode -- only the answer
generation step changes.

---

## 6. Managing Credentials

API credentials are stored in Windows Credential Manager, encrypted
with DPAPI and tied to your Windows login.

### Store Credentials

```powershell
rag-store-key              # API key (hidden input)
rag-store-endpoint         # Endpoint URL
```

For Azure OpenAI (optional additional fields):

```powershell
rag-store-deployment       # Azure deployment name
rag-store-api-version      # Azure API version
```

### Check Status

```powershell
rag-cred-status
```

Shows masked previews:

```
API Key:       SET (source: keyring)     sk-abc1...wxyz
API Endpoint:  SET (source: keyring)     https://openrouter.ai/api/v1
```

### Test Connection

```powershell
rag-test-api
```

Sends one test prompt and reports success/failure with latency and
cost estimate.

### Remove Credentials

```powershell
rag-cred-delete
```

### Credential Priority

If credentials exist in multiple places, the highest-priority source
wins:

1. **Windows Credential Manager** (most secure, recommended)
2. **Environment variables** (`HYBRIDRAG_API_KEY`, `OPENAI_API_KEY`)
3. **Config file** (not recommended for secrets)

---

## 7. Performance Profiles

Profiles adjust batch sizes and search parameters for different
hardware.

| Profile | RAM | Batch Size | top_k | Best For |
|---------|-----|-----------|-------|----------|
| `laptop_safe` | 8-16 GB | 16 | 5 | Laptop, conservative |
| `desktop_power` | 32-64 GB | 64 | 10 | Desktop, balanced |
| `server_max` | 64+ GB | 128 | 15 | Workstation, maximum |

### Check and Switch

```powershell
rag-profile                    # Show current profile
rag-profile laptop_safe        # Switch to laptop profile
rag-profile desktop_power      # Switch to desktop profile
rag-profile server_max         # Switch to server profile
```

In the GUI, profiles can be switched from **Admin** >
**Admin Settings**.

---

## 8. Admin Tuning

Open the admin menu from the GUI: **Admin** >
**Admin Settings**. Changes take effect immediately.

### Retrieval Settings

| Setting | Default | What It Does |
|---------|---------|-------------|
| **top_k** | 12 | How many document chunks to retrieve and send to the AI. Higher = more context but more noise. |
| **min_score** | 0.10 | Minimum relevance score. Chunks below this are discarded. Raise if answers include irrelevant info. Lower if queries return no results. |
| **Hybrid search** | ON | Combines meaning search + keyword search. Leave ON for best results. |
| **Reranker** | OFF | Second-pass accuracy model. **Keep OFF for general use** -- enabling it degrades unanswerable, injection, and ambiguous question handling. |

### LLM Settings

| Setting | Default | What It Does |
|---------|---------|-------------|
| **Max tokens** | 2048 | Maximum answer length. 512 for short answers, 4096 for detailed explanations. |
| **Temperature** | 0.05 | Randomness. 0.05 is nearly deterministic (best for factual queries). Above 0.3 gets creative. |
| **Timeout** | 30 sec | How long to wait for the AI. Raise to 120 sec for offline CPU mode. |

### Test Query

The engineering menu includes a test section at the bottom. Type a
question, click **Run Test**, and see the answer immediately. Useful
for testing the effect of slider changes.

---

## 9. REST API

HybridRAG provides a REST API for programmatic access.

### Start the Server

```powershell
rag-server
```

Server starts on `http://localhost:8000`. API documentation is at
`http://localhost:8000/docs`.

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Server status and versions |
| `/status` | GET | Index status, LLM status, gate mode |
| `/config` | GET | Current configuration |
| `/query` | POST | Execute a query |
| `/index` | POST | Start background indexing |
| `/index/status` | GET | Indexing progress |
| `/mode` | POST | Switch offline/online |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the operating frequency?"}'
```

The server binds to localhost only -- it is not accessible from other
machines on the network.

---

## 10. Supported File Formats

### Fully Supported (49+ formats)

**Documents**: PDF, DOCX, PPTX, XLSX, DOC (legacy), RTF

**Email**: EML, MSG, MBOX

**Text**: TXT, MD, CSV, JSON, XML, LOG, YAML, INI, CFG, CONF,
PROPERTIES, REG

**Web**: HTML, HTM

**Images (OCR)**: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP, WMF, EMF, PSD

**CAD/Engineering**: DXF, STP/STEP, IGS/IGES, STL

**Diagrams**: VSDX

**Security**: EVTX (Windows Event Log), PCAP/PCAPNG (network captures),
CER/CRT/PEM (certificates)

**Database**: ACCDB, MDB (Access)

### Placeholder (Recognized, Not Yet Parsed)

PRT, SLDPRT, ASM, SLDASM (SolidWorks), DWG/DWT (AutoCAD), MPP
(MS Project), VSD (legacy Visio), ONE (OneNote), OST (Outlook), EPS

These file types appear in search results by filename but have no
content extraction yet.

---

## 11. Tips for Better Answers

### Writing Better Questions

| Instead of | Try |
|-----------|-----|
| "Tell me about the manual" | "What is the calibration procedure for the XR-7?" |
| "frequency power modulation" | "What is the operating frequency of the antenna?" |
| "What do you know?" | "According to the installation guide, what access level is required?" |

### Tuning for Your Use Case

| Symptom | Adjustment |
|---------|-----------|
| "No results found" on queries that should work | Lower `min_score` from 0.10 to 0.05 |
| Answers include irrelevant information | Raise `min_score` to 0.25-0.30 |
| Answer is missing context from multiple docs | Raise `top_k` from 5 to 12-15 |
| Answers are too verbose/wandering | Lower `top_k` from 12 to 5-8 |
| Need faster answers | Switch to online mode |
| Need better accuracy | Switch to online mode with a larger model |

### Choosing the Right Use Case (GUI)

The use case dropdown in the GUI selects the best AI model for your
question type. When in doubt:

- **Engineering / STEM** is the safe default for technical documents
- **General AI** for broad questions that cross categories
- **Logistics Analyst** for part numbers, BOMs, and data lookups
- **Program Management** for schedules, reports, and summaries

---

## 12. Common Problems and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `rag-query` says "command not found" | Environment not loaded | Run `. .\start_hybridrag.ps1` |
| "No results found" | Documents not indexed | Run `rag-index` first |
| "Query engine not initialized" (GUI) | Backends still loading | Wait for `[OK] Backends attached to GUI` in terminal (30-60 sec) |
| Ollama shows "Offline" in status bar | Ollama not running | Open separate terminal, run `ollama serve` |
| "Cannot switch to online mode" | Credentials missing | Run `rag-store-key` and `rag-store-endpoint` |
| `[FAIL] 401 / Unauthorized` | Invalid API key | Run `rag-store-key` with a valid key |
| `[FAIL] 404 / NotFound` | Wrong endpoint URL | Check with `rag-cred-status`, re-run `rag-store-endpoint` |
| `[FAIL] Timeout` | LLM too slow | Raise timeout in Admin menu (120 sec for offline CPU) |
| Out of memory during indexing | Batch size too large | Switch to `laptop_safe` profile |
| Slow indexing | Normal on CPU-only machines | Switch to `desktop_power` profile or run overnight |
| Progress bar does not move | Processing a very large file | Normal for 500-page PDFs. Wait or click Stop. |
| Corporate proxy blocks Ollama | Proxy intercepting localhost | Verify `NO_PROXY=localhost,127.0.0.1` is set |
| Answer quality is poor | Multiple possible causes | See "Tuning for Your Use Case" above |

### Diagnostics

```powershell
rag-diag               # Full diagnostic suite
rag-diag --verbose     # Detailed output
rag-diag --test-embed  # Test embedding model live
rag-status             # Quick health check
rag-paths              # Verify configured paths
```

### Getting Help

If diagnostics show failures you cannot resolve:

1. Check the [SHORTCUT_SHEET.md](SHORTCUT_SHEET.md) for known issues
2. Review the terminal output for `[FAIL]` messages with fix hints
3. Run `rag-diag --verbose` and review the detailed report
