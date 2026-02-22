# HybridRAG3 -- Technical Theory of Operation

Last Updated: 2026-02-21

---

## 1. System Architecture Overview

HybridRAG3 is a local-first Retrieval-Augmented Generation (RAG) system
with dual-mode LLM routing (offline via Ollama, online via OpenAI-compatible
API), hybrid search (vector + BM25 via Reciprocal Rank Fusion), a 5-layer
hallucination guard, and a centralized network gate enforcing zero-trust
outbound access control.

```
                        INDEXING PIPELINE
+-----------+     +----------+     +---------+     +-------------+
|  Source    | --> |  Parser  | --> | Chunker | --> |  Embedder   |
|  Files    |     | Registry |     | (1200c, |     | MiniLM-L6   |
| (.pdf,    |     | (24 ext) |     |  200lap)|     | (384-dim)   |
| .docx,..) |     +----------+     +---------+     +-------------+
+-----------+                                             |
                                                          v
                                                 +----------------+
                                                 |  VectorStore   |
                                                 | SQLite + FTS5  |
                                                 | Memmap float16 |
                                                 +----------------+
                                                          |
                        QUERY PIPELINE                    v
+-----------+     +-----------+     +---------+     +-------------+
|  User     | --> | Embedder  | --> |Retriever| --> |  Query      |
|  Query    |     | (same     |     | Hybrid  |     |  Engine     |
|           |     |  model)   |     | RRF k=60|     | + LLM call  |
+-----------+     +-----------+     +---------+     +-------------+
                                                          |
                                                          v
                                                 +----------------+
                                                 | Hallucination  |
                                                 | Guard (5-layer)|
                                                 | (online only)  |
                                                 +----------------+
```

**Design priorities**: Offline operation, crash safety, low RAM usage,
full auditability, zero external server dependencies.

---

## 2. Module Dependency Graph

```
boot.py  (entry point -- constructs all services)
  |-- config.py         (YAML loader, dataclass validation)
  |-- credentials.py    (Windows Credential Manager / env var resolution)
  |-- network_gate.py   (URL allowlist, 3-mode access control)
  |-- api_client_factory.py  (builds httpx client with gate integration)
  |-- embedder.py       (sentence-transformers model wrapper)
  |-- vector_store.py   (SQLite + memmap dual store)
  |-- chunker.py        (text splitter with boundary detection)
  |-- indexer.py        (orchestrates parse -> chunk -> embed -> store)
  |-- retriever.py      (hybrid search: vector + BM25 + RRF)
  |-- query_engine.py   (orchestrates search -> context -> LLM -> answer)
  |-- llm_router.py     (Ollama or API routing, raw httpx)
  +-- hallucination_guard/  (5-layer verification, online mode only)

parsers/registry.py  (extension -> parser class mapping)
  |-- pdf_parser.py          (pdfplumber extraction)
  |-- pdf_ocr_fallback.py    (Tesseract fallback for scanned PDFs)
  |-- office_docx_parser.py  (python-docx paragraph extraction)
  |-- office_pptx_parser.py  (python-pptx slide/shape extraction)
  |-- office_xlsx_parser.py  (openpyxl row extraction, read-only mode)
  |-- eml_parser.py          (stdlib email + attachment extraction)
  |-- image_parser.py        (Tesseract OCR)
  |-- plain_text_parser.py   (direct UTF-8 read)
  +-- text_parser.py         (routing parser, delegates by extension)

gui/                         (tkinter desktop application, dark/light theme)
  |-- app.py                 (main window, panel composition)
  |-- theme.py               (dark/light theme definitions, toggle logic)
  |-- stubs.py               (temporary stubs for Window 2 model routing)
  |-- launch_gui.py          (entry point, boot + background loading)
  +-- panels/
      |-- query_panel.py     (question input, answer display, metrics)
      |-- index_panel.py     (folder picker, progress bar, start/stop)
      |-- status_bar.py      (live system health indicators)
      +-- engineering_menu.py (tuning sliders, profile switch, test query)

api/                         (FastAPI REST server)
  |-- server.py              (lifespan management, app factory)
  |-- routes.py              (endpoint handlers)
  +-- models.py              (Pydantic request/response schemas)
```

---

## 3. Indexing Pipeline

### 3.1 Parser Registry

`src/parsers/registry.py` maps 24+ file extensions to parser classes.
Each parser implements:

```python
def parse(self, file_path: str) -> str
def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]
```

Supported formats: PDF, DOCX, PPTX, XLSX, DOC (legacy), RTF, EML, MSG,
MBOX, HTML, TXT, MD, CSV, JSON, XML, LOG, YAML, INI, PNG/JPG/TIFF/BMP/
GIF/WEBP (OCR), DXF, STP/STEP, IGS/IGES, STL, VSDX, EVTX, PCAP, CER/
CRT/PEM, ACCDB/MDB.

All parsers are lazy-imported to avoid pulling heavy dependencies when
not needed. Every parser wraps its work in try/except and returns
`("", {"error": "..."})` on failure -- a corrupted file never crashes
the pipeline.

### 3.2 Chunker

`src/core/chunker.py` splits raw text into overlapping chunks.

**Parameters:**
- `chunk_size`: 1200 characters (default). Tuned for all-MiniLM-L6-v2
  which performs best on 200-500 word passages.
- `overlap`: 200 characters. Ensures facts near chunk boundaries are
  not lost.

**Boundary detection** (priority order):
1. Paragraph break (`\n\n`) in the second half of the chunk window
2. Sentence end (`. `) in the second half
3. Any newline in the second half
4. Hard cut at `chunk_size` (last resort)

**Heading prepend**: The chunker searches backward up to 2000 characters
for the nearest section heading (ALL CAPS line, numbered section like
"3.2.1 Signal Processing", or line ending with `:`) and prepends it as
`[SECTION] Heading\n`. This preserves document structure across chunks.

### 3.3 Embedder

`src/core/embedder.py` wraps `sentence-transformers/all-MiniLM-L6-v2`.

- Output: 384-dimensional normalized float32 vectors (each chunk becomes
  a list of 384 numbers that act like GPS coordinates in "meaning space"
  -- similar meanings land at nearby coordinates)
- Dimension read from model at load time (never hardcoded)
- Batch embedding for indexing (`embed_batch`), single for queries
  (`embed_query`)
- Model loaded once, held in memory (~100 MB), released with `close()`
- HuggingFace Hub downloads blocked at runtime via `HF_HUB_OFFLINE=1`
  and `TRANSFORMERS_OFFLINE=1`; model must be pre-cached

### 3.4 VectorStore (Dual Storage)

`src/core/vector_store.py` manages two coordinated backends:

**SQLite** (`hybridrag.sqlite3`):
- `chunks` table: id, text, source_path, chunk_index, metadata JSON
- `chunks_fts` FTS5 virtual table: auto-synchronized, provides BM25
  keyword search via SQLite full-text search engine
- `index_runs` table: run audit trail (run_id, timestamps, counts)
- Uses `INSERT OR IGNORE` with deterministic chunk IDs for crash-safe
  restarts (same file + position = same ID)

**Memmap** (`embeddings.f16.dat` + `embeddings_meta.json`):
- Raw float16 matrix of shape `[N, 384]` memory-mapped via numpy
- Disk-backed: the OS loads only the pages being read, like reading
  specific pages from a book without loading the entire book into memory
- 8 GB RAM laptop can search 10M+ embeddings
- JSONDecodeError guard on meta file load: corrupted JSON triggers
  reinitialization instead of crash

**Why two systems**: SQLite handles structured queries. Memmap handles
millions of vectors without loading them all into RAM.

**Why float16**: Halves storage (0.75 GB vs 1.5 GB per million chunks)
with negligible quality loss. Like rounding GPS coordinates to 3 decimal
places instead of 6 -- you lose sub-meter precision but still find the
right neighborhood.

**Why memmap over FAISS**: Simpler, no C++ dependencies, sufficient for
< 500K chunks. Migration to FAISS IVF planned for scale-out (see
`docs/research/FAISS_MIGRATION_PLAN.md`).

### 3.5 Indexer Orchestration

`src/core/indexer.py` ties the pipeline together:

1. Scan source folder recursively for supported extensions
2. Compute file hash (size + mtime) for change detection
3. Skip files whose hash matches stored hash (already indexed)
4. Parse to raw text via ParserRegistry
5. Process in 200K character blocks to cap peak RAM
6. Chunk text into overlapping segments
7. Embed chunks in batches
8. Store chunks in SQLite and embeddings in memmap
9. Garbage collect between files to bound RAM usage
10. Delete orphaned chunks (source file deleted since last run)
11. Rebuild FTS5 index

**Anti-sleep**: On Windows, `SetThreadExecutionState` prevents the OS
from sleeping during long indexing runs (6+ hours overnight).

---

## 4. Query Pipeline

### 4.1 Retriever (Hybrid Search)

`src/core/retriever.py` implements three search strategies:

**Vector search**: Query embedding dot-producted against memmap in
blocks of 25,000 rows. Returns top candidates by cosine similarity.
Block-based scanning avoids loading the full embedding matrix.

**BM25 keyword search**: FTS5 OR-logic query against `chunks_fts`.
OR-logic (not AND) ensures partial matches are returned. Critical for
exact terms: part numbers, acronyms, technical jargon.

**Hybrid search (default)**: Both searches run, then results are merged
via Reciprocal Rank Fusion (RRF). RRF works like combining two judges'
rankings: if Judge A ranks a chunk #1 and Judge B ranks it #3, that chunk
scores higher than one ranked #5 by both. The formula:

```
rrf_score(chunk) = sum( 1 / (k + rank_i) )  for each list i
```

where `k = 60` (standard from the original RRF paper). RRF scores are
multiplied by 30 and capped at 1.0 to normalize into the same range as
cosine similarity, enabling a single `min_score` threshold.

**Optional cross-encoder reranker**: Retrieves `reranker_top_n` (20)
candidates, reranks with cross-encoder. Disabled by default. WARNING:
enabling for multi-type evaluation destroys unanswerable (100->76%),
injection (100->46%), and ambiguous (100->82%) scores.

**Tunable parameters:**

| Setting | Default | Purpose |
|---------|---------|---------|
| `hybrid_search` | true | Enable vector + BM25 fusion |
| `top_k` | 12 | Chunks sent to LLM |
| `min_score` | 0.10 | Minimum similarity to include |
| `rrf_k` | 60 | RRF smoothing constant |
| `reranker_enabled` | false | Cross-encoder reranking |
| `reranker_top_n` | 20 | Candidates for reranker |

### 4.2 Query Engine

`src/core/query_engine.py` orchestrates the full query:

1. Embed user query via `embedder.embed_query()`
2. Retrieve top-K chunks via `retriever.search()`
3. Build context string from retrieved chunks
4. Construct LLM prompt using 9-rule source-bounded generation
5. Route to LLM via `llm_router` (offline or online)
6. Calculate token cost estimate (online mode)
7. Return `QueryResult(answer, sources, tokens, cost, latency, mode)`

**9-rule prompt system** (`_build_prompt()`, ~line 239):
- Priority: injection/refusal > ambiguity > accuracy > formatting
- Rule 5 (injection): refer to false claims generically, never name them
- Rule 8 (source quality): filters indexed test metadata
- Rule 9 (exact line): subordinate to Rule 4 (ambiguity)

**Failure paths**: 0 results returns "no relevant documents found"
without calling LLM. LLM timeout still returns search results with
error flag. Every path returns a valid `QueryResult` -- no exceptions
propagate.

### 4.3 LLM Router

`src/core/llm_router.py` routes to the appropriate backend:

- **Offline (OllamaRouter)**: HTTP POST to `localhost:11434/api/generate`.
  Default timeout 600s (CPU inference is slow).
- **Online (APIRouter)**: HTTP POST to OpenAI-compatible
  `/v1/chat/completions`. Uses `openai` SDK (v1.45.1). Supports Azure
  OpenAI and standard OpenAI endpoints with deployment discovery.

Network Gate is checked before every outbound connection.

**Deployment discovery** (online mode):
- `_deployment_cache`: caches available deployments
- `is_azure_endpoint()`: detects Azure vs standard OpenAI
- `get_available_deployments()`: lists chat/embedding models

---

## 5. Hallucination Guard

`src/core/hallucination_guard/` -- 6 files, each under 500 lines.
Active only in online mode.

| Layer | Module | Function |
|-------|--------|----------|
| 1 | `prompt_hardener.py` | Injects grounding instructions into system prompt |
| 2a | `claim_extractor.py` | Splits response into individual factual claims |
| 2b | `nli_verifier.py` | NLI model checks each claim vs source chunks |
| 3-4 | `response_scoring.py` | Scores faithfulness, constructs safe response |
| 5 | `dual_path.py` | Optional dual-model consensus for critical queries |

**Configuration:**
- `threshold`: 0.80 default (minimum faithfulness score)
- `failure_action`: "block" (replace with safe response) or "warn" (flag)
- `shortcircuit_pass`: 5 (skip remaining checks after N consecutive passes)
- `shortcircuit_fail`: 3 (abort after N consecutive failures)
- `enable_dual_path`: false (opt-in for critical queries)

**Built-In Test**: Runs on first import (< 50ms, no model loading, no
network). Validates all guard components are importable and intact.

---

## 6. Security Architecture

### 6.1 Network Gate

`src/core/network_gate.py` -- Centralized outbound access control.

| Mode | Allowed Destinations | Use Case |
|------|---------------------|----------|
| `offline` | `localhost`, `127.0.0.1` only | Default. offline use. |
| `online` | Localhost + configured API endpoint | Daily use on network |
| `admin` | Unrestricted (with logging) | Maintenance only |

`gate.check_allowed(url, purpose, caller)` raises `NetworkBlockedError`
if URL is not in allowlist. Works like a building security desk: every
visitor (URL) is checked against the guest list, and every visit
(allowed or denied) is written in the log book.

### 6.2 Three-Layer Network Lockdown

| Layer | Mechanism | Blocks |
|-------|-----------|--------|
| 1. PowerShell | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | HuggingFace model downloads |
| 2. Python | `os.environ` enforcement before import | HuggingFace in any Python process |
| 3. Application | NetworkGate URL allowlist | All other outbound URLs |

All three must fail before unauthorized data leaves the machine.

### 6.3 Credential Management

`src/security/credentials.py` resolves API keys (priority order):
1. Windows Credential Manager (DPAPI encrypted, tied to Windows login)
2. Environment variable (`HYBRIDRAG_API_KEY`)
3. Config file (not recommended, logged as warning)

Extended credential fields: api_key, endpoint, deployment, api_version.
`source_*` fields track provenance. Keys never logged in full --
`key_preview()` returns masked form (`sk-...xxxx`).

---

## 7. Boot Pipeline

`src/core/boot.py` -- Single entry point for initialization.

1. Record `boot_timestamp` (ISO format)
2. Load YAML configuration
3. Resolve credentials via `credentials.py`
4. Validate config + credentials together
5. Validate endpoint URL format (`http://` or `https://` prefix)
6. Configure NetworkGate to appropriate mode
7. Build API client (if online + credentials available)
8. Probe Ollama (if offline configured)
9. Return `BootResult` with `success`, `online_available`,
   `offline_available`, `warnings[]`, `errors[]`, and `summary()`

Never crashes on missing credentials -- marks mode as unavailable and
continues. Like a car that starts even if the GPS is not connected.
Offline mode always works even without API configuration.

---

## 8. GUI Architecture

`src/gui/` -- Tkinter desktop application (Python stdlib, zero deps).

**Startup sequence:**
1. Boot pipeline runs (2-4 seconds)
2. Window opens immediately
3. Heavy backends (embedder, vector store, query engine) load in a
   background thread via `queue.Queue` + `root.after(100, poll)` pattern
4. Panels become functional when backends finish

**Panels:**
- **Query Panel**: Use-case dropdown, model auto-selection, question
  input, answer display with sources, latency/token/cost metrics
- **Index Panel**: Folder picker, Start/Stop, progress bar, status
- **Status Bar**: Live 5-second refresh -- Ollama status, LLM model,
  Network Gate mode (color-coded green/red)
- **Engineering Menu**: Retrieval sliders (top_k, min_score, rrf_k),
  LLM tuning (temperature, timeout), profile switching

**Threading safety**: All background work uses `queue.Queue` for
thread-to-GUI communication. `threading.Event` for cancellation.
Never `after_idle()` (known Tcl memory-exhaustion hazard).

---

## 9. REST API

`src/api/server.py` -- FastAPI with lifespan management.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Server status + versions |
| `/status` | GET | Index status, LLM status, Gate mode |
| `/config` | GET | Current configuration |
| `/query` | POST | Execute a query |
| `/index` | POST | Start background indexing |
| `/index/status` | GET | Indexing progress |
| `/mode` | POST | Switch OFFLINE/ONLINE |

Binds to `127.0.0.1:8000` only (no network exposure). TestClient MUST
use context manager: `with TestClient(app) as client:` for lifespan.

---

## 10. Exception Hierarchy

`src/core/exceptions.py` -- Typed tree rooted at `HybridRAGError`.

Every exception includes `fix_suggestion: str` and `error_code: str`.

| Exception | Code | When Raised |
|-----------|------|-------------|
| `ConfigError` | CONF-* | Invalid YAML, missing fields |
| `AuthRejectedError` | AUTH-001 | 401/403 from API |
| `EndpointNotConfiguredError` | NET-002 | API endpoint missing |
| `NetworkBlockedError` | NET-001 | NetworkGate denied connection |
| `EmbeddingError` | EMB-* | Model load failure, dimension mismatch |
| `IndexingError` | IDX-001 | Unrecoverable file error |

---

## 11. Configuration System

`src/core/config.py` loads from `config/default_config.yaml`.

**Nested dataclasses** for type safety:
- `PathsConfig` -- database, embeddings_cache, source_folder
- `EmbeddingConfig` -- model_name, dimension, batch_size, device
- `ChunkingConfig` -- chunk_size, overlap, max_heading_len
- `OllamaConfig` -- base_url, model, timeout_seconds, context_window
- `APIConfig` -- endpoint, model, max_tokens, temperature
- `RetrievalConfig` -- top_k, min_score, hybrid_search, rrf_k
- `CostConfig` -- track_enabled, daily_budget_usd
- `SecurityConfig` -- audit_logging, pii_sanitization
- `HallucinationGuardConfig` -- thresholds, failure_action

**Environment variable overrides**: `HYBRIDRAG_<SECTION>_<KEY>`.

**Hardware profiles** (`config/profiles.yaml`):

| Profile | RAM | Batch | Top_K |
|---------|-----|-------|-------|
| `laptop_safe` | 8-16 GB | 16 | 5 |
| `desktop_power` | 32-64 GB | 64 | 10 |
| `server_max` | 64+ GB | 128 | 15 |

---

## 12. Diagnostic Framework

`src/diagnostic/` -- 3-tier test and monitoring system.

| Tier | Module | What It Tests |
|------|--------|--------------|
| Health | `health_tests.py` | 15 pipeline checks (DB, model, paths) |
| Component | `component_tests.py` | Individual unit tests |
| Performance | `perf_benchmarks.py` | Embedding speed, search latency, RAM |

`fault_analysis.py`: Automated fault hypothesis engine. Classifies by
severity, generates fix suggestions, tracks fault history.

---

## 13. Storage Layout

```
hybridrag.sqlite3
|-- chunks           (id, text, source_path, chunk_index, metadata JSON)
|-- chunks_fts       (FTS5 virtual table, auto-synced with chunks)
|-- index_runs       (run_id, start_time, end_time, file counts)
+-- query_log        (planned: query audit trail)

embeddings.f16.dat   (raw float16 matrix, shape [N, 384])
embeddings_meta.json ({"dim": 384, "count": N, "dtype": "float16"})
```

---

## 14. Model Compliance

All offline models must pass regulatory review before deployment.
Full audit: `docs/enterprise_MODEL_AUDIT.md`.

**Approved publishers**: Microsoft (MIT), Mistral AI (Apache 2.0),
Google (Apache 2.0), NVIDIA (Apache 2.0).

**Banned**: All China-origin (Alibaba, DeepSeek, BAAI). Meta/Llama
(license restrictions). See `docs/waiver_cheat_sheet_v4b.xlsx`.

Model definitions: `scripts/_model_meta.py`, `scripts/_set_model.py`.
Default offline model: `phi4-mini` (`config/default_config.yaml`).
9 use-case profiles: sw, eng, pm, sys, log, draft, fe, cyber, gen.

---

## 15. Evaluation System

**Protected files** (NEVER modify):
- `scripts/run_eval.py`, `tools/eval_runner.py`
- `tools/score_results.py`, `tools/run_all.py`
- `Eval/*.json`

**Scoring formulas:**
- `run_eval.py`: overall = 0.7 * fact + 0.3 * behavior
- `score_results.py`: overall = 0.45 * behavior + 0.35 * fact + 0.20 * citation

Fact matching is case-insensitive substring. Exact spacing matters.
Injection trap: AES_RE regex catches "AES-512" anywhere in answer text.

**Current results**: 98% pass rate on 400-question golden set.

---

## 16. Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|-----------|
| Embedding speed | ~100 chunks/sec | CPU, all-MiniLM-L6-v2 |
| Vector search | < 100 ms | 40K chunks, block scan |
| FTS5 keyword search | < 10 ms | 40K chunks |
| Index skip (unchanged) | < 1 sec | Hash-based detection |
| RAM (indexing) | ~500 MB | Model + active block buffers |
| RAM (search) | ~300 MB | Model + memmap overhead |
| Disk per 1M chunks | ~0.75 GB | float16 embeddings only |
| Online query latency | 2-5 sec | API via configured endpoint |
| Offline query latency | 5-180 sec | Ollama, hardware dependent |

---

## 17. Scale-Out Path

Current memmap brute-force search is O(N) and will not scale beyond
~500K vectors without unacceptable latency. Planned migration:

- **Phase 1**: `faiss-cpu` with `IVF256,SQ8` as drop-in replacement
- **Phase 2**: `IVF4096,SQ8` for 50M+ vectors (~18.6 GB, 90-95% recall)
- **Phase 3**: GPU-accelerated FAISS on dual RTX 3090 workstation
  (requires WSL2 or native Linux -- no Windows GPU FAISS support)

Full analysis: `docs/research/FAISS_MIGRATION_PLAN.md`.

---

## 18. Key Dependencies

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| torch | 2.10.0 | BSD-3 | Tensor computation |
| sentence-transformers | 2.7.0 | Apache 2.0 | Embedding model |
| transformers | 4.57.6 | Apache 2.0 | Tokenization |
| numpy | 1.26.4 | BSD-3 | Numerical arrays, memmap |
| scikit-learn | 1.8.0 | BSD-3 | Distance metrics |
| pdfplumber | 0.11.9 | MIT | PDF extraction |
| python-docx | 1.2.0 | MIT | Word documents |
| python-pptx | 1.0.2 | MIT | PowerPoint |
| openpyxl | 3.1.5 | MIT | Excel |
| httpx | 0.28.1 | BSD-3 | HTTP client |
| openai | 1.45.1 | MIT | OpenAI/Azure SDK |
| fastapi | 0.115.0 | MIT | REST API framework |
| uvicorn | 0.41.0 | BSD-3 | ASGI server |
| keyring | 24.3.0 | MIT | Windows Credential Manager |
| cryptography | 44.0.2 | Apache/BSD | Encryption |
| pydantic | 2.11.1 | MIT | Data validation |
| structlog | 24.4.0 | Apache 2.0 | Structured logging |
| PyYAML | 6.0.2 | MIT | YAML parsing |
| tiktoken | 0.8.0 | MIT | Token counting |
