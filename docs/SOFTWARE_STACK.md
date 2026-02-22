# HybridRAG v3 -- Software Stack Decisions

Last updated: 2026-02-22

This document records every major technology choice in HybridRAG v3,
what alternatives were evaluated, and the reasoning behind each decision.

---

## Table of Contents

1. [Core Runtime](#1-core-runtime)
2. [Embedding Model](#2-embedding-model)
3. [Vector Storage](#3-vector-storage)
4. [Retrieval Strategy](#4-retrieval-strategy)
5. [Chunking Strategy](#5-chunking-strategy)
6. [LLM Models -- Offline](#6-llm-models----offline)
7. [LLM Models -- Online](#7-llm-models----online)
8. [GUI Framework](#8-gui-framework)
9. [REST API Framework](#9-rest-api-framework)
10. [Document Parsing](#10-document-parsing)
11. [Credential Storage](#11-credential-storage)
12. [Network Security](#12-network-security)
13. [Logging](#13-logging)
14. [LLM Client SDK](#14-llm-client-sdk)
15. [HTTP Client](#15-http-client)
16. [Configuration](#16-configuration)
17. [Prompt Strategy](#17-prompt-strategy)
18. [Banned / Excluded Technologies](#18-banned--excluded-technologies)

---

## 1. Core Runtime

| | |
|---|---|
| **Chosen** | Python 3.10 |
| **License** | PSF (permissive) |

### Why Python

- Universal data science / ML ecosystem (torch, transformers, numpy)
- sentence-transformers and Ollama both have first-class Python support
- Runs on every target machine (work laptop, home PC, workstation)
- tkinter ships with stdlib -- zero-install GUI

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Rust** | Fast, memory safe, small binaries | No sentence-transformers, no tkinter, steep learning curve | Rejected -- wrong ecosystem |
| **Node.js / TypeScript** | Electron for GUI, good async | No torch/numpy, weak ML library support | Rejected -- wrong ecosystem |
| **Go** | Fast compilation, easy deployment | No ML ecosystem, no embeddings support | Rejected |
| **C# / .NET** | Good Windows integration | Weak ML library support, ONNX only for embeddings | Rejected |

---

## 2. Embedding Model

| | |
|---|---|
| **Chosen** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Dimensions** | 384 |
| **Size** | ~80 MB download |
| **License** | Apache 2.0 |
| **Speed** | ~100 chunks/second on CPU |

### Why all-MiniLM-L6-v2

- Small footprint -- loads in seconds on 8 GB laptop
- Pre-normalized vectors (dot product = cosine similarity, very fast)
- 384 dimensions is a good balance of quality vs storage cost
- Millions of downloads, well-tested, stable
- Runs entirely on CPU (no GPU required)

### Alternatives Considered

| Alternative | Dims | Size | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| **all-mpnet-base-v2** | 768 | ~420 MB | Slightly better quality | 2x storage, 2x slower search | Rejected -- not worth 2x cost |
| **e5-large-v2** | 1024 | ~1.3 GB | Much better quality | Needs GPU, 2.7x storage | Rejected -- hardware constraint |
| **OpenAI text-embedding-ada-002** | 1536 | API call | Excellent quality | Requires internet + API key, 4x storage | Rejected -- offline-first |
| **nomic-embed-text-v1.5** | 768 | ~274 MB | Comparable, Apache 2.0 | 2x storage, needs trust_remote_code | Evaluated but not adopted |
| **BGE models (BAAI)** | various | various | Good benchmarks | China-origin, NDAA banned | Banned |

### Configuration

```yaml
embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384
  device: cpu
  batch_size: 16
```

---

## 3. Vector Storage

| | |
|---|---|
| **Chosen** | SQLite + NumPy memmap (custom hybrid) |
| **Metadata** | SQLite with FTS5 full-text search |
| **Vectors** | NumPy memory-mapped files (float16) |
| **License** | Public domain (SQLite), BSD (NumPy) |

### Why This Design

- SQLite handles metadata + full-text search natively (FTS5 BM25)
- NumPy memmap stores vectors on disk, loads only requested rows into RAM
- float16 halves storage: 1M chunks = 750 MB disk (vs 1.5 GB at float32)
- Zero external services -- no database server to install or manage
- WAL mode for concurrent reads during indexing
- Crash-safe: INSERT OR IGNORE prevents duplicates on recovery

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **ChromaDB** | Easy API, built-in persistence | Dependency hell, required C++ compiler on Windows, alpha quality | Rejected -- install failures |
| **LanceDB** | Rust-backed, fast | Alpha quality, .vector import errors, Rust binary issues | Rejected -- too unstable |
| **FAISS (Facebook)** | Best-in-class search speed | Complex Windows install, no built-in persistence, Meta origin | Rejected -- install complexity (future option for workstation) |
| **Pinecone** | Managed, scales infinitely | Cloud-only, requires internet, vendor lock-in | Rejected -- offline-first |
| **Weaviate** | Full-featured vector DB | Requires Docker / server process, heavy | Rejected -- too heavyweight |
| **Milvus** | Enterprise-grade | Requires Docker, massive footprint | Rejected -- overkill |
| **Qdrant** | Good API, Rust-backed | Requires server process | Rejected -- extra dependency |
| **Pure SQLite (vectors in BLOB)** | Single file | Terrible for large numeric arrays, slow similarity search | Rejected -- performance |

### Storage Math

| Scenario | float16 (chosen) | float32 | Savings |
|---|---|---|---|
| 40,000 chunks (current) | 30 MB | 60 MB | 50% |
| 500,000 chunks (medium) | 375 MB | 750 MB | 50% |
| 1,000,000 chunks (large) | 750 MB | 1.5 GB | 50% |

---

## 4. Retrieval Strategy

| | |
|---|---|
| **Chosen** | Hybrid search (Vector + Keyword) via Reciprocal Rank Fusion |
| **Vector** | Cosine similarity on normalized embeddings |
| **Keyword** | SQLite FTS5 with BM25 ranking |
| **Fusion** | RRF: score = 1 / (k + rank + 1), k=60 |

### Why Hybrid Search

- Vector search captures meaning ("operating temperature" matches "thermal limit")
- Keyword search captures exact terms (part numbers, acronyms, model names)
- RRF combines ranks (not raw scores) -- parameter-free and robust
- Chunks appearing in both search paths float to the top

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Vector-only** | Simpler code | Misses exact part numbers, acronyms | Rejected -- too many misses on engineering docs |
| **Keyword-only (BM25)** | Fast, no embeddings needed | Misses paraphrases and synonyms | Rejected -- weak semantic understanding |
| **Linear score combination** (0.7*vec + 0.3*kw) | Tunable weights | Requires careful weight tuning, fragile | Rejected -- RRF is parameter-free |
| **Always-on cross-encoder reranker** | Best accuracy | ~100x slower than bi-encoder, ~8s on CPU | Available but disabled by default |

### Configuration

```yaml
retrieval:
  hybrid_search: true
  top_k: 12
  min_score: 0.1
  rrf_k: 60
  reranker_enabled: false       # Destroys unanswerable/injection/ambiguous scores
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  reranker_top_n: 20
```

### Reranker Warning

Enabling the reranker for the full eval set destroys behavioral scores:
- Unanswerable: 100% -> 76%
- Injection: 100% -> 46%
- Ambiguous: 100% -> 82%

The reranker is available for single-type factual queries but must NEVER
be enabled for the multi-type evaluation suite.

---

## 5. Chunking Strategy

| | |
|---|---|
| **Chosen** | Smart boundary detection + overlapping character-based chunks |
| **Chunk size** | 1,200 characters (~200-300 words) |
| **Overlap** | 200 characters |
| **Boundary detection** | Paragraph breaks > sentence ends > newlines > hard cut |

### Why This Approach

- Fixed-size cuts produce broken sentences mid-thought
- Overlap ensures facts spanning boundaries appear complete in at least one chunk
- Smart boundary detection keeps each chunk as a coherent thought
- Section heading prepend (looks back 2,000 chars) preserves document structure
- Designed for engineering PDFs with deep hierarchical structure

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Fixed-size chunks** | Simplest implementation | Cuts mid-sentence, breaks context | Rejected |
| **Recursive/tree splitting** (LangChain style) | Handles nested structure | Over-engineered, hard to debug | Rejected |
| **Sentence-level chunks** | Clean boundaries | Too small (~50 chars), lose context | Rejected |
| **Token-based splitting** | Aligns with LLM context | Requires tokenizer, adds complexity | Rejected |
| **Semantic chunking** (embedding similarity) | Best coherence | Very slow (embeds every sentence), complex | Rejected -- too slow for 40K files |

### Configuration

```yaml
chunking:
  chunk_size: 1200
  overlap: 200
  max_heading_len: 160
```

---

## 6. LLM Models -- Offline

| | |
|---|---|
| **Runtime** | Ollama (localhost:11434) |
| **Default model** | phi4-mini (3.8B, Microsoft, MIT) |
| **License requirement** | MIT or Apache 2.0, US/EU origin only |

### Approved Offline Stack (5 models, ~26 GB total)

| Model | Size | VRAM | ENG | GEN | License | Origin | Role |
|---|---|---|---|---|---|---|---|
| **phi4-mini** | 2.3 GB | 5.5 GB | 52 | 48 | MIT | Microsoft/USA | Primary for 7/9 profiles |
| **mistral:7b** | 4.1 GB | 5.5 GB | 40 | 48 | Apache 2.0 | Mistral/France | Alt for eng/sys/fe/cyber |
| **phi4:14b-q4_K_M** | 9.1 GB | 11 GB | 72 | 65 | MIT | Microsoft/USA | Logistics primary, CAD alt |
| **gemma3:4b** | 3.3 GB | 4.0 GB | 50 | 46 | Apache 2.0 | Google/USA | PM fast summarization |
| **mistral-nemo:12b** | 7.1 GB | 10 GB | 48 | 55 | Apache 2.0 | Mistral+NVIDIA | Upgrade for sw/eng/sys/cyber/gen (128K ctx) |

### Future (requires 24+ GB VRAM)

| Model | Size | VRAM | License | Role |
|---|---|---|---|---|
| **mistral-small3.1:24b** | 14 GB | 16 GB | Apache 2.0 | Replaces phi4-mini on workstation |

### Why Ollama

- Single binary, runs on Windows/Mac/Linux
- Manages model downloads, quantization, GPU offloading
- Simple REST API (localhost:11434/api/generate)
- Supports all approved model formats (GGUF)
- No Python dependency (separate process)

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **llama.cpp directly** | Fastest, most control | No model management, CLI-only | Rejected -- Ollama wraps it better |
| **vLLM** | Production-grade, batching | Requires Linux + NVIDIA GPU | Rejected -- no Windows support |
| **HuggingFace Transformers (direct)** | Full control | Slow on CPU, complex setup, high RAM | Rejected -- Ollama is simpler |
| **LM Studio** | Nice GUI | Not scriptable, no REST API for integration | Rejected -- can't automate |
| **text-generation-inference** | Fast, Docker-based | Requires Docker + Linux + GPU | Rejected -- deployment complexity |
| **LocalAI** | OpenAI-compatible API | Less mature than Ollama, smaller community | Evaluated but not adopted |

### Configuration

```yaml
ollama:
  base_url: http://localhost:11434
  model: phi4-mini
  context_window: 8192
  timeout_seconds: 600
```

---

## 7. LLM Models -- Online

| | |
|---|---|
| **API gateway** | OpenRouter (openrouter.ai/api/v1) |
| **Protocol** | OpenAI-compatible chat completions API |
| **Also supports** | Azure OpenAI, direct OpenAI |

### Recommended Online Models by Use Case

| Use Case | Primary | Alt |
|---|---|---|
| Software Engineering | AI assistant-sonnet-4 | gpt-4.1 |
| Engineering / STEM | AI assistant-sonnet-4 | gpt-4o |
| Systems Admin | AI assistant-sonnet-4 | gpt-4o |
| Drafting / CAD | AI assistant-sonnet-4 | gpt-4o |
| Logistics | gpt-4o | gpt-4.1 |
| Program Management | gpt-4o-mini | gpt-4.1-mini |
| Field Engineer | AI assistant-sonnet-4 | gpt-4o |
| Cybersecurity | AI assistant-sonnet-4 | gpt-4o |
| General AI | gpt-4o | AI assistant-sonnet-4 |

### Why OpenRouter

- Single API key accesses 100+ models from multiple providers
- OpenAI-compatible API (same SDK, same code path)
- Pay-per-token, no monthly commitment
- Transparent pricing per model
- Fallback: also supports direct Azure OpenAI and standard OpenAI

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Direct OpenAI API** | Official, reliable | One provider only, no model diversity | Supported as fallback |
| **Azure OpenAI** | Enterprise, compliance | Complex setup, deployment management | Supported as primary at work |
| **Together AI** | Cheap, open models | Smaller catalog | Evaluated, not primary |
| **Groq** | Extremely fast inference | Limited model selection | Not evaluated |
| **Self-hosted API** | Full control | Requires server hardware | Future option with workstation |

---

## 8. GUI Framework

| | |
|---|---|
| **Chosen** | tkinter (Python standard library) |
| **Theme** | Custom dark/light toggle |
| **License** | PSF (included with Python) |

### Why tkinter

- Zero additional dependencies (in Python stdlib)
- Works on every machine including restricted work laptops
- No entry needed in requirements.txt
- No npm, no Electron, no web server
- Suitable for prototype / internal tool

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **PyQt5 / PySide6** | Modern look, rich widgets | ~60 MB dependency, GPL/LGPL licensing complexity | Rejected -- dependency weight |
| **wxPython** | Native look per OS | Complex build, C++ dependency | Rejected -- install issues |
| **Dear PyGui** | GPU-accelerated, modern | Requires GPU context, less mature | Rejected -- GPU dependency |
| **Electron (JS)** | Web-based, beautiful | 100+ MB, requires Node.js, two languages | Rejected -- massive overhead |
| **Streamlit** | Easy dashboards | Requires web server, browser-based | Rejected -- not a standalone app |
| **Gradio** | ML-focused, easy | Web server, browser-based, limited customization | Rejected |
| **CustomTkinter** | Modern tkinter look | Extra dependency, less stable | Evaluated -- possible future upgrade |

### Layout

```
+------------------------------------------+
| Title Bar: Mode toggle + Theme toggle    |
+------------------------------------------+
| Query Panel: Use case, model, Q&A        |
|   [Use case dropdown] [Model: auto]      |
|   [Question entry............] [Ask]     |
|   [Answer text area.................]    |
|   Sources: file1.pdf (3 chunks)          |
|   Latency: 850ms | Tokens: 200/30       |
+------------------------------------------+
| Index Panel: Folder, progress, controls  |
|   [Source folder: ...........] [Browse]  |
|   [Start Indexing] [Stop]                |
|   [==========>        ] 45/120 files     |
+------------------------------------------+
| Status: LLM: phi4-mini | Ollama: Ready  |
+------------------------------------------+
```

---

## 9. REST API Framework

| | |
|---|---|
| **Chosen** | FastAPI 0.115.0 |
| **ASGI server** | Uvicorn 0.41.0 |
| **Validation** | Pydantic 2.11.1 |
| **License** | MIT (FastAPI), BSD (Uvicorn) |

### Why FastAPI

- Automatic OpenAPI/Swagger docs at /docs
- Pydantic validation built-in (request/response models)
- Async support for concurrent requests
- Lightweight (~2 MB total with dependencies)
- Type hints = self-documenting code

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Health check |
| GET | /status | System status |
| GET | /config | Current configuration |
| POST | /query | Submit a RAG query |
| POST | /index | Start indexing a folder |
| GET | /index/status | Indexing progress |
| POST | /mode | Switch online/offline |

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Flask** | Simple, mature, huge ecosystem | No async, no auto-docs, no validation | Rejected -- FastAPI is better fit |
| **Django REST Framework** | Full-featured, ORM | Massive overhead for a simple API | Rejected -- overkill |
| **Starlette (raw)** | Lightweight, async | No auto-validation, no auto-docs | Rejected -- FastAPI wraps it |
| **aiohttp** | Async, flexible | No auto-docs, more boilerplate | Rejected |
| **Bottle** | Minimal | No async, tiny ecosystem | Rejected |

### Configuration

```
Bind: 127.0.0.1:8000 (localhost only)
Launch: rag-server (PowerShell) or python -m src.api.server
```

---

## 10. Document Parsing

### Parser Stack by File Type

| Format | Library | Version | License | Notes |
|---|---|---|---|---|
| **PDF** | pypdf (primary) + pdfplumber (fallback) | 6.6.2 / 0.11.9 | BSD / MIT | Dual extraction: fast first, robust second |
| **PDF OCR** | pytesseract + pdf2image + Pillow | 0.3.13 / 1.17.0 / 12.1.0 | Apache 2.0 / MIT / HPND | Only when text extraction fails (<20 chars) |
| **DOCX** | python-docx | 1.2.0 | MIT | Paragraphs joined with double newlines |
| **XLSX** | openpyxl | 3.1.5 | MIT | Read-only mode, pipe-delimited cells |
| **PPTX** | python-pptx | 1.0.2 | MIT | Slide-tagged text extraction |
| **HTML** | html.parser (stdlib) | -- | PSF | Built-in, no dependency |
| **Plain text** | Built-in I/O | -- | PSF | .txt, .md, .csv, .json, .xml, .log, .yaml |
| **Images** | pytesseract + Pillow | 0.3.13 / 12.1.0 | Apache 2.0 / HPND | OCR for .png, .jpg, .tif, .bmp |

### PDF Strategy: Dual Extraction Pipeline

```
PDF File
  |
  v
[1] pypdf (fast, reads text objects directly)
  |
  +---> Got text? --> Done
  |
  v
[2] pdfplumber (slower, handles complex layouts)
  |
  +---> Got text? --> Done
  |
  v
[3] OCR fallback (pytesseract, very slow)
  |
  +---> Return whatever OCR found
```

### Why This Multi-Library Approach

- pypdf is fast but fails on complex layouts and some fonts
- pdfplumber is slower but handles tables, unusual encoding
- OCR is last resort -- 100x slower but handles scanned documents
- Each step only runs if previous step failed
- One bad page never crashes the entire file

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **PyMuPDF (fitz)** | Very fast, good quality | AGPL license (viral), C dependency | Rejected -- license |
| **Apache Tika** | Handles everything | Requires Java runtime | Rejected -- Java dependency |
| **Unstructured.io** | ML-based layout analysis | Heavy dependencies, GPU preferred | Rejected -- too heavy |
| **Docling (IBM)** | Document understanding AI | New, complex, GPU needed | Rejected -- experimental |
| **LlamaParse** | Cloud-based, excellent quality | Requires internet + API key | Rejected -- offline-first |

---

## 11. Credential Storage

| | |
|---|---|
| **Chosen** | Windows Credential Manager (via keyring library) |
| **Fallback** | Environment variables (multiple aliases) |
| **Last resort** | Config file dictionary |

### Resolution Order

1. **Windows Credential Manager** (most secure -- encrypted by OS)
2. **Environment variables** (checks 4-6 aliases per credential type)
3. **Config file** (least preferred for secrets)

### Why Keyring + Windows Credential Manager

- OS-level encryption (DPAPI on Windows)
- Survives reboots without re-entry
- No plaintext files on disk
- Same credential store used by Git, VS Code, etc.
- Python `keyring` library is cross-platform

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **.env file** | Simple, portable | Plaintext on disk, git-leak risk | Rejected -- security |
| **python-dotenv** | Easy .env loading | Same plaintext problem | Rejected |
| **HashiCorp Vault** | Enterprise-grade | Server process, massive overhead | Rejected -- overkill |
| **AWS Secrets Manager** | Cloud-native | Requires AWS, internet | Rejected -- offline-first |
| **Encrypted JSON file** | Portable | Key management problem (where to store the key?) | Rejected |
| **Environment variables only** | No library needed | Lost on reboot, awkward UX | Supported as fallback |

---

## 12. Network Security

| | |
|---|---|
| **Chosen** | Centralized Network Gate (singleton access control) |
| **Default mode** | OFFLINE (fail-closed) |
| **Architecture** | 3-layer lockdown |

### Three-Layer Security Model

| Layer | Location | Mechanism | Purpose |
|---|---|---|---|
| 1 | start_hybridrag.ps1 | HF_HUB_OFFLINE=1 env var | Block HuggingFace downloads at session level |
| 2 | embedder.py | os.environ.setdefault at import | Block HuggingFace if launched outside start script |
| 3 | config.py | Empty API endpoint default | Prevent accidental data exfiltration through LLM path |

### Network Gate Modes

| Mode | Allowed Destinations | Use Case |
|---|---|---|
| **OFFLINE** | localhost only (127.0.0.1, ::1, 0.0.0.0) | Default, offline |
| **ONLINE** | localhost + configured API endpoint | When user explicitly enables |
| **ADMIN** | Unrestricted (with audit warnings) | Maintenance only |

### Why a Centralized Gate

- Before: network access scattered across 5+ modules, each with own policy
- If one module forgot to check, data could leak
- Central gate: every outbound call passes through one checkpoint
- Full audit trail: every allowed AND denied connection logged

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **OS-level firewall rules** | System-wide | Requires admin, affects other apps | Complementary, not primary |
| **Per-module URL checks** | Decentralized | Easy to forget, inconsistent policy | Rejected -- gate is better |
| **Proxy server** | Full traffic inspection | Heavy infrastructure, complex setup | Rejected -- overkill |
| **No network control** | Simpler code | Security violation for offline-first design | Rejected |

---

## 13. Logging

| | |
|---|---|
| **Chosen** | structlog + Python logging (hybrid) |
| **Format** | Structured JSON |
| **License** | MIT (structlog) |

### Log Files (Separate by Purpose)

| File | Content |
|---|---|
| app_YYYY-MM-DD.log | General events (indexing, queries, startup) |
| error_YYYY-MM-DD.log | Errors and failures |
| audit_YYYY-MM-DD.log | Security events (gate checks, credential access) |
| cost_YYYY-MM-DD.log | API cost tracking per query |

### Why Structured Logging

- Plain text: `Indexed file manual.pdf in 2.3 seconds`
- Structured: `{"event": "file_indexed", "file": "manual.pdf", "seconds": 2.3}`
- JSON is searchable with jq, parseable by dashboards, machine-readable
- Audit trail for security review

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Python logging (plain text)** | Built-in, simple | Not machine-parseable, hard to search | Rejected -- need structured output |
| **loguru** | Beautiful console output | Less structured, no JSON by default | Rejected |
| **Serilog-style** (.NET) | Great structured logging | Wrong ecosystem | N/A |
| **ELK Stack** (Elasticsearch) | Full search and dashboards | Massive infrastructure | Rejected -- overkill |
| **print statements** | Simplest | No structure, no levels, no files | Rejected |

---

## 14. LLM Client SDK

| | |
|---|---|
| **Chosen** | openai SDK 1.45.1 |
| **License** | Apache 2.0 |
| **Usage** | Azure OpenAI + standard OpenAI + OpenRouter |

### Why the Official openai SDK

- Handles URL construction, auth headers, retries automatically
- Same SDK works for Azure, OpenAI, and OpenAI-compatible services
- Eliminated two painful bugs from manual httpx calls:
  - 401 Unauthorized (wrong auth header format)
  - 404 Not Found (URL path doubling on Azure)
- Same approach used in enterprise example code

### Why Version 1.45.1 (Not Latest)

- Store-approved version (enterprise software policy)
- Zero code changes needed vs latest
- All 131+ tests pass

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **Raw httpx calls** | No SDK dependency | Manual URL/header building caused 401/404 bugs | Rejected -- replaced |
| **LangChain** | Abstractions for RAG pipeline | 100+ transitive dependencies, changes weekly | Rejected -- dependency hell |
| **LlamaIndex** | RAG-specific abstractions | Same dependency problem as LangChain | Rejected |
| **litellm** | Unified API for 100+ providers | Extra dependency, proxy complexity | Evaluated, not needed |

---

## 15. HTTP Client

| | |
|---|---|
| **Chosen** | httpx 0.28.1 |
| **License** | BSD |
| **Usage** | Ollama communication, deployment discovery |

### Why httpx

- Modern async + sync support
- Required by the openai SDK (already a dependency)
- Connection pooling with persistent clients
- Clean timeout handling

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **requests** | Ubiquitous, simple API | No async, no HTTP/2 | Still in requirements (used by some transitive deps) |
| **urllib3** | Low-level, reliable | Too low-level for application code | Used transitively by requests |
| **aiohttp** | Full async | Different API style, not needed for sync calls | Rejected |

---

## 16. Configuration

| | |
|---|---|
| **Chosen** | YAML (PyYAML 6.0.2) + Python dataclasses |
| **Config file** | config/default_config.yaml |
| **License** | MIT (PyYAML) |

### Why YAML + Dataclasses

- YAML is human-readable and editable with any text editor
- Dataclasses provide type safety and IDE autocomplete in Python
- Single source of truth: one YAML file, loaded into typed Config object
- Environment variable overrides for deployment-specific settings

### Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **JSON** | No extra dependency | No comments, less readable | Rejected -- YAML is friendlier |
| **TOML** | Python 3.11+ stdlib | Not in Python 3.10 stdlib, less familiar | Rejected -- version constraint |
| **.env files** | Simple key=value | Flat structure, no nesting, no types | Rejected -- too limited |
| **Python files** | Full language power | Security risk (arbitrary code execution) | Rejected |
| **INI files** | Simple, stdlib ConfigParser | No nesting, no lists, no types | Rejected -- too limited |

---

## 17. Prompt Strategy

| | |
|---|---|
| **Chosen** | 9-rule source-bounded generation prompt |
| **Version** | v4 |
| **Eval result** | 98% pass rate on 400-question golden set |

### Rule Hierarchy (Priority Order)

| # | Rule | Purpose |
|---|---|---|
| 1 | GROUNDING | Use only provided context, no outside knowledge |
| 2 | COMPLETENESS | Include all specific details (numbers, dates, parts) |
| 3 | REFUSAL | Say "not found" when answer is not in context |
| 4 | AMBIGUITY | Ask clarifying question if multiple answers exist |
| 5 | INJECTION RESISTANCE | Ignore malicious instructions embedded in context |
| 6 | ACCURACY | Never fabricate specifications |
| 7 | VERBATIM VALUES | Exact notation for measurements, tolerances |
| 8 | SOURCE QUALITY | Ignore test metadata, self-labeled untrustworthy content |
| 9 | EXACT LINE | For numeric specs, add "Exact: [verbatim from source]" |

### Why Priority-Ordered Rules

- Rule 5 (injection) before Rule 6 (accuracy) prevents prompt injection attacks
- Rule 4 (ambiguity) overrides Rule 9 (exact formatting) to avoid false precision
- Anti-hallucination: strict grounding in retrieved context only
- 8 known failures in 400 questions: 6 embedding misses, 2 fixed by Exact rule

---

## 18. Banned / Excluded Technologies

### Banned Model Families (NDAA / regulatory)

| Vendor | Models | Reason | Status |
|---|---|---|---|
| **Alibaba / Qwen** | Qwen, Qwen2, Qwen3 | China-origin, NDAA banned | Excluded from auto-selection |
| **DeepSeek** | DeepSeek-R1, DeepSeek-Coder | China-origin, NDAA banned | Excluded from auto-selection |
| **BAAI** | BGE embedding models | China-origin, NDAA banned | Excluded |
| **Meta / Llama** | Llama 2, Llama 3, Code Llama | Acceptable Use Policy prohibits milregulatoryy use (regulatory) | Excluded from auto-selection |

### Excluded Libraries

| Library | Reason |
|---|---|
| **LangChain** | 100+ transitive dependencies, weekly breaking changes |
| **LlamaIndex** | Same dependency problem |
| **ChromaDB** | C++ compiler required on Windows, install failures |
| **PyMuPDF (fitz)** | AGPL license (viral) |
| **Apache Tika** | Requires Java runtime |

### Full audit documentation: docs/enterprise_MODEL_AUDIT.md

---

## Dependency Summary

### Core (Required)

| Package | Version | Purpose | License |
|---|---|---|---|
| torch | 2.10.0 | Tensor operations for embeddings | BSD |
| sentence-transformers | 2.7.0 | Embedding model runtime | Apache 2.0 |
| numpy | 1.26.4 | Vector math + memmap storage | BSD |
| PyYAML | 6.0.2 | Configuration loading | MIT |
| httpx | 0.28.1 | HTTP client (Ollama + discovery) | BSD |
| openai | 1.45.1 | LLM API client SDK | Apache 2.0 |
| structlog | 24.4.0 | Structured JSON logging | MIT |
| cryptography | 44.0.2 | Credential encryption support | Apache 2.0 / BSD |

### Document Parsing

| Package | Version | Purpose | License |
|---|---|---|---|
| pypdf | 6.6.2 | PDF text extraction (primary) | BSD |
| pdfplumber | 0.11.9 | PDF text extraction (fallback) | MIT |
| python-docx | 1.2.0 | Word document parsing | MIT |
| openpyxl | 3.1.5 | Excel spreadsheet parsing | MIT |
| python-pptx | 1.0.2 | PowerPoint parsing | MIT |
| pytesseract | 0.3.13 | OCR wrapper | Apache 2.0 |
| Pillow | 12.1.0 | Image processing for OCR | HPND |
| lxml | 6.0.2 | XML parsing (used by document libs) | BSD |

### API Server

| Package | Version | Purpose | License |
|---|---|---|---|
| fastapi | 0.115.0 | REST API framework | MIT |
| uvicorn | 0.41.0 | ASGI server | BSD |
| pydantic | 2.11.1 | Request/response validation | MIT |

### GUI

| Package | Version | Purpose | License |
|---|---|---|---|
| tkinter | (stdlib) | GUI framework | PSF |

---

## Hardware Targets

| Machine | Specs | Primary Model | Notes |
|---|---|---|---|
| **Work Laptop** | i7-12700H, 16 GB RAM, Intel Iris Xe (512 MB) | phi4-mini (CPU) | All models run on CPU only |
| **Home PC** | (development machine) | phi4-mini | Development and testing |
| **Workstation** (arriving) | Dual RTX 3090 (48 GB GPU), 64 GB RAM | phi4:14b-q4_K_M | Can run all 5 approved models on GPU |
