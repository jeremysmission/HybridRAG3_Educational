# HybridRAG v3 Architecture

Last Updated: 2026-02-07

## 1. System Overview

HybridRAG is a local-first Retrieval Augmented Generation system designed for
research and engineering workflows. It indexes large document collections
and answers natural language queries using retrieved context.

Design priorities: offline operation, crash safety, RAM efficiency, auditability.

## 2. High-Level Data Flow
```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                      │
│                                                          │
│  Source Files ──→ Parser Registry ──→ Raw Text           │
│                    (PDF/DOCX/PPTX/                       │
│                     XLSX/EML/OCR/TXT)                    │
│                                                          │
│  Raw Text ──→ Chunker ──→ Chunks (1200 chars, 200 lap)  │
│                 (paragraph/sentence                      │
│                  boundary splitting,                     │
│                  heading prepend)                        │
│                                                          │
│  Chunks ──→ Embedder ──→ float32 vectors (384-dim)      │
│              (all-MiniLM-L6-v2)                          │
│                                                          │
│  Storage:                                                │
│    SQLite ← chunk text, metadata, FTS5 index             │
│    Memmap ← float16 embeddings (disk-backed)             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                         │
│                                                          │
│  User Query ──→ Embedder ──→ query vector (384-dim)      │
│                                                          │
│  Hybrid Search:                                          │
│    ├── Vector Search (cosine sim via memmap dot product)  │
│    ├── BM25 Keyword Search (FTS5 OR-logic)               │
│    └── Reciprocal Rank Fusion (RRF, k=60)                │
│                                                          │
│  Top-K Chunks ──→ Context Builder ──→ LLM Prompt         │
│                                                          │
│  LLM (Ollama offline / GPT API online) ──→ Answer        │
└─────────────────────────────────────────────────────────┘
```

## 3. Module Responsibilities

### src/core/config.py
Central configuration management. Loads from YAML with environment variable
overrides. Type-safe dataclasses for all settings. Validation at startup
catches misconfiguration before any work begins.

### src/core/indexer.py
The workhorse. Scans source folder, dispatches files to parsers, chunks text,
computes embeddings in batches, stores results. Features:
- Block-based processing (200K char blocks) for RAM safety
- Determisecurity standardic chunk IDs (same file + position = same ID every time)
- INSERT OR IGNORE for crash-safe restarts
- Retry logic with exponential backoff for network drive reads
- File hash methods for change detection (size + mtime)
- Garbage collection between files

### src/core/chunker.py
Splits raw text into overlapping chunks at paragraph or sentence boundaries.
Prepends detected section headings to each chunk so context survives chunking.
Configurable chunk_size (1200) and overlap (200).

### src/core/embedder.py
Wraps sentence-transformers all-MiniLM-L6-v2. Produces 384-dimensional
normalized float32 vectors. Dimension read from model (never hardcoded).
Batch embedding for indexing, single embedding for queries.

### src/core/vector_store.py
Dual storage system:
- SQLite: chunk text, metadata, FTS5 full-text index, run history
- Memmap: float16 embedding matrix on disk (numpy memory-mapped file)

Why two systems: SQLite handles structured queries efficiently. Memmap handles
millions of embedding vectors without loading them all into RAM. A laptop with
8GB RAM can search 10M+ embeddings because numpy reads only the rows needed.

### src/core/retriever.py
Hybrid search pipeline combining two ranking signals:

1. **Vector search**: Query embedding dot-producted against memmap in blocks.
   Returns top candidates by cosine similarity.

2. **BM25 keyword search**: FTS5 OR-logic query against chunk text.
   Critical for exact matches (part numbers, acronyms, technical terms).

3. **Reciprocal Rank Fusion (RRF)**: Merges both ranked lists using
   score = 1/(k + rank). Chunks appearing in both lists get boosted.
   k=60 is the standard smoothing constant from the original RRF paper.

Optional cross-encoder reranker available but disabled by default
(adds latency, minimal benefit at current corpus size).

### src/core/query_engine.py
Orchestrates the full query pipeline: retrieve chunks → build context →
construct prompt → call LLM → calculate cost → log result. Returns
structured QueryResult with answer, sources, token counts, and latency.

### src/core/llm_router.py
Routes to the appropriate LLM backend. Offline mode calls Ollama's
/api/generate endpoint. Online mode calls OpenAI-compatible /v1/chat/completions.
Explicit HTTP calls via httpx — no SDK dependencies, full control over
timeouts and error handling.

### src/parsers/registry.py
Maps file extensions to parser classes. Each parser implements .parse(path)
returning raw text. Registry pattern makes adding new formats trivial:
register the extension and parser class, everything else is automatic.

### src/tools/run_index_once.py
Production entry point for indexing. Handles:
- Windows anti-sleep (SetThreadExecutionState)
- Run tracking with unique run_id
- Progress logging with ETA calculation
- Auto FTS5 rebuild after indexing
- Stale file cleanup (removes chunks for deleted source files)
- Structured error logging

## 4. Storage Architecture
```
hybridrag.sqlite3
├── chunks          ← id, text, source_path, chunk_index, metadata
├── chunks_fts      ← FTS5 virtual table (auto-synced with chunks)
├── index_runs      ← Run audit trail (run_id, timestamps, counts)
└── query_log       ← Query audit trail (planned)

embeddings.f16.dat  ← Raw float16 matrix, shape [N, 384]
embeddings_meta.json ← {"dim": 384, "count": N, "dtype": "float16"}
```

Why float16: Halves storage (0.75GB vs 1.5GB per million chunks) with
negligible quality loss for cosine similarity on normalized vectors.

## 5. Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite over Postgres | Single-file, zero-config, portable, XCOPY-deployable |
| Memmap over FAISS | Simpler, no C++ dependencies, sufficient for <10M chunks |
| all-MiniLM-L6-v2 | 80MB, CPU-fast, 384-dim, pre-normalized, battle-tested |
| FTS5 OR-logic | AND-logic missed relevant chunks; OR with BM25 ranking catches more |
| RRF over linear combination | Rank-based fusion is robust to score scale differences |
| httpx over openai SDK | Zero magic, explicit timeouts, works in restricted environments |
| Determisecurity standardic chunk IDs | Crash-safe restart without duplicates |
| Block-based chunking | Process 200K char blocks to cap RAM on 500-page PDFs |
| Anti-sleep API | Prevent Windows from sleeping during 6+ hour overnight runs |
| Structured logging | Machine-parseable logs with run_id for audit and debugging |

## 6. Security Posture

- Offline mode: zero network access required
- Online mode: outbound HTTPS (443) only, authenticated with API key
- API keys loaded from environment variables, never in config files or logs
- No inbound ports, no listeners, no custom protocols
- SQLite database is local file — access controlled by OS file permissions
- Determisecurity standardic behavior: same input always produces same output
- All operations logged with run_id for audit trail
- Compatible with ACAS scans (standard Python libraries, no exotic binaries)

## 7. Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|-----------|
| Embedding speed | ~100 chunks/sec | CPU, all-MiniLM-L6-v2 |
| Vector search | <100ms | 2000 chunks, block scan |
| FTS5 search | <10ms | 2000 chunks |
| Index skip (unchanged) | <1 sec | 3 files, chunk-existence check |
| RAM usage (indexing) | ~500MB | Model + active blocks |
| RAM usage (search) | ~300MB | Model + memmap overhead |
| Disk per 1M chunks | ~0.75 GB | float16 embeddings only |

## 8. Future Roadmap

- Source-bounded generation (prevent hallucination)
- PII scanning and redaction
- GUI with engineering config menu
- Multi-user authentication and query audit
- Audio/video indexing (Whisper transcription)
- Task Scheduler integration for autonomous polling
- SQLCipher encryption at rest