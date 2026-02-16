# HybridRAG System State

Last Updated: 2026-02-07

## Current Version: v3 (Hybrid Search)

## Working Features

### Indexing Pipeline
- Crash-safe indexing with deterministic chunk IDs (INSERT OR IGNORE)
- Block-based text processing (200K char blocks) for RAM safety
- Skip unchanged files on restart (chunk-existence check)
- Anti-sleep: Windows SetThreadExecutionState prevents sleep during overnight runs
- Retry logic: _process_file_with_retry() with 3 attempts and exponential backoff
- File hash methods: _compute_file_hash() and _file_changed() (size+mtime)
- Garbage collection between files (gc.collect)
- Auto FTS5 rebuild after every indexing run
- Stale file cleanup: removes chunks for deleted source files

### Search Pipeline
- Hybrid search: vector similarity + BM25 keyword matching
- Reciprocal Rank Fusion (RRF) with configurable rrf_k=60
- FTS5 OR-logic: query words joined with OR for broader recall
- Optional cross-encoder reranker (disabled by default)
- Configurable min_score threshold and top_k

### LLM Integration
- Offline: Ollama (llama3, localhost:11434)
- Online: GPT-3.5 Turbo API with cost tracking
- Configurable timeout (180s default)

### Parsers
- PDF (pdfplumber), DOCX, PPTX, XLSX
- EML (email), plain text formats
- Image OCR via Tesseract (PNG, JPG, TIF, BMP, GIF, WEBP)

### Infrastructure
- SQLite metadata + text storage
- Memory-mapped float16 embeddings (scales to millions of chunks on 8GB RAM)
- FTS5 full-text search index (auto-rebuilt)
- Structured logging with run_id tracking
- Run tracker with SQLite audit trail

## Current Index Stats
- Database: HybridRAG_IndexedData/hybridrag.sqlite3
- Chunks: 1855 from 3 files
- Embeddings: 13712 rows in memmap
- FTS5: Synced (auto-rebuild active)

## Indexed Documents
1. Digisonde 4D Manual (1387 chunks)
2. Handbook of Ionogram Interpretation (462 chunks)
3. USRP B200/B210 Getting Started Guide (6 chunks)

## Current Config
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Chunk size: 1200 chars, overlap: 200 chars
- Retrieval: top_k=5, min_score=0.20, hybrid_search=true
- Timeout: 180 seconds
- Mode: offline (Ollama)

## Known Limitations (from audit)
- Hash-based change detection methods exist but not wired into skip logic
- No resource cleanup (close methods) for embedder and vector store
- No post-index validation to catch binary garbage chunk