# HybridRAG Performance Baseline

Last Updated: 2026-02-07

## Current Index

| Metric | Value |
|--------|-------|
| Total files | 3 |
| Total chunks | 1855 |
| Total embeddings | 13,712 (memmap rows) |
| Database size | ~15 MB |
| FTS5 status | Synced (auto-rebuild) |

## Indexed Documents

| Document | Chunks | Characters |
|----------|--------|-----------|
| Digisonde 4D Manual | 1,387 | 585,355 |
| Handbook of Ionogram Interpretation | 462 | 320,143 |
| USRP B200/B210 Getting Started Guide | 6 | 5,819 |

## Indexing Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Full index (3 files) | ~45 sec | Includes model load (~3s) |
| Skip run (no changes) | <1 sec | Chunk-existence check |
| Embedding throughput | ~100 chunks/sec | CPU, all-MiniLM-L6-v2 |
| FTS rebuild | <50ms | After indexing |
| Stale cleanup | <10ms | No stale files |

## Search Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Hybrid search (vector + BM25) | <200ms | 1855 chunks, RRF fusion |
| Vector-only search | <100ms | Block scan, cosine similarity |
| FTS5 keyword search | <10ms | OR-logic, BM25 ranking |
| End-to-end query (offline) | 60-120 sec | Llama3 on CPU via Ollama |

## Retrieval Quality

| Setting | Value |
|---------|-------|
| Search mode | Hybrid (vector + BM25 RRF) |
| top_k | 5 |
| min_score | 0.20 |
| rrf_k | 60 |
| reranker | Disabled |

### Observed Behavior
- Hybrid search returns relevant chunks for both semantic queries
  ("What is the operating frequency?") and keyword queries ("USRP B210")
- FTS5 OR-logic fixed prior issue where AND-logic missed relevant chunks
- min_score=0.20 filters noise while retaining useful results
- top_k=5 balances context quality vs LLM response time on CPU

## Resource Usage

| Resource | Indexing | Querying |
|----------|---------|---------|
| RAM | ~500 MB | ~300 MB |
| CPU | High (embedding) | Moderate (search) then high (LLM) |
| Disk (database) | ~15 MB | Read-only |
| Disk (embeddings) | ~10 MB | Memory-mapped, minimal RAM |
| Network | None (offline) | None (offline) |

## Known Baselines for Regression Testing

These values should remain stable unless code changes affect retrieval:
```
FTS match for "USRP": 6 chunks
FTS match for "digisonde": 203 chunks
Total chunks in database: 1855
Total embedding rows: 13,712
```

## How to Reproduce
```powershell
# Activate environment
. .\start_hybridrag.ps1

# Run indexing
rag-index

# Verify FTS
python -c "
import sqlite3
db = r'{PROJECT_ROOT}\data\hybridrag.sqlite3'
conn = sqlite3.connect(db)
print('USRP:', conn.execute(\"\"\"SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'USRP'\"\"\").fetchone()[0])
print('digisonde:', conn.execute(\"\"\"SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'digisonde'\"\"\").fetchone()[0])
print('total:', conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0])
conn.close()
"
```