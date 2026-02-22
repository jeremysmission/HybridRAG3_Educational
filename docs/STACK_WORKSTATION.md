# HybridRAG v3 -- Workstation Stack (64 GB RAM / 2x RTX 3090)

Last updated: 2026-02-22

Hardware: 16-thread CPU, 64 GB RAM, 2x NVIDIA RTX 3090 FE (24 GB VRAM each, 48 GB total), 2 TB NVMe SSD (7,250 MB/s)

This is the "production" configuration -- every component scaled up to
exploit the available hardware. The guiding principle: **throughput and
quality, constrained only by approved software.**

---

## At a Glance

| Component | Laptop (8 GB) | Workstation (64 GB) | Why It Changed |
|---|---|---|---|
| Embedding model | all-MiniLM-L6-v2 (384d) | all-MiniLM-L6-v2 (384d) | Same -- re-indexing 40K files not worth quality delta |
| Embedding device | CPU | **cuda** | 24 GB VRAM can host embedder + LLM simultaneously |
| Embedding batch size | 16 | **64** | 4x throughput, no swapping risk |
| LLM model | phi4-mini (3.8B) | **phi4:14b-q4_K_M (14B)** | 4x stronger reasoning, fits 11 GB VRAM on GPU 1 |
| LLM alt model | (none) | **mistral-nemo:12b** | 128K context, fits GPU 2 for concurrent serving |
| Context window | 8,192 | **16,384** | 2x more retrieval context for complex queries |
| Retrieval top_k | 5 | **12** | More chunks = better recall, RAM is no longer scarce |
| Chunk block size | 200,000 | **500,000** | 2.5x larger blocks, faster indexing |
| Max concurrent files | 1 | **2** | Parallel file processing |
| GC between blocks | Yes | **No** | 64 GB RAM, no memory pressure |
| Reranker | Disabled | **Still disabled** | Destroys behavioral eval scores regardless of hardware |

---

## What This Hardware Unlocks

### 1. GPU-Accelerated LLM Inference

phi4:14b-q4_K_M on RTX 3090 (24 GB VRAM):
- **40-60 tok/s** (vs 2 tok/s on CPU) -- 20-30x speedup
- 200-token answer: ~3-5 seconds (vs 60-120 seconds on laptop CPU)
- KV cache fits in VRAM: 16K context window is comfortable

phi4-mini on second RTX 3090:
- **80-120 tok/s** -- fast enough for concurrent serving
- Can serve a second user while GPU 1 handles the first

| Query Type | Laptop (CPU) | Workstation (GPU) | Speedup |
|---|---|---|---|
| Easy (short answer) | ~10 s | ~2 s | 5x |
| Hard (long reasoning) | ~25 s | ~5 s | 5x |
| 2 concurrent users | ~50 s (serial) | ~5 s (parallel GPUs) | 10x |

### 2. GPU-Accelerated Embeddings

sentence-transformers on CUDA with batch_size=64:
- **~400 chunks/sec** (vs ~100 on CPU) -- 4x speedup
- Indexing 39,602 chunks: ~1.5 minutes (vs ~7 minutes on laptop)
- Cold start: ~5s (GPU loads model faster than CPU)

### 3. Dual GPU Serving

| GPU | Model Loaded | VRAM Used | Purpose |
|---|---|---|---|
| GPU 0 | phi4:14b-q4_K_M | ~11 GB | Primary LLM (strongest reasoning) |
| GPU 1 | phi4-mini or mistral-nemo:12b | ~5.5-10 GB | Concurrent serving or alt model |

Ollama supports `CUDA_VISIBLE_DEVICES` to pin models to specific GPUs.
With `OLLAMA_NUM_PARALLEL=2`, two queries can run simultaneously on separate GPUs.

### 4. Large Index Support

| Index Scale | Chunks | Vector File (f16) | Search RAM | Feasible? |
|---|---|---|---|---|
| Current (1,345 files) | 39,602 | 30 MB | ~50 MB | Trivial |
| Medium (10K files) | ~300,000 | 225 MB | ~400 MB | Easy |
| Large (700 GB source) | ~8,400,000 | 6.45 GB | ~8 GB | Yes -- fits in 64 GB RAM |
| XL (2 TB source) | ~24,000,000 | 18.4 GB | ~24 GB | Tight but possible |

### 5. Concurrent Users

| Users | Retrieval | LLM (phi4-mini, Ollama) | LLM (phi4:14b, Ollama) | Total |
|---|---|---|---|---|
| 1 | 0.5 s | 2.8 s | 4.8 s | 3.3-5.3 s |
| 2 | 0.8 s | 5.6 s | 9.7 s | 6.4-10.5 s |
| 4 | 0.9 s | 8.4 s | 14.6 s | 9.3-15.5 s |
| 8 | 1.2 s | 14.5 s | 25.0 s | 15.7-26.2 s |
| 10 | 1.5 s | 16.9 s | 29.2 s | 18.4-30.7 s |

With vLLM instead of Ollama (continuous batching):
- 4 users: 7.3 s total (22% faster than Ollama)
- 10 users: 11.7 s total (36% faster than Ollama)

---

## Model Selection by Use Case (Workstation)

| Use Case | Primary | Alt | Temperature | Context | top_k |
|---|---|---|---|---|---|
| Software Engineering | phi4-mini | mistral:7b | 0.1 | 16,384 | 8 |
| Engineering / STEM | phi4-mini | mistral:7b | 0.1 | 16,384 | 8 |
| Systems Admin | phi4-mini | mistral:7b | 0.1 | 16,384 | 8 |
| Drafting / CAD | phi4-mini | phi4:14b-q4_K_M | 0.05 | 16,384 | 8 |
| Logistics | **phi4:14b-q4_K_M** | phi4-mini | 0.0 | 8,192 | 10 |
| Program Management | phi4-mini | gemma3:4b | 0.25 | 8,192 | 5 |
| Field Engineer | phi4-mini | mistral:7b | 0.1 | 16,384 | 8 |
| Cybersecurity | phi4-mini | mistral:7b | 0.1 | 16,384 | 8 |
| General AI | mistral:7b | phi4-mini | 0.3 | 8,192 | 5 |

When the upgrade models (mistral-nemo:12b) are loaded, they replace the
alt model for sw/eng/sys/cyber/gen profiles (128K context, stronger reasoning).

---

## What Stays the Same as Laptop

| Component | Value | Why No Change |
|---|---|---|
| Embedding model | all-MiniLM-L6-v2 | Changing dims requires full re-index of 40K chunks |
| Chunk size | 1,200 chars | Optimal for engineering PDFs regardless of hardware |
| Overlap | 200 chars | Same rationale |
| Retrieval method | Hybrid RRF | Algorithm is hardware-independent |
| Reranker | Disabled | Destroys eval scores -- this is a data quality issue, not hardware |
| GUI | tkinter | Works fine, no reason to change |
| API framework | FastAPI | Works fine at this scale |
| Security model | 3-layer gate | Security doesn't scale down |
| Prompt strategy | 9-rule v4 | Prompt is model-dependent, not hardware-dependent |

---

## Future Upgrades Possible on This Hardware

### 1. FAISS GPU Index

Replace NumPy memmap brute-force search with FAISS IVF on GPU:
- 8.4M vectors: search in ~50 ms (vs ~1.5 s with memmap scan)
- Requires re-indexing to build IVF clusters
- FAISS GPU needs ~18.6 GB VRAM for 8.4M vectors (fits one RTX 3090)

### 2. vLLM Continuous Batching

Replace Ollama with vLLM for multi-user serving:
- 3-5x throughput improvement at 4+ concurrent users
- Requires WSL2 or Linux (vLLM does not support native Windows)
- Continuous batching shares KV cache across concurrent requests

### 3. mistral-small3.1:24b

24B model with Apache 2.0 license, 128K context:
- Fits one RTX 3090 (24 GB VRAM)
- Replaces phi4-mini as primary for all profiles
- Strongest approved reasoning model available

### 4. Larger Embedding Model

If re-indexing is acceptable:
- nomic-embed-text-v1.5 (768d): 10-15% better recall, 2x storage
- snowflake-arctic-embed-l (1024d): 15-25% better recall, 2.7x storage
- Both fit in GPU VRAM for accelerated embedding

---

## Configuration File (Workstation)

```yaml
# config/default_config.yaml -- workstation-optimized
mode: offline
embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384
  device: cuda
  batch_size: 64
ollama:
  base_url: http://localhost:11434
  model: phi4:14b-q4_K_M
  context_window: 16384
  timeout_seconds: 120
chunking:
  chunk_size: 1200
  overlap: 200
  max_heading_len: 160
indexing:
  block_chars: 500000
  max_chars_per_file: 2000000
retrieval:
  hybrid_search: true
  top_k: 12
  min_score: 0.1
  rrf_k: 60
  reranker_enabled: false
```

---

## Bottom Line

The workstation transforms HybridRAG from a "demo that works" into a
"tool you can actually use." Query latency drops from 15-25s to 3-5s,
indexing is 4x faster, and the system can serve 2-4 concurrent users
with acceptable latency.

The dual RTX 3090 setup means you can run the 14B model (strongest
reasoning) on GPU 0 and a fast 3.8B model on GPU 1 simultaneously --
one for quality queries, one for quick lookups.

For 8+ concurrent users, consider switching from Ollama to vLLM
(requires WSL2/Linux) for continuous batching support.
