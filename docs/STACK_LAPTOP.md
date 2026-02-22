# HybridRAG v3 -- Laptop Stack (8 GB RAM / 512 MB VRAM)

Last updated: 2026-02-22

Hardware: i7-12700H, 15.7 GB RAM (usable ~13 GB), Intel Iris Xe (512 MB VRAM), NVMe SSD

This is the "toaster" configuration -- every choice optimized for survival
on constrained hardware. The guiding principle: **it must run, not run fast.**

---

## At a Glance

| Component | Choice | Why This, Not Something Better |
|---|---|---|
| Embedding model | all-MiniLM-L6-v2 (384d, 80 MB) | Only model that loads in seconds on 8 GB |
| Embedding device | CPU | 512 MB VRAM cannot fit any transformer model |
| Embedding batch size | 16 | Larger batches cause disk swapping |
| Vector storage | SQLite + NumPy memmap (float16) | Zero-server, disk-backed, half the RAM of float32 |
| LLM model | phi4-mini (3.8B, Q4) | Only approved model that fits in RAM |
| LLM runtime | Ollama (CPU inference) | Single binary, manages memory automatically |
| Context window | 8,192 tokens | Larger windows spike RAM past available limit |
| Retrieval top_k | 5 | Fewer chunks = less prompt = faster inference |
| Reranker | Disabled | Cross-encoder needs ~500 MB RAM on top of LLM |
| GUI | tkinter (stdlib) | Zero install, zero RAM overhead |
| Chunk block size | 200,000 chars | Processes ~100 pages at a time to cap RAM |
| GC between blocks | Yes | Aggressive garbage collection prevents leak |
| Max concurrent files | 1 | Serial only -- parallel indexing would OOM |

---

## What Hurts on This Hardware

### 1. LLM Inference Speed: 8-12 tok/s

phi4-mini runs entirely on CPU. A typical 200-token answer takes 15-25 seconds.
There is no way to speed this up without GPU VRAM. Ollama's `OLLAMA_NUM_PARALLEL=1`
means queries are strictly sequential -- second user waits for first to finish.

| Query Type | Retrieval | LLM | Total |
|---|---|---|---|
| Easy (short answer) | ~200 ms | ~10 s | ~10 s |
| Hard (long reasoning) | ~200 ms | ~25 s | ~25 s |
| With online API instead | ~200 ms | ~1.5 s | ~1.7 s |

### 2. Embedding Speed: ~100 chunks/sec

all-MiniLM-L6-v2 on CPU embeds about 100 chunks per second.
Indexing 1,345 files (39,602 chunks) takes ~7 minutes.
The embedding model itself takes ~16 seconds to cold-start (torch + weights).

### 3. RAM Pressure During Indexing

| Component | Resident RAM |
|---|---|
| Python process | ~300 MB |
| sentence-transformers + torch | ~800 MB |
| Ollama (phi4-mini loaded) | ~4.5 GB |
| SQLite + memmap overhead | ~200 MB |
| **Total** | **~5.8 GB** |

That leaves ~8 GB for Windows + other apps. Larger batch sizes or parallel
processing cause swapping to disk, which tanks throughput by 10x.

### 4. No GPU Offload

Intel Iris Xe has 512 MB shared VRAM. Even the smallest transformer model
(all-MiniLM at 80 MB weights) would compete with Windows display for VRAM.
Everything runs on CPU. This is the single biggest performance constraint.

---

## Design Decisions Forced by Hardware

### Embedding: all-MiniLM-L6-v2 (Not Nomic, Not Snowflake)

| Model | Dims | RAM | Quality | Verdict on Laptop |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | ~400 MB | Good | Chosen -- fits comfortably |
| nomic-embed-text-v1.5 | 768 | ~800 MB | Better | Rejected -- doubles RAM + doubles vector storage |
| snowflake-arctic-embed-l | 1024 | ~1.5 GB | Best | Rejected -- would OOM during indexing |
| all-mpnet-base-v2 | 768 | ~900 MB | Good+ | Rejected -- same problem as nomic |

384 dimensions means each chunk costs 768 bytes (float16).
At 40,000 chunks that is 30 MB of vector data on disk.
At 768 dims it would be 60 MB; at 1024 dims, 80 MB.
The disk difference is minor but the RAM difference during search
(loading blocks of 25,000 vectors) is significant.

### LLM: phi4-mini (Not Mistral 7B, Not Nemo 12B)

| Model | Params | RAM (loaded) | Speed (CPU) | Verdict on Laptop |
|---|---|---|---|---|
| **phi4-mini** | 3.8B | ~4.5 GB | ~10 tok/s | Chosen -- fits with headroom |
| mistral:7b | 7B | ~6.5 GB | ~5 tok/s | Rejected -- barely fits, 2x slower |
| mistral-nemo:12b | 12B | ~10 GB | ~3 tok/s | Rejected -- OOM guaranteed |
| phi4:14b-q4_K_M | 14B | ~11 GB | ~2 tok/s | Rejected -- needs 11 GB VRAM |
| gemma3:4b | 4B | ~5 GB | ~8 tok/s | Possible alt -- slightly larger than phi4-mini |

phi4-mini is the only model where Ollama + Python + embedder + the model
all fit in 13 GB usable RAM without swapping.

### Vector Storage: float16 Memmap (Not float32, Not FAISS)

float16 halves memory pressure during search. When scanning 25,000 vectors:
- float16: 25,000 x 384 x 2 bytes = 18.75 MB per block
- float32: 25,000 x 384 x 4 bytes = 37.5 MB per block

On a RAM-constrained machine, this difference matters when multiple blocks
are scanned during a single query.

FAISS would be faster for search but requires a complex Windows build chain
and Meta is a banned vendor origin for the FAISS library (though it would
be acceptable as a search algorithm, not a model).

### Context Window: 8K (Not 16K, Not 128K)

Larger context windows mean Ollama allocates more KV-cache RAM upfront.
At 8K tokens with phi4-mini, the KV cache is manageable.
At 16K tokens, it would add ~500 MB to Ollama's resident footprint.
At 128K (mistral-nemo), it would need several GB for KV alone.

### Retrieval: top_k=5 (Not 12, Not 15)

Each retrieved chunk adds ~200-300 tokens to the prompt.
At top_k=12, the prompt is ~3,000+ tokens, increasing LLM processing time.
At top_k=5, the prompt stays under 1,500 tokens -- faster inference and
less RAM for the KV cache.

The 98% eval pass rate was achieved with top_k=12, but that was with
a cloud API. On the laptop, top_k=5 is the practical limit.

---

## What This Configuration Cannot Do

| Capability | Why Not |
|---|---|
| Serve 2+ concurrent users | Ollama is serial; second user waits 20-60s |
| Run reranker | Cross-encoder adds ~500 MB on top of existing load |
| Use larger embedding model | Would need to drop LLM from RAM to fit |
| Stream LLM output | Possible but adds complexity for marginal UX gain on 10s queries |
| Run FAISS GPU search | No usable GPU |
| Index 700+ GB source data | Would take ~12 hours and produce multi-GB vector files |

---

## Optimization Tricks Used

1. **Eager embedder preload** -- start loading torch + model weights at t=0, before
   boot/config/GUI. The 16s model load overlaps with the 2s GUI setup.

2. **Embedder cache** -- keep the Embedder instance alive across Reset clicks.
   The 8s model-load is paid once per process lifetime.

3. **Warm encode** -- fire a dummy `embed_query("warmup")` after load so the
   first real query pays zero lazy-init cost.

4. **Block-based indexing** -- process documents in 200K character blocks.
   RAM usage stays flat regardless of document size.

5. **Aggressive GC** -- `gc.collect()` between indexing blocks and on embedder close
   to reclaim torch tensor memory immediately.

6. **float16 vectors** -- half the storage and search memory of float32 with
   negligible quality loss (cosine similarity difference < 0.001).

7. **Deferred GUI init** -- `self.after(100, ...)` defers network-dependent init
   so the window renders instantly.

---

## Configuration File (Laptop)

```yaml
# config/default_config.yaml -- laptop-optimized
mode: offline
embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384
  device: cpu
  batch_size: 16
ollama:
  base_url: http://localhost:11434
  model: phi4-mini
  context_window: 8192
  timeout_seconds: 600
chunking:
  chunk_size: 1200
  overlap: 200
  max_heading_len: 160
indexing:
  block_chars: 200000
  max_chars_per_file: 2000000
retrieval:
  hybrid_search: true
  top_k: 5
  min_score: 0.1
  rrf_k: 60
  reranker_enabled: false
```

---

## Bottom Line

This laptop can run the full HybridRAG pipeline -- indexing, querying, GUI,
offline mode, online mode -- but it is slow. The primary value of the laptop
stack is **portability and demo capability**, not throughput. For actual
multi-user production work, the workstation stack is required.

The online mode (API) transforms the laptop experience: query latency drops
from 15-25s to 1.5-3s because the LLM inference happens in the cloud.
Everything else (embedding, retrieval, GUI) runs identically.
