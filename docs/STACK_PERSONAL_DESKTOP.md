# HybridRAG v3 -- Personal Desktop Stack (128 GB RAM / 48 GB VRAM)

Last updated: 2026-02-22

Target build: 128 GB RAM, 48 GB VRAM, personal desktop
Preference: portable, modular, open source

This is the "dream machine" configuration -- unconstrained by enterprise
policy, optimized for maximum local AI capability. The guiding principle:
**open source everything, run the best models that fit, no cloud needed.**

---

## Hardware Recommendations

### GPU: How to Get 48 GB VRAM

| Option | VRAM | Pooled? | tok/s (70B Q4) | Power | Price (Feb 2026) | Verdict |
|---|---|---|---|---|---|---|
| **Dual RTX 3090 + NVLink** | 48 GB | Yes | ~15-20 | ~700W | ~$1,900 | Best value for pooled 48 GB |
| Single RTX 5090 | 32 GB | N/A | ~27 | 575W | ~$3,000-5,000 | Fastest single card, but only 32 GB |
| Dual RTX 5090 | 64 GB | No (sharded) | ~27 | 1,150W | ~$6,000-10,000 | Overkill, no NVLink pooling |
| AMD Radeon PRO W7900 | 48 GB | N/A | ~11 | 295W | ~$3,500-4,000 | True 48 GB single card, but ROCm ecosystem |
| RTX PRO 6000 Blackwell | 96 GB | N/A | Unknown | 600W | ~$8,500 | Future-proof but expensive |

### Recommended: Dual RTX 3090 + NVLink

**Why this wins for your use case:**

- Only consumer GPU with NVLink support (pools VRAM into unified 48 GB)
- Best price-per-GB of VRAM on the market (~$40/GB vs ~$100+/GB for RTX 5090)
- Proven for LLM inference -- massive community support, every framework works
- 48 GB pooled = run 70B Q4 models as a single GPU workload
- Well-matched to 128 GB RAM (CPU offload for models > 48 GB VRAM)

**The RTX 5090 alternative:** If you can live with 32 GB instead of 48 GB,
a single RTX 5090 gives ~2x the bandwidth (1,792 GB/s vs 936 GB/s) and
simpler build. But 32 GB limits you to ~50B Q4 models max, and no NVLink
means a second 5090 doesn't pool VRAM.

### Full Build Recommendation

| Component | Recommendation | Est. Price |
|---|---|---|
| **GPU** | 2x RTX 3090 FE + NVLink bridge | ~$1,900 |
| **RAM** | 128 GB DDR5-5600 (4x 32 GB) | ~$300 |
| **CPU** | AMD Ryzen 9 7950X (16c/32t) or Intel i9-14900K | ~$450 |
| **PSU** | 1200W 80+ Platinum (Corsair HX1200 or similar) | ~$250 |
| **Motherboard** | X670E with 2x PCIe 4.0 x16 slots + 4x DDR5 | ~$350 |
| **Storage** | 2 TB NVMe Gen4 (Samsung 990 Pro or similar) | ~$150 |
| **Case** | Full tower with good airflow (be quiet! Dark Base) | ~$200 |
| **Cooling** | 360mm AIO for CPU + stock GPU coolers | ~$150 |
| **Total** | | **~$3,750** |

**Power budget:** 700W (GPUs) + 125W (CPU) + 50W (other) = ~875W peak.
The 1200W PSU provides 37% headroom.

---

## Software Stack (All Open Source)

### Inference Engines

| Engine | License | Role | Why |
|---|---|---|---|
| **llama.cpp / llama-server** | MIT | Primary interactive inference | Most portable. GGUF format. Mixed CPU/GPU offload. Runs everywhere. |
| **vLLM** | Apache 2.0 | Multi-user serving | Continuous batching, PagedAttention (60-80% less VRAM waste), tensor parallelism across dual GPUs. 3-5x throughput vs Ollama at 4+ users. |
| **Ollama** | MIT | Quick testing / model management | Pull-and-run convenience. Your project already uses it. Good for single-user interactive. |

**Recommendation:** Keep Ollama for development/testing (already integrated).
Add llama.cpp for maximum flexibility (GGUF ecosystem, mixed offload).
Deploy vLLM when you need multi-user serving or batch inference.

All three are MIT or Apache 2.0 licensed. All three support CUDA.

### Why Not These

| Engine | License | Issue |
|---|---|---|
| TensorRT-LLM | Apache 2.0 | Requires per-GPU model compilation. Least portable. NVIDIA-only. |
| HF TGI | Apache 2.0 | Heavier than vLLM for same purpose. Less community adoption. |
| ExLlamaV2 | MIT | Good but narrower format support (EXL2 only). |

---

## LLM Models: What 48 GB VRAM Unlocks

### VRAM Budget (48 GB pooled via NVLink)

| Quantization | Bytes/Param | Max Model Size | Quality vs FP16 |
|---|---|---|---|
| FP16 | 2.0 | ~22B | 100% (baseline) |
| Q8_0 | 1.0 | ~42B | ~99.5% |
| Q6_K | 0.78 | ~55B | ~99% |
| Q5_K_M | 0.68 | ~60B | ~98% |
| Q4_K_M | 0.57 | ~75B | ~96% |

### Approved Models by Tier (NO Llama, NO Qwen, NO DeepSeek, NO BAAI)

#### Flagship (run at Q8 or FP16 in 48 GB)

| Model | Params | Quant | VRAM | License | Origin | Context | Why |
|---|---|---|---|---|---|---|---|
| **Mistral Small 3.1** | 24B | Q8 | ~28 GB | Apache 2.0 | Mistral/France | 128K | Best all-around. Outperforms GPT-4o Mini. Multimodal. |
| **Mistral Small 3.1** | 24B | FP16 | ~55 GB | Apache 2.0 | Mistral/France | 128K | Needs CPU offload for ~7 GB. Max quality. |
| **Phi-4** | 14B | FP16 | ~28 GB | MIT | Microsoft/USA | 16K | Strongest STEM reasoning per parameter. |
| **Devstral Small 2** | 24B | Q8 | ~28 GB | Apache 2.0 | Mistral/France | 128K | Code-specialized Mistral. Top coding benchmarks. |
| **Gemma 2 27B** | 27B | Q8 | ~30 GB | Apache 2.0 | Google/USA | 8K | Strong reasoning. Apache 2.0 (not restrictive Gemma 3 ToU). |

#### Sweet Spot (highest quality that fits with room for other workloads)

| Model | Params | Quant | VRAM | License | Role |
|---|---|---|---|---|---|
| **Mistral Small 3.1 24B** | 24B | Q8 | ~28 GB | Apache 2.0 | Primary for everything |
| **Phi-4 14B** | 14B | FP16 | ~28 GB | MIT | Secondary for STEM/code |
| **Mistral Nemo 12B** | 12B | FP16 | ~24 GB | Apache 2.0 | Fast model, 128K context |

With 48 GB VRAM, you can keep Mistral Small 3.1 (28 GB Q8) loaded on GPU 0
and Mistral Nemo 12B (24 GB FP16) loaded on GPU 1 simultaneously. Two models
ready to serve different use cases without reload.

#### At Q4 (push the limits)

| Model | Params | Quant | VRAM | License | Note |
|---|---|---|---|---|---|
| **Mistral Small 3.1** | 24B | Q4 | ~14 GB | Apache 2.0 | Fits with massive headroom |
| **Gemma 2 27B** | 27B | Q4 | ~18 GB | Apache 2.0 | 8K context limit |
| **Gemma 3 27B** | 27B | Q4 | ~14 GB | Gemma ToU | Restrictive license (remote kill). Use Gemma 2 instead. |

Note: There are NO approved-vendor 70B models available (Mistral Large 2 at 123B
is too large, and the 70B parameter class is dominated by Llama which is banned).
The practical ceiling for approved vendors is ~27B at high quality or ~24B at FP16.

### Comparison: Laptop vs Workstation vs Personal Desktop

| | Laptop (512 MB) | Workstation (48 GB) | Personal Desktop (48 GB) |
|---|---|---|---|
| Best model | phi4-mini 3.8B Q4 | phi4:14b Q4 | Mistral Small 3.1 24B Q8 |
| Quality tier | Basic (52 ENG / 48 GEN) | Strong (72 ENG / 65 GEN) | Near-frontier (~85 ENG / ~82 GEN) |
| Inference speed | ~10 tok/s | ~40-60 tok/s | ~30-50 tok/s (Q8 24B) |
| Context window | 8K | 16K | 128K |
| Concurrent models | 0 (barely fits 1) | 1-2 | 2-3 |

---

## Embedding Models: What 128 GB RAM + GPU Unlocks

### Recommended (Approved Vendors Only)

| Model | Params | Dims | License | Origin | MTEB Score | Why |
|---|---|---|---|---|---|---|
| **Snowflake Arctic Embed L v2.0** | 303M | up to 1024 | Apache 2.0 | Snowflake/USA | Top-tier | Best retrieval benchmarks. MRL support (compress dimensions). GPU-accelerated. |
| **Microsoft E5-large-instruct** | 560M | 1024 | MIT | Microsoft/USA | Top-tier | Instruction-tunable. 100 languages. Task-customizable prompts. |
| **Nomic Embed Text v2 MoE** | 475M (305M active) | 768 | Apache 2.0 | Nomic AI/USA | Strong | First MoE embedding. Efficient inference. Matryoshka dims. |
| **all-MiniLM-L6-v2** | 22M | 384 | Apache 2.0 | Community | Good | Current choice. Fast, lightweight. Keep as fallback. |

### Recommendation: Snowflake Arctic Embed L v2.0

- Apache 2.0 license, USA-based vendor (Snowflake)
- Top MTEB retrieval scores (not just average -- specifically retrieval)
- Matryoshka Representation Learning: store 256d or 512d vectors but train at 1024d
  (query with 1024d at search time for quality, store at 512d for space)
- GPU-accelerated: >100 docs/sec on a single GPU, sub-10ms query latency
- 8192 token input context (handles entire chunks without truncation)

**Migration note:** Changing embedding model requires full re-indexing.
At 128 GB RAM with GPU acceleration, re-indexing 40K chunks takes ~2 minutes.
Re-indexing 8.4M chunks (700 GB source) would take ~2-3 hours.

### Storage Impact

| Model | Dims | Per-chunk (f16) | 40K chunks | 8.4M chunks |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 768 bytes | 30 MB | 6.45 GB |
| Nomic Embed v2 | 768 | 1,536 bytes | 60 MB | 12.9 GB |
| Snowflake Arctic L | 1024 | 2,048 bytes | 80 MB | 17.2 GB |
| Snowflake Arctic L (MRL 512d) | 512 | 1,024 bytes | 40 MB | 8.6 GB |

With 128 GB RAM and NVMe SSD, even the 17 GB vector file is trivial.
MRL at 512d gives 90%+ of the quality at half the storage.

---

## Vector Database: Scale-Up Options

### Current: SQLite + NumPy Memmap

Works perfectly at 40K chunks. At 8.4M chunks, brute-force memmap search
takes ~1.5 seconds per query. Still functional but worth upgrading.

### Recommended Upgrade Path

| Database | License | GPU Accel | Query @ 10M vecs | Best For |
|---|---|---|---|---|
| **FAISS** | MIT | Yes (GPU index + search) | 1-5 ms | Raw speed, in-memory indexes |
| **LanceDB** | Apache 2.0 | Yes (GPU indexing) | 40-60 ms | Embedded, disk-backed, full CRUD |
| **Qdrant** | Apache 2.0 | No native GPU | 20-30 ms | Metadata filtering, production API |

### Recommendation: FAISS for Search + SQLite for Metadata

Keep the current hybrid architecture but replace NumPy memmap with FAISS:

- **FAISS IVF4096,SQ8** index on GPU: 1-5 ms search at 10M vectors
- SQLite continues to hold metadata, text, and FTS5 keyword index
- 128 GB RAM easily holds the entire FAISS index in memory
- GPU-accelerated index building (minutes, not hours)
- MIT license, well-proven, massive community

**Why not LanceDB?** LanceDB is excellent and more "batteries-included" than FAISS,
but FAISS gives lower latency (1-5 ms vs 40-60 ms) and your existing SQLite
metadata layer already provides what LanceDB's extra features would add.
If you wanted to eliminate the SQLite dependency entirely, LanceDB would be
the better choice.

**Why not Qdrant?** Requires a server process. For a personal desktop, embedded
is preferable (no Docker, no service management). FAISS and LanceDB are both
in-process.

---

## Complete Stack Comparison: All Three Tiers

| Component | Laptop (8 GB) | Workstation (64 GB) | Personal Desktop (128 GB) |
|---|---|---|---|
| **GPU** | Intel Iris Xe (512 MB) | 2x RTX 3090 (48 GB) | 2x RTX 3090 (48 GB) |
| **RAM** | 15.7 GB | 64 GB | 128 GB |
| **Embedding** | all-MiniLM-L6-v2 (384d, CPU) | all-MiniLM-L6-v2 (384d, CUDA) | Snowflake Arctic L v2 (1024d, CUDA) |
| **Embed speed** | ~100 chunks/s | ~400 chunks/s | ~1,000+ chunks/s |
| **LLM** | phi4-mini 3.8B (CPU) | phi4:14b Q4 (GPU) | Mistral Small 3.1 24B Q8 (GPU) |
| **LLM speed** | ~10 tok/s | ~40-60 tok/s | ~30-50 tok/s |
| **Context** | 8K tokens | 16K tokens | 128K tokens |
| **Vector store** | SQLite + memmap | SQLite + memmap | FAISS GPU + SQLite |
| **Search latency** | ~200 ms | ~200 ms | ~5 ms |
| **Inference engine** | Ollama | Ollama | Ollama + vLLM + llama.cpp |
| **Concurrent models** | 1 (barely) | 1-2 | 2-3 |
| **Concurrent users** | 1 | 2-4 | 4-10 (vLLM) |
| **Query latency** | 10-25 s | 3-5 s | 2-4 s |
| **Index 40K chunks** | ~7 min | ~1.5 min | ~40 sec |
| **Max index size** | ~500K chunks | ~8M chunks | ~24M+ chunks |

---

## Configuration File (Personal Desktop)

```yaml
# config/default_config.yaml -- personal desktop optimized
mode: offline
embedding:
  model_name: Snowflake/snowflake-arctic-embed-l-v2.0
  dimension: 1024            # Or 512 with MRL for space savings
  device: cuda
  batch_size: 128
ollama:
  base_url: http://localhost:11434
  model: mistral-small3.1:24b-instruct-2503-q8_0
  context_window: 131072     # 128K context
  timeout_seconds: 120
chunking:
  chunk_size: 1200
  overlap: 200
  max_heading_len: 160
indexing:
  block_chars: 1000000       # 1M chars per block
  max_chars_per_file: 2000000
retrieval:
  hybrid_search: true
  top_k: 15                  # More chunks, plenty of context budget
  min_score: 0.1
  rrf_k: 60
  reranker_enabled: false    # Still disabled for eval integrity
```

---

## Open Source License Summary

Every component in this stack is MIT or Apache 2.0:

| Component | Tool | License |
|---|---|---|
| LLM inference | Ollama / llama.cpp / vLLM | MIT / MIT / Apache 2.0 |
| Primary LLM | Mistral Small 3.1 24B | Apache 2.0 |
| Secondary LLM | Phi-4 14B | MIT |
| Fast LLM | Mistral Nemo 12B | Apache 2.0 |
| Coding LLM | Devstral Small 2 24B | Apache 2.0 |
| Embedding model | Snowflake Arctic Embed L v2.0 | Apache 2.0 |
| Embedding runtime | sentence-transformers | Apache 2.0 |
| Vector search | FAISS | MIT |
| Metadata store | SQLite | Public domain |
| API layer | FastAPI + Uvicorn | MIT / BSD |
| Configuration | PyYAML | MIT |
| Logging | structlog | MIT |
| GUI | tkinter | PSF |

No proprietary licenses. No cloud dependencies. No phone-home.
Everything runs on your hardware, in your house, forever.

---

## What This Build Cannot Do (and What Would Fix It)

| Limitation | Why | Fix |
|---|---|---|
| Run 70B+ approved models at high quality | No approved 70B model exists (Llama banned, Qwen banned) | Wait for Mistral/Microsoft 70B release |
| FP16 on 27B+ models | 27B FP16 = ~54 GB, exceeds 48 GB VRAM | CPU offload (~7 GB to RAM), or upgrade to RTX PRO 6000 (96 GB) |
| Train / fine-tune models | 48 GB VRAM too small for training 24B+ | LoRA/QLoRA fits, full fine-tune needs 4x A100 |
| Serve 20+ concurrent users | Ollama is serial, vLLM peaks around 10-15 | Deploy multiple vLLM instances across GPUs |
