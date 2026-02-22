#!/usr/bin/env python3
# ============================================================================
# HybridRAG -- Multi-User Workstation Stress Test Simulation
# ============================================================================
# FILE: tests/stress_test_workstation_simulation.py
#
# WHAT THIS FILE DOES (plain English):
#   Simulates a realistic multi-user workstation scenario to predict
#   query response times under concurrent load. Models the ENTIRE
#   RAG pipeline: embedding, vector search, BM25, RRF fusion,
#   reranker (optional), context building, and LLM inference.
#
#   Tests both offline (Ollama local models) and online (API endpoint)
#   modes with varying user counts: 10, 8, 6, 4, 3, 2 simultaneous.
#
# HARDWARE PROFILE:
#   - CPU: Multi-core workstation (assumed 16 threads)
#   - RAM: 64 GB
#   - GPU: NVIDIA 12 GB VRAM
#   - Storage: 2 TB HDD
#   - Source data: 700 GB miscellaneous formats
#
# METHODOLOGY:
#   This is a SIMULATION, not a live benchmark. We model each pipeline
#   stage with measured/documented latency values scaled by data size
#   and concurrency. The simulation accounts for:
#     - GPU memory contention (only one model in VRAM at a time)
#     - CPU thread contention for embedding and search
#     - SQLite read concurrency (readers don't block readers)
#     - Memmap I/O pressure on HDD vs SSD
#     - LLM token generation throughput (GPU-bound)
#     - API rate limits and network latency (online mode)
#
# HOW TO RUN:
#   python tests/stress_test_workstation_simulation.py
#
# INTERNET ACCESS: NONE (pure calculation)
# ============================================================================

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ============================================================================
# HARDWARE PROFILE
# ============================================================================
# Non-programmer note:
#   These numbers describe the workstation hardware. Each value affects
#   different parts of the pipeline. RAM determines how much data we
#   can hold in memory. VRAM determines LLM model size and speed.
#   Storage speed affects how fast we read embeddings from disk.
# ============================================================================

class HardwareProfile:
    """Workstation hardware specification."""

    def __init__(
        self,
        name: str = "Workstation",
        cpu_threads: int = 16,
        ram_gb: float = 64.0,
        gpu_vram_gb: float = 12.0,
        gpu_name: str = "NVIDIA 12GB",
        storage_type: str = "NVMe SSD",
        storage_model: str = "Generic NVMe",
        storage_read_mbps: float = 3500,
        storage_iops: int = 500000,
        network_mbps: float = 100,
    ):
        self.name = name
        self.cpu_threads = cpu_threads
        self.ram_gb = ram_gb
        self.gpu_vram_gb = gpu_vram_gb
        self.gpu_name = gpu_name
        self.storage_type = storage_type
        self.storage_model = storage_model
        self.storage_read_mbps = storage_read_mbps
        self.storage_iops = storage_iops
        self.network_mbps = network_mbps


# Pre-built hardware profiles
PERSONAL_WORKSTATION = HardwareProfile(
    name="Personal Workstation (Home Build)",
    cpu_threads=16,           # AMD Ryzen 7 7700X -- 8 cores / 16 threads, 5.4 GHz boost
    ram_gb=128.0,             # 128 GB DDR5 6000 MT/s (2x G.SKILL Flare X5 64GB kits)
    gpu_vram_gb=48.0,         # 2x RTX 3090 FE = 24 GB GDDR6X each = 48 GB total
    gpu_name="2x RTX 3090 FE (NVLink)",
    storage_type="NVMe SSD",
    storage_model="Samsung 990 EVO Plus 2TB (PCIe Gen5x2/Gen4x4)",
    storage_read_mbps=7250,   # Samsung 990 EVO Plus sequential read
    storage_iops=1000000,     # PCIe Gen5 class random IOPS
)
# Personal workstation build:
#   Mobo:  ASUS TUF Gaming B650-PLUS WiFi (AM5, PCIe 5.0 M.2)
#   PSU:   Corsair RM1200x Shift 1200W (ATX 3.1, 80+ Gold)
#   Case:  Lian Li Dynamic EVO XL (E-ATX, up to 3x 420mm rad)
#   Cool:  Thermalright Peerless Assassin 120 SE + 5x Arctic P14 140mm
#   GPU:   2x NVIDIA GeForce RTX 3090 Founders Edition (Renewed)
#          - 24 GB GDDR6X each, 936 GB/s mem bandwidth each
#          - 10,496 CUDA cores each, 350W TDP each
#          - NVLink bridge supported (112.5 GB/s bidirectional)
#   RAM:   4x 32GB G.SKILL Flare X5 DDR5 6000 CL30 (AMD EXPO)
#
# KEY ADVANTAGES:
#   - 48 GB total VRAM = can run 70B parameter models (q4 quantized)
#   - Dual GPU with vLLM tensor parallel = 2x throughput
#   - Or: serve TWO different models simultaneously (one per GPU)
#   - 128 GB RAM = handles 2TB+ indexes entirely in memory
#   - RTX 3090 GDDR6X at 936 GB/s = ~3.7x baseline memory bandwidth
#   - 1200W PSU handles both GPUs at full load (2x 350W + system)

WORK_LAPTOP = HardwareProfile(
    name="Work Laptop (Demo)",
    cpu_threads=28,           # Intel Core Ultra 7 265HX: 8P (HT) + 12E = 28 threads
    ram_gb=64.0,              # 64 GB SODIMM DDR5 6400 MT/s
    gpu_vram_gb=12.0,         # NVIDIA RTX Pro 3000 Blackwell, 12 GB GDDR7
    gpu_name="RTX Pro 3000 Blackwell (992 AI TOPS)",
    storage_type="NVMe SSD",
    storage_model="WD PC SN8000S SED SanDisk (PCIe Gen4)",
    storage_read_mbps=5545,   # WD SN8000S sequential read
    storage_iops=700000,      # Estimated PCIe Gen4 random IOPS
    network_mbps=100,
)
# Work laptop extras (not in HardwareProfile but used in simulation):
#   NPU: Intel AI Boost, 11 TOPS, 36 GB shared memory
#   CPU: Intel Core Ultra 7 265HX (8P + 12E, 5.3 GHz turbo)
#   Cache: L1 2MB, L2 36MB, L3 30MB (68 MB total cache!)
#   GPU mem bandwidth: 672 GB/s (GDDR7) -- ~3x faster than desktop GDDR6


# ============================================================================
# INDEX PROFILE (derived from 700 GB source data)
# ============================================================================
# Non-programmer note:
#   700 GB of mixed documents produces a LOT of text chunks. We estimate
#   based on real-world ratios:
#     - Average file yields ~50 KB of extractable text
#     - Binary formats (CAD, images) yield much less text
#     - Each chunk is ~1200 characters with 200-char overlap
#     - Each chunk produces one 384-dim embedding (768 bytes in float16)
# ============================================================================

class IndexProfile:
    """Estimated index size from 700 GB source data."""

    def __init__(self, source_gb: float = 700.0):
        self.source_gb = source_gb

        # ---------- Estimation logic ----------
        # Mixed format corpus: ~60% is parseable text-bearing formats
        # Average text yield: ~2% of file size for mixed (PDFs, DOCX, CAD, images)
        # This is conservative -- pure text files yield 100%, images yield <1%
        self.parseable_fraction = 0.60
        self.text_yield_ratio = 0.02  # 2% of file bytes become extractable text

        effective_gb = source_gb * self.parseable_fraction
        text_gb = effective_gb * self.text_yield_ratio
        text_chars = text_gb * 1e9  # chars (approx 1 byte per char for ASCII)

        # Chunking: 1200 chars per chunk, 200 overlap = 1000 net chars per chunk
        self.chunk_size = 1200
        self.overlap = 200
        net_chars_per_chunk = self.chunk_size - self.overlap
        self.total_chunks = int(text_chars / net_chars_per_chunk)

        # Embeddings: 384-dim float16 = 768 bytes per embedding
        self.embedding_dim = 384
        self.bytes_per_embedding = self.embedding_dim * 2  # float16
        self.embeddings_size_gb = (
            self.total_chunks * self.bytes_per_embedding / 1e9
        )

        # SQLite DB: ~500 bytes per chunk (text + metadata + FTS index)
        self.sqlite_size_gb = self.total_chunks * 500 / 1e9

        # Total index size
        self.total_index_gb = self.embeddings_size_gb + self.sqlite_size_gb

    def summary(self) -> str:
        return (
            f"Source data: {self.source_gb:.0f} GB\n"
            f"Estimated chunks: {self.total_chunks:,}\n"
            f"Embeddings file: {self.embeddings_size_gb:.2f} GB "
            f"({self.embedding_dim}-dim float16)\n"
            f"SQLite DB: {self.sqlite_size_gb:.2f} GB\n"
            f"Total index: {self.total_index_gb:.2f} GB"
        )


# ============================================================================
# PIPELINE STAGE LATENCY MODELS
# ============================================================================
# Non-programmer note:
#   Each stage of the query pipeline has a cost in time. Some stages
#   are CPU-bound (embedding, search), some are GPU-bound (LLM), and
#   some are I/O-bound (reading embeddings from disk). When multiple
#   users query simultaneously, these resources get shared.
#
#   The key insight: the LLM inference stage is BY FAR the slowest
#   part. Everything else (search, embedding, context building) is
#   less than 1 second combined. The LLM can take 5-60 seconds
#   depending on mode and model size.
# ============================================================================

def query_embedding_latency(hw: HardwareProfile, concurrent: int) -> float:
    """
    Time to embed the user's query text (single sentence).

    all-MiniLM-L6-v2 on CPU: ~10-20ms for single sentence.
    With concurrent users, CPU contention adds overhead.
    """
    base_ms = 15  # Single query embedding on modern CPU
    # CPU contention: threads share cores
    contention = 1.0 + max(0, (concurrent - hw.cpu_threads)) * 0.1
    return (base_ms * contention) / 1000  # seconds


def vector_search_latency(
    hw: HardwareProfile, idx: IndexProfile, concurrent: int
) -> float:
    """
    Time for cosine similarity search across all embeddings.

    Memmap block scan: reads embeddings in blocks from disk.
    On HDD, this is I/O bound. On NVMe SSD, it is CPU bound.

    The search involves two components:
      1. I/O: reading the memmap file from disk
      2. CPU: numpy vectorized dot products (BLAS-optimized)

    Baseline measurement: ~100ms for 2000 chunks (PERFORMANCE_BASELINE.md).
    That 100ms includes both I/O and CPU overhead.

    CPU dot products scale sub-linearly with numpy BLAS because:
      - SIMD vectorization processes 8-16 floats per instruction
      - Block scanning amortizes overhead across large matrices
      - numpy uses multi-threaded BLAS for large matrix ops
    At 8.4M chunks (384-dim), numpy computes all dot products in ~1-3s.
    """
    # ---------- I/O component ----------
    # Reading the memmap embedding file from disk
    embeddings_mb = idx.embeddings_size_gb * 1024
    # OS page cache: warm cache from prior queries
    cache_hit = 0.7 if idx.embeddings_size_gb < hw.ram_gb * 0.3 else 0.3
    io_read_mb = embeddings_mb * (1 - cache_hit)
    io_time_ms = (io_read_mb / hw.storage_read_mbps) * 1000

    # ---------- CPU component ----------
    # numpy BLAS-optimized dot product: matrix (N x 384) @ vector (384,)
    # Measured throughput: ~50M dot products/second on modern 16-thread CPU
    # This accounts for SIMD, cache locality, and multi-threading
    dot_products_per_second = 50_000_000
    cpu_time_ms = (idx.total_chunks / dot_products_per_second) * 1000

    # ---------- Concurrency ----------
    if hw.storage_type == "HDD":
        io_contention = 1.0 + (concurrent - 1) * 0.15
        cpu_contention = 1.0 + max(0, (concurrent - 4)) * 0.1
    elif "SSD" in hw.storage_type:
        io_contention = 1.0 + (concurrent - 1) * 0.01  # NVMe negligible
        # CPU contention: concurrent numpy dot products share CPU threads
        cpu_contention = 1.0 + max(0, (concurrent - 4)) * 0.15
    else:
        io_contention = 1.0
        cpu_contention = 1.0

    total_ms = (cpu_time_ms * cpu_contention) + (io_time_ms * io_contention)

    return total_ms / 1000  # seconds


def bm25_search_latency(
    hw: HardwareProfile, idx: IndexProfile, concurrent: int
) -> float:
    """
    FTS5 BM25 keyword search in SQLite.

    SQLite FTS5 is very fast. Baseline: <10ms for 2000 chunks.
    Scales sub-linearly (FTS5 uses inverted index, not linear scan).
    SQLite allows concurrent readers (WAL mode).
    """
    baseline_ms = 10
    # FTS5 scales with log(N) not N (inverted index)
    scale = math.log10(max(idx.total_chunks, 1)) / math.log10(2000)
    # SQLite reader concurrency is excellent in WAL mode
    contention = 1.0 + (concurrent - 1) * 0.02
    return (baseline_ms * scale * contention) / 1000


def rrf_fusion_latency(concurrent: int) -> float:
    """
    Reciprocal Rank Fusion: merge two ranked lists.
    Pure in-memory, negligible (~1ms).
    """
    return 0.001 * concurrent  # trivial


def reranker_latency(
    hw: HardwareProfile, concurrent: int, top_k: int = 12
) -> float:
    """
    Cross-encoder reranker (ms-marco-MiniLM-L-6-v2).

    Reranker scores each of top_k chunks against the query.
    On CPU: ~30ms per pair. On GPU: ~5ms per pair.
    But GPU is shared with LLM, so we assume CPU for reranker.
    """
    ms_per_pair_cpu = 30
    total_ms = ms_per_pair_cpu * top_k
    # CPU contention
    contention = 1.0 + max(0, (concurrent - 4)) * 0.2
    return (total_ms * contention) / 1000


def context_build_latency() -> float:
    """Build prompt from retrieved chunks. Pure string ops, <5ms."""
    return 0.005


def llm_inference_offline(
    hw: HardwareProfile, concurrent: int, model: str
) -> Tuple[float, str]:
    """
    LLM inference via Ollama on local GPU.

    THIS IS THE BOTTLENECK. The GPU can only run one inference at a time.
    Ollama queues requests -- concurrent users must wait in line.

    Token generation speed depends heavily on GPU MEMORY BANDWIDTH:
      - Desktop GDDR6 (RTX 4070): ~256 GB/s -> ~30 tok/s for 8B model
      - Laptop GDDR7 (RTX Pro 3000 Blackwell): ~672 GB/s -> ~70 tok/s for 8B

    LLM inference is memory-bandwidth-bound (not compute-bound).
    Each generated token requires reading all model weights from VRAM.
    Faster VRAM = proportionally faster token generation.

    Average response: ~300-500 tokens (1-2 paragraphs with sources).
    Prompt processing (input tokens): ~1000-2000 tokens.
    """
    # GPU architecture determines throughput scaling
    # Baseline: desktop GDDR6 at ~256 GB/s memory bandwidth
    # Blackwell GDDR7 at ~672 GB/s = ~2.6x bandwidth = ~2.3x actual throughput
    # (not perfectly linear due to compute overhead, but close)
    # GPU architecture determines throughput scaling
    # Baseline: older desktop 12GB GDDR6 at ~256 GB/s memory bandwidth
    # LLM token gen is memory-bandwidth-bound (reading model weights per token)
    is_blackwell = "Blackwell" in hw.gpu_name or "RTX Pro 3000" in hw.gpu_name
    is_dual_3090 = "3090" in hw.gpu_name
    num_gpus = 2 if ("2x" in hw.gpu_name or "dual" in hw.gpu_name.lower()) else 1

    if is_dual_3090:
        # RTX 3090 GDDR6X: 936 GB/s per GPU = ~3.7x baseline
        # Dual GPUs: Ollama uses ONE GPU at a time (serial, no tensor parallel)
        # But having 2 GPUs means we can alternate / load-balance
        bw_multiplier = 3.7
        gpu_tag = "RTX 3090 3.7x"
    elif is_blackwell and "Desktop" in hw.gpu_name:
        bw_multiplier = 2.5
        gpu_tag = "Blackwell Desktop 2.5x"
    elif is_blackwell:
        bw_multiplier = 2.3
        gpu_tag = "Blackwell Laptop 2.3x"
    else:
        bw_multiplier = 1.0
        gpu_tag = ""

    # Model-specific throughput (tokens per second on BASELINE older 12 GB GPU)
    model_throughput = {
        "phi4-mini":         {"prompt_tps": 200, "gen_tps": 30, "note": "Primary for most profiles"},
        "phi4:14b-q4_K_M":   {"prompt_tps": 120, "gen_tps": 17, "note": "14B model"},
        "mistral-small3.1:24b": {"prompt_tps": 80,  "gen_tps": 12, "note": "24B model (needs 16GB VRAM)"},
        "mistral-small3.1:24b-q4":     {"prompt_tps": 40,  "gen_tps": 6,  "note": "24B quantized (needs 16GB+ VRAM, tensor parallel)"},
        "mistral:7b":        {"prompt_tps": 180, "gen_tps": 25, "note": "Baseline model, good throughput"},
        "gemma3:4b":         {"prompt_tps": 300, "gen_tps": 45, "note": "Small, fast"},
    }

    specs = model_throughput.get(model, model_throughput["phi4-mini"])
    # Scale throughput by GPU memory bandwidth
    effective_prompt_tps = specs["prompt_tps"] * bw_multiplier
    effective_gen_tps = specs["gen_tps"] * bw_multiplier
    tag = f" [{gpu_tag}]" if gpu_tag else ""
    specs = {
        "prompt_tps": effective_prompt_tps,
        "gen_tps": effective_gen_tps,
        "note": specs["note"] + tag,
    }

    # Typical query context
    input_tokens = 1500   # ~5 chunks * 300 tokens each
    output_tokens = 400   # ~1-2 paragraphs with citations

    # Single-user inference time
    prompt_time = input_tokens / specs["prompt_tps"]
    gen_time = output_tokens / specs["gen_tps"]
    single_user_time = prompt_time + gen_time

    # CRITICAL: GPU is a SERIAL BOTTLENECK (per GPU)
    # Ollama processes one request at a time per GPU.
    # With dual GPUs, Ollama can run 2 models or load-balance.
    # Effective parallelism = number of GPUs.
    # With N concurrent users and G GPUs:
    #   avg_queue_wait = ((N/G - 1) / 2) * single_user_time
    effective_concurrent = max(1, concurrent / num_gpus)
    avg_queue_wait = ((effective_concurrent - 1) / 2) * single_user_time
    total_time = avg_queue_wait + single_user_time

    return total_time, specs["note"]


def llm_inference_online(
    hw: HardwareProfile, concurrent: int, model: str
) -> Tuple[float, str]:
    """
    LLM inference via cloud API (OpenRouter/Azure/OpenAI).

    Online mode is MUCH faster because the cloud has massive GPU clusters.
    Multiple users can query in parallel without local GPU contention.

    Latency components:
      - Network round trip: ~50-100ms (corporate LAN to internet)
      - API queue time: ~0-500ms (depends on provider load)
      - Prompt processing: ~0.5-2s (cloud GPU, very fast)
      - Token generation: ~1-4s for 400 tokens (cloud GPU)
      - Total: typically 2-6 seconds per query

    Rate limits:
      - OpenRouter: 200 req/min (free), 500 req/min (paid)
      - Azure OpenAI: 120 req/min typical enterprise
      - All providers: concurrent requests are parallel
    """
    model_latency = {
        "gpt-4o":               {"first_token_ms": 800,  "gen_tps": 80,  "note": "Fast flagship"},
        "gpt-4o-mini":          {"first_token_ms": 400,  "gen_tps": 120, "note": "Very fast, cost-efficient"},
        "gpt-4.1":              {"first_token_ms": 600,  "gen_tps": 90,  "note": "Latest GPT-4 series"},
        "gpt-4.1-mini":         {"first_token_ms": 350,  "gen_tps": 130, "note": "Fastest GPT-4.1"},
        "AI assistant-sonnet-4":      {"first_token_ms": 700,  "gen_tps": 70,  "note": "AI provider flagship"},
        "AI assistant-haiku-4":       {"first_token_ms": 300,  "gen_tps": 150, "note": "Fastest AI assistant"},
        "gpt-3.5-turbo":        {"first_token_ms": 300,  "gen_tps": 100, "note": "Legacy, very fast"},
    }

    specs = model_latency.get(model, model_latency["gpt-4o"])

    output_tokens = 400
    network_rtt_ms = 80  # Corporate LAN

    # Time to first token (includes prompt processing)
    ttft = (specs["first_token_ms"] + network_rtt_ms) / 1000

    # Token generation time
    gen_time = output_tokens / specs["gen_tps"]

    # Online: requests are parallel on cloud side
    # Slight degradation from rate limiting at high concurrency
    rate_limit_factor = 1.0
    if concurrent > 8:
        rate_limit_factor = 1.1  # Mild throttling
    elif concurrent > 5:
        rate_limit_factor = 1.05

    total_time = (ttft + gen_time) * rate_limit_factor
    return total_time, specs["note"]


def llm_inference_vllm_server(
    hw: HardwareProfile, concurrent: int, model: str
) -> Tuple[float, str]:
    """
    LLM inference via vLLM on a DEDICATED GPU server (Docker or bare metal).

    vLLM vs Ollama -- why it matters for multi-user:
      Ollama: ONE request at a time on GPU. Users wait in a queue.
      vLLM:   CONTINUOUS BATCHING. Multiple requests processed on GPU
              simultaneously. The GPU stays busy 100% of the time and
              throughput scales much better with concurrent users.

    Setup options:
      1. Docker with NVIDIA Container Toolkit (easiest)
         docker run --gpus all -p 8000:8000 vllm/vllm-openai \\
             --model microsoft/Phi-4-mini-instruct
      2. Bare metal: pip install vllm && python -m vllm.entrypoints.openai.api_server
      3. Dedicated server: separate machine on LAN with bigger GPU

    Throughput model (12 GB VRAM, phi4-mini equivalent):
      - vLLM batch efficiency: ~2.5x throughput vs serial Ollama
      - With continuous batching, concurrent requests share GPU compute
      - Throughput degrades gently, not linearly like Ollama queue
    """
    # GPU architecture scaling (same as offline)
    is_blackwell = "Blackwell" in hw.gpu_name or "RTX Pro 3000" in hw.gpu_name
    is_dual_3090 = "3090" in hw.gpu_name
    num_gpus = 2 if ("2x" in hw.gpu_name or "dual" in hw.gpu_name.lower()) else 1

    if is_dual_3090:
        bw_multiplier = 3.7
        gpu_tag = "2x RTX 3090"
    elif is_blackwell and "Desktop" in hw.gpu_name:
        bw_multiplier = 2.5
        gpu_tag = "Blackwell Desktop"
    elif is_blackwell:
        bw_multiplier = 2.3
        gpu_tag = "Blackwell Laptop"
    else:
        bw_multiplier = 1.0
        gpu_tag = ""

    # Model throughput under vLLM continuous batching (BASELINE desktop 12 GB GPU)
    model_throughput = {
        "phi4-mini": {
            "single_tps": 30,
            "batch_factor": 2.5,
            "max_batch": 4,        # Per GPU
            "note": "vLLM continuous batching",
        },
        "phi4:14b-q4_K_M": {
            "single_tps": 17,
            "batch_factor": 2.0,
            "max_batch": 2,
            "note": "vLLM batching, 14B model",
        },
        "mistral-small3.1:24b": {
            "single_tps": 12,
            "batch_factor": 2.0,
            "max_batch": 3,        # 16GB model in 24GB = room for batching
            "note": "vLLM 24B model, fits in one 24GB GPU",
        },
        "mistral-small3.1:24b-q4": {
            "single_tps": 6,
            "batch_factor": 1.8,
            "max_batch": 2,        # Tensor parallel across 2 GPUs, tight
            "note": "vLLM 24B-q4 tensor parallel across 2 GPUs",
        },
    }

    specs = model_throughput.get(model, model_throughput["phi4-mini"])
    specs = dict(specs)  # copy
    specs["single_tps"] = specs["single_tps"] * bw_multiplier
    # Dual GPUs with vLLM: can run tensor parallel (1 big model across 2)
    # or serve 2 models independently (doubles max_batch)
    if num_gpus > 1 and model != "mistral-small3.1:24b-q4":
        # Non-tensor-parallel models: serve from both GPUs independently
        specs["max_batch"] = specs["max_batch"] * num_gpus
        specs["note"] += f" [2 GPUs independent, {specs['max_batch']} batch]"
    elif num_gpus > 1:
        # 24B-q4 model: tensor parallel, single model across both GPUs
        specs["note"] += f" [tensor parallel 2 GPUs]"
    if gpu_tag:
        specs["note"] += f" [{gpu_tag} {bw_multiplier:.1f}x]"

    input_tokens = 1500
    output_tokens = 400

    # Single-user inference (same as Ollama)
    prompt_time = input_tokens / 200  # prompt processing ~200 tok/s
    gen_time = output_tokens / specs["single_tps"]
    single_user_time = prompt_time + gen_time

    # vLLM continuous batching model:
    # Up to max_batch users processed simultaneously with batch_factor speedup
    # Beyond max_batch, requests queue but with batch_factor throughput
    effective_batch = min(concurrent, specs["max_batch"])
    queued_users = max(0, concurrent - specs["max_batch"])

    # Time for a batched request (slightly slower per-request due to sharing)
    batch_overhead = 1.0 + (effective_batch - 1) * 0.15  # 15% overhead per extra
    batched_time = single_user_time * batch_overhead / specs["batch_factor"]

    # Queue wait for overflow beyond max_batch
    batches_ahead = queued_users / max(specs["max_batch"], 1) / 2  # avg wait
    queue_wait = batches_ahead * batched_time

    total_time = queue_wait + batched_time
    return total_time, specs["note"]


# ============================================================================
# FULL PIPELINE SIMULATION
# ============================================================================

def simulate_query(
    hw: HardwareProfile,
    idx: IndexProfile,
    concurrent: int,
    mode: str,  # "offline" or "online"
    model: str,
    reranker: bool = True,
) -> Dict:
    """
    Simulate a complete query pipeline and return timing breakdown.
    """
    stages = {}

    # Stage 1: Query embedding
    stages["1_query_embed"] = query_embedding_latency(hw, concurrent)

    # Stage 2: Vector search
    stages["2_vector_search"] = vector_search_latency(hw, idx, concurrent)

    # Stage 3: BM25 search
    stages["3_bm25_search"] = bm25_search_latency(hw, idx, concurrent)

    # Stage 4: RRF fusion
    stages["4_rrf_fusion"] = rrf_fusion_latency(concurrent)

    # Stage 5: Reranker (optional)
    if reranker:
        stages["5_reranker"] = reranker_latency(hw, concurrent)
    else:
        stages["5_reranker"] = 0.0

    # Stage 6: Context building
    stages["6_context_build"] = context_build_latency()

    # Stage 7: LLM inference (THE BOTTLENECK)
    if mode == "offline":
        llm_time, llm_note = llm_inference_offline(hw, concurrent, model)
    elif mode == "vllm":
        llm_time, llm_note = llm_inference_vllm_server(hw, concurrent, model)
    else:
        llm_time, llm_note = llm_inference_online(hw, concurrent, model)
    stages["7_llm_inference"] = llm_time

    total = sum(stages.values())
    retrieval_total = sum(v for k, v in stages.items() if k != "7_llm_inference")

    return {
        "stages": stages,
        "total_seconds": total,
        "retrieval_seconds": retrieval_total,
        "llm_seconds": llm_time,
        "llm_note": llm_note,
        "concurrent": concurrent,
        "mode": mode,
        "model": model,
    }


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def _print_hw_profile(hw: HardwareProfile):
    """Print a hardware profile summary."""
    print(f"  Name:    {hw.name}")
    print(f"  CPU:     {hw.cpu_threads} threads")
    print(f"  RAM:     {hw.ram_gb:.0f} GB")
    print(f"  GPU:     {hw.gpu_name} ({hw.gpu_vram_gb:.0f} GB VRAM)")
    print(f"  Storage: {hw.storage_model} ({hw.storage_read_mbps:.0f} MB/s)")


def main():
    hw = PERSONAL_WORKSTATION
    hw_laptop = WORK_LAPTOP
    idx_700 = IndexProfile(700.0)
    idx_2000 = IndexProfile(2000.0)

    print("=" * 78)
    print("HybridRAG Multi-User Workstation Stress Test Simulation")
    print("=" * 78)
    print(f"Date: {datetime.now().isoformat()}")
    print()

    # ---- Hardware Summary ----
    print("HARDWARE PROFILES")
    print("-" * 40)
    _print_hw_profile(hw)
    print()
    _print_hw_profile(hw_laptop)
    print()

    # ---- Index Summary (700 GB) ----
    print("INDEX PROFILE (700 GB source data)")
    print("-" * 40)
    print(f"  {idx_700.summary()}")
    print()

    # ---- Index Summary (2 TB) ----
    print("INDEX PROFILE (2 TB source data)")
    print("-" * 40)
    print(f"  {idx_2000.summary()}")
    print()

    user_counts = [10, 8, 6, 4, 3, 2]

    # ==================================================================
    # OFFLINE MODE SIMULATION (700 GB) -- Ollama (serial GPU queue)
    # ==================================================================
    print()
    print("=" * 78)
    print("SCENARIO 1: PERSONAL WORKSTATION -- OFFLINE (Ollama) -- 700 GB")
    print("=" * 78)
    print()
    print("NOTE: Personal workstation is for HOME development/testing ONLY.")
    print("Cannot bring to work. Cannot comingle work data on it.")
    print("Used for scalability testing and offline AI experimentation.")
    print()
    print("2x RTX 3090 FE = 48 GB VRAM total. Ollama can load-balance across GPUs.")
    print("936 GB/s memory bandwidth per GPU = 3.7x faster than baseline GDDR6.")
    print()
    print("Using phi4-mini as primary (covers most use cases)")
    print()

    offline_results_700 = []
    for n in user_counts:
        r = simulate_query(hw, idx_700, n, "offline", "phi4-mini", reranker=True)
        offline_results_700.append(r)

    _print_results_table("OFFLINE (phi4-mini) -- 700 GB", offline_results_700)

    # Show phi4 (logistics profile) -- slower due to larger model
    print()
    print("  phi4:14b-q4_K_M (logistics profile) -- slower, tighter VRAM fit:")
    phi4_results = []
    for n in user_counts:
        r = simulate_query(hw, idx_700, n, "offline", "phi4:14b-q4_K_M", reranker=True)
        phi4_results.append(r)
    _print_results_table("OFFLINE (phi4:14b) -- 700 GB", phi4_results)

    # Show larger models that dual 3090 enables
    if hw.gpu_vram_gb >= 24:
        print()
        print("  mistral-small3.1:24b (fits in one 24 GB GPU, massive quality upgrade):")
        q32_results = []
        for n in user_counts:
            r = simulate_query(hw, idx_700, n, "offline", "mistral-small3.1:24b", reranker=True)
            q32_results.append(r)
        _print_results_table("OFFLINE (mistral-small3.1:24b) -- 700 GB", q32_results)

    if hw.gpu_vram_gb >= 40:
        print()
        print("  mistral-small3.1:24b-q4 (tensor parallel across 2x 3090, 40 GB needed):")
        l70_results = []
        for n in user_counts:
            r = simulate_query(hw, idx_700, n, "offline", "mistral-small3.1:24b-q4", reranker=True)
            l70_results.append(r)
        _print_results_table("OFFLINE (mistral-small3.1:24b-q4) -- 700 GB", l70_results)

    # ==================================================================
    # vLLM SERVER MODE (700 GB) -- continuous batching
    # ==================================================================
    print()
    print("=" * 78)
    print("SCENARIO 2: PERSONAL WORKSTATION -- vLLM SERVER -- 700 GB")
    print("=" * 78)
    print()
    print("vLLM replaces Ollama. Same GPU, same model, but continuous batching")
    print("means multiple users are processed on GPU simultaneously instead of")
    print("waiting in a serial queue. Can run as Docker container or systemd service.")
    print()
    print("Setup: docker run --gpus all -p 8000:8000 vllm/vllm-openai \\")
    print("           --model microsoft/Phi-4-mini-instruct")
    print()

    vllm_results_700 = []
    for n in user_counts:
        r = simulate_query(hw, idx_700, n, "vllm", "phi4-mini", reranker=True)
        vllm_results_700.append(r)

    _print_results_table("vLLM SERVER (phi4-mini) -- 700 GB", vllm_results_700)

    print()
    print("  phi4:14b-q4_K_M via vLLM (logistics profile):")
    vllm_phi4_results = []
    for n in user_counts:
        r = simulate_query(hw, idx_700, n, "vllm", "phi4:14b-q4_K_M", reranker=True)
        vllm_phi4_results.append(r)
    _print_results_table("vLLM SERVER (phi4:14b) -- 700 GB", vllm_phi4_results)

    # Show larger models via vLLM on dual 3090
    if hw.gpu_vram_gb >= 24:
        print()
        print("  mistral-small3.1:24b via vLLM (fits one GPU, other GPU serves parallel):")
        vllm_q32 = []
        for n in user_counts:
            r = simulate_query(hw, idx_700, n, "vllm", "mistral-small3.1:24b", reranker=True)
            vllm_q32.append(r)
        _print_results_table("vLLM SERVER (mistral-small3.1:24b) -- 700 GB", vllm_q32)

    if hw.gpu_vram_gb >= 40:
        print()
        print("  mistral-small3.1:24b-q4 via vLLM (tensor parallel across 2x 3090):")
        vllm_l70 = []
        for n in user_counts:
            r = simulate_query(hw, idx_700, n, "vllm", "mistral-small3.1:24b-q4", reranker=True)
            vllm_l70.append(r)
        _print_results_table("vLLM SERVER (mistral-small3.1:24b-q4) -- 700 GB", vllm_l70)

    # ==================================================================
    # ONLINE MODE SIMULATION (700 GB)
    # ==================================================================
    print()
    print("=" * 78)
    print("SCENARIO 3: PERSONAL WORKSTATION -- ONLINE (Cloud API) -- 700 GB")
    print("=" * 78)
    print()
    print("Using gpt-4o as primary, gpt-4o-mini for PM profile")
    print()

    online_results_700 = []
    for n in user_counts:
        r = simulate_query(hw, idx_700, n, "online", "gpt-4o", reranker=True)
        online_results_700.append(r)

    _print_results_table("ONLINE (gpt-4o) -- 700 GB", online_results_700)

    # gpt-4o-mini (faster, cheaper)
    print()
    print("  gpt-4o-mini (PM/general profile) -- faster, cheaper:")
    mini_results = []
    for n in user_counts:
        r = simulate_query(hw, idx_700, n, "online", "gpt-4o-mini", reranker=True)
        mini_results.append(r)
    _print_results_table("ONLINE (gpt-4o-mini) -- 700 GB", mini_results)

    # ==================================================================
    # 2 TB SOURCE DATA
    # ==================================================================
    print()
    print("=" * 78)
    print("SCENARIO 4: PERSONAL WORKSTATION -- 2 TB SOURCE DATA")
    print("=" * 78)
    print()
    print(f"  {idx_2000.summary()}")
    print()

    offline_results_2000 = []
    for n in user_counts:
        r = simulate_query(hw, idx_2000, n, "offline", "phi4-mini", reranker=True)
        offline_results_2000.append(r)

    _print_results_table("OFFLINE/Ollama (phi4-mini) -- 2 TB", offline_results_2000)

    vllm_results_2000 = []
    for n in user_counts:
        r = simulate_query(hw, idx_2000, n, "vllm", "phi4-mini", reranker=True)
        vllm_results_2000.append(r)

    _print_results_table("vLLM SERVER (phi4-mini) -- 2 TB", vllm_results_2000)

    online_results_2000 = []
    for n in user_counts:
        r = simulate_query(hw, idx_2000, n, "online", "gpt-4o", reranker=True)
        online_results_2000.append(r)

    _print_results_table("ONLINE (gpt-4o) -- 2 TB", online_results_2000)

    # ==================================================================
    # WORK LAPTOP (DEMO) -- RTX Pro 3000 Blackwell + NPU
    # ==================================================================
    print()
    print("=" * 78)
    print("SCENARIO 5: WORK LAPTOP (RTX Pro 3000 Blackwell, 64 GB DDR5)")
    print("=" * 78)
    print()
    print("  CPU:     Intel Core Ultra 7 265HX (8P+12E, 28 threads, 5.3 GHz)")
    print("  RAM:     64 GB SODIMM DDR5 6400 MT/s (~51.2 GB/s bandwidth)")
    print("  GPU:     NVIDIA RTX Pro 3000 Blackwell, 12 GB GDDR7")
    print("           5,888 CUDA cores, 184 Tensor Cores (5th gen)")
    print("           992 AI TOPS, 672 GB/s memory bandwidth")
    print("  NPU:     Intel AI Boost, 11 TOPS, 36 GB shared memory")
    print("  Cache:   L1 2MB + L2 36MB + L3 30MB = 68 MB total")
    print("  Storage: WD PC SN8000S SED (5,545 MB/s)")
    print()
    print("  GDDR7 at 672 GB/s = ~2.6x the bandwidth of desktop GDDR6.")
    print("  LLM inference is memory-bandwidth-bound, so token generation")
    print("  scales roughly proportionally: ~2.3x faster per token.")
    print()

    idx_demo = IndexProfile(700.0)

    # Offline Ollama on Blackwell GPU
    print("  --- OFFLINE (Ollama on Blackwell GPU) ---")
    laptop_counts = [10, 8, 6, 4, 3, 2, 1]
    laptop_offline_results = []
    for n in laptop_counts:
        r = simulate_query(hw_laptop, idx_demo, n, "offline", "phi4-mini", reranker=True)
        laptop_offline_results.append(r)
    _print_results_table("LAPTOP OLLAMA (phi4-mini Blackwell) -- 700 GB", laptop_offline_results)

    print()
    print("  phi4:14b on Blackwell (logistics profile):")
    laptop_phi4_results = []
    for n in laptop_counts:
        r = simulate_query(hw_laptop, idx_demo, n, "offline", "phi4:14b-q4_K_M", reranker=True)
        laptop_phi4_results.append(r)
    _print_results_table("LAPTOP OLLAMA (phi4:14b Blackwell) -- 700 GB", laptop_phi4_results)

    # vLLM on Blackwell GPU
    print()
    print("  --- vLLM SERVER on Blackwell GPU ---")
    laptop_vllm_results = []
    for n in laptop_counts:
        r = simulate_query(hw_laptop, idx_demo, n, "vllm", "phi4-mini", reranker=True)
        laptop_vllm_results.append(r)
    _print_results_table("LAPTOP vLLM (phi4-mini Blackwell) -- 700 GB", laptop_vllm_results)

    # Online API
    print()
    print("  --- ONLINE (Cloud API) ---")
    laptop_online_results = []
    for n in laptop_counts:
        r = simulate_query(hw_laptop, idx_demo, n, "online", "gpt-4o", reranker=True)
        laptop_online_results.append(r)
    _print_results_table("LAPTOP ONLINE (gpt-4o) -- 700 GB", laptop_online_results)

    # Side-by-side laptop comparison
    print()
    print("=" * 78)
    print("LAPTOP 10-USER COMPARISON (700 GB)")
    print("=" * 78)
    l_off = laptop_offline_results[0]  # 10 users
    l_vllm = laptop_vllm_results[0]
    l_on = laptop_online_results[0]
    print(f"  {'Mode':<30s}  {'Retrieval':>10s}  {'LLM':>10s}  {'TOTAL':>10s}  {'Rating':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, r in [
        ("Ollama (serial, Blackwell)", l_off),
        ("vLLM (batched, Blackwell)", l_vllm),
        ("Cloud API (gpt-4o)", l_on),
    ]:
        total = r["total_seconds"]
        rating = ("Excellent" if total < 10 else "Good" if total < 20
                  else "Acceptable" if total < 45 else "Slow" if total < 90
                  else "Poor" if total < 180 else "Unusable")
        print(f"  {label:<30s}  {r['retrieval_seconds']:>9.1f}s  "
              f"{r['llm_seconds']:>9.1f}s  {total:>9.1f}s  {rating:>10s}")

    # NPU analysis
    print()
    print("=" * 78)
    print("NPU POTENTIAL (Intel AI Boost, 11 TOPS, 36 GB shared)")
    print("=" * 78)
    print()
    print("  Current NPU capabilities (via OpenVINO):")
    print("    - Embedding model (all-MiniLM-L6-v2): CAN offload to NPU")
    print("      Frees CPU for BM25/reranker while NPU handles embeddings")
    print("      Expected: ~5ms query embedding (faster than CPU)")
    print("    - Reranker (cross-encoder): Possible via OpenVINO ONNX export")
    print("      Would free CPU entirely for search operations")
    print("    - LLM inference: NOT practical on NPU yet")
    print("      11 TOPS is too slow for 8B+ parameter models")
    print("      CPU still faster than NPU for 4-bit quantized LLMs")
    print()
    print("  Future NPU potential (2026-2027 drivers + frameworks):")
    print("    - Intel NITRO framework: experimental LLM inference on NPU")
    print("    - 36 GB shared memory could hold smaller models (1-3B)")
    print("    - Hybrid GPU+NPU: GPU runs main LLM, NPU runs draft model")
    print("      for speculative decoding (2x generation speed)")
    print("    - NPU handles all pre/post-processing while GPU does inference")
    print()

    print("  RAM check (64 GB DDR5 6400 MT/s):")
    headroom = hw_laptop.ram_gb - idx_demo.embeddings_size_gb - 8
    print(f"    Embeddings: {idx_demo.embeddings_size_gb:.1f} GB")
    print(f"    OS + models: ~8 GB")
    print(f"    Headroom: {headroom:.0f} GB")
    print(f"    [OK] Plenty of room. DDR5 6400 = ~51 GB/s bandwidth.")

    # ==================================================================
    # BOTTLENECK ANALYSIS
    # ==================================================================
    print()
    print("=" * 78)
    print("BOTTLENECK ANALYSIS")
    print("=" * 78)
    print()

    # Show stage breakdown for 10-user offline (Ollama)
    r10 = offline_results_700[0]
    print("Pipeline breakdown (10 users, Ollama, phi4-mini, 700 GB, NVMe SSD):")
    print("-" * 60)
    for stage, secs in r10["stages"].items():
        pct = (secs / r10["total_seconds"]) * 100
        bar = "#" * int(pct / 2)
        print(f"  {stage:20s}  {secs:7.2f}s  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':20s}  {r10['total_seconds']:7.2f}s")
    print()
    print(f"  Retrieval (stages 1-6): {r10['retrieval_seconds']:.2f}s "
          f"({r10['retrieval_seconds']/r10['total_seconds']*100:.1f}%)")
    print(f"  LLM inference (stage 7): {r10['llm_seconds']:.2f}s "
          f"({r10['llm_seconds']/r10['total_seconds']*100:.1f}%)")
    print()
    print("  >> With NVMe SSD, retrieval is now FAST.")
    print("  >> LLM inference still dominates due to Ollama's serial GPU queue.")
    print()

    # Show vLLM comparison
    rv10 = vllm_results_700[0]
    print("Pipeline breakdown (10 users, vLLM, phi4-mini, 700 GB, NVMe SSD):")
    print("-" * 60)
    for stage, secs in rv10["stages"].items():
        pct = (secs / rv10["total_seconds"]) * 100
        bar = "#" * int(pct / 2)
        print(f"  {stage:20s}  {secs:7.2f}s  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':20s}  {rv10['total_seconds']:7.2f}s")
    print()
    print(f"  >> vLLM continuous batching: {r10['llm_seconds']:.1f}s -> "
          f"{rv10['llm_seconds']:.1f}s ({r10['llm_seconds']/max(rv10['llm_seconds'],0.1):.1f}x faster)")
    print()

    # Side-by-side summary
    ro10 = online_results_700[0]
    print("=" * 78)
    print("10-USER COMPARISON (700 GB, NVMe SSD)")
    print("=" * 78)
    print(f"  {'Mode':<25s}  {'Retrieval':>10s}  {'LLM':>10s}  {'TOTAL':>10s}  {'Rating':>10s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for label, r in [
        ("Ollama (serial queue)", r10),
        ("vLLM (continuous batch)", rv10),
        ("Cloud API (gpt-4o)", ro10),
    ]:
        total = r["total_seconds"]
        rating = ("Excellent" if total < 10 else "Good" if total < 20
                  else "Acceptable" if total < 45 else "Slow" if total < 90
                  else "Poor" if total < 180 else "Unusable")
        print(f"  {label:<25s}  {r['retrieval_seconds']:>9.1f}s  "
              f"{r['llm_seconds']:>9.1f}s  {total:>9.1f}s  {rating:>10s}")

    # ==================================================================
    # 700 GB vs 2 TB COMPARISON
    # ==================================================================
    print()
    print("=" * 78)
    print("700 GB vs 2 TB COMPARISON (10 users, offline)")
    print("=" * 78)
    print()
    r_700 = offline_results_700[0]
    r_2000 = offline_results_2000[0]
    print(f"  {'Metric':<30s}  {'700 GB':>10s}  {'2 TB':>10s}  {'Change':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")
    print(f"  {'Chunks':.<30s}  {idx_700.total_chunks:>10,}  {idx_2000.total_chunks:>10,}  "
          f"{idx_2000.total_chunks/idx_700.total_chunks:.1f}x")
    print(f"  {'Embeddings file':.<30s}  {idx_700.embeddings_size_gb:>9.1f}G  {idx_2000.embeddings_size_gb:>9.1f}G  "
          f"{idx_2000.embeddings_size_gb/idx_700.embeddings_size_gb:.1f}x")
    vs_700 = r_700['stages']['2_vector_search']
    vs_2000 = r_2000['stages']['2_vector_search']
    print(f"  {'Vector search (s)':.<30s}  {vs_700:>10.2f}  "
          f"{vs_2000:>10.2f}  "
          f"{vs_2000/max(vs_700,0.001):.1f}x")
    print(f"  {'BM25 search (s)':.<30s}  {r_700['stages']['3_bm25_search']:>10.3f}  "
          f"{r_2000['stages']['3_bm25_search']:>10.3f}  "
          f"{r_2000['stages']['3_bm25_search']/max(r_700['stages']['3_bm25_search'],0.001):.1f}x")
    print(f"  {'LLM inference (s)':.<30s}  {r_700['llm_seconds']:>10.2f}  "
          f"{r_2000['llm_seconds']:>10.2f}  {'same':>10s}")
    print(f"  {'Total response (s)':.<30s}  {r_700['total_seconds']:>10.2f}  "
          f"{r_2000['total_seconds']:>10.2f}  "
          f"{r_2000['total_seconds']/r_700['total_seconds']:.1f}x")
    print()
    print("  KEY INSIGHT: With NVMe SSD, retrieval stays fast even at 2 TB.")
    print("  The embeddings file grows to ~18 GB but NVMe handles it well.")
    print("  LLM inference time stays the same (it doesn't depend on index size).")
    print("  vLLM is the biggest remaining improvement for multi-user offline.")

    # ==================================================================
    # WHAT BREAKS AT 2 TB
    # ==================================================================
    print()
    print("=" * 78)
    print("WHAT BREAKS AT 2 TB")
    print("=" * 78)
    print()
    ram_for_embeddings = idx_2000.embeddings_size_gb
    ram_for_models = 6.0  # Embedding model + overhead
    ram_for_sqlite = idx_2000.sqlite_size_gb * 0.3  # WAL cache
    ram_for_os = 4.0
    ram_total_needed = ram_for_embeddings + ram_for_models + ram_for_sqlite + ram_for_os
    print(f"  RAM budget at 2 TB:")
    print(f"    Embeddings (memmap cache):  {ram_for_embeddings:.1f} GB")
    print(f"    ML models in memory:        {ram_for_models:.1f} GB")
    print(f"    SQLite + FTS5 cache:        {ram_for_sqlite:.1f} GB")
    print(f"    OS + applications:          {ram_for_os:.1f} GB")
    print(f"    TOTAL NEEDED:               {ram_total_needed:.1f} GB")
    print(f"    AVAILABLE:                  {hw.ram_gb:.0f} GB")
    if ram_total_needed < hw.ram_gb:
        print(f"    STATUS: [OK] Fits in RAM ({hw.ram_gb - ram_total_needed:.1f} GB headroom)")
    else:
        print(f"    STATUS: [WARN] Tight ({ram_total_needed - hw.ram_gb:.1f} GB over)")
    print()
    print("  ISSUES AT 2 TB:")
    print(f"    1. Vector search: {r_2000['stages']['2_vector_search']:.1f}s "
          "on NVMe SSD (fast, no bottleneck)")
    print(f"    2. Indexing time: ~{idx_2000.total_chunks / 100 / 3600:.0f} hours "
          "to build index from scratch")
    print(f"    3. Reranker CPU load increases with chunk count")
    print(f"    4. FTS5 index rebuild takes longer after indexing")
    print(f"    5. LLM inference (Ollama): still serial GPU bottleneck")
    print(f"       -> vLLM continuous batching solves this")

    # ==================================================================
    # IMPROVEMENT RECOMMENDATIONS
    # ==================================================================
    print()
    print("=" * 78)
    print("IMPROVEMENT RECOMMENDATIONS (priority order)")
    print("=" * 78)
    print()

    improvements = [
        {
            "rank": 1,
            "what": "Replace Ollama with vLLM (Docker or bare metal)",
            "cost": "Free (Apache 2.0 license)",
            "impact": "BIGGEST win for multi-user offline. vLLM continuous batching "
                      "processes multiple GPU requests simultaneously instead of "
                      "queuing serially. Docker setup: one command. "
                      "Requires Linux or WSL2 for GPU passthrough.",
            "offline_speedup": "3-5x throughput at 10 concurrent users",
            "online_speedup": "N/A (already using cloud batching)",
            "how": "docker run --gpus all -p 8000:8000 vllm/vllm-openai "
                   "--model microsoft/Phi-4-mini-instruct",
        },
        {
            "rank": 2,
            "what": "Switch to FAISS or Hnswlib for vector search",
            "cost": "Free (code change, adds dependency)",
            "impact": "Replace brute-force memmap scan with approximate nearest "
                      "neighbor (ANN) index. Searches 8M chunks in <50ms instead "
                      "of 1-2s. Critical for scaling to 2 TB. "
                      "faiss-cpu is BSD licensed, hnswlib is Apache.",
            "offline_speedup": "Vector search drops from ~1.3s to <50ms",
            "online_speedup": "Same benefit for retrieval stage",
            "how": "pip install faiss-cpu; swap memmap scan for FAISS IVF index",
        },
        {
            "rank": 3,
            "what": "Enable embedding cache (query-level caching)",
            "cost": "Free (code change)",
            "impact": "Cache recent query embeddings + search results. If users ask "
                      "similar questions, skip retrieval entirely. 80% cache hit rate "
                      "for teams asking related questions about same documents.",
            "offline_speedup": "Retrieval drops to ~0ms for cached queries",
            "online_speedup": "Same benefit for retrieval stage",
            "how": "LRU dict keyed by query hash, invalidate on re-index",
        },
        {
            "rank": 4,
            "what": "Upgrade GPU to 24 GB VRAM (RTX 4090 / A5000)",
            "cost": "$1,200-2,000",
            "impact": "Enables mistral-small3.1:24b (much better quality), 2x faster token "
                      "generation, and model stays in VRAM without swapping. "
                      "More VRAM also means vLLM can batch more concurrent requests.",
            "offline_speedup": "2-3x faster inference, better answer quality",
            "online_speedup": "No change (cloud GPU already fast)",
            "how": "Hardware upgrade, swap GPU card",
        },
        {
            "rank": 5,
            "what": "Add request queuing with priority (software change)",
            "cost": "Free (code change)",
            "impact": "FastAPI backend with asyncio queue. Prevents GPU starvation. "
                      "Priority queue lets urgent queries skip ahead. "
                      "Shows estimated wait time in UI.",
            "offline_speedup": "Better UX, not faster raw throughput",
            "online_speedup": "Prevents rate limit errors under burst load",
            "how": "asyncio.PriorityQueue in FastAPI middleware",
        },
        {
            "rank": 6,
            "what": "Precompute common queries (scheduled batch)",
            "cost": "Free (code change)",
            "impact": "Run top-50 anticipated queries overnight, cache results. "
                      "Morning users get instant answers for common questions.",
            "offline_speedup": "Instant for precomputed queries",
            "online_speedup": "Same benefit, also saves API cost",
            "how": "Scheduled task that runs queries from a seed list",
        },
        {
            "rank": 7,
            "what": "Add second GPU (multi-GPU inference)",
            "cost": "$800-2,000",
            "impact": "Two 12 GB GPUs can serve two models simultaneously, "
                      "halving queue wait. Or one 24 GB model via tensor parallel. "
                      "vLLM supports tensor parallel natively.",
            "offline_speedup": "2x concurrent throughput",
            "online_speedup": "No change",
            "how": "Hardware upgrade; vLLM --tensor-parallel-size 2",
        },
        {
            "rank": 8,
            "what": "Dedicated inference server (separate machine)",
            "cost": "$2,000-5,000 (used workstation with GPU)",
            "impact": "Offload all LLM inference to a separate machine on LAN. "
                      "Main workstation handles only retrieval. Both machines "
                      "work at full speed without competing for resources.",
            "offline_speedup": "Near-online-mode speed for local inference",
            "online_speedup": "N/A",
            "how": "Second machine running vLLM, accessible via LAN HTTP API",
        },
    ]

    for imp in improvements:
        print(f"  #{imp['rank']}. {imp['what']}")
        print(f"     Cost: {imp['cost']}")
        print(f"     Impact: {imp['impact']}")
        print(f"     Offline gain: {imp['offline_speedup']}")
        print(f"     Online gain: {imp['online_speedup']}")
        if imp.get("how"):
            print(f"     How: {imp['how']}")
        print()

    # ==================================================================
    # SAVE REPORT
    # ==================================================================
    report_path = PROJECT_ROOT / "docs" / "WORKSTATION_STRESS_TEST.md"
    _write_report(report_path, hw, idx_700, idx_2000,
                  offline_results_700, phi4_results,
                  vllm_results_700, vllm_phi4_results,
                  online_results_700, mini_results,
                  offline_results_2000, vllm_results_2000,
                  online_results_2000, improvements)
    print(f"Report saved: {report_path}")
    print()
    print("=" * 78)
    print("SIMULATION COMPLETE")
    print("=" * 78)

    return 0


def _print_results_table(title: str, results: List[Dict]):
    """Print a formatted results table."""
    print(f"  {'Users':>5s}  {'Retrieval':>10s}  {'LLM':>10s}  {'TOTAL':>10s}  {'Rating':>10s}")
    print(f"  {'-----':>5s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}")
    for r in results:
        total = r["total_seconds"]
        if total < 10:
            rating = "Excellent"
        elif total < 20:
            rating = "Good"
        elif total < 45:
            rating = "Acceptable"
        elif total < 90:
            rating = "Slow"
        elif total < 180:
            rating = "Poor"
        else:
            rating = "Unusable"

        print(f"  {r['concurrent']:>5d}  "
              f"{r['retrieval_seconds']:>9.1f}s  "
              f"{r['llm_seconds']:>9.1f}s  "
              f"{total:>9.1f}s  "
              f"{rating:>10s}")


def _write_report(path, hw, idx700, idx2000,
                  off700, phi4, vllm700, vllm_phi4,
                  on700, mini,
                  off2000, vllm2000, on2000, improvements):
    """Write markdown report."""
    lines = [
        "# Workstation Stress Test Simulation Results",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        "",
        "## Hardware Profile",
        "",
        f"| Component | Spec |",
        f"|-----------|------|",
        f"| CPU | {hw.cpu_threads} threads |",
        f"| RAM | {hw.ram_gb:.0f} GB |",
        f"| GPU | {hw.gpu_name} ({hw.gpu_vram_gb:.0f} GB VRAM) |",
        f"| Storage | 2 TB {hw.storage_type} ({hw.storage_read_mbps:.0f} MB/s) |",
        "",
        "## Index Profile",
        "",
        f"| Metric | 700 GB Source | 2 TB Source |",
        f"|--------|---------------|-------------|",
        f"| Chunks | {idx700.total_chunks:,} | {idx2000.total_chunks:,} |",
        f"| Embeddings | {idx700.embeddings_size_gb:.2f} GB | {idx2000.embeddings_size_gb:.2f} GB |",
        f"| SQLite DB | {idx700.sqlite_size_gb:.2f} GB | {idx2000.sqlite_size_gb:.2f} GB |",
        "",
    ]

    for title, results in [
        ("Offline/Ollama (phi4-mini) -- 700 GB", off700),
        ("Offline/Ollama (phi4:14b) -- 700 GB", phi4),
        ("vLLM Server (phi4-mini) -- 700 GB", vllm700),
        ("vLLM Server (phi4:14b) -- 700 GB", vllm_phi4),
        ("Online (gpt-4o) -- 700 GB", on700),
        ("Online (gpt-4o-mini) -- 700 GB", mini),
        ("Offline/Ollama (phi4-mini) -- 2 TB", off2000),
        ("vLLM Server (phi4-mini) -- 2 TB", vllm2000),
        ("Online (gpt-4o) -- 2 TB", on2000),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"| Users | Retrieval | LLM | Total | Rating |")
        lines.append(f"|-------|-----------|-----|-------|--------|")
        for r in results:
            total = r["total_seconds"]
            rating = ("Excellent" if total < 10 else "Good" if total < 20
                      else "Acceptable" if total < 45 else "Slow" if total < 90
                      else "Poor" if total < 180 else "Unusable")
            lines.append(
                f"| {r['concurrent']} | {r['retrieval_seconds']:.1f}s | "
                f"{r['llm_seconds']:.1f}s | {total:.1f}s | {rating} |"
            )
        lines.append("")

    lines.append("## Improvement Recommendations")
    lines.append("")
    for imp in improvements:
        lines.append(f"### #{imp['rank']}. {imp['what']}")
        lines.append(f"- **Cost:** {imp['cost']}")
        lines.append(f"- **Impact:** {imp['impact']}")
        lines.append(f"- **Offline gain:** {imp['offline_speedup']}")
        lines.append(f"- **Online gain:** {imp['online_speedup']}")
        lines.append("")

    lines.extend(["---", "*Generated by stress_test_workstation_simulation.py*"])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
