#!/usr/bin/env python3
# ============================================================================
# HybridRAG3 -- Live Indexing + RAGAS-Style Evaluation Test
# ============================================================================
# PURPOSE:
#   1. Index real source data from {SOURCE_DIR}
#   2. Monitor indexing performance (speed, memory, errors)
#   3. Query the index with 20 domain-specific questions
#   4. Rate each answer RAGAS-style (faithfulness, relevance, completeness)
#   5. Display everything live for visual follow-along
#
# USAGE:
#   python tests/live_indexing_test.py
#
# ENVIRONMENT:
#   Reads HYBRIDRAG_DATA_DIR and HYBRIDRAG_INDEX_FOLDER from env or uses
#   defaults: {DATA_DIR} and {SOURCE_DIR}
# ============================================================================

from __future__ import annotations

import os
import sys
import time
import json
import psutil
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set env vars BEFORE any imports that might read them
os.environ.setdefault("HYBRIDRAG_DATA_DIR", r"{DATA_DIR}")
os.environ.setdefault("HYBRIDRAG_INDEX_FOLDER", r"{SOURCE_DIR}")

# Allow network for online mode (OpenRouter)
os.environ["HYBRIDRAG_ADMIN_MODE"] = "0"


# ============================================================================
# Console formatting helpers
# ============================================================================

def hr(char="=", width=78):
    print(char * width)

def banner(title, char="="):
    hr(char)
    print(f"  {title}")
    hr(char)

def section(title):
    print()
    hr("-")
    print(f"  {title}")
    hr("-")

def ok(msg):
    print(f"  [OK]   {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")

def warn(msg):
    print(f"  [WARN] {msg}")

def info(msg):
    print(f"  [INFO] {msg}")

def progress_bar(current, total, width=40, label=""):
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {label}" if label else ""
    print(f"\r  [{bar}] {current}/{total} ({pct:.0%}){suffix}", end="", flush=True)


# ============================================================================
# Performance monitor
# ============================================================================

@dataclass
class PerfSnapshot:
    timestamp: float
    ram_used_mb: float
    ram_pct: float
    cpu_pct: float


@dataclass
class IndexingMetrics:
    start_time: float = 0.0
    files_processed: int = 0
    files_skipped: int = 0
    files_errored: int = 0
    total_chunks: int = 0
    preflight_blocked: int = 0
    perf_snapshots: List[PerfSnapshot] = field(default_factory=list)
    file_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def snapshot(self):
        proc = psutil.Process()
        mem = proc.memory_info()
        self.perf_snapshots.append(PerfSnapshot(
            timestamp=time.time(),
            ram_used_mb=mem.rss / (1024 * 1024),
            ram_pct=proc.memory_percent(),
            cpu_pct=proc.cpu_percent(interval=0.1),
        ))

    @property
    def peak_ram_mb(self):
        if not self.perf_snapshots:
            return 0
        return max(s.ram_used_mb for s in self.perf_snapshots)

    @property
    def avg_file_time(self):
        if not self.file_times:
            return 0
        return sum(self.file_times) / len(self.file_times)

    @property
    def chunks_per_sec(self):
        elapsed = time.time() - self.start_time if self.start_time else 1
        return self.total_chunks / max(elapsed, 0.001)


# ============================================================================
# Progress callback with live display
# ============================================================================

class LiveProgressCallback:
    """Prints live progress during indexing."""

    def __init__(self, metrics: IndexingMetrics):
        self.metrics = metrics
        self._file_start_time = 0.0

    def on_file_start(self, file_path: str, file_num: int, total_files: int):
        self._file_start_time = time.time()
        name = Path(file_path).name
        # Truncate long names
        if len(name) > 45:
            name = name[:42] + "..."
        progress_bar(file_num, total_files, label=name)

    def on_file_complete(self, file_path: str, chunks_created: int):
        elapsed = time.time() - self._file_start_time
        self.metrics.files_processed += 1
        self.metrics.total_chunks += chunks_created
        self.metrics.file_times.append(elapsed)
        # Take periodic snapshots (every 25 files)
        if self.metrics.files_processed % 25 == 0:
            self.metrics.snapshot()

    def on_file_skipped(self, file_path: str, reason: str):
        self.metrics.files_skipped += 1
        if "preflight" in reason:
            self.metrics.preflight_blocked += 1

    def on_indexing_complete(self, total_chunks: int, elapsed_seconds: float):
        print()  # End the progress bar line
        self.metrics.total_chunks = total_chunks
        self.metrics.snapshot()

    def on_error(self, file_path: str, error: str):
        self.metrics.files_errored += 1
        name = Path(file_path).name
        self.metrics.errors.append(f"{name}: {error}")


# ============================================================================
# Indexing Progress Callback adapter
# ============================================================================

class ProgressAdapter:
    """
    Adapts LiveProgressCallback to IndexingProgressCallback interface.

    The Indexer expects an object with specific method signatures from
    IndexingProgressCallback. This adapter wraps our LiveProgressCallback.
    """

    def __init__(self, live_cb: LiveProgressCallback):
        self._cb = live_cb

    def on_file_start(self, file_path, file_num, total_files):
        self._cb.on_file_start(file_path, file_num, total_files)

    def on_file_complete(self, file_path, chunks_created):
        self._cb.on_file_complete(file_path, chunks_created)

    def on_file_skipped(self, file_path, reason):
        self._cb.on_file_skipped(file_path, reason)

    def on_indexing_complete(self, total_chunks, elapsed_seconds):
        self._cb.on_indexing_complete(total_chunks, elapsed_seconds)

    def on_error(self, file_path, error):
        self._cb.on_error(file_path, error)


# ============================================================================
# RAGAS-style evaluation
# ============================================================================

@dataclass
class QuestionResult:
    question: str
    domain: str
    answer: str
    sources: list
    chunks_used: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    mode: str
    error: Optional[str]
    # RAGAS-style scores (0.0 - 1.0)
    retrieval_score: float = 0.0     # Did retrieval find relevant chunks?
    answer_presence: float = 0.0     # Was an answer returned (not error)?
    answer_quality: float = 0.0      # Length/substance proxy
    source_diversity: float = 0.0    # Multiple source files?
    overall_score: float = 0.0       # Weighted average


def rate_answer(result: QuestionResult) -> QuestionResult:
    """
    Compute RAGAS-style scores for a query result.

    Without ground-truth answers, we use proxy metrics:
      - retrieval_score: Based on chunks retrieved and their count
      - answer_presence: Binary -- did we get a real answer?
      - answer_quality:  Based on answer length (proxy for substance)
      - source_diversity: Based on number of unique source files
    """
    # Retrieval score: 0 if no chunks, scales up to 1.0
    if result.chunks_used >= 5:
        result.retrieval_score = 1.0
    elif result.chunks_used >= 3:
        result.retrieval_score = 0.8
    elif result.chunks_used >= 1:
        result.retrieval_score = 0.5
    else:
        result.retrieval_score = 0.0

    # Answer presence: 1.0 if real answer, 0.0 if error
    if result.error or not result.answer:
        result.answer_presence = 0.0
    elif "no relevant information" in result.answer.lower():
        result.answer_presence = 0.2
    elif "error" in result.answer.lower()[:50]:
        result.answer_presence = 0.1
    else:
        result.answer_presence = 1.0

    # Answer quality proxy: based on answer length
    ans_len = len(result.answer)
    if ans_len > 500:
        result.answer_quality = 1.0
    elif ans_len > 200:
        result.answer_quality = 0.8
    elif ans_len > 50:
        result.answer_quality = 0.5
    elif ans_len > 0:
        result.answer_quality = 0.3
    else:
        result.answer_quality = 0.0

    # Source diversity: unique source files
    unique_sources = set()
    for s in result.sources:
        if isinstance(s, dict) and "path" in s:
            unique_sources.add(s["path"])
    n_sources = len(unique_sources)
    if n_sources >= 3:
        result.source_diversity = 1.0
    elif n_sources >= 2:
        result.source_diversity = 0.7
    elif n_sources >= 1:
        result.source_diversity = 0.4
    else:
        result.source_diversity = 0.0

    # Overall: weighted average
    result.overall_score = (
        result.retrieval_score * 0.25 +
        result.answer_presence * 0.35 +
        result.answer_quality * 0.25 +
        result.source_diversity * 0.15
    )

    return result


def score_label(score: float) -> str:
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Acceptable"
    elif score >= 0.3:
        return "Poor"
    else:
        return "Fail"


# ============================================================================
# Test questions (20 questions across all source domains)
# ============================================================================

QUESTIONS = [
    # Domain: Digisonde / Ionosphere
    {
        "q": "What is a digisonde and what frequencies does it use to probe the ionosphere?",
        "domain": "Digisonde",
    },
    {
        "q": "How does the digisonde antenna system work and what type of antenna does it use?",
        "domain": "Digisonde",
    },
    # Domain: Ionogram interpretation
    {
        "q": "How do you identify the critical frequency of the F2 layer on an ionogram?",
        "domain": "Ionogram",
    },
    # Domain: USRP / SDR
    {
        "q": "What are the key specifications and sample rates of the USRP B200 and B210?",
        "domain": "USRP/SDR",
    },
    # Domain: Technical Personnel Management
    {
        "q": "What are the main topics covered in the Technical Personnel Management course?",
        "domain": "Personnel Mgmt",
    },
    {
        "q": "What are best practices for managing technical professionals in engineering organizations?",
        "domain": "Personnel Mgmt",
    },
    # Domain: Project Management
    {
        "q": "What are the key environments that affect how projects operate?",
        "domain": "Project Mgmt",
    },
    {
        "q": "What is the role of the project manager according to PMI standards?",
        "domain": "Project Mgmt",
    },
    {
        "q": "What are the main processes involved in project integration management?",
        "domain": "Project Mgmt",
    },
    # Domain: Python
    {
        "q": "What new features were introduced in Python 3.14?",
        "domain": "Python",
    },
    {
        "q": "How does the Python match statement work for pattern matching?",
        "domain": "Python",
    },
    # Domain: Logic Design
    {
        "q": "How are Karnaugh maps used to simplify Boolean expressions in digital logic?",
        "domain": "Logic Design",
    },
    {
        "q": "What are the fundamental logic gates and their truth tables?",
        "domain": "Logic Design",
    },
    # Domain: AI Workstation
    {
        "q": "What hardware specifications were recommended for the AI workstation procurement?",
        "domain": "AI Workstation",
    },
    # Domain: JHU Advanced Technology (newly copied)
    {
        "q": "What topics and modules are covered in the JHU Advanced Technology course?",
        "domain": "Adv Technology",
    },
    {
        "q": "How has the number of internet users worldwide changed over the past two decades?",
        "domain": "Adv Technology",
    },
    {
        "q": "What are the key applications and properties of nanotechnology?",
        "domain": "Adv Technology",
    },
    {
        "q": "What trends are shown in US R&D investment data?",
        "domain": "Adv Technology",
    },
    {
        "q": "What degree programs does the JHU Engineering Management department offer?",
        "domain": "Adv Technology",
    },
    # Domain: Stack Exchange / General
    {
        "q": "What is the difference between supervised and unsupervised machine learning?",
        "domain": "Stack Exchange",
    },
]


# ============================================================================
# Phase 1: Environment setup and validation
# ============================================================================

def phase1_setup():
    banner("PHASE 1: ENVIRONMENT SETUP")

    data_dir = os.environ.get("HYBRIDRAG_DATA_DIR", "")
    source_dir = os.environ.get("HYBRIDRAG_INDEX_FOLDER", "")

    info(f"Data dir:   {data_dir}")
    info(f"Source dir:  {source_dir}")
    info(f"Project:     {PROJECT_ROOT}")

    # Validate paths
    if not source_dir or not Path(source_dir).is_dir():
        fail(f"Source folder not found: {source_dir}")
        return None, None
    ok(f"Source folder exists")

    # Count files
    all_files = list(Path(source_dir).rglob("*"))
    file_count = sum(1 for f in all_files if f.is_file())
    info(f"Total files in source: {file_count}")

    # Ensure data dir exists
    os.makedirs(data_dir, exist_ok=True)
    ok(f"Data directory ready")

    # System info
    section("System Info")
    mem = psutil.virtual_memory()
    info(f"CPU cores:   {psutil.cpu_count(logical=True)}")
    info(f"RAM total:   {mem.total / (1024**3):.1f} GB")
    info(f"RAM free:    {mem.available / (1024**3):.1f} GB")
    info(f"RAM used:    {mem.percent:.0f}%")

    # Disk info for data dir
    disk = psutil.disk_usage(data_dir)
    info(f"Disk free:   {disk.free / (1024**3):.1f} GB")

    return data_dir, source_dir


# ============================================================================
# Phase 2: Indexing with live monitoring
# ============================================================================

def phase2_indexing(data_dir: str, source_dir: str):
    banner("PHASE 2: INDEXING WITH LIVE MONITORING")

    section("Loading HybridRAG components")

    # Import components
    from src.core.config import load_config, validate_config, ensure_directories
    from src.core.vector_store import VectorStore
    from src.core.embedder import Embedder
    from src.core.chunker import Chunker, ChunkerConfig
    from src.core.indexer import Indexer

    # Load config
    config = load_config(str(PROJECT_ROOT))
    ok(f"Config loaded (mode={config.mode})")

    # Validate
    errors = validate_config(config)
    if errors:
        for e in errors:
            # Skip non-fatal for indexing (API endpoint errors don't affect indexing)
            if "SEC-001" in e or "endpoint" in e.lower():
                warn(e[:80])
            else:
                fail(e[:80])

    # Ensure directories
    ensure_directories(config)
    ok("Directories ready")

    # Initialize components
    info("Loading embedding model (all-MiniLM-L6-v2)...")
    t0 = time.time()
    embedder = Embedder(config.embedding.model_name)
    ok(f"Embedder loaded in {time.time() - t0:.1f}s (dim={embedder.dimension})")

    info("Connecting vector store...")
    vs = VectorStore(
        db_path=config.paths.database,
        embedding_dim=config.embedding.dimension,
    )
    vs.connect()
    ok(f"Vector store connected: {config.paths.database}")

    chunker_config = ChunkerConfig(
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap,
    )
    chunker = Chunker(chunker_config)
    ok(f"Chunker ready (size={config.chunking.chunk_size}, overlap={config.chunking.overlap})")

    indexer = Indexer(config, vs, embedder, chunker)
    ok("Indexer initialized")

    # Run indexing
    section("Indexing source data")
    info(f"Source: {source_dir}")
    info(f"Output: {data_dir}")
    print()

    metrics = IndexingMetrics(start_time=time.time())
    metrics.snapshot()

    live_cb = LiveProgressCallback(metrics)
    adapter = ProgressAdapter(live_cb)

    try:
        result = indexer.index_folder(
            folder_path=source_dir,
            progress_callback=adapter,
            recursive=True,
        )
    except Exception as e:
        fail(f"Indexing failed: {e}")
        traceback.print_exc()
        indexer.close()
        return None, None, None

    metrics.snapshot()
    elapsed = time.time() - metrics.start_time

    # Display results
    section("Indexing Results")
    info(f"Files scanned:     {result['total_files_scanned']}")
    info(f"Files indexed:     {result['total_files_indexed']}")
    info(f"Files re-indexed:  {result['total_files_reindexed']}")
    info(f"Files skipped:     {result['total_files_skipped']}")
    info(f"Chunks created:    {result['total_chunks_added']}")
    info(f"Time elapsed:      {elapsed:.1f}s")
    info(f"Throughput:        {metrics.chunks_per_sec:.1f} chunks/sec")
    if metrics.file_times:
        info(f"Avg file time:     {metrics.avg_file_time:.2f}s")
    info(f"Peak RAM:          {metrics.peak_ram_mb:.0f} MB")

    if result['preflight_blocked']:
        warn(f"Preflight blocked: {len(result['preflight_blocked'])} files")
        for path, reason in result['preflight_blocked'][:10]:
            name = Path(path).name
            warn(f"  {name}: {reason}")
        if len(result['preflight_blocked']) > 10:
            warn(f"  ... and {len(result['preflight_blocked']) - 10} more")

    if metrics.errors:
        warn(f"Errors: {len(metrics.errors)}")
        for err in metrics.errors[:10]:
            warn(f"  {err[:80]}")

    # Check index health
    section("Index Health Check")
    try:
        cursor = vs.conn.execute("SELECT COUNT(*) FROM chunks")
        total_rows = cursor.fetchone()[0]
        info(f"Total chunks in DB: {total_rows}")

        cursor = vs.conn.execute(
            "SELECT COUNT(DISTINCT source_path) FROM chunks"
        )
        total_sources = cursor.fetchone()[0]
        info(f"Unique source files: {total_sources}")

        # Check memmap
        memmap_ok, memmap_msg = vs.mem_store.paths_ok()
        if memmap_ok:
            ok(f"Memmap store: {vs.mem_store.count} embeddings")
        else:
            fail(f"Memmap store: {memmap_msg}")

        # DB file size
        db_size = os.path.getsize(config.paths.database)
        info(f"SQLite size: {db_size / (1024*1024):.1f} MB")

        # Memmap file size
        dat_path = vs.mem_store.dat_path
        if os.path.exists(dat_path):
            dat_size = os.path.getsize(dat_path)
            info(f"Memmap size: {dat_size / (1024*1024):.1f} MB")

    except Exception as e:
        fail(f"Health check error: {e}")

    return config, vs, embedder


# ============================================================================
# Phase 3: RAGAS-style query evaluation
# ============================================================================

def phase3_evaluation(config, vs, embedder):
    banner("PHASE 3: RAGAS-STYLE QUERY EVALUATION (20 QUESTIONS)")

    from src.core.query_engine import QueryEngine
    from src.core.llm_router import LLMRouter

    section("Initializing query pipeline")

    # Create LLM router
    info(f"Mode: {config.mode}")
    info(f"Endpoint: {config.api.endpoint}")
    info(f"Model: {config.api.model}")

    try:
        llm_router = LLMRouter(config)
        ok("LLM router initialized")
    except Exception as e:
        fail(f"LLM router failed: {e}")
        return []

    # Create query engine
    query_engine = QueryEngine(config, vs, embedder, llm_router)
    ok("Query engine ready")

    # Run 20 questions
    section("Running 20 evaluation questions")
    print()

    results: List[QuestionResult] = []

    for i, q_info in enumerate(QUESTIONS, 1):
        question = q_info["q"]
        domain = q_info["domain"]

        # Display question
        print(f"  Q{i:02d} [{domain}]")
        print(f"       {question}")

        try:
            t0 = time.time()
            qr = query_engine.query(question)
            elapsed_ms = (time.time() - t0) * 1000

            result = QuestionResult(
                question=question,
                domain=domain,
                answer=qr.answer,
                sources=qr.sources,
                chunks_used=qr.chunks_used,
                latency_ms=qr.latency_ms,
                tokens_in=qr.tokens_in,
                tokens_out=qr.tokens_out,
                cost_usd=qr.cost_usd,
                mode=qr.mode,
                error=qr.error,
            )
            result = rate_answer(result)
            results.append(result)

            # Display answer preview
            answer_preview = qr.answer[:200].replace("\n", " ")
            if len(qr.answer) > 200:
                answer_preview += "..."

            score_text = score_label(result.overall_score)
            print(f"       Answer: {answer_preview}")
            print(f"       Chunks: {qr.chunks_used} | "
                  f"Latency: {qr.latency_ms:.0f}ms | "
                  f"Score: {result.overall_score:.2f} ({score_text})")

            if qr.sources:
                src_names = []
                for s in qr.sources[:3]:
                    if isinstance(s, dict) and "path" in s:
                        src_names.append(Path(s["path"]).name)
                if src_names:
                    print(f"       Sources: {', '.join(src_names)}")

            if qr.error:
                print(f"       Error: {qr.error}")

            print()

        except Exception as e:
            fail(f"Q{i:02d} crashed: {e}")
            results.append(QuestionResult(
                question=question,
                domain=domain,
                answer="",
                sources=[],
                chunks_used=0,
                latency_ms=0,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0,
                mode=config.mode,
                error=str(e),
            ))
            print()

    return results


# ============================================================================
# Phase 4: Summary report
# ============================================================================

def phase4_report(results: List[QuestionResult]):
    banner("PHASE 4: EVALUATION SUMMARY")

    if not results:
        fail("No results to report")
        return

    # Score table
    section("Score Card")
    print()
    print(f"  {'#':>3}  {'Domain':<16} {'Retr':>4} {'Pres':>4} "
          f"{'Qual':>4} {'Div':>4} {'TOTAL':>5}  {'Rating':<10} "
          f"{'Latency':>8}")
    print(f"  {'---':>3}  {'--------':<16} {'----':>4} {'----':>4} "
          f"{'----':>4} {'----':>4} {'-----':>5}  {'------':<10} "
          f"{'-------':>8}")

    total_score = 0.0
    total_latency = 0.0
    total_cost = 0.0
    domain_scores: Dict[str, List[float]] = {}

    for i, r in enumerate(results, 1):
        rating = score_label(r.overall_score)
        print(f"  {i:3d}  {r.domain:<16} "
              f"{r.retrieval_score:4.2f} {r.answer_presence:4.2f} "
              f"{r.answer_quality:4.2f} {r.source_diversity:4.2f} "
              f"{r.overall_score:5.2f}  {rating:<10} "
              f"{r.latency_ms:7.0f}ms")

        total_score += r.overall_score
        total_latency += r.latency_ms
        total_cost += r.cost_usd

        if r.domain not in domain_scores:
            domain_scores[r.domain] = []
        domain_scores[r.domain].append(r.overall_score)

    avg_score = total_score / len(results)
    avg_latency = total_latency / len(results)

    print()
    hr("-")
    print(f"  {'AVG':>3}  {'OVERALL':<16} "
          f"{'':>4} {'':>4} {'':>4} {'':>4} "
          f"{avg_score:5.2f}  {score_label(avg_score):<10} "
          f"{avg_latency:7.0f}ms")
    hr("-")

    # Domain breakdown
    section("Score by Domain")
    for domain, scores in sorted(domain_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"  {domain:<20} {avg:.2f} ({score_label(avg)}) "
              f"[{len(scores)} question(s)]")

    # Cost summary
    section("Cost Summary")
    info(f"Total API cost: ${total_cost:.4f}")
    info(f"Avg cost/query: ${total_cost/max(len(results),1):.4f}")

    # Performance summary
    section("Performance Summary")
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    if latencies:
        info(f"Avg latency:  {sum(latencies)/len(latencies):.0f}ms")
        info(f"Min latency:  {min(latencies):.0f}ms")
        info(f"Max latency:  {max(latencies):.0f}ms")
        info(f"P50 latency:  {sorted(latencies)[len(latencies)//2]:.0f}ms")

    # Pass/Fail summary
    section("Pass/Fail Summary")
    excellent = sum(1 for r in results if r.overall_score >= 0.9)
    good = sum(1 for r in results if 0.7 <= r.overall_score < 0.9)
    acceptable = sum(1 for r in results if 0.5 <= r.overall_score < 0.7)
    poor = sum(1 for r in results if 0.3 <= r.overall_score < 0.5)
    fail_count = sum(1 for r in results if r.overall_score < 0.3)

    info(f"Excellent (>=0.9): {excellent}")
    info(f"Good (0.7-0.9):    {good}")
    info(f"Acceptable (0.5-0.7): {acceptable}")
    info(f"Poor (0.3-0.5):    {poor}")
    info(f"Fail (<0.3):       {fail_count}")

    overall_rating = score_label(avg_score)
    print()
    banner(f"OVERALL RATING: {avg_score:.2f} / 1.00 -- {overall_rating}")

    return avg_score


# ============================================================================
# Main
# ============================================================================

def main():
    start_time = time.time()
    banner("HYBRIDRAG3 LIVE INDEXING + RAGAS-STYLE EVALUATION")
    info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info(f"Python:  {sys.version.split()[0]}")
    info(f"Project: {PROJECT_ROOT}")
    print()

    # Phase 1: Setup
    data_dir, source_dir = phase1_setup()
    if not data_dir:
        fail("Setup failed -- cannot continue")
        return 1

    # Phase 2: Indexing
    config, vs, embedder = phase2_indexing(data_dir, source_dir)
    if config is None:
        fail("Indexing failed -- cannot continue to evaluation")
        return 1

    # Phase 3: Query evaluation
    results = phase3_evaluation(config, vs, embedder)

    # Phase 4: Report
    avg_score = phase4_report(results)

    # Cleanup
    section("Cleanup")
    try:
        if vs:
            vs.close()
        if embedder:
            embedder.close()
        ok("Resources released")
    except Exception as e:
        warn(f"Cleanup: {e}")

    total_elapsed = time.time() - start_time
    print()
    banner(f"COMPLETE -- Total time: {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
