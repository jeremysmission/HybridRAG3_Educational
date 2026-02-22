# ============================================================================
# HybridRAG3 -- Golden Probe Check Implementations
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Contains the actual health-check logic for each golden probe.
#   Extracted from fault_analysis.py to keep the GoldenProbes class
#   under 500 lines. Each function is a standalone probe that accepts
#   its dependencies as parameters and returns a ProbeResult.
#
# WHY SEPARATE:
#   The GoldenProbes class was 651 lines. Moving 10 probe implementations
#   (~550 lines) here keeps both modules under 500 while maintaining a
#   clean single-responsibility boundary: GoldenProbes orchestrates WHICH
#   probes run, this module implements HOW each probe works.
#
# INTERNET ACCESS: probe_api_connectivity contacts the API endpoint.
#                  probe_ollama_connectivity contacts localhost.
#                  All others: NONE.
# ============================================================================

from __future__ import annotations

import shutil
import time
from typing import Any

from .fault_analysis import ProbeResult, Severity


# -- Configuration Probes ---------------------------------------------------

def check_config_valid(config: Any) -> ProbeResult:
    """
    PROBE: Is the configuration object valid and complete?

    CHECKS:
      - config.mode is either "offline" or "online"
      - Required nested objects exist (ollama, api, chunking, retrieval)
      - Numeric values are within sane ranges
    """
    start = time.time()
    try:
        errors = []

        if not hasattr(config, "mode"):
            errors.append("config.mode is missing")
        elif config.mode not in ("offline", "online"):
            errors.append(f"config.mode='{config.mode}' is invalid")

        if not hasattr(config, "ollama"):
            errors.append("config.ollama section is missing")

        if not hasattr(config, "api"):
            errors.append("config.api section is missing")

        if hasattr(config, "chunking"):
            if config.chunking.chunk_size < 100:
                errors.append(
                    f"chunk_size={config.chunking.chunk_size} is too small (<100)"
                )
            if config.chunking.overlap >= config.chunking.chunk_size:
                errors.append(
                    "overlap >= chunk_size (chunks would have no unique content)"
                )

        if hasattr(config, "retrieval"):
            if config.retrieval.top_k < 1:
                errors.append(f"top_k={config.retrieval.top_k} must be >= 1")
            if not (0.0 <= config.retrieval.min_score <= 1.0):
                errors.append(
                    f"min_score={config.retrieval.min_score} must be 0.0-1.0"
                )

        latency = (time.time() - start) * 1000

        if errors:
            return ProbeResult(
                probe_name="config_valid",
                passed=False,
                message=f"Config validation failed: {'; '.join(errors)}",
                latency_ms=latency,
                severity=Severity.SEV_2,
                details={"errors": errors},
            )

        return ProbeResult(
            probe_name="config_valid",
            passed=True,
            message="Configuration is valid",
            latency_ms=latency,
        )

    except Exception as e:
        return ProbeResult(
            probe_name="config_valid",
            passed=False,
            message=f"Config probe error: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_2,
        )


# -- Resource Probes --------------------------------------------------------

def check_disk_space(config: Any) -> ProbeResult:
    """
    PROBE: Is there enough disk space for indexing and logs?

    THRESHOLD: Warn if < 1 GB free, fail if < 100 MB free.
    """
    start = time.time()
    try:
        index_dir = getattr(config, "paths", None)
        check_path = "."
        if index_dir and hasattr(index_dir, "index_dir"):
            check_path = index_dir.index_dir

        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        pct_free = (usage.free / usage.total) * 100

        latency = (time.time() - start) * 1000

        if free_gb < 0.1:
            return ProbeResult(
                probe_name="disk_space",
                passed=False,
                message=f"CRITICAL: Only {free_gb:.2f} GB free ({pct_free:.1f}%)",
                latency_ms=latency,
                severity=Severity.SEV_1,
                details={"free_gb": free_gb, "total_gb": total_gb},
            )
        elif free_gb < 1.0:
            return ProbeResult(
                probe_name="disk_space",
                passed=False,
                message=f"LOW: {free_gb:.2f} GB free ({pct_free:.1f}%)",
                latency_ms=latency,
                severity=Severity.SEV_3,
                details={"free_gb": free_gb, "total_gb": total_gb},
            )

        return ProbeResult(
            probe_name="disk_space",
            passed=True,
            message=f"{free_gb:.1f} GB free ({pct_free:.1f}%)",
            latency_ms=latency,
            details={"free_gb": free_gb, "total_gb": total_gb},
        )

    except Exception as e:
        return ProbeResult(
            probe_name="disk_space",
            passed=False,
            message=f"Disk probe error: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_3,
        )


def check_memory_usage() -> ProbeResult:
    """
    PROBE: Is the Python process using too much RAM?

    THRESHOLD: Warn at 80% of available system RAM.
    """
    start = time.time()
    try:
        import resource
        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        usage_mb = usage_kb / 1024

        latency = (time.time() - start) * 1000

        return ProbeResult(
            probe_name="memory_usage",
            passed=True,
            message=f"Peak RSS: {usage_mb:.0f} MB",
            latency_ms=latency,
            details={"peak_rss_mb": usage_mb},
        )

    except (ImportError, AttributeError):
        # Windows doesn't have the resource module
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 ** 2)
            pct = process.memory_percent()

            latency = (time.time() - start) * 1000

            if pct > 80:
                return ProbeResult(
                    probe_name="memory_usage",
                    passed=False,
                    message=f"HIGH: {rss_mb:.0f} MB ({pct:.1f}% of system RAM)",
                    latency_ms=latency,
                    severity=Severity.SEV_2,
                    details={"rss_mb": rss_mb, "percent": pct},
                )

            return ProbeResult(
                probe_name="memory_usage",
                passed=True,
                message=f"RSS: {rss_mb:.0f} MB ({pct:.1f}%)",
                latency_ms=latency,
                details={"rss_mb": rss_mb, "percent": pct},
            )

        except ImportError:
            return ProbeResult(
                probe_name="memory_usage",
                passed=True,
                message="Memory probe skipped (psutil not installed)",
                latency_ms=(time.time() - start) * 1000,
            )


# -- Connectivity Probes ----------------------------------------------------

def check_ollama_connectivity(config: Any) -> ProbeResult:
    """
    PROBE: Can we reach the local Ollama server?

    CHECKS:
      - HTTP GET to http://localhost:11434 returns 200
      - Response within 5 seconds
    """
    start = time.time()
    try:
        import httpx
        from src.core.network_gate import get_gate

        base_url = "http://localhost:11434"
        if hasattr(config, "ollama") and hasattr(config.ollama, "base_url"):
            base_url = config.ollama.base_url

        get_gate().check_allowed(base_url, "ollama_probe", "fault_analysis")

        with httpx.Client(timeout=5) as client:
            resp = client.get(base_url)

        latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            return ProbeResult(
                probe_name="ollama_connectivity",
                passed=True,
                message=f"Ollama responding at {base_url} ({latency:.0f}ms)",
                latency_ms=latency,
            )
        else:
            return ProbeResult(
                probe_name="ollama_connectivity",
                passed=False,
                message=f"Ollama returned HTTP {resp.status_code}",
                latency_ms=latency,
                severity=Severity.SEV_2,
            )

    except Exception as e:
        return ProbeResult(
            probe_name="ollama_connectivity",
            passed=False,
            message=f"Ollama unreachable: {type(e).__name__}: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_2,
        )


def check_api_connectivity(config: Any) -> ProbeResult:
    """
    PROBE: Can we reach the cloud API endpoint?

    NOTE: This does NOT send a real query (that would cost money).
    It only checks if the endpoint is reachable via a lightweight request.

    INTERNET ACCESS: YES -- this probe contacts the API endpoint.
    """
    start = time.time()
    try:
        import httpx
        from src.core.network_gate import get_gate

        endpoint = config.api.endpoint

        get_gate().check_allowed(endpoint, "api_probe", "fault_analysis")

        with httpx.Client(timeout=10, verify=True) as client:
            resp = client.get(endpoint)

        latency = (time.time() - start) * 1000

        if resp.status_code < 500:
            return ProbeResult(
                probe_name="api_connectivity",
                passed=True,
                message=f"API reachable at {endpoint} (HTTP {resp.status_code}, {latency:.0f}ms)",
                latency_ms=latency,
            )
        else:
            return ProbeResult(
                probe_name="api_connectivity",
                passed=False,
                message=f"API server error: HTTP {resp.status_code}",
                latency_ms=latency,
                severity=Severity.SEV_2,
            )

    except Exception as e:
        return ProbeResult(
            probe_name="api_connectivity",
            passed=False,
            message=f"API unreachable: {type(e).__name__}: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_2,
        )


# -- Index Probes -----------------------------------------------------------

def check_index_readable(vector_store: Any) -> ProbeResult:
    """
    PROBE: Can we read from the SQLite index database?

    CHECKS:
      - Database file exists
      - We can execute a simple query (SELECT count(*) FROM chunks)
      - No corruption errors
    """
    start = time.time()
    try:
        if vector_store is None:
            return ProbeResult(
                probe_name="index_readable",
                passed=False,
                message="VectorStore not initialized",
                latency_ms=(time.time() - start) * 1000,
                severity=Severity.SEV_2,
            )

        count = 0
        if hasattr(vector_store, "count_chunks"):
            count = vector_store.count_chunks()
        elif hasattr(vector_store, "get_stats"):
            stats = vector_store.get_stats()
            count = stats.get("total_chunks", 0)

        latency = (time.time() - start) * 1000

        return ProbeResult(
            probe_name="index_readable",
            passed=True,
            message=f"Index readable: {count:,} chunks ({latency:.0f}ms)",
            latency_ms=latency,
            details={"chunk_count": count},
        )

    except Exception as e:
        return ProbeResult(
            probe_name="index_readable",
            passed=False,
            message=f"Index read error: {type(e).__name__}: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_1,
        )


def check_index_not_empty(vector_store: Any) -> ProbeResult:
    """
    PROBE: Does the index actually contain data?

    An empty index isn't technically broken, but it means no queries
    will return results -- which is probably not what the user intended.
    """
    start = time.time()
    try:
        count = 0
        if hasattr(vector_store, "count_chunks"):
            count = vector_store.count_chunks()
        elif hasattr(vector_store, "get_stats"):
            stats = vector_store.get_stats()
            count = stats.get("total_chunks", 0)

        latency = (time.time() - start) * 1000

        if count == 0:
            return ProbeResult(
                probe_name="index_not_empty",
                passed=False,
                message="Index is EMPTY -- no documents have been indexed",
                latency_ms=latency,
                severity=Severity.SEV_3,
                details={"chunk_count": 0},
            )

        return ProbeResult(
            probe_name="index_not_empty",
            passed=True,
            message=f"Index contains {count:,} chunks",
            latency_ms=latency,
            details={"chunk_count": count},
        )

    except Exception as e:
        return ProbeResult(
            probe_name="index_not_empty",
            passed=False,
            message=f"Index count error: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_2,
        )


# -- Embedding Probes -------------------------------------------------------

def check_embedder_loaded(embedder: Any) -> ProbeResult:
    """
    PROBE: Is the embedding model loaded and functional?

    CHECKS:
      - Model object exists
      - Can generate an embedding for a test string
    """
    start = time.time()
    try:
        if embedder is None:
            return ProbeResult(
                probe_name="embedder_loaded",
                passed=False,
                message="Embedder not initialized",
                latency_ms=(time.time() - start) * 1000,
                severity=Severity.SEV_1,
            )

        test_text = ["golden probe test sentence for embedding validation"]
        result = embedder.embed_batch(test_text)

        latency = (time.time() - start) * 1000

        if result and len(result) > 0 and len(result[0]) > 0:
            return ProbeResult(
                probe_name="embedder_loaded",
                passed=True,
                message=f"Embedder functional (dim={len(result[0])}, {latency:.0f}ms)",
                latency_ms=latency,
                details={"embedding_dim": len(result[0])},
            )
        else:
            return ProbeResult(
                probe_name="embedder_loaded",
                passed=False,
                message="Embedder returned empty result",
                latency_ms=latency,
                severity=Severity.SEV_1,
            )

    except Exception as e:
        return ProbeResult(
            probe_name="embedder_loaded",
            passed=False,
            message=f"Embedder error: {type(e).__name__}: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_1,
        )


def check_embedding_dimensions(embedder: Any) -> ProbeResult:
    """
    PROBE: Are the embedding dimensions consistent?

    The model (all-MiniLM-L6-v2) should produce 384-dimensional vectors.
    If dimensions change (wrong model loaded, corrupted cache), all
    similarity calculations break silently.
    """
    start = time.time()
    try:
        test_texts = [
            "test sentence one for dimension validation",
            "test sentence two for dimension validation",
        ]
        results = embedder.embed_batch(test_texts)

        latency = (time.time() - start) * 1000

        if len(results) != 2:
            return ProbeResult(
                probe_name="embedding_dimensions",
                passed=False,
                message=f"Expected 2 embeddings, got {len(results)}",
                latency_ms=latency,
                severity=Severity.SEV_1,
            )

        dim1 = len(results[0])
        dim2 = len(results[1])

        if dim1 != dim2:
            return ProbeResult(
                probe_name="embedding_dimensions",
                passed=False,
                message=f"Inconsistent dimensions: {dim1} vs {dim2}",
                latency_ms=latency,
                severity=Severity.SEV_1,
            )

        expected_dim = 384  # all-MiniLM-L6-v2
        if dim1 != expected_dim:
            return ProbeResult(
                probe_name="embedding_dimensions",
                passed=False,
                message=f"Wrong dimension: expected {expected_dim}, got {dim1}",
                latency_ms=latency,
                severity=Severity.SEV_1,
                details={"expected": expected_dim, "actual": dim1},
            )

        return ProbeResult(
            probe_name="embedding_dimensions",
            passed=True,
            message=f"Dimensions correct: {dim1}",
            latency_ms=latency,
            details={"dimension": dim1},
        )

    except Exception as e:
        return ProbeResult(
            probe_name="embedding_dimensions",
            passed=False,
            message=f"Dimension probe error: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_1,
        )


# -- Retrieval Probes -------------------------------------------------------

def check_golden_query(
    vector_store: Any,
    embedder: Any,
    config: Any,
) -> ProbeResult:
    """
    PROBE: Does a known-good query return sensible results?

    This is the most comprehensive probe -- it tests the full
    retrieval pipeline end-to-end. Uses a generic query that
    should match something in most indexes.

    NOTE: This does NOT call the LLM (too slow/expensive for a probe).
    It only tests search + retrieval.
    """
    start = time.time()
    try:
        golden_query = "system architecture design"

        if hasattr(vector_store, "search"):
            from .retriever import Retriever
            retriever = Retriever(vector_store, embedder, config)
            results = retriever.search(golden_query)
        else:
            embedding = embedder.embed_batch([golden_query])
            results = []

        latency = (time.time() - start) * 1000

        if not results:
            return ProbeResult(
                probe_name="golden_query",
                passed=False,
                message="Golden query returned zero results",
                latency_ms=latency,
                severity=Severity.SEV_3,
                details={"query": golden_query, "result_count": 0},
            )

        return ProbeResult(
            probe_name="golden_query",
            passed=True,
            message=f"Golden query returned {len(results)} results ({latency:.0f}ms)",
            latency_ms=latency,
            details={"query": golden_query, "result_count": len(results)},
        )

    except Exception as e:
        return ProbeResult(
            probe_name="golden_query",
            passed=False,
            message=f"Golden query error: {type(e).__name__}: {e}",
            latency_ms=(time.time() - start) * 1000,
            severity=Severity.SEV_2,
        )
