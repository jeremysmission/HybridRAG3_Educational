# HybridRAG3 -- Stable Interfaces

This document defines the public APIs that the GUI and external tools may
depend on. Interfaces marked **STABLE** will not change without a version
bump. Interfaces marked **UNSTABLE** may change between sessions.

Last updated: 2026-02-21

---

## 1. Boot Pipeline

**Module:** `src/core/boot.py`
**Status:** STABLE

```python
from src.core.boot import boot_hybridrag, BootResult

result: BootResult = boot_hybridrag(config_path=None)

# BootResult fields:
#   boot_timestamp: str        -- ISO timestamp of boot (e.g., "2026-02-20 14:30:00")
#   success: bool              -- True if at least one mode is available
#   online_available: bool     -- True if API client was built
#   offline_available: bool    -- True if Ollama responds
#   api_client: Any            -- ApiClient instance or None
#   config: dict               -- Loaded config dictionary
#   credentials: Any           -- ApiCredentials instance or None
#   warnings: List[str]        -- Non-fatal issues
#   errors: List[str]          -- Fatal issues
#   summary() -> str           -- Human-readable report
```

---

## 2. Configuration

**Module:** `src/core/config.py`
**Status:** STABLE

```python
from src.core.config import Config, load_config

config: Config = load_config(project_dir=".", config_filename="default_config.yaml")

# Nested sub-configs (all dataclasses):
#   config.paths       -> PathsConfig     (database, embeddings_cache, source_folder)
#   config.embedding   -> EmbeddingConfig  (model_name, dimension, batch_size, device)
#   config.chunking    -> ChunkingConfig   (chunk_size, overlap, max_heading_len)
#   config.ollama      -> OllamaConfig     (base_url, model, timeout_seconds, context_window)
#   config.api         -> APIConfig        (endpoint, model, max_tokens, temperature, ...)
#   config.retrieval   -> RetrievalConfig  (top_k, min_score, hybrid_search, rrf_k, ...)
#   config.indexing    -> IndexingConfig   (supported_extensions, excluded_dirs, ocr_*)
#   config.cost        -> CostConfig       (track_enabled, daily_budget_usd, ...)
#   config.security    -> SecurityConfig   (audit_logging, pii_sanitization)
#   config.hallucination_guard -> HallucinationGuardConfig
```

---

## 3. Indexing

**Module:** `src/core/indexer.py`
**Status:** STABLE

```python
from src.core.indexer import Indexer, IndexingProgressCallback

indexer = Indexer(config, vector_store, embedder, chunker)

# Returns dict with consistent keys:
result: dict = indexer.index_folder(
    folder_path,
    progress_callback=None,   # Optional IndexingProgressCallback
    recursive=True,
)

# Result keys:
#   total_files_scanned: int
#   total_files_indexed: int
#   total_files_skipped: int
#   total_files_reindexed: int
#   total_chunks_added: int
#   preflight_blocked: int
#   elapsed_seconds: float

# Callback interface (all methods optional, no-op by default):
#   on_file_start(file_path, file_num, total_files)
#   on_file_complete(file_path, chunks_created)
#   on_file_skipped(file_path, reason)
#   on_indexing_complete(total_chunks, elapsed_seconds)
#   on_error(file_path, error)
```

---

## 4. Vector Store

**Module:** `src/core/vector_store.py`
**Status:** STABLE

```python
from src.core.vector_store import VectorStore, ChunkMetadata

store = VectorStore(db_path="path/to/db.sqlite3", embedding_dim=384)
store.connect()

# Write
store.add_embeddings(embeddings, metadata_list, texts, file_hash="")

# Read
results: List[dict] = store.search(query_vec, top_k=8)
# Each result: {score, source_path, chunk_index, text}

fts_results: List[dict] = store.fts_search(query_text, top_k=20)
# Each result: {text, source_path, chunk_index, rank}

stats: dict = store.get_stats()
# Keys: chunk_count, source_count, embedding_count, embedding_dim

# Maintenance
deleted: int = store.delete_chunks_by_source(source_path)
file_hash: str = store.get_file_hash(source_path)
store.close()
```

---

## 5. Embedder

**Module:** `src/core/embedder.py`
**Status:** STABLE

```python
from src.core.embedder import Embedder

embedder = Embedder(model_name="all-MiniLM-L6-v2")

# Batch embedding (for indexing)
vectors: np.ndarray = embedder.embed_batch(texts)
# Shape: (N, 384), dtype: float32

# Single query embedding (for search)
vector: np.ndarray = embedder.embed_query(text)
# Shape: (384,), dtype: float32

embedder.close()  # Release ~100MB model from RAM
```

---

## 6. Query Engine

**Module:** `src/core/query_engine.py`
**Status:** STABLE

```python
from src.core.query_engine import QueryEngine, QueryResult

engine = QueryEngine(config, vector_store, embedder, llm_router)
result: QueryResult = engine.query("What is the operating frequency?")

# QueryResult fields:
#   answer: str           -- LLM-generated answer
#   sources: List[dict]   -- Source documents used
#   chunks_used: int      -- Number of chunks sent to LLM
#   tokens_in: int        -- Input tokens consumed
#   tokens_out: int       -- Output tokens generated
#   cost_usd: float       -- Estimated cost
#   latency_ms: float     -- Total query time
#   mode: str             -- "offline" or "online"
#   error: str            -- Error message (empty if OK)
```

---

## 7. LLM Router

**Module:** `src/core/llm_router.py`
**Status:** STABLE

```python
from src.core.llm_router import LLMRouter, LLMResponse

router = LLMRouter(config, api_key=None)
response: Optional[LLMResponse] = router.query("prompt text")

# LLMResponse fields:
#   text: str          -- Generated text
#   tokens_in: int     -- Input tokens
#   tokens_out: int    -- Output tokens
#   model: str         -- Which model answered
#   latency_ms: float  -- Response time

status: dict = router.get_status()
# Keys: mode, api_configured, api_endpoint, ollama_available

# Deployment discovery (auto-detects Azure vs OpenAI)
from src.core.llm_router import get_available_deployments, refresh_deployments

deployments: list = get_available_deployments()
# Azure: GET {base}/openai/deployments -> list of deployment IDs
# OpenAI: GET {endpoint}/models -> list of model IDs
# Returns [] on failure. Results are cached until refresh_deployments().

fresh: list = refresh_deployments()
# Clears cache, re-probes endpoint, returns fresh list
```

---

## 8. Retriever

**Module:** `src/core/retriever.py`
**Status:** STABLE

```python
from src.core.retriever import Retriever, SearchHit

retriever = Retriever(vector_store, embedder, config)
hits: List[SearchHit] = retriever.search("query text")

# SearchHit fields:
#   score: float        -- Relevance (0.0-1.0)
#   source_path: str    -- File path of source document
#   chunk_index: int    -- Chunk position in source file
#   text: str           -- Chunk text content

context: str = retriever.build_context(hits)  # For LLM prompt
sources: List[dict] = retriever.get_sources(hits)  # Grouped by file
```

---

## 9. Network Gate

**Module:** `src/core/network_gate.py`
**Status:** STABLE

```python
from src.core.network_gate import (
    get_gate, configure_gate, NetworkGate, NetworkBlockedError, NetworkMode
)

# Configure at boot (done by boot.py)
gate: NetworkGate = configure_gate(
    mode="offline",           # "offline", "online", or "admin"
    api_endpoint="",          # Allowed endpoint URL
    allowed_prefixes=[],      # Additional allowed URL prefixes
)

# Check before any network call
gate = get_gate()
gate.check_allowed(url, purpose, caller)
# Returns None if allowed, raises NetworkBlockedError if blocked

# Non-raising version
allowed: bool = gate.is_allowed(url)

# Audit
log: List[NetworkAuditEntry] = gate.get_audit_log(last_n=50)
report: str = gate.status_report()
```

---

## 10. Parser Registry

**Module:** `src/parsers/registry.py`
**Status:** STABLE

```python
from src.parsers.registry import REGISTRY, ParserInfo

# Look up parser for a file extension
info: Optional[ParserInfo] = REGISTRY.get(".pdf")
if info:
    parser = info.parser_cls()
    text: str = parser.parse(file_path)
    text, details = parser.parse_with_details(file_path)

# List all supported extensions
extensions: List[str] = REGISTRY.supported_extensions()
# Currently: 24 extensions (.bmp, .csv, .docx, .eml, .gif, .htm, .html,
#   .ini, .jpeg, .jpg, .json, .log, .md, .pdf, .png, .pptx, .tif, .tiff,
#   .txt, .webp, .xlsx, .xml, .yaml, .yml)
```

---

## 11. HTTP Parser

**Module:** `src/parsers/http_parser.py`
**Status:** STABLE

```python
from src.parsers.http_parser import HttpParser

parser = HttpParser(timeout=30, max_bytes=10_000_000, verify_ssl=True)
text, details = parser.fetch_and_parse(url, purpose="intranet_fetch")

# On success: text is extracted content, details has metadata
# On failure: text is "", details["error"] has the error message
# NetworkGate is checked automatically before fetching
```

---

## 12. Credentials

**Module:** `src/security/credentials.py`
**Status:** STABLE

```python
from src.security.credentials import (
    resolve_credentials, ApiCredentials,
    credential_status, store_api_key, store_endpoint,
    store_deployment, store_api_version, clear_credentials,
    validate_endpoint,
    KEYRING_SERVICE, KEYRING_KEY_NAME, KEYRING_ENDPOINT_NAME,
    KEYRING_DEPLOYMENT_NAME, KEYRING_API_VERSION_NAME,
    KEY_ENV_ALIASES, ENDPOINT_ENV_ALIASES,
    DEPLOYMENT_ENV_ALIASES, API_VERSION_ENV_ALIASES,
)

creds: ApiCredentials = resolve_credentials(config_dict=None)

# ApiCredentials fields:
#   api_key: str              -- The API key (or empty)
#   endpoint: str             -- The API endpoint URL (or empty)
#   deployment: str           -- Azure deployment name (or empty)
#   api_version: str          -- Azure API version (or empty)
#   has_key: bool             -- True if api_key is non-empty
#   has_endpoint: bool        -- True if endpoint is non-empty
#   is_online_ready: bool     -- True if both key and endpoint are set
#   key_preview: str          -- Masked key for logging ("sk-...xxxx")
#   source_key: str           -- Where key was found (keyring/env/config)
#   source_endpoint: str      -- Where endpoint was found
#   source_deployment: str    -- Where deployment was found
#   source_api_version: str   -- Where api_version was found

# Status check (for PowerShell wrappers)
status: dict = credential_status()
# Keys: api_key_set, api_endpoint_set, deployment_set, api_version_set,
#        api_key_source, api_endpoint_source, deployment_source, api_version_source

# Store individual values
store_api_key("sk-...")
store_endpoint("https://company.openai.azure.com")
store_deployment("gpt-4o")
store_api_version("2024-02-02")

# Clear all four keyring entries
clear_credentials()
```

---

## 13. Fault Analysis

**Module:** `src/core/fault_analysis.py`
**Status:** STABLE

```python
from src.core.fault_analysis import (
    init_fault_analysis, report_fault, run_health_check,
    FaultAnalysisEngine, Severity, ErrorClass
)

# Initialize once at startup
engine = init_fault_analysis(config, log_dir="logs")

# Report errors from any module
fault = report_fault(
    exception=e,
    source_module="my_module",
    source_function="my_function",
    user_context="what the user was doing",
)

# Run health probes
results: List[ProbeResult] = run_health_check()

# Query fault history
summary: dict = engine.get_summary()
recent: List[FaultEvent] = engine.get_recent_faults(n=10)
```

---

## 14. Model Selection

**Module:** `scripts/_model_meta.py`
**Status:** STABLE

```python
from scripts._model_meta import (
    USE_CASES,             # 9 use cases with work_only flag
    RECOMMENDED_OFFLINE,   # Per use-case offline model recommendations
    RECOMMENDED_ONLINE,    # Per use-case cloud API recommendations
    PERSONAL_FUTURE,       # Models needing >12GB VRAM (recognized, not auto-selected)
    KNOWN_MODELS,          # 46 models with dual tier scores
    use_case_score,        # Compute blended score for a use case
    lookup_known_model,    # Look up model in knowledge base
    select_best_model,     # Pick best model from live deployment list
    get_routing_table,     # Map all use cases to best available models
)

# Use case score (blended eng + gen weights)
score: int = use_case_score(tier_eng=85, tier_gen=70, uc_key="eng")

# Recommended model for a use case
rec = RECOMMENDED_OFFLINE["eng"]
# {primary, alt, temperature, context, reranker, top_k}

# Select best model from live deployments (auto-excludes banned families)
best: str = select_best_model("sw", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
# Returns "gpt-4o" (highest eng score for SW use case)

# Build complete routing table (use_case_key -> model_id)
table: dict = get_routing_table(["gpt-4o", "gpt-4o-mini"])
# {"sw": "gpt-4o", "eng": "gpt-4o", "pm": "gpt-4o", ...}
```

---

## 15. Hallucination Guard

**Module:** `src/core/hallucination_guard/__init__.py`
**Status:** STABLE

```python
from src.core.hallucination_guard import (
    guard_response, harden_prompt, HallucinationGuard, GuardConfig,
    GuardResult, ClaimResult, ClaimVerdict, GUARD_VERSION,
)

# GUARD_VERSION: str  -- Package version (currently "1.1.0")

# Before LLM call: harden the prompt
pkg: dict = harden_prompt(system_prompt, query, chunks, source_files)
# Keys: "system", "user"

# After LLM call: verify the response
result: GuardResult = guard_response(response_text, chunks, query)
# GuardResult fields:
#   is_safe: bool           -- True if faithfulness >= threshold
#   faithfulness: float     -- 0.0-1.0 overall score
#   original_response: str  -- The LLM's raw response
#   safe_response: str      -- Conservative fallback (if not safe)
#   claims: List[ClaimResult]  -- Per-claim verification results
```

---

## 16. Exceptions

**Module:** `src/core/exceptions.py`
**Status:** STABLE

```python
from src.core.exceptions import (
    HybridRAGError,        # Base class for all custom exceptions
    ConfigError,           # Invalid config (CONF-*)
    AuthRejectedError,     # 401/403 from API (AUTH-001)
    NetworkBlockedError,   # Gate denied connection (NET-001)
    EndpointNotConfiguredError,  # Missing endpoint (NET-002)
    EmbeddingError,        # Model issues (EMB-*)
    IndexingError,         # Unrecoverable file error (IDX-001)
)

# All exceptions have:
#   fix_suggestion: str | None  -- Human-readable fix
#   error_code: str | None      -- Machine-readable code (e.g., "IDX-001")

# IndexingError also has:
#   file_path: str | None       -- The file that caused the error
```

---

## Stability Legend

- **STABLE**: Public API will not change without a version bump. Safe for
  GUI and external tools to depend on.
- **UNSTABLE**: Interface may change between sessions. Do not build
  persistent dependencies on these APIs.

Currently all documented interfaces are STABLE. No modules are flagged
UNSTABLE at this time.
