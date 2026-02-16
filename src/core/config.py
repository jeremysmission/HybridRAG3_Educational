# ============================================================================
# HybridRAG -- Configuration (src/core/config.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   This is the "single source of truth" for every setting in HybridRAG.
#   Instead of scattering settings across environment variables, magic
#   numbers buried in code, or ad-hoc defaults, EVERYTHING lives here.
#
# HOW IT WORKS:
#   1. Python "dataclasses" define every setting with a sensible default
#   2. A YAML file (config/default_config.yaml) can override those defaults
#   3. Environment variables can override YAML (for machine-specific paths)
#
#   Priority: env vars > YAML file > hardcoded defaults
#
# WHY DATACLASSES INSTEAD OF A PLAIN DICTIONARY:
#   - Typo protection: config.paths.database gives an error if you mistype;
#     config["paths"]["databse"] silently returns None and breaks later
#   - IDE autocomplete: your editor shows available settings as you type
#   - Documentation: each field has a type and default value visible here
#
# WHY NOT A CONFIG CLASS WITH load() METHOD (like the old ConfigManager):
#   - Functions are simpler than classes when there's no state to manage
#   - load_config() is a pure function: input path, output Config object
#   - Easier to test, easier to understand, fewer things to break
#
# USAGE:
#   from src.core.config import load_config
#   config = load_config(".")                      # load from project dir
#   config = load_config(".", "custom.yaml")       # load specific file
#   print(config.paths.database)                   # typed access
#   print(config.embedding.batch_size)             # IDE shows all options
#
# CHANGES:
#   2026-02-07: SEC-001 FIX -- API endpoint default changed from public
#               OpenAI URL to empty string. Online mode now requires
#               explicit endpoint configuration. Added endpoint validation.
#   2026-02-07: Reranker enabled by default for technical document accuracy.
#   2026-02-14: SAFETY NET -- _dict_to_dataclass now warns on YAML key
#               mismatches instead of silently dropping them. Catches the
#               class of bug where YAML says "timeout" but Python expects
#               "timeout_seconds", and the default wins without warning.
# ============================================================================

from __future__ import annotations

import os
import sys
import yaml
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# -------------------------------------------------------------------
# Sub-configs: each one maps to a section in the YAML file
# -------------------------------------------------------------------

@dataclass
class PathsConfig:
    """
    Where HybridRAG reads and writes data.

    These can be set three ways (in priority order):
      1. Environment variables (HYBRIDRAG_DATA_DIR, HYBRIDRAG_INDEX_FOLDER)
      2. YAML config file
      3. Hardcoded defaults (empty strings = not set)

    Environment variables win because different machines have different
    paths, and you don't want to commit machine-specific paths to git.
    """
    database: str = ""             # Full path to hybridrag.sqlite3
    embeddings_cache: str = ""     # Folder where memmap files live
    source_folder: str = ""        # Folder containing documents to index

    def __post_init__(self) -> None:
        # __post_init__ runs automatically after the dataclass is created.
        # We use it to apply environment variable overrides.

        # If HYBRIDRAG_DATA_DIR is set, use it to build database + cache paths
        data_dir = os.getenv("HYBRIDRAG_DATA_DIR")
        if data_dir:
            if not self.database:
                self.database = os.path.join(data_dir, "hybridrag.sqlite3")
            if not self.embeddings_cache:
                self.embeddings_cache = data_dir

        # If HYBRIDRAG_INDEX_FOLDER is set, use it as the source folder
        source_env = os.getenv("HYBRIDRAG_INDEX_FOLDER")
        if source_env and not self.source_folder:
            self.source_folder = source_env

        # Clean up paths: expand %VARIABLES% and normalize slashes
        if self.database:
            self.database = os.path.normpath(os.path.expandvars(self.database))
        if self.embeddings_cache:
            self.embeddings_cache = os.path.normpath(os.path.expandvars(self.embeddings_cache))
        if self.source_folder:
            self.source_folder = os.path.normpath(os.path.expandvars(self.source_folder))


@dataclass
class EmbeddingConfig:
    """
    Embedding model settings.

    all-MiniLM-L6-v2 was chosen because:
      - Small (~80MB download), fast on CPU
      - 384 dimensions = good balance of quality vs storage
      - Pre-normalized vectors = cosine similarity is just a dot product
      - Widely used, well-tested, stable across platforms

    Alternatives considered:
      - all-mpnet-base-v2: better quality but 768-dim (2x storage, slower)
      - e5-large: even better quality but needs GPU to be practical
      - OpenAI ada-002: excellent but requires API calls ($) and network
    """
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384           # Must match model output (384 for MiniLM)
    batch_size: int = 16           # Chunks embedded at once. Higher = faster, more RAM
    device: str = "cpu"            # "cpu" or "cuda" (GPU)

    def __post_init__(self) -> None:
        # Allow env var override for batch size (useful for tuning per-machine)
        env_batch = os.getenv("HYBRIDRAG_EMBED_BATCH")
        if env_batch:
            self.batch_size = int(env_batch)


@dataclass
class ChunkingConfig:
    """
    Text chunking settings.

    Why character-based instead of token-based:
      - Predictable memory usage (1 char = 1 byte, always)
      - No dependency on a tokenizer (tiktoken adds complexity)
      - For retrieval quality, chunk SIZE matters more than exact token count
      - Defence manuals have tables/diagrams that confuse tokenizers anyway

    Why 1200 chars with 200 overlap:
      - 1200 chars = about 200-300 tokens = a solid paragraph
      - Small enough for the embedding model to capture meaning
      - Large enough to contain a complete thought
      - 200 char overlap prevents losing context at chunk boundaries
    """
    chunk_size: int = 1200        # Characters per chunk
    overlap: int = 200            # Character overlap between consecutive chunks
    max_heading_len: int = 160    # Max length of heading prefix on chunks


@dataclass
class OllamaConfig:
    """
    Local LLM (Ollama) settings for offline mode.

    Ollama runs on your machine -- no internet needed, no API costs.
    """
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout_seconds: int = 120     # How long to wait for a response
    context_window: int = 8192     # Max tokens the model can see at once


@dataclass
class APIConfig:
    """
    Online LLM (OpenAI-compatible API) settings.

    Used when mode="online". Requires internet and an API key.

    SEC-001 FIX (2026-02-07):
      The endpoint previously defaulted to "https://api.openai.com/v1/chat/completions".
      In a restricted environment, this meant that switching to online mode without
      explicitly configuring the endpoint would silently send queries (containing
      sensitive document content) to OpenAI's public servers.

      The default is now an empty string. Online mode will refuse to start
      unless the endpoint is explicitly set in the YAML config or via the
      HYBRIDRAG_API_ENDPOINT environment variable. This is a "fail closed"
      design -- if you forget to configure it, nothing leaks.

    HOW TO CONFIGURE FOR ONLINE MODE:
      Option 1 -- YAML (config/default_config.yaml):
        api:
          endpoint: "https://your-company-api.internal/v1/chat/completions"

      Option 2 -- Environment variable (start_hybridrag.ps1):
        $env:HYBRIDRAG_API_ENDPOINT = "https://your-company-api.internal/v1/chat/completions"
    """
    endpoint: str = ""             # EMPTY BY DEFAULT -- must be explicitly configured
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2048
    temperature: float = 0.1       # Low = more focused/deterministic answers
    timeout_seconds: int = 30

    # Azure-specific settings (ignored for non-Azure providers)
    # These can be set in YAML, env vars, or extracted from URL.
    # If empty, the system will try env vars (AZURE_OPENAI_DEPLOYMENT,
    # AZURE_OPENAI_API_VERSION) then fall back to sensible defaults.
    deployment: str = ""           # Azure deployment name (e.g., "gpt-35-turbo")
    api_version: str = ""          # Azure API version (e.g., "2024-02-02")

    # URL allowlist: if non-empty, endpoint MUST start with one of these
    # prefixes or online mode will refuse to start. This prevents
    # accidental or malicious redirection to unauthorized servers.
    # Set in YAML or leave empty to allow any explicitly-configured URL.
    allowed_endpoint_prefixes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Allow env var override for endpoint
        env_endpoint = os.getenv("HYBRIDRAG_API_ENDPOINT")
        if env_endpoint:
            self.endpoint = env_endpoint.strip()


@dataclass
class CostConfig:
    """
    API cost tracking settings.

    Only relevant when mode="online". Tracks how much you're spending
    so you don't accidentally run up a big bill during testing.
    """
    track_enabled: bool = True
    input_cost_per_1k: float = 0.0015    # GPT-3.5 Turbo pricing (per 1K tokens)
    output_cost_per_1k: float = 0.002
    daily_budget_usd: float = 5.0        # Warn if daily spend exceeds this


@dataclass
class RetrievalConfig:
    """
    Search and retrieval settings.

    These control how search works at query time. HybridRAG supports
    three levels of retrieval sophistication:

    1. Vector only (hybrid_search=False, reranker_enabled=False)
       - Fastest, uses only semantic similarity
       - Good for general questions

    2. Hybrid search (hybrid_search=True, reranker_enabled=False)
       - Combines vector similarity with BM25 keyword search
       - Uses Reciprocal Rank Fusion (RRF) to merge results
       - Better for exact terms, part numbers, acronyms
       - Recommended default for technical documentation

    3. Hybrid + reranker (hybrid_search=True, reranker_enabled=True)
       - Adds a cross-encoder model that re-reads each candidate
       - Most accurate but adds 1-2 seconds per query
       - Toggle on/off via config or future GUI switch
    """
    top_k: int = 8                 # How many chunks to send to the LLM
    min_score: float = 0.20        # Minimum similarity score (0-1) to include
    block_rows: int = 25000        # Memmap rows loaded per search block (RAM control)
    lex_boost: float = 0.06        # Legacy lexical boost (used when hybrid_search=False)

    # --- Hybrid search (BM25 + vector fusion) ---
    hybrid_search: bool = True     # True = use BM25+vector, False = vector only
    rrf_k: int = 60                # RRF constant (higher = less aggressive merging)

    # --- Cross-encoder reranker ---
    # CHANGED 2026-02-07: Default ON for technical document accuracy.
    # The 1-2 second penalty is acceptable for engineering reference lookups.
    # Toggle off via YAML or future GUI engineering menu for speed testing.
    reranker_enabled: bool = True   # Cross-encoder reranker ON by default
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 20       # Retrieve this many candidates, rerank, keep top_k

    # --- Retrieval gate (source-bounded generation) ---
    min_chunks: int = 1            # Minimum chunks required before calling LLM

    def __post_init__(self) -> None:
        env_block = os.getenv("HYBRIDRAG_RETRIEVAL_BLOCK_ROWS")
        if env_block:
            self.block_rows = int(env_block)


@dataclass
class IndexingConfig:
    """
    Indexing pipeline settings.

    Controls which files get indexed and how large files are handled.
    """
    max_chars_per_file: int = 2_000_000   # Clamp files larger than this (safety)
    block_chars: int = 200_000             # Process text in blocks of this size

    # Which file types the indexer will attempt to parse
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".csv", ".json", ".xml", ".log",
        ".pdf", ".docx", ".pptx", ".xlsx", ".eml",
        ".yaml", ".yml", ".ini",
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
    ])

    # Folders to skip during recursive scanning
    excluded_dirs: List[str] = field(default_factory=lambda: [
        ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode",
        "node_modules", ".pytest_cache", ".mypy_cache",
    ])

    # OCR settings (for scanned PDFs)
    ocr_fallback: bool = False        # Disabled by default (slow, needs Tesseract)
    ocr_max_pages: int = 200
    ocr_dpi: int = 200
    ocr_timeout_per_page: int = 20    # Seconds before giving up on one page
    ocr_lang: str = "eng"

    def __post_init__(self) -> None:
        env_ocr = os.getenv("HYBRIDRAG_OCR_FALLBACK")
        if env_ocr:
            self.ocr_fallback = env_ocr.strip() in ("1", "true", "True", "yes")


@dataclass
class SecurityConfig:
    """
    Security settings.

    Placeholder for future phases (encryption, PII filtering, etc.)
    """
    audit_logging: bool = True
    pii_sanitization: bool = False


# -------------------------------------------------------------------
# Master Config -- the one object that holds everything
# -------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration object for HybridRAG.

    Every component receives this object (or a sub-config from it)
    so there is one source of truth for all settings.

    Example:
        config = load_config(".")
        print(config.mode)                    # "offline"
        print(config.paths.database)          # "C:\\...\\hybridrag.sqlite3"
        print(config.embedding.batch_size)    # 16
    """
    mode: str = "offline"   # "offline" (Ollama) or "online" (OpenAI API)

    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    api: APIConfig = field(default_factory=APIConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


# -------------------------------------------------------------------
# Helper: YAML dict -> dataclass (with safety net)
# -------------------------------------------------------------------

def _dict_to_dataclass(cls, data: dict):
    """
    Build a dataclass from a dictionary, ignoring unknown keys.

    Why ignore unknown keys?
      If someone adds a new field to their YAML that we haven't defined
      yet in code, we don't want the whole system to crash. We just
      skip that key and use the default value instead.

    SAFETY NET (2026-02-14):
      If a YAML key does NOT match any dataclass field name, print a
      loud warning to stderr. This catches typos and name mismatches
      like "timeout" vs "timeout_seconds" that previously caused
      silent fallback to defaults -- which led to the Ollama 120s
      timeout bug where the YAML said 180 but Python used 120.

      We also try to suggest the closest matching field name so the
      user knows what to change. This uses simple substring matching
      (not a full fuzzy library) to keep dependencies at zero.

    How it works step by step:
      1. Get the set of valid field names from the dataclass definition
      2. Walk through every key in the YAML dictionary
      3. If the key matches a field name, include it
      4. If NOT, print a [WARN] and try to suggest the right name
      5. Build the dataclass with only the matched keys
      6. Any unmatched fields get their hardcoded default values
    """
    # Step 1: Get valid field names from the dataclass
    known_fields = {f.name for f in dataclasses.fields(cls)}

    # Step 2-4: Sort YAML keys into matched vs unmatched
    filtered = {}
    for k, v in data.items():
        if k in known_fields:
            # This YAML key matches a Python field -- use it
            filtered[k] = v
        else:
            # This YAML key does NOT match anything -- warn loudly
            # Try to suggest a similar field name (catches timeout/timeout_seconds)
            suggestion = ""
            for field_name in known_fields:
                # Check if one string contains the other
                if k in field_name or field_name in k:
                    suggestion = " Did you mean '" + field_name + "'?"
                    break
            print(
                "  [WARN] config/" + cls.__name__ + ": YAML key '"
                + k + "' is not a recognized setting"
                + " -- IGNORED (using default)." + suggestion,
                file=sys.stderr,
            )

    # Step 5-6: Build dataclass with matched keys; unmatched get defaults
    return cls(**filtered)


# -------------------------------------------------------------------
# Main entry point: load_config()
# -------------------------------------------------------------------

def load_config(
    project_dir: str = ".",
    config_filename: str = "default_config.yaml",
) -> Config:
    """
    Load configuration from YAML file, with defaults and env var overrides.

    Resolution order (highest priority first):
      1. Environment variables (applied inside each dataclass __post_init__)
      2. YAML file values
      3. Hardcoded defaults (defined in the dataclass fields above)

    Parameters
    ----------
    project_dir : str
        Path to the HybridRAG project root folder.

    config_filename : str
        Name of the YAML config file inside the config/ subfolder.

    Returns
    -------
    Config
        Fully resolved configuration object ready for use.
    """
    # Look for the YAML file at: <project_dir>/config/<config_filename>
    config_path = Path(project_dir) / "config" / config_filename

    # Load YAML if it exists; otherwise use all defaults
    yaml_data: dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                yaml_data = raw

    # Build the Config object from YAML sections
    # Each section (like "paths", "embedding") maps to a sub-dataclass.
    # If a section is missing from YAML, we get an empty dict,
    # which means "use all defaults for that section."
    config = Config(
        mode=yaml_data.get("mode", "offline"),
        paths=_dict_to_dataclass(PathsConfig, yaml_data.get("paths", {})),
        embedding=_dict_to_dataclass(EmbeddingConfig, yaml_data.get("embedding", {})),
        chunking=_dict_to_dataclass(ChunkingConfig, yaml_data.get("chunking", {})),
        ollama=_dict_to_dataclass(OllamaConfig, yaml_data.get("ollama", {})),
        api=_dict_to_dataclass(APIConfig, yaml_data.get("api", {})),
        cost=_dict_to_dataclass(CostConfig, yaml_data.get("cost", {})),
        retrieval=_dict_to_dataclass(RetrievalConfig, yaml_data.get("retrieval", {})),
        indexing=_dict_to_dataclass(IndexingConfig, yaml_data.get("indexing", {})),
        security=_dict_to_dataclass(SecurityConfig, yaml_data.get("security", {})),
    )

    # --- Auto-configure the network gate ---
    # The gate must be configured BEFORE any network calls. By doing it
    # here in load_config(), it happens automatically no matter which
    # code path loads the config (boot.py, diagnostics, toolkit, etc.).
    #
    # Admin mode override: set HYBRIDRAG_ADMIN_MODE=1 for maintenance tasks
    # (pip install, model downloads). This is intentionally an env var,
    # not a YAML setting, so it can't be accidentally left on.
    try:
        from src.core.network_gate import configure_gate
        admin_override = os.environ.get(
            "HYBRIDRAG_ADMIN_MODE", ""
        ).strip().lower() in ("1", "true", "yes")

        gate_mode = "admin" if admin_override else config.mode

        configure_gate(
            mode=gate_mode,
            api_endpoint=config.api.endpoint,
            allowed_prefixes=config.api.allowed_endpoint_prefixes,
        )
    except Exception:
        # If gate setup fails, we continue -- it defaults to OFFLINE
        # (fail-closed). This is enterprise-in-depth, not a hard dependency.
        pass

    return config


def validate_config(config: Config) -> List[str]:
    """
    Check a Config object for problems. Returns a list of error messages.
    Empty list = everything is valid.

    Call this after load_config() if you want to fail fast with clear messages
    instead of cryptic errors deep in the pipeline.
    """
    errors: List[str] = []

    if config.mode not in ("offline", "online"):
        errors.append(
            "Invalid mode: '" + config.mode + "'. Must be 'offline' or 'online'."
        )

    if not config.paths.database:
        errors.append(
            "paths.database is empty. "
            "Set HYBRIDRAG_DATA_DIR env var or configure in YAML."
        )

    if config.embedding.dimension < 1:
        errors.append(
            "Invalid embedding dimension: " + str(config.embedding.dimension)
        )

    if config.chunking.chunk_size < 100:
        errors.append(
            "chunk_size too small: " + str(config.chunking.chunk_size)
            + ". Minimum 100."
        )

    if config.chunking.overlap >= config.chunking.chunk_size:
        errors.append("chunking.overlap must be less than chunking.chunk_size")

    # --- SEC-001: API endpoint validation ---
    # If online mode is active, the endpoint MUST be explicitly configured.
    # This prevents accidental data leakage to public servers.
    if config.mode == "online":
        if not config.api.endpoint:
            errors.append(
                "SEC-001: API endpoint is empty but mode is 'online'. "
                "Set api.endpoint in YAML or HYBRIDRAG_API_ENDPOINT env var. "
                "This is a security requirement -- the system will not send "
                "queries without an explicitly configured endpoint."
            )
        elif config.api.endpoint == "https://api.openai.com/v1/chat/completions":
            errors.append(
                "SEC-001: API endpoint is the public OpenAI URL. "
                "In a restricted environment, set this to your organization's "
                "internal API endpoint to prevent data exfiltration."
            )

        # If an allowlist is configured, validate the endpoint against it
        if config.api.allowed_endpoint_prefixes and config.api.endpoint:
            allowed = config.api.allowed_endpoint_prefixes
            if not any(config.api.endpoint.startswith(p) for p in allowed):
                errors.append(
                    "SEC-001: API endpoint '" + config.api.endpoint
                    + "' does not match any allowed prefix: "
                    + str(allowed)
                    + ". Update allowed_endpoint_prefixes in YAML or fix the endpoint."
                )

    return errors


def ensure_directories(config: Config) -> None:
    """
    Create the data directories if they don't exist yet.

    Call this before indexing so the first run doesn't crash
    trying to write to a folder that hasn't been created.
    """
    if config.paths.database:
        db_dir = os.path.dirname(config.paths.database)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    if config.paths.embeddings_cache:
        os.makedirs(config.paths.embeddings_cache, exist_ok=True)
