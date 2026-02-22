# ============================================================================
# conftest.py -- Shared Test Fixtures for HybridRAG3 Test Suite
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Pytest automatically loads this file before any test runs.
#   It provides:
#     1. sys.path setup so "from src.core.X import Y" works from any test
#     2. Shared fake config objects used by all test files
#     3. FakeLLMResponse dataclass used across test files
#
# WHY SEPARATE:
#   Previously all fixtures lived in test_hybridrag3.py (1785 lines).
#   Splitting into focused files makes each section readable and
#   diagnosable independently.
#
# INTERNET ACCESS: NONE
# ============================================================================

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# -- sys.path setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# SECTION 0: FAKE CONFIG AND HELPER OBJECTS
# ============================================================================
#
# WHY THIS EXISTS:
#   Every module in HybridRAG3 takes a Config object in its constructor.
#   Instead of loading the real YAML config (which would require file paths
#   and all the real modules), we create a lightweight fake that has just
#   the attributes each module needs.
#
#   Think of it like a movie prop -- it looks like the real thing from the
#   outside, but it's hollow on the inside. The code under test doesn't
#   know the difference.
# ============================================================================

@dataclass
class FakeOllamaConfig:
    """Fake config section for Ollama settings."""
    base_url: str = "http://localhost:11434"
    model: str = "phi4-mini"
    timeout_seconds: int = 120


@dataclass
class FakeAPIConfig:
    """Fake config section for API settings."""
    endpoint: str = "https://openrouter.ai/api/v1"
    model: str = "gpt-3.5-turbo"
    key: str = ""
    provider: str = "openai"
    api_version: str = "2024-02-01"
    deployment: str = ""
    auth_scheme: str = "api_key"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_seconds: int = 30


@dataclass
class FakeCostConfig:
    """Fake config section for cost calculation."""
    input_cost_per_1k: float = 0.0015
    output_cost_per_1k: float = 0.002


@dataclass
class FakeChunkingConfig:
    """Fake config section for chunking settings."""
    chunk_size: int = 1200
    overlap: int = 200
    max_heading_len: int = 160


@dataclass
class FakeRetrievalConfig:
    """Fake config section for retrieval settings."""
    top_k: int = 5
    min_score: float = 0.3
    hybrid: bool = True
    reranker: bool = False


@dataclass
class FakeIndexingConfig:
    """Fake config section for indexing settings."""
    max_chars_per_file: int = 2_000_000
    block_chars: int = 200_000
    supported_extensions: list = field(default_factory=lambda: [
        ".txt", ".md", ".csv", ".json", ".xml", ".log", ".pdf",
        ".docx", ".pptx", ".xlsx", ".eml",
    ])
    excluded_dirs: list = field(default_factory=lambda: [
        ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode",
        "node_modules",
    ])


@dataclass
class FakeConfig:
    """
    Lightweight fake of the full HybridRAG3 Config object.

    WHY:
      The real Config class loads from YAML and has complex initialization.
      Tests just need something that "quacks like a Config" -- same attributes,
      same structure, but no file I/O.
    """
    mode: str = "offline"
    ollama: FakeOllamaConfig = field(default_factory=FakeOllamaConfig)
    api: FakeAPIConfig = field(default_factory=FakeAPIConfig)
    cost: FakeCostConfig = field(default_factory=FakeCostConfig)
    chunking: FakeChunkingConfig = field(default_factory=FakeChunkingConfig)
    retrieval: FakeRetrievalConfig = field(default_factory=FakeRetrievalConfig)
    indexing: FakeIndexingConfig = field(default_factory=FakeIndexingConfig)


# -- Fake LLMResponse (matches the real dataclass in llm_router.py) ---------
# We define it here so tests don't need to import from the real module.
# This avoids circular imports if the real module has import issues.
@dataclass
class FakeLLMResponse:
    """Matches LLMResponse from llm_router.py."""
    text: str
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: float


# ============================================================================
# SECTION 1: LLM ROUTER TESTS
# ============================================================================
#
# WHAT WE'RE TESTING:
#   The llm_router.py "switchboard" -- it decides whether to talk to
#   Ollama (offline) or the cloud API (online) and wraps the answer
#   in a standard LLMResponse object.
#
# WHAT WE MOCK:
#   - httpx.Client (for Ollama HTTP calls)
#   - openai.OpenAI / openai.AzureOpenAI (for API calls)
#   - keyring (for credential storage)
#   - get_app_logger (to suppress log output during tests)
#
# TEST CATEGORIES:
#   1. OllamaRouter tests (offline mode)
#   2. APIRouter tests (online mode)
#   3. LLMRouter orchestration tests (mode switching)
#   4. Credential resolution tests
#   5. Error handling tests
# ============================================================================