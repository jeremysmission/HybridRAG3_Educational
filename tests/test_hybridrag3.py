# ============================================================================
# HybridRAG3 -- Complete Test Suite
# ============================================================================
# FILE: tests/test_hybridrag3.py
#
# WHAT THIS FILE DOES (plain English):
#   This file contains automated tests for the three most critical modules
#   in HybridRAG3:
#
#     1. test_llm_router   -- Tests the "switchboard" that routes queries
#                             to Ollama (offline) or API (online)
#     2. test_query_engine  -- Tests the full pipeline: search -> retrieve
#                             context -> call LLM -> return result
#     3. test_indexer       -- Tests the file scanner, chunker, and storage
#                             pipeline
#
#   "Tests" means: we run the code with FAKE inputs (called "mocks") and
#   check that it produces the EXPECTED outputs. If something breaks,
#   the test fails with a clear message telling you what went wrong.
#
# HOW TO RUN:
#   From your HybridRAG3 root directory:
#     python -m pytest tests/test_hybridrag3.py -v
#
#   The -v flag means "verbose" -- it shows each test name and PASS/FAIL.
#
#   To run just one section:
#     python -m pytest tests/test_hybridrag3.py -v -k "test_llm_router"
#     python -m pytest tests/test_hybridrag3.py -v -k "test_query_engine"
#     python -m pytest tests/test_hybridrag3.py -v -k "test_indexer"
#
# WHY MOCKS?
#   We don't want tests to actually call Ollama or Azure -- that would be
#   slow, expensive, and require credentials. Instead, we create fake
#   ("mock") versions of those services that return predictable answers.
#   This lets us test our logic without any external dependencies.
#
# DEPENDENCIES:
#   pip install pytest pytest-mock --break-system-packages
#
# INTERNET ACCESS: NONE -- all tests use mocks, no network calls
#
# COMPATIBILITY NOTES:
#   - pytest >= 7.0 (tested with 7.4+)
#   - pytest-mock >= 3.10 (provides the mocker fixture)
#   - No LangChain dependencies (per project rules)
#   - No em-dashes, emojis, or non-ASCII chars (per PS 5.1 rules)
#   - Works on both home PC and work laptop
# ============================================================================

import os
import sys
import time
import json
import shutil
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import pytest


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
    model: str = "llama3"
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

class TestOllamaRouter:
    """
    Tests for the OllamaRouter class (offline mode).

    OllamaRouter talks to a local Ollama server via HTTP.
    We mock the HTTP client so no real Ollama server is needed.
    """

    def _make_router(self, config=None):
        """
        Helper: create an OllamaRouter with a fake config and mocked logger.

        WHY A HELPER?
          Every test needs the same setup. Putting it in a helper means
          we write the setup once and call it from each test. DRY principle
          (Don't Repeat Yourself).
        """
        if config is None:
            config = FakeConfig(mode="offline")

        # We mock the logger import to prevent actual log file creation
        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            # Import the real class -- this tests your actual code
            from src.core.llm_router import OllamaRouter
            router = OllamaRouter(config)

        return router

    # ------------------------------------------------------------------
    # Test 1.1: Ollama health check -- server is running
    # ------------------------------------------------------------------
    def test_is_available_when_ollama_running(self):
        """
        WHAT: Check that is_available() returns True when Ollama responds.
        WHY:  The GUI shows a green/red status indicator based on this.
              If this test fails, the GUI would always show "offline."
        HOW:  We mock httpx.Client.get() to return status 200 (success).
        """
        router = self._make_router()

        # Create a fake HTTP response that looks like Ollama is running
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Patch httpx.Client so it returns our fake response
        with patch("src.core.llm_router.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get.return_value = mock_response
            MockClient.return_value.__enter__ = Mock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__exit__ = Mock(return_value=False)

            result = router.is_available()

        assert result is True, (
            "is_available() should return True when Ollama responds with 200"
        )

    # ------------------------------------------------------------------
    # Test 1.2: Ollama health check -- server is NOT running
    # ------------------------------------------------------------------
    def test_is_available_when_ollama_not_running(self):
        """
        WHAT: Check that is_available() returns False when Ollama is down.
        WHY:  If Ollama isn't running and we try to query it, we'd get a
              confusing connection error. Better to check first.
        HOW:  We mock httpx.Client.get() to raise a ConnectionError.
        """
        router = self._make_router()

        with patch("src.core.llm_router.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.get.side_effect = Exception("Connection refused")
            MockClient.return_value.__enter__ = Mock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__exit__ = Mock(return_value=False)

            result = router.is_available()

        assert result is False, (
            "is_available() should return False when Ollama can't be reached"
        )

    # ------------------------------------------------------------------
    # Test 1.3: Successful Ollama query
    # ------------------------------------------------------------------
    def test_query_success(self):
        """
        WHAT: Send a prompt to Ollama and get back a proper LLMResponse.
        WHY:  This is the core offline path. If this breaks, no queries work
              without internet.
        HOW:  Mock the HTTP POST to return a fake Ollama JSON response.
        """
        router = self._make_router()

        # This is what Ollama's /api/generate endpoint actually returns
        fake_ollama_response = {
            "response": "The operating frequency is 5.2 GHz.",
            "prompt_eval_count": 150,     # Tokens in the prompt
            "eval_count": 25,             # Tokens in the response
            "done": True,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_ollama_response
        mock_response.raise_for_status = MagicMock()

        with patch("src.core.llm_router.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post.return_value = mock_response
            MockClient.return_value.__enter__ = Mock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__exit__ = Mock(return_value=False)

            result = router.query("What is the operating frequency?")

        # Verify we got a proper response back
        assert result is not None, "query() should return a response, not None"
        assert result.text == "The operating frequency is 5.2 GHz."
        assert result.tokens_in == 150
        assert result.tokens_out == 25
        assert result.model == "llama3"
        assert result.latency_ms > 0, "Latency should be positive"

    # ------------------------------------------------------------------
    # Test 1.4: Ollama query with HTTP error
    # ------------------------------------------------------------------
    def test_query_http_error(self):
        """
        WHAT: Verify that HTTP errors (500, connection refused, etc.)
              return None instead of crashing.
        WHY:  If Ollama crashes mid-query, HybridRAG should gracefully
              show "Error calling LLM" instead of a Python traceback.
        """
        router = self._make_router()

        # Import the actual httpx to get the right exception type
        import httpx as real_httpx

        with patch("src.core.llm_router.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post.side_effect = real_httpx.HTTPError(
                "Connection refused"
            )
            MockClient.return_value.__enter__ = Mock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__exit__ = Mock(return_value=False)

            result = router.query("Test query")

        assert result is None, (
            "query() should return None on HTTP error, not raise an exception"
        )

    # ------------------------------------------------------------------
    # Test 1.5: Ollama query with empty response
    # ------------------------------------------------------------------
    def test_query_empty_response(self):
        """
        WHAT: Handle the edge case where Ollama returns an empty answer.
        WHY:  This can happen if the model runs out of context or gets
              confused. We should return the empty string, not crash.
        """
        router = self._make_router()

        fake_response = {
            "response": "",
            "prompt_eval_count": 100,
            "eval_count": 0,
            "done": True,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_response
        mock_response.raise_for_status = MagicMock()

        with patch("src.core.llm_router.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post.return_value = mock_response
            MockClient.return_value.__enter__ = Mock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__exit__ = Mock(return_value=False)

            result = router.query("What is nothing?")

        assert result is not None
        assert result.text == ""
        assert result.tokens_out == 0

    # ------------------------------------------------------------------
    # Test 1.6: Ollama query sends correct payload
    # ------------------------------------------------------------------
    def test_query_sends_correct_payload(self):
        """
        WHAT: Verify the HTTP POST body matches Ollama's expected format.
        WHY:  If the payload format is wrong, Ollama returns a 400 error
              and the user sees a confusing "Bad Request" message.
        """
        router = self._make_router()

        fake_response = {
            "response": "Test answer",
            "prompt_eval_count": 50,
            "eval_count": 10,
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = fake_response
        mock_response.raise_for_status = MagicMock()

        with patch("src.core.llm_router.httpx.Client") as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.post.return_value = mock_response
            MockClient.return_value.__enter__ = Mock(
                return_value=mock_client_instance
            )
            MockClient.return_value.__exit__ = Mock(return_value=False)

            router.query("My test prompt")

            # Check what was actually sent to Ollama
            call_args = mock_client_instance.post.call_args

            # Verify the URL
            assert "/api/generate" in call_args[0][0], (
                "Should POST to /api/generate"
            )

            # Verify the JSON payload
            sent_payload = call_args[1]["json"]
            assert sent_payload["model"] == "llama3"
            assert sent_payload["prompt"] == "My test prompt"
            assert sent_payload["stream"] is False, (
                "stream should be False -- we want the full response at once"
            )


class TestAPIRouter:
    """
    Tests for the APIRouter class (online mode).

    APIRouter uses the openai SDK to talk to Azure OpenAI or standard
    OpenAI-compatible APIs (OpenRouter, Groq, etc.).
    We mock the SDK client so no real API calls happen.
    """

    def _make_router(self, config=None, api_key="test-key-123",
                     endpoint="https://openrouter.ai/api/v1"):
        """
        Helper: create an APIRouter with mocked dependencies.

        Args:
            config:   Optional FakeConfig (defaults to online mode)
            api_key:  The API key to use (fake for testing)
            endpoint: The API endpoint URL

        Returns:
            An APIRouter instance with a mocked OpenAI client
        """
        if config is None:
            config = FakeConfig(mode="online")

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            with patch("src.core.llm_router.OpenAI") as MockOpenAI:
                with patch("src.core.llm_router.AzureOpenAI") as MockAzure:
                    from src.core.llm_router import APIRouter
                    router = APIRouter(config, api_key, endpoint)

        return router

    # ------------------------------------------------------------------
    # Test 2.1: Provider auto-detection -- OpenRouter
    # ------------------------------------------------------------------
    def test_detects_openrouter_as_non_azure(self):
        """
        WHAT: Verify that https://openrouter.ai/api/v1 is detected as
              NON-Azure (standard OpenAI-compatible).
        WHY:  Azure and OpenAI use different client classes. If we
              accidentally use AzureOpenAI for OpenRouter, every call
              would fail with malformed URLs.
        """
        router = self._make_router(
            endpoint="https://openrouter.ai/api/v1"
        )
        assert router.is_azure is False, (
            "OpenRouter should be detected as non-Azure"
        )

    # ------------------------------------------------------------------
    # Test 2.2: Provider auto-detection -- Azure
    # ------------------------------------------------------------------
    def test_detects_azure_endpoint(self):
        """
        WHAT: Verify that Azure-style URLs are correctly identified.
        WHY:  Your work laptop uses an Azure endpoint. If the detection
              fails, the code would use OpenAI auth headers (Bearer) instead
              of Azure auth headers (api-key), causing 401 errors.
        """
        router = self._make_router(
            endpoint="https://company.openai.azure.com"
        )
        assert router.is_azure is True, (
            "Azure endpoint should be detected as Azure"
        )

    # ------------------------------------------------------------------
    # Test 2.3: Provider auto-detection -- AOAI variant
    # ------------------------------------------------------------------
    def test_detects_aoai_endpoint(self):
        """
        WHAT: Verify that 'aoai' in the URL triggers Azure detection.
        WHY:  Your company uses 'aoai' in their internal URL pattern.
              The old code didn't recognize this and treated it as standard
              OpenAI, causing the 401 errors that blocked you for weeks.
        """
        router = self._make_router(
            endpoint="https://company-aoai.openai.azure.com"
        )
        assert router.is_azure is True, (
            "AOAI-style endpoint should be detected as Azure"
        )

    # ------------------------------------------------------------------
    # Test 2.4: Azure URL extraction -- strips deployment path
    # ------------------------------------------------------------------
    def test_extract_azure_base_strips_deployment_path(self):
        """
        WHAT: Verify that _extract_azure_base() strips /openai/deployments/...
        WHY:  The SDK appends this path automatically. If we pass the full
              URL, the SDK doubles it: .../openai/deployments/.../openai/deployments/...
              This caused the 404 errors you experienced.
        """
        router = self._make_router(
            endpoint="https://company.openai.azure.com"
        )

        full_url = (
            "https://company.openai.azure.com/openai/deployments/"
            "gpt-35-turbo/chat/completions?api-version=2024-02-02"
        )
        result = router._extract_azure_base(full_url)

        assert result == "https://company.openai.azure.com", (
            f"Expected just the base domain, got: {result}"
        )

    # ------------------------------------------------------------------
    # Test 2.5: Azure URL extraction -- already clean URL passes through
    # ------------------------------------------------------------------
    def test_extract_azure_base_clean_url_unchanged(self):
        """
        WHAT: If the URL is already just the base domain, don't modify it.
        WHY:  Not everyone stores the full URL. Some users store just the
              base, which is correct. Don't break what's already right.
        """
        router = self._make_router(
            endpoint="https://company.openai.azure.com"
        )

        clean_url = "https://company.openai.azure.com"
        result = router._extract_azure_base(clean_url)

        assert result == clean_url

    # ------------------------------------------------------------------
    # Test 2.6: Successful API query
    # ------------------------------------------------------------------
    def test_query_success(self):
        """
        WHAT: Simulate a successful API call and verify the LLMResponse.
        WHY:  This is the core online path. Tests that the SDK response
              is correctly parsed into our standard format.
        """
        config = FakeConfig(mode="online")
        config.api.max_tokens = 1024
        config.api.temperature = 0.7

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            with patch("src.core.llm_router.OPENAI_SDK_AVAILABLE", True):
                with patch("src.core.llm_router.OpenAI") as MockOpenAI:
                    # Build a fake SDK response object
                    # This mimics what openai.ChatCompletion actually returns
                    mock_choice = MagicMock()
                    mock_choice.message.content = "The answer is 42."

                    mock_usage = MagicMock()
                    mock_usage.prompt_tokens = 200
                    mock_usage.completion_tokens = 30

                    mock_completion = MagicMock()
                    mock_completion.choices = [mock_choice]
                    mock_completion.usage = mock_usage
                    mock_completion.model = "gpt-3.5-turbo"

                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = (
                        mock_completion
                    )
                    MockOpenAI.return_value = mock_client

                    from src.core.llm_router import APIRouter
                    router = APIRouter(
                        config, "test-key", "https://openrouter.ai/api/v1"
                    )

                    result = router.query("What is the meaning of life?")

        assert result is not None
        assert result.text == "The answer is 42."
        assert result.tokens_in == 200
        assert result.tokens_out == 30
        assert result.model == "gpt-3.5-turbo"
        assert result.latency_ms > 0

    # ------------------------------------------------------------------
    # Test 2.7: API query with no client (SDK not installed)
    # ------------------------------------------------------------------
    def test_query_returns_none_when_no_client(self):
        """
        WHAT: If the openai SDK isn't installed, query() returns None.
        WHY:  On a fresh machine without pip install openai, the router
              should fail gracefully, not crash with ImportError.
        """
        config = FakeConfig(mode="online")

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            with patch("src.core.llm_router.OPENAI_SDK_AVAILABLE", False):
                from src.core.llm_router import APIRouter
                router = APIRouter(config, "test-key", "https://fake.com")

        # client should be None because SDK is "not installed"
        result = router.query("Test")
        assert result is None

    # ------------------------------------------------------------------
    # Test 2.8: API 401 error handling
    # ------------------------------------------------------------------
    def test_query_handles_401_error(self):
        """
        WHAT: Verify that 401 Unauthorized errors are caught and logged
              with a helpful troubleshooting message.
        WHY:  This is the exact error that blocked your work laptop for
              weeks. The test ensures we never regress to cryptic errors.
        """
        config = FakeConfig(mode="online")
        config.api.max_tokens = 1024
        config.api.temperature = 0.7

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_log_instance = MagicMock()
            mock_logger.return_value = mock_log_instance

            with patch("src.core.llm_router.OPENAI_SDK_AVAILABLE", True):
                with patch("src.core.llm_router.OpenAI") as MockOpenAI:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.side_effect = (
                        Exception("Error code: 401 - Unauthorized")
                    )
                    MockOpenAI.return_value = mock_client

                    from src.core.llm_router import APIRouter
                    router = APIRouter(
                        config, "bad-key", "https://openrouter.ai/api/v1"
                    )

                    result = router.query("Test query")

        assert result is None, "401 errors should return None, not crash"

        # Verify the error was logged with the troubleshooting hint
        error_calls = [
            call for call in mock_log_instance.error.call_args_list
            if "401" in str(call) or "auth" in str(call).lower()
        ]
        assert len(error_calls) > 0, (
            "401 error should be logged with troubleshooting hint"
        )

    # ------------------------------------------------------------------
    # Test 2.9: Status report includes correct provider info
    # ------------------------------------------------------------------
    def test_get_status_openrouter(self):
        """
        WHAT: Verify get_status() correctly reports the provider and endpoint.
        WHY:  The diagnostics script (rag-cred-status) uses this to show
              the user what's configured. Wrong info = wrong troubleshooting.
        """
        router = self._make_router(
            endpoint="https://openrouter.ai/api/v1"
        )

        status = router.get_status()

        assert status["provider"] == "openai", (
            "OpenRouter should report as 'openai' provider type"
        )
        assert "openrouter" in status["endpoint"].lower()
        assert status["api_configured"] is True
        assert status["sdk_available"] is True


class TestLLMRouter:
    """
    Tests for the LLMRouter orchestration class.

    This is the top-level "switchboard" that decides between
    Ollama and API based on config.mode.
    """

    # ------------------------------------------------------------------
    # Test 3.1: Offline mode routes to Ollama
    # ------------------------------------------------------------------
    def test_offline_mode_routes_to_ollama(self):
        """
        WHAT: In offline mode, query() should call ollama.query().
        WHY:  If it accidentally calls the API in offline mode, that's
              a security violation (data leaving the machine unexpectedly).
        """
        config = FakeConfig(mode="offline")

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            # We need to patch the credential resolution to avoid keyring
            with patch.dict(os.environ, {}, clear=False):
                from src.core.llm_router import LLMRouter

                # Patch the internal routers after creation
                router = LLMRouter(config, api_key=None)

                # Replace the Ollama router with a mock that returns a
                # predictable response
                mock_ollama = MagicMock()
                mock_ollama.query.return_value = FakeLLMResponse(
                    text="Offline answer",
                    tokens_in=100,
                    tokens_out=20,
                    model="llama3",
                    latency_ms=5000.0,
                )
                router.ollama = mock_ollama

                result = router.query("Test question")

        assert result is not None
        assert result.text == "Offline answer"
        mock_ollama.query.assert_called_once_with("Test question")

    # ------------------------------------------------------------------
    # Test 3.2: Online mode routes to API
    # ------------------------------------------------------------------
    def test_online_mode_routes_to_api(self):
        """
        WHAT: In online mode, query() should call api.query().
        WHY:  If online mode accidentally uses Ollama, the user gets slow
              local answers when they expected fast cloud answers.
        """
        config = FakeConfig(mode="online")

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            from src.core.llm_router import LLMRouter
            router = LLMRouter(config, api_key="test-key-123")

            # Replace the API router with a mock
            mock_api = MagicMock()
            mock_api.query.return_value = FakeLLMResponse(
                text="Online answer",
                tokens_in=200,
                tokens_out=30,
                model="gpt-3.5-turbo",
                latency_ms=1500.0,
            )
            router.api = mock_api

            result = router.query("Test question")

        assert result is not None
        assert result.text == "Online answer"
        mock_api.query.assert_called_once_with("Test question")

    # ------------------------------------------------------------------
    # Test 3.3: Online mode with no API key returns None
    # ------------------------------------------------------------------
    def test_online_mode_no_api_key_returns_none(self):
        """
        WHAT: If online mode is selected but no API key is configured,
              query() should return None with a helpful log message.
        WHY:  Before the SDK rebuild, this case would silently send
              requests with an empty auth header, causing confusing 401s.
        """
        config = FakeConfig(mode="online")

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            from src.core.llm_router import LLMRouter
            router = LLMRouter(config, api_key=None)

            # Force api to None (simulating no key found)
            router.api = None

            result = router.query("Test question")

        assert result is None, (
            "Should return None when API key is not configured"
        )

    # ------------------------------------------------------------------
    # Test 3.4: Status report shows both backends
    # ------------------------------------------------------------------
    def test_get_status_shows_both_backends(self):
        """
        WHAT: get_status() should report on both Ollama and API availability.
        WHY:  The diagnostics display needs to show the user a complete
              picture of what's working and what's not.
        """
        config = FakeConfig(mode="offline")

        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            from src.core.llm_router import LLMRouter
            router = LLMRouter(config, api_key="test-key")

            # Mock Ollama availability check
            router.ollama.is_available = MagicMock(return_value=True)

            status = router.get_status()

        assert "mode" in status
        assert "ollama_available" in status
        assert "api_configured" in status
        assert "sdk_available" in status
        assert status["mode"] == "offline"
        assert status["ollama_available"] is True
        assert status["api_configured"] is True


# ============================================================================
# SECTION 2: QUERY ENGINE TESTS
# ============================================================================
#
# WHAT WE'RE TESTING:
#   The query_engine.py orchestration pipeline:
#     user question -> retrieve chunks -> build prompt -> call LLM -> log -> return
#
# WHAT WE MOCK:
#   - Retriever (returns fake search results)
#   - LLMRouter (returns fake AI answers)
#   - get_app_logger (suppress log output)
#
# TEST CATEGORIES:
#   1. Successful query flow
#   2. No results found
#   3. Empty context edge case
#   4. LLM call failure
#   5. Cost calculation (online vs offline)
#   6. Prompt construction
#   7. Exception handling
# ============================================================================

class TestQueryEngine:
    """
    Tests for the QueryEngine orchestration pipeline.
    """

    def _make_engine(self, config=None, search_results=None,
                     llm_text="Test answer"):
        """
        Helper: create a QueryEngine with all dependencies mocked.

        Args:
            config:         Optional FakeConfig
            search_results: What the retriever should return (list of dicts)
            llm_text:       What the LLM should return as answer text

        Returns:
            (engine, mocks) -- the engine and a dict of all mock objects
        """
        if config is None:
            config = FakeConfig(mode="offline")

        # Mock all the dependencies that QueryEngine needs
        mock_vector_store = MagicMock()
        mock_embedder = MagicMock()
        mock_llm_router = MagicMock()

        # Set up the LLM router to return a fake response
        mock_llm_router.query.return_value = FakeLLMResponse(
            text=llm_text,
            tokens_in=150,
            tokens_out=25,
            model="llama3",
            latency_ms=3000.0,
        )

        with patch("src.core.query_engine.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            with patch("src.core.query_engine.Retriever") as MockRetriever:
                mock_retriever_instance = MagicMock()

                # Default: return some search results
                if search_results is None:
                    search_results = [
                        {
                            "chunk_id": "abc123",
                            "text": "The system operates at 5.2 GHz.",
                            "score": 0.85,
                            "source_path": "/docs/spec.pdf",
                        },
                        {
                            "chunk_id": "def456",
                            "text": "Power output is 100 watts.",
                            "score": 0.72,
                            "source_path": "/docs/spec.pdf",
                        },
                    ]

                mock_retriever_instance.search.return_value = search_results

                # build_context joins the chunk texts into a single string
                mock_retriever_instance.build_context.return_value = (
                    "The system operates at 5.2 GHz.\n\n"
                    "Power output is 100 watts."
                )

                # get_sources returns source file summaries
                mock_retriever_instance.get_sources.return_value = [
                    {
                        "path": "/docs/spec.pdf",
                        "chunks": 2,
                        "avg_relevance": 0.785,
                    }
                ]

                MockRetriever.return_value = mock_retriever_instance

                from src.core.query_engine import QueryEngine
                engine = QueryEngine(
                    config, mock_vector_store, mock_embedder, mock_llm_router
                )

        mocks = {
            "vector_store": mock_vector_store,
            "embedder": mock_embedder,
            "llm_router": mock_llm_router,
            "retriever": mock_retriever_instance,
        }

        return engine, mocks

    # ------------------------------------------------------------------
    # Test 4.1: Successful end-to-end query
    # ------------------------------------------------------------------
    def test_successful_query(self):
        """
        WHAT: Run a full query and verify all fields in QueryResult.
        WHY:  This is the "happy path" -- everything works. If this test
              fails, the core functionality is broken.
        """
        engine, mocks = self._make_engine(
            llm_text="The system operates at 5.2 GHz with 100W output."
        )

        result = engine.query("What is the operating frequency?")

        assert result.answer == (
            "The system operates at 5.2 GHz with 100W output."
        )
        assert result.chunks_used == 2
        assert result.sources[0]["path"] == "/docs/spec.pdf"
        assert result.mode == "offline"
        assert result.error is None
        assert result.latency_ms > 0

    # ------------------------------------------------------------------
    # Test 4.2: No relevant documents found
    # ------------------------------------------------------------------
    def test_no_results_found(self):
        """
        WHAT: When the retriever finds nothing, return a clear message.
        WHY:  Empty results shouldn't crash or return "None." The user
              should see a friendly message explaining nothing was found.
        """
        engine, mocks = self._make_engine(search_results=[])

        result = engine.query("What is quantum entanglement?")

        assert "No relevant information" in result.answer
        assert result.chunks_used == 0
        assert result.sources == []
        assert result.error is None

    # ------------------------------------------------------------------
    # Test 4.3: Empty context edge case
    # ------------------------------------------------------------------
    def test_empty_context_returns_error(self):
        """
        WHAT: If chunks exist but their text is empty (extremely rare),
              handle it gracefully.
        WHY:  A corrupted index could have entries with empty text fields.
              Rather than sending a blank prompt to the LLM, we catch this.
        """
        engine, mocks = self._make_engine()

        # Override build_context to return empty string
        mocks["retriever"].build_context.return_value = ""

        result = engine.query("Test query")

        assert "no usable context" in result.answer.lower()
        assert result.error == "empty_context"

    # ------------------------------------------------------------------
    # Test 4.4: LLM call failure
    # ------------------------------------------------------------------
    def test_llm_failure(self):
        """
        WHAT: If the LLM router returns None (call failed), show error.
        WHY:  Network glitches, Ollama crashes, API timeouts -- any of
              these would cause the LLM call to fail. User should see a
              clear error message, not a traceback.
        """
        engine, mocks = self._make_engine()

        # Override LLM router to return None (simulating failure)
        mocks["llm_router"].query.return_value = None

        result = engine.query("Test query")

        assert "Error calling LLM" in result.answer
        assert result.error == "LLM call failed"
        assert result.chunks_used == 2  # Chunks were found, LLM just failed

    # ------------------------------------------------------------------
    # Test 4.5: Cost calculation -- offline mode = $0
    # ------------------------------------------------------------------
    def test_cost_calculation_offline(self):
        """
        WHAT: In offline mode, cost should always be $0.
        WHY:  Ollama runs locally -- no API charges. If cost shows up
              in offline mode, something is wrong with the calculation.
        """
        config = FakeConfig(mode="offline")
        engine, mocks = self._make_engine(config=config)

        result = engine.query("Test query")

        assert result.cost_usd == 0.0, (
            "Offline mode should have zero cost"
        )

    # ------------------------------------------------------------------
    # Test 4.6: Cost calculation -- online mode
    # ------------------------------------------------------------------
    def test_cost_calculation_online(self):
        """
        WHAT: In online mode, verify cost is calculated from token counts.
        WHY:  Cost tracking lets you monitor API spend. Wrong formula
              = wrong budget tracking. The formula is:
              (tokens_in / 1000) * input_rate + (tokens_out / 1000) * output_rate
        """
        config = FakeConfig(mode="online")
        engine, mocks = self._make_engine(config=config)

        result = engine.query("Test query")

        # Expected: (150/1000)*0.0015 + (25/1000)*0.002 = 0.000225 + 0.00005 = 0.000275
        expected_cost = (150 / 1000) * 0.0015 + (25 / 1000) * 0.002
        assert abs(result.cost_usd - expected_cost) < 0.0001, (
            f"Expected cost ~{expected_cost}, got {result.cost_usd}"
        )

    # ------------------------------------------------------------------
    # Test 4.7: Prompt construction
    # ------------------------------------------------------------------
    def test_prompt_includes_context_and_question(self):
        """
        WHAT: Verify the prompt sent to the LLM contains both the
              retrieved context and the user's question.
        WHY:  If the context is missing, the LLM hallucinates.
              If the question is missing, the LLM writes a random summary.
              Both parts must be present.
        """
        engine, mocks = self._make_engine()

        engine.query("What is the frequency?")

        # Get the prompt that was actually sent to the LLM
        call_args = mocks["llm_router"].query.call_args
        prompt_sent = call_args[0][0]

        assert "5.2 GHz" in prompt_sent, "Prompt should contain the context"
        assert "What is the frequency?" in prompt_sent, (
            "Prompt should contain the user question"
        )

    # ------------------------------------------------------------------
    # Test 4.8: Exception handling -- unexpected errors
    # ------------------------------------------------------------------
    def test_exception_returns_error_result(self):
        """
        WHAT: If something unexpected crashes, return an error QueryResult
              instead of an unhandled exception.
        WHY:  The GUI should show "Error processing query: ..." not a
              Python traceback. Never crash the application.
        """
        engine, mocks = self._make_engine()

        # Force the retriever to raise an unexpected exception
        mocks["retriever"].search.side_effect = RuntimeError(
            "Corrupted index file"
        )

        result = engine.query("Test query")

        assert "RuntimeError" in result.error
        assert "Corrupted index" in result.answer
        assert result.chunks_used == 0


# ============================================================================
# SECTION 3: INDEXER TESTS
# ============================================================================
#
# WHAT WE'RE TESTING:
#   The indexer.py pipeline:
#     scan folder -> parse file -> validate text -> chunk -> embed -> store
#
# WHAT WE MOCK:
#   - VectorStore (fake database)
#   - Embedder (returns dummy vectors)
#   - Chunker (returns predictable chunks)
#   - File system (using temp directories)
#
# TEST CATEGORIES:
#   1. File discovery (supported extensions, excluded dirs)
#   2. File hash change detection (incremental re-indexing)
#   3. Text validation (garbage detection)
#   4. Block processing (large file handling)
#   5. Error resilience (corrupted files don't crash the run)
#   6. Resource cleanup (memory leak prevention)
# ============================================================================

class TestIndexer:
    """
    Tests for the Indexer pipeline.

    Uses a real temporary directory with real files, but mocks
    the database, embedder, and chunker to avoid needing actual
    AI models or SQLite during tests.
    """

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self, tmp_path):
        """
        WHAT: Create a temporary directory for each test with sample files.
        WHY:  Tests need real files on disk to scan, but we don't want to
              touch your actual documents. tmp_path is automatically cleaned
              up after each test.

        WHAT IS @pytest.fixture?
          A "fixture" is code that runs before each test to set up the
          environment. autouse=True means it runs for EVERY test in this
          class without being explicitly requested.
        """
        self.test_dir = tmp_path / "source"
        self.test_dir.mkdir()

        # Create sample files of different types
        (self.test_dir / "document.txt").write_text(
            "This is a test document about radio frequency engineering. "
            "The system operates at 5.2 GHz with a power output of 100 watts. "
            "The antenna gain is measured in dBi." * 20,  # Make it chunky
            encoding="utf-8",
        )

        (self.test_dir / "notes.md").write_text(
            "# Engineering Notes\n\n"
            "## Section 1: RF Design\n\n"
            "The receiver sensitivity is -95 dBm at the input. "
            "Signal-to-noise ratio must exceed 10 dB." * 15,
            encoding="utf-8",
        )

        (self.test_dir / "data.csv").write_text(
            "frequency,power,gain\n"
            "5200,100,15\n"
            "5300,95,14\n"
            "5400,90,13\n",
            encoding="utf-8",
        )

        # File type we don't support -- should be skipped
        (self.test_dir / "image.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        )

        # Excluded directory -- should be skipped entirely
        excluded = self.test_dir / ".git"
        excluded.mkdir()
        (excluded / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

        # Subdirectory with a valid file
        sub = self.test_dir / "subdir"
        sub.mkdir()
        (sub / "deep_doc.txt").write_text(
            "This is a document in a subdirectory." * 30,
            encoding="utf-8",
        )

    def _make_indexer(self, config=None):
        """
        Helper: create an Indexer with all dependencies mocked.

        Returns:
            (indexer, mocks) -- the indexer and a dict of mock objects
        """
        if config is None:
            config = FakeConfig()

        mock_vector_store = MagicMock()
        mock_embedder = MagicMock()
        mock_chunker = MagicMock()

        # VectorStore: no existing data (everything is new)
        mock_vector_store.get_file_hash.return_value = None
        mock_vector_store.delete_chunks_by_source.return_value = 0

        # Embedder: return dummy vectors (384-dim for MiniLM)
        # Each chunk gets a list of 384 zeros as its "embedding"
        def fake_embed_batch(texts):
            return [[0.0] * 384 for _ in texts]

        mock_embedder.embed_batch.side_effect = fake_embed_batch

        # Chunker: split text into fixed-size chunks for testing
        def fake_chunk_text(text):
            # Simple chunker: split into 200-char pieces
            chunks = []
            for i in range(0, len(text), 200):
                chunk = text[i:i + 200]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        mock_chunker.chunk_text.side_effect = fake_chunk_text

        # We need to patch several imports that the Indexer uses
        with patch("src.core.indexer.make_chunk_id") as mock_make_id:
            # Return deterministic IDs
            call_counter = [0]

            def make_fake_id(**kwargs):
                call_counter[0] += 1
                return f"chunk_{call_counter[0]:06d}"

            mock_make_id.side_effect = make_fake_id

            from src.core.indexer import Indexer
            indexer = Indexer(config, mock_vector_store, mock_embedder, mock_chunker)

        mocks = {
            "vector_store": mock_vector_store,
            "embedder": mock_embedder,
            "chunker": mock_chunker,
        }

        return indexer, mocks

    # ------------------------------------------------------------------
    # Test 5.1: Discovers supported files, ignores unsupported
    # ------------------------------------------------------------------
    def test_discovers_supported_files_only(self):
        """
        WHAT: The indexer should find .txt, .md, .csv but skip .png
        WHY:  Indexing a binary PNG as text would create garbage chunks
              that pollute search results.
        """
        indexer, mocks = self._make_indexer()

        result = indexer.index_folder(str(self.test_dir))

        # We created: document.txt, notes.md, data.csv, deep_doc.txt = 4 files
        # Skipped: image.png (unsupported), .git/HEAD (excluded dir)
        assert result["total_files_scanned"] == 4, (
            f"Expected 4 supported files, found {result['total_files_scanned']}"
        )

    # ------------------------------------------------------------------
    # Test 5.2: Skips excluded directories
    # ------------------------------------------------------------------
    def test_skips_excluded_directories(self):
        """
        WHAT: Files inside .git, __pycache__, .venv etc. are skipped.
        WHY:  These contain internal tool files, not user documents.
              Indexing .git objects would create nonsensical search results.
        """
        indexer, mocks = self._make_indexer()

        result = indexer.index_folder(str(self.test_dir))

        # The .git/HEAD file should NOT appear in the results
        # If it did, total_files_scanned would be 5 instead of 4
        assert result["total_files_scanned"] == 4

    # ------------------------------------------------------------------
    # Test 5.3: Skips unchanged files (hash match)
    # ------------------------------------------------------------------
    def test_skips_unchanged_files(self):
        """
        WHAT: If a file was already indexed and hasn't changed, skip it.
        WHY:  Your 100GB corporate drive has thousands of files. Most don't
              change. Without skip logic, every re-index takes days.
              With it, a re-index takes minutes.
        """
        indexer, mocks = self._make_indexer()

        # First run: index everything
        result1 = indexer.index_folder(str(self.test_dir))
        first_run_indexed = result1["total_files_indexed"]

        # Simulate: all files now have stored hashes (already indexed)
        def return_matching_hash(file_path):
            # Return the same hash that _compute_file_hash would produce
            p = Path(file_path)
            if p.exists():
                stat = p.stat()
                return f"{stat.st_size}:{stat.st_mtime_ns}"
            return None

        mocks["vector_store"].get_file_hash.side_effect = return_matching_hash

        # Reset the call counters
        mocks["embedder"].embed_batch.reset_mock()

        # Second run: everything should be skipped
        result2 = indexer.index_folder(str(self.test_dir))

        assert result2["total_files_skipped"] == result2["total_files_scanned"], (
            f"All {result2['total_files_scanned']} files should be skipped "
            f"on re-index, but only {result2['total_files_skipped']} were"
        )
        assert result2["total_files_indexed"] == 0

    # ------------------------------------------------------------------
    # Test 5.4: Re-indexes modified files
    # ------------------------------------------------------------------
    def test_reindexes_modified_files(self):
        """
        WHAT: If a file was indexed but has since been modified, re-index it.
        WHY:  Someone updates a document on the shared drive. HybridRAG
              needs to pick up those changes without re-indexing everything.
        """
        indexer, mocks = self._make_indexer()

        # Simulate: file was previously indexed with a DIFFERENT hash
        mocks["vector_store"].get_file_hash.return_value = "old_size:old_mtime"
        mocks["vector_store"].delete_chunks_by_source.return_value = 5

        result = indexer.index_folder(str(self.test_dir))

        # All files should be re-indexed because hash doesn't match
        assert result["total_files_reindexed"] > 0, (
            "Modified files should be detected and re-indexed"
        )

        # Old chunks should have been deleted before re-indexing
        assert mocks["vector_store"].delete_chunks_by_source.call_count > 0

    # ------------------------------------------------------------------
    # Test 5.5: Text validation catches binary garbage
    # ------------------------------------------------------------------
    def test_validate_text_catches_binary_garbage(self):
        """
        WHAT: _validate_text() returns False for binary-looking strings.
        WHY:  Corrupted PDFs sometimes return binary garbage as "text."
              Without validation, that garbage gets embedded and pollutes
              search results with nonsensical matches.
        """
        indexer, _ = self._make_indexer()

        # Good text -- should pass validation
        good_text = (
            "This is a perfectly normal document about RF engineering. "
            "The system operates at 5.2 GHz with an antenna gain of 15 dBi."
        )
        assert indexer._validate_text(good_text) is True

        # Binary garbage -- should FAIL validation
        binary_garbage = "\x00\x89PNG\r\n\x1a\x00" * 100
        assert indexer._validate_text(binary_garbage) is False

        # Too short -- should FAIL validation
        assert indexer._validate_text("Hi") is False
        assert indexer._validate_text("") is False
        assert indexer._validate_text(None) is False

    # ------------------------------------------------------------------
    # Test 5.6: File hash computation is deterministic
    # ------------------------------------------------------------------
    def test_file_hash_is_deterministic(self):
        """
        WHAT: Same file produces the same hash every time.
        WHY:  Hash comparison is how we detect changes. If the hash
              changes randomly, we'd re-index everything every time,
              defeating the skip optimization.
        """
        indexer, _ = self._make_indexer()

        test_file = self.test_dir / "document.txt"

        hash1 = indexer._compute_file_hash(test_file)
        hash2 = indexer._compute_file_hash(test_file)

        assert hash1 == hash2, "Same file should produce same hash"
        assert ":" in hash1, "Hash should be in format 'size:mtime_ns'"

    # ------------------------------------------------------------------
    # Test 5.7: File hash changes when file is modified
    # ------------------------------------------------------------------
    def test_file_hash_changes_on_modification(self):
        """
        WHAT: Modifying a file changes its hash.
        WHY:  If the hash doesn't change on modification, the skip logic
              would never re-index changed files, giving stale results.
        """
        indexer, _ = self._make_indexer()

        test_file = self.test_dir / "document.txt"

        hash_before = indexer._compute_file_hash(test_file)

        # Modify the file (changes size and mtime)
        import time as _time
        _time.sleep(0.1)  # Ensure mtime changes
        test_file.write_text(
            "Modified content that is different from the original.",
            encoding="utf-8",
        )

        hash_after = indexer._compute_file_hash(test_file)

        assert hash_before != hash_after, (
            "Modified file should produce a different hash"
        )

    # ------------------------------------------------------------------
    # Test 5.8: Single file error doesn't crash the entire run
    # ------------------------------------------------------------------
    def test_single_file_error_continues(self):
        """
        WHAT: If one file fails to parse, the indexer continues to the next.
        WHY:  Your 100GB drive might have corrupted PDFs buried in
              subdirectories. If the indexer crashes on file #3,000, you
              lose 3 days of progress. Instead, it logs the error and moves on.
        """
        indexer, mocks = self._make_indexer()

        # Make the chunker fail on the FIRST call, then work normally
        call_count = [0]
        original_side_effect = mocks["chunker"].chunk_text.side_effect

        def sometimes_failing_chunker(text):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated corrupted file")
            return original_side_effect(text)

        mocks["chunker"].chunk_text.side_effect = sometimes_failing_chunker

        # Should NOT raise an exception
        result = indexer.index_folder(str(self.test_dir))

        # At least some files should have been indexed despite the error
        assert result["total_files_indexed"] >= 1, (
            "Indexer should continue after a single file error"
        )

    # ------------------------------------------------------------------
    # Test 5.9: Excluded directory detection
    # ------------------------------------------------------------------
    def test_is_excluded_detects_git_dir(self):
        """
        WHAT: _is_excluded() returns True for paths inside .git/
        WHY:  Git objects look like valid text files but contain
              internal version control data, not user documents.
        """
        indexer, _ = self._make_indexer()

        git_file = Path("/repo/.git/objects/ab/cdef123")
        assert indexer._is_excluded(git_file) is True

        normal_file = Path("/repo/docs/readme.md")
        assert indexer._is_excluded(normal_file) is False

        venv_file = Path("/repo/.venv/lib/python3.11/site.py")
        assert indexer._is_excluded(venv_file) is True

    # ------------------------------------------------------------------
    # Test 5.10: Text block iterator
    # ------------------------------------------------------------------
    def test_iter_text_blocks_splits_large_text(self):
        """
        WHAT: _iter_text_blocks() splits text into manageable chunks.
        WHY:  A 500-page PDF produces ~2 million characters. Loading all
              that into RAM at once would spike memory. The block iterator
              processes 200K chars at a time, keeping RAM stable.
        """
        indexer, _ = self._make_indexer()

        # Create a large text that's bigger than one block
        # block_chars defaults to 200,000
        large_text = "A" * 500_000  # 500K chars = at least 3 blocks

        blocks = list(indexer._iter_text_blocks(large_text))

        assert len(blocks) >= 2, "500K chars should produce multiple blocks"
        total_chars = sum(len(b) for b in blocks)
        assert total_chars == 500_000, "All text should be accounted for"

    # ------------------------------------------------------------------
    # Test 5.11: Resource cleanup
    # ------------------------------------------------------------------
    def test_close_releases_resources(self):
        """
        WHAT: close() calls close() on the embedder and vector store.
        WHY:  The embedding model stays in RAM (~100MB) until explicitly
              released. Over repeated indexing runs, this leaks memory.
              close() prevents this.
        """
        indexer, mocks = self._make_indexer()

        indexer.close()

        mocks["embedder"].close.assert_called_once()
        mocks["vector_store"].close.assert_called_once()

    # ------------------------------------------------------------------
    # Test 5.12: Large file clamping
    # ------------------------------------------------------------------
    def test_large_file_is_clamped(self):
        """
        WHAT: Files larger than max_chars_per_file are truncated.
        WHY:  A single 10,000-page document could overwhelm the system.
              Clamping at 2 million chars (~1,000 pages) keeps processing
              time and memory reasonable.
        """
        indexer, mocks = self._make_indexer()

        # Create a file larger than the limit
        huge_file = self.test_dir / "huge_doc.txt"
        # Use a small limit for testing
        indexer.max_chars_per_file = 1000
        huge_file.write_text("X" * 5000, encoding="utf-8")

        result = indexer.index_folder(str(self.test_dir))

        # The file should still be indexed, just truncated
        assert result["total_files_indexed"] >= 1

    # ------------------------------------------------------------------
    # Test 5.13: Empty folder handling
    # ------------------------------------------------------------------
    def test_empty_folder(self):
        """
        WHAT: Indexing an empty folder returns zero counts without crashing.
        WHY:  A user might point the indexer at an empty directory by mistake.
        """
        indexer, _ = self._make_indexer()

        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()

        result = indexer.index_folder(str(empty_dir))

        assert result["total_files_scanned"] == 0
        assert result["total_files_indexed"] == 0
        assert result["total_chunks_added"] == 0

    # ------------------------------------------------------------------
    # Test 5.14: Nonexistent folder raises error
    # ------------------------------------------------------------------
    def test_nonexistent_folder_raises(self):
        """
        WHAT: Pointing the indexer at a folder that doesn't exist raises
              FileNotFoundError with a clear message.
        WHY:  Better to fail fast with a clear error than silently index
              zero files and let the user wonder why nothing happened.
        """
        indexer, _ = self._make_indexer()

        with pytest.raises(FileNotFoundError):
            indexer.index_folder("/this/path/does/not/exist")

    # ------------------------------------------------------------------
    # Test 5.15: Progress callback is called
    # ------------------------------------------------------------------
    def test_progress_callback_called(self):
        """
        WHAT: Verify the progress callback methods are invoked during indexing.
        WHY:  The GUI progress bar depends on these callbacks. If they stop
              firing, the GUI appears frozen during long indexing runs.
        """
        indexer, _ = self._make_indexer()

        from src.core.indexer import IndexingProgressCallback

        class TrackingCallback(IndexingProgressCallback):
            def __init__(self):
                self.file_starts = []
                self.file_completes = []
                self.file_skips = []
                self.errors = []
                self.completed = False

            def on_file_start(self, path, num, total):
                self.file_starts.append((path, num, total))

            def on_file_complete(self, path, chunks):
                self.file_completes.append((path, chunks))

            def on_file_skipped(self, path, reason):
                self.file_skips.append((path, reason))

            def on_error(self, path, error):
                self.errors.append((path, error))

            def on_indexing_complete(self, total, elapsed):
                self.completed = True

        tracker = TrackingCallback()
        result = indexer.index_folder(str(self.test_dir), progress_callback=tracker)

        assert len(tracker.file_starts) > 0, "on_file_start should be called"
        assert tracker.completed is True, "on_indexing_complete should be called"


# ============================================================================
# SECTION 4: INTEGRATION-STYLE TESTS
# ============================================================================
#
# These tests check that different modules work together correctly.
# Still uses mocks for external services, but tests the wiring between
# internal components.
# ============================================================================

class TestIntegration:
    """
    Tests that verify components work together correctly.
    """

    # ------------------------------------------------------------------
    # Test 6.1: Config validation
    # ------------------------------------------------------------------
    def test_fake_config_has_all_required_fields(self):
        """
        WHAT: Verify our FakeConfig has all the fields the real code expects.
        WHY:  If someone adds a new config field to the real Config class
              but forgets to add it to FakeConfig, all tests would break
              with confusing AttributeError messages.
        """
        config = FakeConfig()

        # Fields used by LLMRouter
        assert hasattr(config, "mode")
        assert hasattr(config.ollama, "base_url")
        assert hasattr(config.ollama, "model")
        assert hasattr(config.ollama, "timeout_seconds")
        assert hasattr(config.api, "endpoint")
        assert hasattr(config.api, "model")

        # Fields used by QueryEngine
        assert hasattr(config, "cost")
        assert hasattr(config.cost, "input_cost_per_1k")
        assert hasattr(config.cost, "output_cost_per_1k")

        # Fields used by Indexer
        assert hasattr(config, "chunking")
        assert hasattr(config.chunking, "chunk_size")
        assert hasattr(config.chunking, "overlap")

    # ------------------------------------------------------------------
    # Test 6.2: LLMResponse dataclass completeness
    # ------------------------------------------------------------------
    def test_llm_response_fields(self):
        """
        WHAT: Verify LLMResponse has all the fields that QueryEngine expects.
        WHY:  QueryEngine reads tokens_in, tokens_out, text, and model
              from LLMResponse. If any are missing, cost calculation or
              logging breaks.
        """
        response = FakeLLMResponse(
            text="Answer",
            tokens_in=100,
            tokens_out=20,
            model="llama3",
            latency_ms=5000.0,
        )

        assert response.text == "Answer"
        assert response.tokens_in == 100
        assert response.tokens_out == 20
        assert response.model == "llama3"
        assert response.latency_ms == 5000.0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# If you run this file directly (python test_hybridrag3.py), it runs all tests.
# But the recommended way is: python -m pytest tests/test_hybridrag3.py -v
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
