# ============================================================================
# test_api_router.py -- Tests for APIRouter (online/cloud API path)
# ============================================================================
#
# COVERS:
#   TestAPIRouter -- online path: queries to Azure OpenAI / OpenRouter
#
# RUN:
#   python -m pytest tests/test_api_router.py -v
#
# INTERNET ACCESS: NONE -- all external calls are mocked
# ============================================================================

import time
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pytest
# Import shared fixtures from conftest.py in the same directory.
# WHY this style: avoids requiring tests/__init__.py to exist.
# Works on both home PC and work laptop regardless of package structure.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from conftest import FakeConfig, FakeLLMResponse

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

        # WHY mock get_gate here:
        #   The gate singleton starts in OFFLINE mode. APIRouter.__init__
        #   calls get_gate().check_allowed() on the endpoint URL.
        #   In offline mode, openrouter.ai is blocked -> self.client = None.
        #   Mocking the gate to allow all URLs lets __init__ complete normally.
        mock_gate = MagicMock()
        mock_gate.check_allowed.return_value = None  # Never blocks

        with patch("src.core.llm_router.get_gate", return_value=mock_gate):
            with patch("src.core.llm_router.get_app_logger") as mock_logger:
                mock_logger.return_value = MagicMock()

                with patch("openai.OpenAI") as MockOpenAI:
                    with patch("openai.AzureOpenAI") as MockAzure:
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

        # WHY mock get_gate here:
        #   query() calls get_gate().check_allowed() per-call (mode may have
        #   changed mid-session). Without the mock, offline mode blocks the
        #   call and returns None before the mocked OpenAI client is reached.
        mock_gate = MagicMock()
        mock_gate.check_allowed.return_value = None

        # WHY mock time.time here:
        #   The mock HTTP call completes in < 1 microsecond. time.time()
        #   returns a float with ~microsecond resolution, so the delta
        #   rounds to 0.0 ms. We return t+0.05 on the second call to
        #   simulate a realistic 50ms latency in the test.
        _time_calls = []
        def fake_time():
            _time_calls.append(1)
            return 1000.000 if len(_time_calls) == 1 else 1000.050

        with patch("src.core.llm_router.get_gate", return_value=mock_gate):
            with patch("src.core.llm_router.get_app_logger") as mock_logger:
                mock_logger.return_value = MagicMock()

                with patch("openai.OpenAI") as MockOpenAI:
                    with patch("src.core.llm_router.time.time", side_effect=fake_time):
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

        import sys
        with patch("src.core.llm_router.get_app_logger") as mock_logger:
            mock_logger.return_value = MagicMock()

            with patch.dict(sys.modules, {"openai": None}):
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

        mock_gate = MagicMock()
        mock_gate.check_allowed.return_value = None

        mock_log_instance = MagicMock()

        with patch("src.core.llm_router.get_gate", return_value=mock_gate):
            with patch("src.core.llm_router.get_app_logger") as mock_logger:
                mock_logger.return_value = mock_log_instance

                with patch("openai.OpenAI") as MockOpenAI:
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


