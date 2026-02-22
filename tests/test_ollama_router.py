# ============================================================================
# test_ollama_router.py -- Tests for OllamaRouter and LLMRouter
# ============================================================================
#
# COVERS:
#   TestOllamaRouter -- offline path: queries to local Ollama server
#   TestLLMRouter    -- orchestration: mode switching between Ollama and API
#
# RUN:
#   python -m pytest tests/test_ollama_router.py -v
#
# INTERNET ACCESS: NONE -- all external calls are mocked
# ============================================================================

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pytest
# Import shared fixtures from conftest.py in the same directory.
# WHY this style: avoids requiring tests/__init__.py to exist.
# Works on both home PC and work laptop regardless of package structure.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from conftest import FakeConfig, FakeLLMResponse

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

        # WHY mock time.time here:
        #   The mocked HTTP call completes in < 1 microsecond, so the
        #   real time.time() delta rounds to 0.0 ms. We return t+0.05
        #   on the second call to simulate 50ms latency.
        _time_calls = []
        def fake_time():
            _time_calls.append(1)
            return 1000.000 if len(_time_calls) == 1 else 1000.050

        with patch("src.core.llm_router.time.time", side_effect=fake_time):
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
        assert result.model == "phi4-mini"
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
            assert sent_payload["model"] == "phi4-mini"
            assert sent_payload["prompt"] == "My test prompt"
            assert sent_payload["stream"] is False, (
                "stream should be False -- we want the full response at once"
            )


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
                    model="phi4-mini",
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

