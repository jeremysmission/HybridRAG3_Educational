# ============================================================================
# test_query_engine.py -- Tests for QueryEngine pipeline
# ============================================================================
#
# COVERS:
#   TestQueryEngine -- full query pipeline: retrieve -> context -> LLM -> result
#
# RUN:
#   python -m pytest tests/test_query_engine.py -v
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
            model="phi4-mini",
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

        # WHY mock time.time here:
        #   The entire query pipeline (retriever + LLM) is mocked, so it
        #   completes in microseconds. time.time() delta rounds to 0.0 ms.
        #   We return t+0.1 on the second call to simulate 100ms latency.
        _time_calls = []
        def fake_time():
            _time_calls.append(1)
            return 1000.000 if len(_time_calls) == 1 else 1000.100

        with patch("src.core.query_engine.time.time", side_effect=fake_time):
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

