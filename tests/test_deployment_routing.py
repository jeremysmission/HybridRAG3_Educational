# ============================================================================
# test_deployment_routing.py -- Tests for deployment discovery + model routing
# ============================================================================
#
# COVERS:
#   Tests 01-04: Deployment discovery (Azure path, OpenAI path, cache, refresh)
#   Tests 05-08: Model selection (best for use case, banned models, unknown)
#   Tests 09-12: Routing table (all use cases, empty list, single model)
#
# RUN:
#   python -m pytest tests/test_deployment_routing.py -v
#
# INTERNET ACCESS: NONE -- all HTTP calls are mocked
# ============================================================================

import sys
import os
import json
import importlib
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# DEPLOYMENT DISCOVERY TESTS (get_available_deployments, refresh_deployments)
# ============================================================================

class TestDeploymentDiscovery:
    """Tests for Azure and OpenAI deployment discovery in llm_router.py."""

    def setup_method(self):
        """Reset credentials module before each test.

        WHY: test_credential_management.py uses importlib.reload() on the
        credentials module inside patch.dict contexts. This can leave the
        module in a state where unittest.mock.patch no longer intercepts
        the 'from ..security.credentials import resolve_credentials' call
        inside get_available_deployments(). Reloading here restores a
        clean module so patching works reliably regardless of test order.
        """
        import src.security.credentials as _cred_mod
        importlib.reload(_cred_mod)

    def _clear_env(self):
        """Remove credential env vars to isolate tests."""
        for var in [
            "HYBRIDRAG_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPEN_AI_KEY",
            "OPENAI_API_KEY", "HYBRIDRAG_API_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
            "OPENAI_API_ENDPOINT", "AZURE_OPENAI_BASE_URL", "OPENAI_BASE_URL",
            "AZURE_OPENAI_DEPLOYMENT", "AZURE_DEPLOYMENT", "OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_NAME", "DEPLOYMENT_NAME", "AZURE_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION", "OPENAI_API_VERSION",
            "API_VERSION",
        ]:
            os.environ.pop(var, None)

    def _make_fake_creds(self, endpoint, api_key="sk-test-key",
                         api_version="2024-02-02"):
        """Build a fake ApiCredentials-like object."""
        creds = MagicMock()
        creds.has_key = bool(api_key)
        creds.has_endpoint = bool(endpoint)
        creds.api_key = api_key
        creds.endpoint = endpoint
        creds.api_version = api_version
        creds.deployment = None
        return creds

    # ------------------------------------------------------------------
    # TEST 01: Azure deployment discovery returns deployment IDs
    # ------------------------------------------------------------------
    def test_01_azure_discovery_returns_deployments(self):
        """Azure endpoint triggers GET /openai/deployments and parses value list."""
        self._clear_env()

        fake_creds = self._make_fake_creds(
            "https://company.openai.azure.com"
        )

        azure_response = {
            "value": [
                {"id": "gpt-4o", "model": "gpt-4o"},
                {"id": "gpt-35-turbo", "model": "gpt-35-turbo"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = azure_response

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        import src.core.llm_router as router_mod
        router_mod._deployment_cache = None  # Clear cache

        with patch("src.security.credentials.resolve_credentials", return_value=fake_creds), \
             patch("httpx.Client", return_value=mock_client):
            result = router_mod.get_available_deployments()

        assert "gpt-4o" in result
        assert "gpt-35-turbo" in result
        assert len(result) == 2
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 02: OpenAI discovery returns model IDs from /models endpoint
    # ------------------------------------------------------------------
    def test_02_openai_discovery_returns_models(self):
        """Non-Azure endpoint triggers GET /models and parses data list."""
        self._clear_env()

        fake_creds = self._make_fake_creds(
            "https://openrouter.ai/api/v1"
        )

        openai_response = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
                {"id": "AI assistant-sonnet-4"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = openai_response

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        import src.core.llm_router as router_mod
        router_mod._deployment_cache = None

        with patch("src.security.credentials.resolve_credentials", return_value=fake_creds), \
             patch("httpx.Client", return_value=mock_client):
            result = router_mod.get_available_deployments()

        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result
        assert "AI assistant-sonnet-4" in result
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 03: Cache returns same result without re-fetching
    # ------------------------------------------------------------------
    def test_03_cache_prevents_refetch(self):
        """Second call returns cached data without hitting the API again."""
        import src.core.llm_router as router_mod
        router_mod._deployment_cache = ["cached-model-a", "cached-model-b"]

        result = router_mod.get_available_deployments()

        assert result == ["cached-model-a", "cached-model-b"]
        # Clean up
        router_mod._deployment_cache = None

    # ------------------------------------------------------------------
    # TEST 04: refresh_deployments() clears cache and re-fetches
    # ------------------------------------------------------------------
    def test_04_refresh_clears_cache_and_refetches(self):
        """refresh_deployments() sets cache to None and calls discovery again."""
        self._clear_env()

        fake_creds = self._make_fake_creds(
            "https://openrouter.ai/api/v1"
        )

        openai_response = {
            "data": [{"id": "refreshed-model"}]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = openai_response

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        import src.core.llm_router as router_mod
        router_mod._deployment_cache = ["stale-model"]

        with patch("src.security.credentials.resolve_credentials", return_value=fake_creds), \
             patch("httpx.Client", return_value=mock_client):
            result = router_mod.refresh_deployments()

        assert "refreshed-model" in result
        assert "stale-model" not in result
        router_mod._deployment_cache = None
        self._clear_env()


# ============================================================================
# MODEL SELECTION TESTS (select_best_model)
# ============================================================================

class TestModelSelection:
    """Tests for select_best_model() in _model_meta.py."""

    # ------------------------------------------------------------------
    # TEST 05: Best model for software engineering (eng-heavy use case)
    # ------------------------------------------------------------------
    def test_05_best_model_for_sw_use_case(self):
        """select_best_model('sw') picks highest eng-score model."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import select_best_model

        deployments = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        result = select_best_model("sw", deployments)

        # gpt-4o has tier_eng=93, gpt-4o-mini=72, gpt-3.5-turbo=50
        assert result == "gpt-4o", f"Expected gpt-4o for SW, got {result}"

    # ------------------------------------------------------------------
    # TEST 06: Best model for general use case (gen-heavy)
    # ------------------------------------------------------------------
    def test_06_best_model_for_gen_use_case(self):
        """select_best_model('gen') picks highest gen-score model."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import select_best_model

        deployments = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        result = select_best_model("gen", deployments)

        # gpt-4o has tier_gen=95, gpt-4o-mini=75, gpt-3.5-turbo=58
        assert result == "gpt-4o", f"Expected gpt-4o for GEN, got {result}"

    # ------------------------------------------------------------------
    # TEST 07: Banned models are excluded from auto-selection
    # ------------------------------------------------------------------
    def test_07_banned_models_excluded(self):
        """Models from banned families (llama, deepseek, qwen) are skipped."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import select_best_model

        deployments = [
            "meta-llama/llama-3.1-405b-instruct",  # Banned
            "deepseek/deepseek-r1",                  # Banned
            "qwen/qwen-2.5-72b",                     # Banned
            "gpt-4o-mini",                            # Allowed
        ]
        result = select_best_model("sw", deployments)

        assert result == "gpt-4o-mini", (
            f"Expected gpt-4o-mini (only non-banned), got {result}"
        )

    # ------------------------------------------------------------------
    # TEST 08: Unknown models get conservative scores, still selectable
    # ------------------------------------------------------------------
    def test_08_unknown_model_gets_conservative_score(self):
        """Models not in KNOWN_MODELS still participate with fallback scores."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import select_best_model

        deployments = ["totally-unknown-model-xyz"]
        result = select_best_model("sw", deployments)

        # Should still return the unknown model (only option)
        assert result == "totally-unknown-model-xyz"


# ============================================================================
# ROUTING TABLE TESTS (get_routing_table)
# ============================================================================

class TestRoutingTable:
    """Tests for get_routing_table() in _model_meta.py."""

    # ------------------------------------------------------------------
    # TEST 09: Routing table has entry for every use case
    # ------------------------------------------------------------------
    def test_09_routing_table_covers_all_use_cases(self):
        """get_routing_table() returns a key for every use case in USE_CASES."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import get_routing_table, USE_CASES

        deployments = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        table = get_routing_table(deployments)

        for uc_key in USE_CASES:
            assert uc_key in table, f"Missing use case key: {uc_key}"
            assert table[uc_key] is not None, (
                f"Use case {uc_key} has no recommended model"
            )

    # ------------------------------------------------------------------
    # TEST 10: Empty deployment list produces all-None routing table
    # ------------------------------------------------------------------
    def test_10_empty_deployments_gives_none_routing(self):
        """get_routing_table([]) returns None for every use case."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import get_routing_table, USE_CASES

        table = get_routing_table([])

        for uc_key in USE_CASES:
            assert uc_key in table
            assert table[uc_key] is None, (
                f"Expected None for {uc_key} with empty deployments"
            )

    # ------------------------------------------------------------------
    # TEST 11: Single model maps to all use cases
    # ------------------------------------------------------------------
    def test_11_single_model_maps_to_all(self):
        """With one non-banned model, every use case maps to it."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import get_routing_table, USE_CASES

        table = get_routing_table(["gpt-4o"])

        for uc_key in USE_CASES:
            assert table[uc_key] == "gpt-4o", (
                f"Expected gpt-4o for {uc_key}, got {table[uc_key]}"
            )

    # ------------------------------------------------------------------
    # TEST 12: Routing table with mixed models picks correctly per use case
    # ------------------------------------------------------------------
    def test_12_mixed_models_route_by_use_case(self):
        """Different use cases may pick different models based on scores."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from _model_meta import get_routing_table

        # gpt-4.1: eng=97, gen=96 (frontier everywhere)
        # gpt-4o-mini: eng=72, gen=75 (budget option)
        # gpt-3.5-turbo: eng=50, gen=58 (legacy)
        table = get_routing_table(["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"])

        # For both eng-heavy and gen-heavy use cases, gpt-4.1 should win
        # because it dominates both dimensions
        assert table["sw"] == "gpt-4.1", f"SW: expected gpt-4.1, got {table['sw']}"
        assert table["gen"] == "gpt-4.1", f"GEN: expected gpt-4.1, got {table['gen']}"
        assert table["pm"] == "gpt-4.1", f"PM: expected gpt-4.1, got {table['pm']}"
