# ============================================================================
# test_credential_management.py -- Tests for credential storage and retrieval
# ============================================================================
#
# COVERS:
#   Tests 01-10: keyring storage, resolve priority, credential_status,
#   clear_credentials, empty string handling
#
# RUN:
#   python -m pytest tests/test_credential_management.py -v
#
# INTERNET ACCESS: NONE -- all keyring calls are mocked
# ============================================================================

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCredentialManagement:
    """Tests for the extended credential system (api_key, endpoint, deployment, api_version)."""

    def _clear_env(self):
        """Remove all credential env vars to isolate keyring tests."""
        env_vars = [
            "HYBRIDRAG_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPEN_AI_KEY",
            "OPENAI_API_KEY", "HYBRIDRAG_API_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
            "OPENAI_API_ENDPOINT", "AZURE_OPENAI_BASE_URL", "OPENAI_BASE_URL",
            "AZURE_OPENAI_DEPLOYMENT", "AZURE_DEPLOYMENT", "OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_NAME", "DEPLOYMENT_NAME", "AZURE_CHAT_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION", "OPENAI_API_VERSION",
            "API_VERSION",
        ]
        for var in env_vars:
            os.environ.pop(var, None)

    # ------------------------------------------------------------------
    # TEST 01: store_deployment() stores value in keyring successfully
    # ------------------------------------------------------------------
    def test_01_store_deployment_in_keyring(self):
        """store_deployment() writes the deployment name to keyring."""
        self._clear_env()
        mock_keyring = MagicMock()
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security.credentials import store_deployment, KEYRING_SERVICE, KEYRING_DEPLOYMENT_NAME
            store_deployment("gpt-4o")
            mock_keyring.set_password.assert_called_once_with(
                KEYRING_SERVICE, KEYRING_DEPLOYMENT_NAME, "gpt-4o"
            )

    # ------------------------------------------------------------------
    # TEST 02: store_api_version() stores value in keyring successfully
    # ------------------------------------------------------------------
    def test_02_store_api_version_in_keyring(self):
        """store_api_version() writes the api version to keyring."""
        self._clear_env()
        mock_keyring = MagicMock()
        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security.credentials import store_api_version, KEYRING_SERVICE, KEYRING_API_VERSION_NAME
            store_api_version("2024-02-02")
            mock_keyring.set_password.assert_called_once_with(
                KEYRING_SERVICE, KEYRING_API_VERSION_NAME, "2024-02-02"
            )

    # ------------------------------------------------------------------
    # TEST 03: resolve_credentials() returns deployment from keyring (priority 1)
    # ------------------------------------------------------------------
    def test_03_resolve_deployment_from_keyring_first(self):
        """Keyring deployment takes priority over env var and URL extraction."""
        self._clear_env()
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = "env-deployment"

        def fake_keyring_get(service, key_name):
            mapping = {
                "azure_deployment": "keyring-deployment",
                "azure_api_key": None,
                "azure_endpoint": None,
                "azure_api_version": None,
            }
            return mapping.get(key_name)

        mock_keyring = MagicMock()
        mock_keyring.get_password = fake_keyring_get

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            # Force reimport to pick up mock
            import importlib
            importlib.reload(creds_mod)
            result = creds_mod.resolve_credentials()

        assert result.deployment == "keyring-deployment", (
            f"Expected keyring value, got: {result.deployment}"
        )
        assert result.source_deployment == "keyring"
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 04: resolve_credentials() falls back to env var if keyring empty
    # ------------------------------------------------------------------
    def test_04_resolve_deployment_from_env_fallback(self):
        """If keyring has no deployment, env var is used."""
        self._clear_env()
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = "env-deploy"

        def fake_keyring_get(service, key_name):
            return None  # keyring is empty

        mock_keyring = MagicMock()
        mock_keyring.get_password = fake_keyring_get

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            result = creds_mod.resolve_credentials()

        assert result.deployment == "env-deploy"
        assert "env:" in result.source_deployment
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 05: resolve_credentials() falls back to URL extraction if env empty
    # ------------------------------------------------------------------
    def test_05_resolve_deployment_from_url_extraction(self):
        """If keyring and env are empty, deployment extracted from endpoint URL."""
        self._clear_env()

        def fake_keyring_get(service, key_name):
            if key_name == "azure_endpoint":
                return "https://company.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
            return None

        mock_keyring = MagicMock()
        mock_keyring.get_password = fake_keyring_get

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            result = creds_mod.resolve_credentials()

        assert result.deployment == "gpt-4o"
        assert result.source_deployment == "extracted_from_url"
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 06: all four values present -- credential_status() reports complete
    # ------------------------------------------------------------------
    def test_06_credential_status_all_present(self):
        """credential_status() includes deployment and api_version when all set."""
        self._clear_env()

        def fake_keyring_get(service, key_name):
            mapping = {
                "azure_api_key": "sk-test-key-12345678",
                "azure_endpoint": "https://company.openai.azure.com",
                "azure_deployment": "gpt-4o",
                "azure_api_version": "2024-02-02",
            }
            return mapping.get(key_name)

        mock_keyring = MagicMock()
        mock_keyring.get_password = fake_keyring_get

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            status = creds_mod.credential_status()

        assert status["api_key_set"] is True
        assert status["api_endpoint_set"] is True
        assert status["deployment_set"] is True
        assert status["api_version_set"] is True
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 07: any one value missing -- credential_status() reports incomplete
    # ------------------------------------------------------------------
    def test_07_credential_status_missing_deployment(self):
        """credential_status() shows deployment_set=False when deployment is missing."""
        self._clear_env()

        def fake_keyring_get(service, key_name):
            mapping = {
                "azure_api_key": "sk-test-key-12345678",
                "azure_endpoint": "https://company.openai.azure.com",
                "azure_deployment": None,
                "azure_api_version": "2024-02-02",
            }
            return mapping.get(key_name)

        mock_keyring = MagicMock()
        mock_keyring.get_password = fake_keyring_get

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            status = creds_mod.credential_status()

        assert status["deployment_set"] is False
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 08: clear_credentials() removes all four keyring values cleanly
    # ------------------------------------------------------------------
    def test_08_clear_credentials_removes_all_four(self):
        """clear_credentials() deletes deployment and api_version from keyring too."""
        self._clear_env()
        mock_keyring = MagicMock()
        mock_keyring.delete_password = MagicMock()

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            creds_mod.clear_credentials()

        deleted_keys = [
            call.args[1]
            for call in mock_keyring.delete_password.call_args_list
        ]
        assert "azure_api_key" in deleted_keys
        assert "azure_endpoint" in deleted_keys
        assert "azure_deployment" in deleted_keys
        assert "azure_api_version" in deleted_keys
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 09: credential_status() output includes deployment and api_version keys
    # ------------------------------------------------------------------
    def test_09_credential_status_has_all_keys(self):
        """credential_status() dict has deployment_set and api_version_set keys."""
        self._clear_env()

        mock_keyring = MagicMock()
        mock_keyring.get_password = MagicMock(return_value=None)

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            status = creds_mod.credential_status()

        assert "deployment_set" in status
        assert "api_version_set" in status
        assert "deployment_source" in status
        assert "api_version_source" in status
        self._clear_env()

    # ------------------------------------------------------------------
    # TEST 10: empty string values treated same as missing
    # ------------------------------------------------------------------
    def test_10_empty_string_treated_as_missing(self):
        """Empty string in keyring is not accepted as a valid value."""
        self._clear_env()

        def fake_keyring_get(service, key_name):
            mapping = {
                "azure_api_key": "",
                "azure_endpoint": "   ",
                "azure_deployment": "",
                "azure_api_version": "  ",
            }
            return mapping.get(key_name)

        mock_keyring = MagicMock()
        mock_keyring.get_password = fake_keyring_get

        with patch.dict("sys.modules", {"keyring": mock_keyring}):
            from src.security import credentials as creds_mod
            import importlib
            importlib.reload(creds_mod)
            result = creds_mod.resolve_credentials()

        assert not result.has_key, "Empty string should not count as having a key"
        assert not result.has_endpoint, "Whitespace should not count as having an endpoint"
        assert result.deployment is None or result.deployment == "", (
            "Empty deployment should be treated as missing"
        )
        self._clear_env()
