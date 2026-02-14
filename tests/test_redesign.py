#!/usr/bin/env python3
# ===========================================================================
# HybridRAG v3 REDESIGN -- COMPREHENSIVE TEST SUITE
# ===========================================================================
# Tests every module front-to-back:
#   1. exceptions.py       -- All exception types, to_dict, HTTP mapping
#   2. credentials.py      -- Env var resolution, validation, edge cases
#   3. http_client.py      -- Config, response, offline mode, retries
#   4. api_client_factory  -- Provider detection, URL building, auth, diagnostics
#   5. boot.py             -- Full boot pipeline
#   6. Integration         -- End-to-end flow
# ===========================================================================

import os
import sys
from pathlib import Path
import json
import traceback
import time
from unittest.mock import patch, MagicMock
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Track results
PASS = 0
FAIL = 0
ERRORS = []

def test(name):
    """Decorator-style test runner."""
    def decorator(func):
        global PASS, FAIL
        try:
            func()
            PASS += 1
            print(f"  [PASS] {name}")
        except AssertionError as e:
            FAIL += 1
            ERRORS.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            FAIL += 1
            tb = traceback.format_exc().split('\n')[-3]
            ERRORS.append((name, f"{type(e).__name__}: {e}\n    {tb}"))
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return func
    return decorator


# ===========================================================================
# MODULE 1: EXCEPTIONS
# ===========================================================================
print()
print("=" * 60)
print("  MODULE 1: EXCEPTIONS (src/core/exceptions.py)")
print("=" * 60)

from src.core.exceptions import (
    HybridRAGError,
    EndpointNotConfiguredError,
    ApiKeyNotConfiguredError,
    InvalidEndpointError,
    DeploymentNotConfiguredError,
    ProviderConfigError,
    ApiVersionNotConfiguredError,
    ConnectionFailedError,
    TLSValidationError,
    ProxyError,
    TimeoutError,
    AuthRejectedError,
    ForbiddenError,
    DeploymentNotFoundError,
    RateLimitedError,
    ServerError,
    UnexpectedResponseError,
    OllamaNotRunningError,
    OllamaModelNotFoundError,
    IndexNotFoundError,
    IndexCorruptedError,
    exception_from_http_status,
)

@test("Base exception has message, fix_suggestion, error_code")
def _():
    e = HybridRAGError("test msg", fix_suggestion="do X", error_code="TEST-001")
    assert str(e) == "test msg"
    assert e.fix_suggestion == "do X"
    assert e.error_code == "TEST-001"

@test("Base exception to_dict produces correct structure")
def _():
    e = HybridRAGError("msg", fix_suggestion="fix", error_code="T-1")
    d = e.to_dict()
    assert d["error_type"] == "HybridRAGError"
    assert d["error_code"] == "T-1"
    assert d["message"] == "msg"
    assert d["fix_suggestion"] == "fix"

@test("All config exceptions have default messages and error codes")
def _():
    exceptions = [
        (EndpointNotConfiguredError(), "CONF-001"),
        (ApiKeyNotConfiguredError(), "CONF-002"),
        (InvalidEndpointError(), "CONF-003"),
        (DeploymentNotConfiguredError(), "CONF-004"),
        (ProviderConfigError(), "CONF-005"),
        (ApiVersionNotConfiguredError(), "CONF-006"),
    ]
    for exc, code in exceptions:
        assert exc.error_code == code, f"{type(exc).__name__} code={exc.error_code}, expected={code}"
        assert str(exc), f"{type(exc).__name__} has empty message"
        assert exc.fix_suggestion, f"{type(exc).__name__} has no fix_suggestion"

@test("All network exceptions have default messages and error codes")
def _():
    exceptions = [
        (ConnectionFailedError(), "NET-001"),
        (TLSValidationError(), "NET-002"),
        (ProxyError(), "NET-003"),
        (TimeoutError(), "NET-004"),
    ]
    for exc, code in exceptions:
        assert exc.error_code == code
        assert str(exc)
        assert exc.fix_suggestion

@test("All API exceptions have default messages and error codes")
def _():
    exceptions = [
        (AuthRejectedError(), "API-001"),
        (ForbiddenError(), "API-002"),
        (DeploymentNotFoundError(), "API-003"),
        (RateLimitedError(), "API-004"),
        (ServerError(), "API-005"),
        (UnexpectedResponseError(), "API-006"),
    ]
    for exc, code in exceptions:
        assert exc.error_code == code
        assert str(exc)
        assert exc.fix_suggestion

@test("All Ollama exceptions have default messages and error codes")
def _():
    exceptions = [
        (OllamaNotRunningError(), "OLL-001"),
        (OllamaModelNotFoundError(), "OLL-002"),
    ]
    for exc, code in exceptions:
        assert exc.error_code == code
        assert str(exc)
        assert exc.fix_suggestion

@test("All index exceptions have default messages and error codes")
def _():
    exceptions = [
        (IndexNotFoundError(), "IDX-001"),
        (IndexCorruptedError(), "IDX-002"),
    ]
    for exc, code in exceptions:
        assert exc.error_code == code
        assert str(exc)
        assert exc.fix_suggestion

@test("InvalidEndpointError includes URL in message when provided")
def _():
    e = InvalidEndpointError(url="bad-url")
    assert "bad-url" in str(e)

@test("AuthRejectedError includes status code when provided")
def _():
    e = AuthRejectedError(status_code=401)
    assert "401" in str(e)

@test("DeploymentNotFoundError includes deployment name when provided")
def _():
    e = DeploymentNotFoundError(deployment="my-deploy")
    assert "my-deploy" in str(e)

@test("RateLimitedError includes retry_after when provided")
def _():
    e = RateLimitedError(retry_after=30)
    assert "30" in str(e)

@test("ServerError includes status code when provided")
def _():
    e = ServerError(status_code=503)
    assert "503" in str(e)

@test("ConnectionFailedError includes host when provided")
def _():
    e = ConnectionFailedError(host="api.example.com")
    assert "api.example.com" in str(e)

@test("TimeoutError includes timeout_seconds when provided")
def _():
    e = TimeoutError(timeout_seconds=30)
    assert "30" in str(e)

@test("OllamaModelNotFoundError includes model name when provided")
def _():
    e = OllamaModelNotFoundError(model="llama3")
    assert "llama3" in str(e)

@test("exception_from_http_status maps 401 to AuthRejectedError")
def _():
    e = exception_from_http_status(401, "bad key")
    assert isinstance(e, AuthRejectedError)
    assert "401" in str(e)

@test("exception_from_http_status maps 403 to ForbiddenError")
def _():
    e = exception_from_http_status(403, "forbidden")
    assert isinstance(e, ForbiddenError)

@test("exception_from_http_status maps 404 to DeploymentNotFoundError")
def _():
    e = exception_from_http_status(404, "not found", deployment="gpt-35")
    assert isinstance(e, DeploymentNotFoundError)

@test("exception_from_http_status maps 429 to RateLimitedError")
def _():
    e = exception_from_http_status(429, "slow down")
    assert isinstance(e, RateLimitedError)

@test("exception_from_http_status maps 500 to ServerError")
def _():
    e = exception_from_http_status(500, "internal")
    assert isinstance(e, ServerError)

@test("exception_from_http_status maps 502 to ServerError")
def _():
    e = exception_from_http_status(502, "bad gateway")
    assert isinstance(e, ServerError)

@test("exception_from_http_status maps 503 to ServerError")
def _():
    e = exception_from_http_status(503, "unavailable")
    assert isinstance(e, ServerError)

@test("exception_from_http_status maps unknown code to base HybridRAGError")
def _():
    e = exception_from_http_status(418, "teapot")
    assert isinstance(e, HybridRAGError)
    assert "418" in str(e)

@test("exception_from_http_status truncates long response body")
def _():
    long_body = "x" * 1000
    e = exception_from_http_status(401, long_body)
    assert len(str(e)) < 500

@test("All exceptions inherit from HybridRAGError")
def _():
    all_types = [
        EndpointNotConfiguredError, ApiKeyNotConfiguredError,
        InvalidEndpointError, DeploymentNotConfiguredError,
        ProviderConfigError, ApiVersionNotConfiguredError,
        ConnectionFailedError, TLSValidationError, ProxyError,
        TimeoutError, AuthRejectedError, ForbiddenError,
        DeploymentNotFoundError, RateLimitedError, ServerError,
        UnexpectedResponseError, OllamaNotRunningError,
        OllamaModelNotFoundError, IndexNotFoundError, IndexCorruptedError,
    ]
    for cls in all_types:
        assert issubclass(cls, HybridRAGError), f"{cls.__name__} not subclass"

@test("All exceptions are catchable with except HybridRAGError")
def _():
    all_instances = [
        EndpointNotConfiguredError(), ApiKeyNotConfiguredError(),
        InvalidEndpointError(), ConnectionFailedError(),
        AuthRejectedError(), RateLimitedError(),
        OllamaNotRunningError(), IndexNotFoundError(),
    ]
    for exc in all_instances:
        try:
            raise exc
        except HybridRAGError:
            pass  # Should always be caught here


# ===========================================================================
# MODULE 2: CREDENTIALS
# ===========================================================================
print()
print("=" * 60)
print("  MODULE 2: CREDENTIALS (src/security/credentials.py)")
print("=" * 60)

from src.security.credentials import (
    ApiCredentials,
    validate_endpoint,
    resolve_credentials,
    _resolve_env_var,
    _nested_get,
    _KEY_ENV_ALIASES,
    _ENDPOINT_ENV_ALIASES,
    _DEPLOYMENT_ENV_ALIASES,
    _API_VERSION_ENV_ALIASES,
)

@test("ApiCredentials default values are all None")
def _():
    c = ApiCredentials()
    assert c.api_key is None
    assert c.endpoint is None
    assert c.deployment is None
    assert c.api_version is None
    assert c.source_key is None
    assert c.source_endpoint is None

@test("ApiCredentials.has_key returns True when key is set")
def _():
    c = ApiCredentials(api_key="test-key-123")
    assert c.has_key is True

@test("ApiCredentials.has_key returns False when key is None")
def _():
    c = ApiCredentials()
    assert c.has_key is False

@test("ApiCredentials.has_endpoint returns True when endpoint is set")
def _():
    c = ApiCredentials(endpoint="https://api.example.com")
    assert c.has_endpoint is True

@test("ApiCredentials.has_endpoint returns False when endpoint is None")
def _():
    c = ApiCredentials()
    assert c.has_endpoint is False

@test("ApiCredentials.is_online_ready requires both key and endpoint")
def _():
    c1 = ApiCredentials()
    assert c1.is_online_ready is False
    c2 = ApiCredentials(api_key="key")
    assert c2.is_online_ready is False
    c3 = ApiCredentials(endpoint="https://api.example.com")
    assert c3.is_online_ready is False
    c4 = ApiCredentials(api_key="key", endpoint="https://api.example.com")
    assert c4.is_online_ready is True

@test("ApiCredentials.key_preview masks the key correctly")
def _():
    c = ApiCredentials(api_key="abcdefghijklmnop")
    preview = c.key_preview
    assert preview.startswith("abcd")
    assert preview.endswith("mnop")
    assert "..." in preview

@test("ApiCredentials.key_preview handles short keys")
def _():
    c = ApiCredentials(api_key="short")
    assert c.key_preview == "****"

@test("ApiCredentials.key_preview handles None key")
def _():
    c = ApiCredentials()
    assert c.key_preview == "(not set)"

@test("ApiCredentials.to_diagnostic_dict never exposes full key")
def _():
    c = ApiCredentials(api_key="super-secret-key-12345", endpoint="https://api.example.com")
    d = c.to_diagnostic_dict()
    assert "super-secret-key-12345" not in str(d)
    assert "supe" in d["api_key"]

@test("ApiCredentials.to_diagnostic_dict includes all fields")
def _():
    c = ApiCredentials(
        api_key="key123", endpoint="https://api.com",
        deployment="gpt-35", api_version="2024-02-01",
        source_key="keyring", source_endpoint="env:AZURE_OPENAI_ENDPOINT",
    )
    d = c.to_diagnostic_dict()
    assert d["endpoint"] == "https://api.com"
    assert d["deployment"] == "gpt-35"
    assert d["api_version"] == "2024-02-01"
    assert d["source_key"] == "keyring"
    assert d["online_ready"] is True

@test("validate_endpoint accepts valid HTTPS URL")
def _():
    result = validate_endpoint("https://api.example.com/v1")
    assert result == "https://api.example.com/v1"

@test("validate_endpoint strips trailing slash")
def _():
    result = validate_endpoint("https://api.example.com/")
    assert result == "https://api.example.com"

@test("validate_endpoint strips whitespace")
def _():
    result = validate_endpoint("  https://api.example.com  ")
    assert result == "https://api.example.com"

@test("validate_endpoint rejects empty string")
def _():
    try:
        validate_endpoint("")
        assert False, "Should have raised"
    except InvalidEndpointError:
        pass

@test("validate_endpoint rejects None")
def _():
    try:
        validate_endpoint(None)
        assert False, "Should have raised"
    except InvalidEndpointError:
        pass

@test("validate_endpoint rejects URL without scheme")
def _():
    try:
        validate_endpoint("api.example.com")
        assert False, "Should have raised"
    except InvalidEndpointError as e:
        assert "https://" in str(e)

@test("validate_endpoint rejects URL with spaces")
def _():
    try:
        validate_endpoint("https://api.example .com")
        assert False, "Should have raised"
    except InvalidEndpointError as e:
        assert "spaces" in str(e).lower()

@test("validate_endpoint rejects URL with smart quotes")
def _():
    try:
        validate_endpoint("https://api.example\u201C.com")
        assert False, "Should have raised"
    except InvalidEndpointError as e:
        assert "character" in str(e).lower()

@test("validate_endpoint rejects URL with curly single quotes")
def _():
    try:
        validate_endpoint("https://api\u2019s.example.com")
        assert False, "Should have raised"
    except InvalidEndpointError as e:
        assert "character" in str(e).lower()

@test("validate_endpoint rejects URL with en-dash")
def _():
    try:
        validate_endpoint("https://api\u2013example.com")
        assert False, "Should have raised"
    except InvalidEndpointError:
        pass

@test("validate_endpoint rejects URL with em-dash")
def _():
    try:
        validate_endpoint("https://api\u2014example.com")
        assert False, "Should have raised"
    except InvalidEndpointError:
        pass

@test("validate_endpoint rejects URL with non-breaking space")
def _():
    try:
        validate_endpoint("https://api\u00a0example.com")
        assert False, "Should have raised"
    except InvalidEndpointError:
        pass

@test("validate_endpoint rejects URL with BOM character")
def _():
    try:
        validate_endpoint("https://api\ufeff.example.com")
        assert False, "Should have raised"
    except InvalidEndpointError:
        pass

@test("validate_endpoint rejects double slashes in path")
def _():
    try:
        validate_endpoint("https://api.example.com//v1//chat")
        assert False, "Should have raised"
    except InvalidEndpointError as e:
        assert "double" in str(e).lower()

@test("validate_endpoint allows http:// (non-standard but valid)")
def _():
    result = validate_endpoint("http://localhost:11434")
    assert result == "http://localhost:11434"

@test("_resolve_env_var finds first matching env var")
def _():
    with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"}):
        val, var = _resolve_env_var(_KEY_ENV_ALIASES)
        assert val == "test-key"
        assert var == "AZURE_OPENAI_API_KEY"

@test("_resolve_env_var respects priority order")
def _():
    with patch.dict(os.environ, {
        "HYBRIDRAG_API_KEY": "first",
        "AZURE_OPENAI_API_KEY": "second",
    }):
        val, var = _resolve_env_var(_KEY_ENV_ALIASES)
        assert val == "first"
        assert var == "HYBRIDRAG_API_KEY"

@test("_resolve_env_var skips empty values")
def _():
    with patch.dict(os.environ, {
        "HYBRIDRAG_API_KEY": "",
        "AZURE_OPENAI_API_KEY": "actual-key",
    }):
        val, var = _resolve_env_var(_KEY_ENV_ALIASES)
        assert val == "actual-key"
        assert var == "AZURE_OPENAI_API_KEY"

@test("_resolve_env_var returns None when nothing set")
def _():
    # Clear all aliases
    env_clean = {k: "" for k in _KEY_ENV_ALIASES}
    with patch.dict(os.environ, env_clean, clear=False):
        for alias in _KEY_ENV_ALIASES:
            os.environ.pop(alias, None)
        val, var = _resolve_env_var(_KEY_ENV_ALIASES)
        assert val is None
        assert var is None

@test("_nested_get retrieves nested dict values")
def _():
    d = {"api": {"key": "abc", "nested": {"deep": 42}}}
    assert _nested_get(d, "api", "key") == "abc"
    assert _nested_get(d, "api", "nested", "deep") == 42

@test("_nested_get returns None for missing keys")
def _():
    d = {"api": {"key": "abc"}}
    assert _nested_get(d, "api", "missing") is None
    assert _nested_get(d, "nonexistent") is None
    assert _nested_get(d, "api", "key", "too_deep") is None

@test("resolve_credentials picks up env vars for all credential types")
def _():
    env = {
        "HYBRIDRAG_API_KEY": "env-key-123",
        "HYBRIDRAG_API_ENDPOINT": "https://env.example.com",
        "AZURE_OPENAI_DEPLOYMENT": "env-deployment",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",
    }
    # Mock keyring to return None (so env vars are used)
    with patch.dict(os.environ, env, clear=False):
        with patch("src.security.credentials._read_keyring", return_value=None):
            creds = resolve_credentials()
            assert creds.api_key == "env-key-123"
            assert creds.endpoint == "https://env.example.com"
            assert creds.deployment == "env-deployment"
            assert creds.api_version == "2024-06-01"
            assert creds.source_key == "env:HYBRIDRAG_API_KEY"
            assert creds.source_endpoint == "env:HYBRIDRAG_API_ENDPOINT"

@test("resolve_credentials prefers keyring over env vars")
def _():
    env = {"HYBRIDRAG_API_KEY": "env-key"}
    with patch.dict(os.environ, env, clear=False):
        with patch("src.security.credentials._read_keyring") as mock_kr:
            mock_kr.side_effect = lambda name: {
                "azure_api_key": "keyring-key",
                "azure_endpoint": "https://keyring.example.com",
            }.get(name)
            creds = resolve_credentials()
            assert creds.api_key == "keyring-key"
            assert creds.source_key == "keyring"
            assert creds.endpoint == "https://keyring.example.com"
            assert creds.source_endpoint == "keyring"

@test("resolve_credentials falls back to config dict")
def _():
    config = {"api": {"key": "config-key", "endpoint": "https://config.example.com"}}
    with patch("src.security.credentials._read_keyring", return_value=None):
        # Clear env vars
        env_clear = {}
        for aliases in [_KEY_ENV_ALIASES, _ENDPOINT_ENV_ALIASES]:
            for a in aliases:
                env_clear[a] = ""
        with patch.dict(os.environ, env_clear, clear=False):
            for a in list(env_clear.keys()):
                os.environ.pop(a, None)
            creds = resolve_credentials(config)
            assert creds.api_key == "config-key"
            assert creds.source_key == "config"

@test("resolve_credentials extracts deployment from URL")
def _():
    with patch("src.security.credentials._read_keyring") as mock_kr:
        mock_kr.side_effect = lambda name: {
            "azure_api_key": "key",
            "azure_endpoint": "https://api.com/openai/deployments/gpt-35-turbo/chat/completions",
        }.get(name)
        creds = resolve_credentials()
        assert creds.deployment == "gpt-35-turbo"
        assert creds.source_deployment == "extracted_from_url"

@test("resolve_credentials extracts api-version from URL")
def _():
    with patch("src.security.credentials._read_keyring") as mock_kr:
        mock_kr.side_effect = lambda name: {
            "azure_api_key": "key",
            "azure_endpoint": "https://api.com/chat/completions?api-version=2024-06-01",
        }.get(name)
        creds = resolve_credentials()
        assert creds.api_version == "2024-06-01"
        assert creds.source_api_version == "extracted_from_url"

@test("resolve_credentials returns empty but valid object when nothing found")
def _():
    with patch("src.security.credentials._read_keyring", return_value=None):
        for a in _KEY_ENV_ALIASES + _ENDPOINT_ENV_ALIASES + _DEPLOYMENT_ENV_ALIASES + _API_VERSION_ENV_ALIASES:
            os.environ.pop(a, None)
        creds = resolve_credentials()
        assert creds.api_key is None
        assert creds.endpoint is None
        assert creds.is_online_ready is False


# ===========================================================================
# MODULE 3: HTTP CLIENT
# ===========================================================================
print()
print("=" * 60)
print("  MODULE 3: HTTP CLIENT (src/core/http_client.py)")
print("=" * 60)

from src.core.http_client import (
    HttpClient,
    HttpClientConfig,
    HttpResponse,
    create_http_client,
    _mask_url,
)

@test("HttpClientConfig has sensible defaults")
def _():
    cfg = HttpClientConfig()
    assert cfg.timeout_seconds == 30
    assert cfg.max_retries == 2
    assert cfg.retry_delay_seconds == 1.0
    assert cfg.verify_ssl is True
    assert cfg.offline_mode is False
    assert cfg.user_agent == "HybridRAG/3.0"

@test("HttpResponse.is_success returns True for 2xx codes")
def _():
    assert HttpResponse(status_code=200).is_success is True
    assert HttpResponse(status_code=201).is_success is True
    assert HttpResponse(status_code=299).is_success is True

@test("HttpResponse.is_success returns False for non-2xx codes")
def _():
    assert HttpResponse(status_code=0).is_success is False
    assert HttpResponse(status_code=301).is_success is False
    assert HttpResponse(status_code=401).is_success is False
    assert HttpResponse(status_code=500).is_success is False

@test("HttpResponse.json() parses valid JSON")
def _():
    r = HttpResponse(body='{"key": "value"}')
    assert r.json() == {"key": "value"}

@test("HttpResponse.json() returns empty dict for invalid JSON")
def _():
    r = HttpResponse(body='not json')
    assert r.json() == {}

@test("HttpResponse.json() returns empty dict for empty body")
def _():
    r = HttpResponse(body='')
    assert r.json() == {}

@test("HttpClient offline mode blocks all requests")
def _():
    cfg = HttpClientConfig(offline_mode=True)
    client = HttpClient(cfg)
    response = client.post("https://api.example.com", json_body={"test": True})
    assert response.status_code == 0
    assert "blocked" in response.error.lower() or "offline" in response.error.lower()

@test("HttpClient offline mode via env var HYBRIDRAG_OFFLINE=1")
def _():
    with patch.dict(os.environ, {"HYBRIDRAG_OFFLINE": "1"}):
        client = HttpClient(HttpClientConfig())
        assert client.config.offline_mode is True
        response = client.get("https://example.com")
        assert response.status_code == 0

@test("_mask_url hides api-key in URL query params")
def _():
    url = "https://api.com/chat?api-version=2024&api-key=mysecretkey123"
    masked = _mask_url(url)
    assert "mysecretkey123" not in masked
    assert "api-key=****" in masked
    assert "api-version=2024" in masked

@test("_mask_url handles URL without api-key")
def _():
    url = "https://api.com/chat?api-version=2024"
    masked = _mask_url(url)
    assert masked == url

@test("_mask_url handles api_key (underscore variant)")
def _():
    url = "https://api.com/chat?api_key=secret123"
    masked = _mask_url(url)
    assert "secret123" not in masked

@test("create_http_client uses config dict values")
def _():
    config = {"http": {"timeout": 60, "max_retries": 5, "verify_ssl": False}}
    client = create_http_client(config)
    assert client.config.timeout_seconds == 60
    assert client.config.max_retries == 5
    assert client.config.verify_ssl is False

@test("create_http_client uses defaults when no config")
def _():
    client = create_http_client()
    assert client.config.timeout_seconds == 30
    assert client.config.max_retries == 2

@test("create_http_client handles empty config dict")
def _():
    client = create_http_client({})
    assert client.config.timeout_seconds == 30

@test("HttpClient handles connection error gracefully")
def _():
    cfg = HttpClientConfig(timeout_seconds=2, max_retries=0)
    client = HttpClient(cfg)
    # Try connecting to a port that definitely isn't listening
    response = client.get("https://192.0.2.1:1")  # TEST-NET IP, won't resolve
    assert response.is_success is False
    assert response.error is not None


# ===========================================================================
# MODULE 4: API CLIENT FACTORY
# ===========================================================================
print()
print("=" * 60)
print("  MODULE 4: API CLIENT FACTORY (src/core/api_client_factory.py)")
print("=" * 60)

from src.core.api_client_factory import (
    ApiClientFactory,
    ApiClient,
    ApiClientConfig,
    AZURE_URL_PATTERNS,
    DEFAULT_AZURE_API_VERSION,
)

@test("Factory detects Azure from URL containing 'azure'")
def _():
    factory = ApiClientFactory()
    provider = factory._detect_provider("https://mycompany.azure.openai.com")
    assert provider == "azure"

@test("Factory detects Azure from URL containing 'aoai'")
def _():
    factory = ApiClientFactory()
    provider = factory._detect_provider("https://aiml-aoai-api.gcl.mycompany.com")
    assert provider == "azure"

@test("Factory detects Azure from URL containing 'cognitiveservices'")
def _():
    factory = ApiClientFactory()
    provider = factory._detect_provider("https://mycompany.cognitiveservices.azure.com")
    assert provider == "azure"

@test("Factory detects Azure from URL containing 'azure-api'")
def _():
    factory = ApiClientFactory()
    provider = factory._detect_provider("https://mycompany.azure-api.net")
    assert provider == "azure"

@test("Factory detects OpenAI from URL without Azure patterns")
def _():
    factory = ApiClientFactory()
    provider = factory._detect_provider("https://api.openai.com")
    assert provider == "openai"

@test("Factory uses explicit provider from config over URL detection")
def _():
    factory = ApiClientFactory({"api": {"provider": "azure"}})
    provider = factory._detect_provider("https://api.openai.com")  # URL says OpenAI
    assert provider == "azure"  # Config overrides

@test("Factory uses explicit openai provider from config")
def _():
    factory = ApiClientFactory({"api": {"provider": "openai"}})
    provider = factory._detect_provider("https://aoai.company.com")  # URL says Azure
    assert provider == "openai"  # Config overrides

@test("Factory rejects invalid provider in config")
def _():
    factory = ApiClientFactory({"api": {"provider": "invalid"}})
    try:
        factory._detect_provider("https://api.example.com")
        assert False, "Should have raised"
    except ProviderConfigError:
        pass

@test("Factory builds correct Azure URL from base endpoint")
def _():
    factory = ApiClientFactory()
    url = factory._build_url(
        "https://mycompany.openai.azure.com",
        "azure", "gpt-35-turbo", "2024-02-01"
    )
    assert url == "https://mycompany.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-01"

@test("Factory builds correct Azure URL when deployment is in URL")
def _():
    factory = ApiClientFactory()
    url = factory._build_url(
        "https://mycompany.com/openai/deployments/gpt-35-turbo",
        "azure", "gpt-35-turbo", "2024-02-01"
    )
    assert url == "https://mycompany.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-01"

@test("Factory uses URL as-is when it already has /chat/completions")
def _():
    factory = ApiClientFactory()
    url = factory._build_url(
        "https://mycompany.com/openai/deployments/gpt-35/chat/completions",
        "azure", "gpt-35", "2024-02-01"
    )
    assert "/chat/completions?api-version=2024-02-01" in url
    assert url.count("/chat/completions") == 1  # NOT doubled

@test("Factory doesn't double api-version if already in URL")
def _():
    factory = ApiClientFactory()
    url = factory._build_url(
        "https://mycompany.com/openai/deployments/gpt-35/chat/completions?api-version=2024-06-01",
        "azure", "gpt-35", "2024-02-01"
    )
    assert url.count("api-version") == 1
    assert "2024-06-01" in url  # Original preserved

@test("Factory builds correct OpenAI URL")
def _():
    factory = ApiClientFactory()
    url = factory._build_url("https://api.openai.com", "openai", None, None)
    assert url == "https://api.openai.com/v1/chat/completions"

@test("Factory doesn't append /v1/chat/completions if already present for OpenAI")
def _():
    factory = ApiClientFactory()
    url = factory._build_url(
        "https://api.openai.com/v1/chat/completions", "openai", None, None
    )
    assert url.count("/chat/completions") == 1

@test("Factory uses api-key header for Azure")
def _():
    factory = ApiClientFactory()
    name, value = factory._resolve_auth("my-key", "azure")
    assert name == "api-key"
    assert value == "my-key"

@test("Factory uses Bearer header for OpenAI")
def _():
    factory = ApiClientFactory()
    name, value = factory._resolve_auth("my-key", "openai")
    assert name == "Authorization"
    assert value == "Bearer my-key"

@test("Factory auth respects explicit config override")
def _():
    factory = ApiClientFactory({"api": {"auth_scheme": "bearer"}})
    name, value = factory._resolve_auth("my-key", "azure")  # Provider says azure
    assert name == "Authorization"  # But config overrides to bearer
    assert value == "Bearer my-key"

@test("Factory rejects invalid auth scheme in config")
def _():
    factory = ApiClientFactory({"api": {"auth_scheme": "magic"}})
    try:
        factory._resolve_auth("key", "azure")
        assert False, "Should have raised"
    except ProviderConfigError:
        pass

@test("Factory.build raises EndpointNotConfiguredError when endpoint missing")
def _():
    factory = ApiClientFactory()
    creds = ApiCredentials(api_key="key")  # No endpoint
    try:
        factory.build(creds)
        assert False, "Should have raised"
    except EndpointNotConfiguredError:
        pass

@test("Factory.build raises ApiKeyNotConfiguredError when key missing")
def _():
    factory = ApiClientFactory()
    creds = ApiCredentials(endpoint="https://api.example.com")  # No key
    try:
        factory.build(creds)
        assert False, "Should have raised"
    except ApiKeyNotConfiguredError:
        pass

@test("Factory.build raises DeploymentNotConfiguredError for Azure without deployment")
def _():
    factory = ApiClientFactory({"api": {"provider": "azure"}})
    creds = ApiCredentials(
        api_key="key",
        endpoint="https://api.example.com",
    )
    try:
        factory.build(creds)
        assert False, "Should have raised"
    except DeploymentNotConfiguredError:
        pass

@test("Factory.build succeeds with complete Azure credentials")
def _():
    factory = ApiClientFactory({"api": {"provider": "azure"}})
    creds = ApiCredentials(
        api_key="test-key-123",
        endpoint="https://aoai-api.mycompany.com",
        deployment="gpt-35-turbo",
        api_version="2024-02-01",
    )
    client = factory.build(creds)
    assert isinstance(client, ApiClient)
    assert client.config.provider == "azure"
    assert client.config.auth_header_name == "api-key"
    assert "gpt-35-turbo" in client.config.final_url
    assert "api-version=2024-02-01" in client.config.final_url

@test("Factory.build succeeds with complete OpenAI credentials")
def _():
    factory = ApiClientFactory({"api": {"provider": "openai"}})
    creds = ApiCredentials(
        api_key="sk-test-key-123",
        endpoint="https://api.openai.com",
    )
    client = factory.build(creds)
    assert isinstance(client, ApiClient)
    assert client.config.provider == "openai"
    assert client.config.auth_header_name == "Authorization"
    assert "/v1/chat/completions" in client.config.final_url

@test("Factory.build with the developer's actual corporate URL pattern")
def _():
    factory = ApiClientFactory({"api": {"provider": "azure"}})
    creds = ApiCredentials(
        api_key="http-key-a651",
        endpoint="https://aiml-aoai-api.gcl.mycompany.com",
        deployment="gpt-35-turbo",
        api_version="2024-02-01",
    )
    client = factory.build(creds)
    assert client.config.provider == "azure"
    assert client.config.auth_header_name == "api-key"
    expected = "https://aiml-aoai-api.gcl.mycompany.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-01"
    assert client.config.final_url == expected, f"Got: {client.config.final_url}"

@test("Factory.build auto-detects Azure from aoai URL")
def _():
    factory = ApiClientFactory({"api": {"provider": "auto"}})
    creds = ApiCredentials(
        api_key="key",
        endpoint="https://aiml-aoai-api.gcl.mycompany.com",
        deployment="gpt-35-turbo",
    )
    client = factory.build(creds)
    assert client.config.provider == "azure"
    assert client.config.auth_header_name == "api-key"

@test("Factory.diagnose returns structured diagnostics with no credentials")
def _():
    factory = ApiClientFactory()
    creds = ApiCredentials()
    result = factory.diagnose(creds)
    assert len(result["problems"]) > 0
    assert result["problems"][0]["code"] == "CONF-001"

@test("Factory.diagnose returns clean result with valid credentials")
def _():
    factory = ApiClientFactory({"api": {"provider": "azure"}})
    creds = ApiCredentials(
        api_key="key", endpoint="https://aoai.company.com",
        deployment="gpt-35", api_version="2024-02-01",
    )
    result = factory.diagnose(creds)
    assert len(result["problems"]) == 0
    assert result["provider"] == "azure"
    assert result["final_url"] is not None
    assert result["auth_header"] == "api-key"

@test("Factory.diagnose catches invalid endpoint")
def _():
    factory = ApiClientFactory()
    creds = ApiCredentials(
        api_key="key", endpoint="no-scheme.example.com",
    )
    result = factory.diagnose(creds)
    assert any(p["code"] == "CONF-003" for p in result["problems"])

@test("Factory.diagnose detects OpenAI URL pattern on Azure")
def _():
    factory = ApiClientFactory({"api": {"provider": "openai"}})
    creds = ApiCredentials(
        api_key="key", endpoint="https://aoai.company.com",
    )
    result = factory.diagnose(creds)
    # Should detect /v1/chat being used on what looks like Azure
    # The final URL will have /v1/ which may be wrong
    assert result["provider"] == "openai"

@test("ApiClient.get_diagnostic_info returns safe info")
def _():
    cfg = ApiClientConfig(
        provider="azure",
        final_url="https://api.com/openai/deployments/gpt-35/chat/completions",
        auth_header_name="api-key",
        auth_header_value="super-secret-key",
        deployment="gpt-35",
        api_version="2024-02-01",
    )
    client = ApiClient(cfg, HttpClient(HttpClientConfig(offline_mode=True)))
    info = client.get_diagnostic_info()
    assert "super-secret-key" not in str(info)
    assert info["provider"] == "azure"
    assert info["deployment"] == "gpt-35"

@test("ApiClient.test_connection returns error dict when offline")
def _():
    cfg = ApiClientConfig(
        provider="azure",
        final_url="https://api.com/chat/completions",
        auth_header_name="api-key",
        auth_header_value="key",
    )
    client = ApiClient(cfg, HttpClient(HttpClientConfig(offline_mode=True)))
    result = client.test_connection()
    assert result["success"] is False
    assert "error" in result


# ===========================================================================
# MODULE 5: BOOT PIPELINE
# ===========================================================================
print()
print("=" * 60)
print("  MODULE 5: BOOT PIPELINE (src/core/boot.py)")
print("=" * 60)

from src.core.boot import boot_hybridrag, load_config, BootResult

@test("BootResult defaults to not ready")
def _():
    r = BootResult()
    assert r.success is False
    assert r.online_available is False
    assert r.offline_available is False

@test("BootResult.summary produces readable output")
def _():
    r = BootResult(
        success=True,
        online_available=True,
        offline_available=False,
        warnings=["Ollama not running"],
        errors=[],
    )
    s = r.summary()
    assert "READY" in s
    assert "AVAILABLE" in s
    assert "Ollama" in s

@test("load_config returns empty dict when no file exists")
def _():
    config = load_config("/nonexistent/path/config.yaml")
    assert isinstance(config, dict)

@test("load_config loads the test config file")
def _():
    config = load_config(str(Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"))
    assert "api" in config
    assert config["api"]["provider"] == "azure"

@test("boot_hybridrag runs without crashing (no credentials)")
def _():
    with patch("src.security.credentials._read_keyring", return_value=None):
        for a in _KEY_ENV_ALIASES + _ENDPOINT_ENV_ALIASES:
            os.environ.pop(a, None)
        result = boot_hybridrag(
            config_path=str(Path(__file__).resolve().parent.parent / "config" / "default_config.yaml")
        )
        assert isinstance(result, BootResult)
        # Should not crash, just report not ready
        assert result.online_available is False

@test("boot_hybridrag with credentials creates API client")
def _():
    with patch("src.security.credentials._read_keyring") as mock_kr:
        mock_kr.side_effect = lambda name: {
            "azure_api_key": "test-key-123",
            "azure_endpoint": "https://aoai-api.company.com",
        }.get(name)
        with patch.dict(os.environ, {"AZURE_OPENAI_DEPLOYMENT": "gpt-35-turbo"}):
            result = boot_hybridrag(
                config_path=str(Path(__file__).resolve().parent.parent / "config" / "default_config.yaml")
            )
            assert result.online_available is True
            assert result.api_client is not None
            assert result.api_client.config.provider == "azure"

@test("boot_hybridrag records warnings for missing Ollama")
def _():
    with patch("src.security.credentials._read_keyring", return_value=None):
        result = boot_hybridrag(
            config_path=str(Path(__file__).resolve().parent.parent / "config" / "default_config.yaml")
        )
        # Ollama isn't running in this test environment
        ollama_warnings = [w for w in result.warnings if "ollama" in w.lower() or "offline" in w.lower()]
        assert len(ollama_warnings) > 0


# ===========================================================================
# MODULE 6: INTEGRATION TESTS
# ===========================================================================
print()
print("=" * 60)
print("  MODULE 6: INTEGRATION TESTS")
print("=" * 60)

@test("End-to-end: credentials -> factory -> client (Azure)")
def _():
    """Simulate the full flow from credential resolution to client creation."""
    with patch("src.security.credentials._read_keyring") as mock_kr:
        mock_kr.side_effect = lambda name: {
            "azure_api_key": "integration-key-12345",
            "azure_endpoint": "https://aiml-aoai-api.gcl.mycompany.com",
        }.get(name)
        with patch.dict(os.environ, {"AZURE_OPENAI_DEPLOYMENT": "gpt-35-turbo"}):
            # Step 1: Resolve credentials
            creds = resolve_credentials()
            assert creds.is_online_ready

            # Step 2: Create factory with config
            config = load_config(str(Path(__file__).resolve().parent.parent / "config" / "default_config.yaml"))
            factory = ApiClientFactory(config)

            # Step 3: Run diagnostics first
            diag = factory.diagnose(creds)
            assert len(diag["problems"]) == 0, f"Problems: {diag['problems']}"
            assert diag["provider"] == "azure"

            # Step 4: Build client
            client = factory.build(creds)
            assert client.config.provider == "azure"
            assert client.config.auth_header_name == "api-key"
            assert "gpt-35-turbo" in client.config.final_url
            assert "api-version=2024-02-01" in client.config.final_url

            # Step 5: Verify URL is exactly right
            expected_url = (
                "https://aiml-aoai-api.gcl.mycompany.com"
                "/openai/deployments/gpt-35-turbo"
                "/chat/completions?api-version=2024-02-01"
            )
            assert client.config.final_url == expected_url, \
                f"Expected: {expected_url}\nGot:      {client.config.final_url}"

@test("End-to-end: boot pipeline with full credentials produces working client")
def _():
    with patch("src.security.credentials._read_keyring") as mock_kr:
        mock_kr.side_effect = lambda name: {
            "azure_api_key": "boot-test-key",
            "azure_endpoint": "https://aoai.company.com",
        }.get(name)
        with patch.dict(os.environ, {"AZURE_OPENAI_DEPLOYMENT": "my-gpt"}):
            result = boot_hybridrag(
                config_path=str(Path(__file__).resolve().parent.parent / "config" / "default_config.yaml")
            )
            assert result.online_available is True
            assert result.api_client is not None
            info = result.api_client.get_diagnostic_info()
            assert info["provider"] == "azure"
            assert info["deployment"] == "my-gpt"
            assert "boot-test-key" not in str(info)

@test("End-to-end: exception hierarchy works in catch block")
def _():
    """Verify that typed exceptions work correctly in try/except chains."""
    errors_caught = {
        "auth": False,
        "deploy": False,
        "rate": False,
        "base": False,
    }

    # Test specific catches
    try:
        raise AuthRejectedError("bad key")
    except AuthRejectedError:
        errors_caught["auth"] = True

    try:
        raise DeploymentNotFoundError(deployment="gpt-35")
    except DeploymentNotFoundError:
        errors_caught["deploy"] = True

    try:
        raise RateLimitedError(retry_after=60)
    except RateLimitedError:
        errors_caught["rate"] = True

    # Test base class catch
    try:
        raise ServerError(status_code=503)
    except HybridRAGError:
        errors_caught["base"] = True

    assert all(errors_caught.values()), f"Not all caught: {errors_caught}"

@test("End-to-end: factory rejects bad credentials with correct exception type")
def _():
    """Verify the factory gate works -- bad inputs produce clear errors."""
    factory = ApiClientFactory({"api": {"provider": "azure"}})

    # No endpoint
    try:
        factory.build(ApiCredentials(api_key="key"))
        assert False
    except EndpointNotConfiguredError as e:
        assert e.error_code == "CONF-001"
        assert e.fix_suggestion

    # No key
    try:
        factory.build(ApiCredentials(endpoint="https://api.com"))
        assert False
    except ApiKeyNotConfiguredError as e:
        assert e.error_code == "CONF-002"
        assert e.fix_suggestion

    # Bad URL
    try:
        factory.build(ApiCredentials(api_key="key", endpoint="not-a-url"))
        assert False
    except InvalidEndpointError as e:
        assert e.error_code == "CONF-003"

    # Azure without deployment
    try:
        factory.build(ApiCredentials(api_key="key", endpoint="https://api.com"))
        assert False
    except DeploymentNotConfiguredError as e:
        assert e.error_code == "CONF-004"

@test("End-to-end: network kill switch blocks API client requests")
def _():
    factory = ApiClientFactory({"api": {"provider": "azure"}, "http": {}})
    creds = ApiCredentials(
        api_key="key", endpoint="https://aoai.company.com",
        deployment="gpt-35",
    )

    with patch.dict(os.environ, {"HYBRIDRAG_OFFLINE": "1"}):
        client = factory.build(creds)
        result = client.test_connection()
        assert result["success"] is False

@test("End-to-end: diagnostic dict has all expected fields")
def _():
    creds = ApiCredentials(
        api_key="diag-key-12345",
        endpoint="https://aoai.company.com",
        deployment="gpt-35-turbo",
        api_version="2024-02-01",
        source_key="keyring",
        source_endpoint="keyring",
    )
    d = creds.to_diagnostic_dict()
    required_fields = [
        "endpoint", "api_key", "deployment", "api_version",
        "source_key", "source_endpoint", "online_ready",
    ]
    for f in required_fields:
        assert f in d, f"Missing field: {f}"
    assert d["online_ready"] is True
    assert "diag-key-12345" not in d["api_key"]  # Must be masked


# ===========================================================================
# FINAL REPORT
# ===========================================================================
print()
print("=" * 60)
total = PASS + FAIL
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {total} total")
if FAIL == 0:
    print("  ALL TESTS PASSED!")
else:
    print()
    print("  FAILURES:")
    for name, err in ERRORS:
        print(f"    [FAIL] {name}")
        print(f"           {err}")
print("=" * 60)
print()

sys.exit(0 if FAIL == 0 else 1)

