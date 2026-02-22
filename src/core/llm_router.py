# ============================================================================
# llm_router.py -- LLM Backend Router (Offline + Online)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   This is the "switchboard" that decides where your AI queries go:
#
#   OFFLINE MODE (no internet needed):
#     Query --> Ollama (local model on your machine) --> Answer
#
#   ONLINE MODE (needs enterprise network or home API key):
#     Query --> Azure OpenAI API (or standard OpenAI) --> Answer
#
#   The rest of HybridRAG only talks to LLMRouter. It never needs
#   to know whether the answer came from Ollama or Azure -- the
#   router handles all of that behind the scenes.
#
# WHY WE CHANGED THIS FILE (February 2026):
#   The old version used raw httpx HTTP calls to talk to Azure. This
#   caused two painful bugs:
#     1. 401 Unauthorized -- wrong auth header format
#     2. 404 Not Found    -- URL path doubling
#   The new version uses the official 'openai' Python SDK, which
#   builds URLs and headers correctly every time. Same SDK your
#   company's own example code uses.
#
# INTERNET ACCESS:
#   OllamaRouter -- NONE (talks to localhost only)
#   APIRouter    -- YES (connects to Azure or OpenAI endpoint)
#
# DEPENDENCIES:
#   - httpx      (for Ollama -- already installed)
#   - openai     (for Azure/OpenAI API -- new addition)
#   - keyring    (optional, for credential storage)
# ============================================================================

import json
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

import httpx

# -- Import the official OpenAI SDK -----------------------------------------
# This is the same library your company's example code uses:
#   from openai import AzureOpenAI
# It handles URL construction, auth headers, retries, and error handling
# automatically. No more hand-building URLs or guessing header formats.
# ---------------------------------------------------------------------------
try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

# -- Import HybridRAG internals ---------------------------------------------
from .config import Config
from .network_gate import get_gate, NetworkBlockedError
from ..monitoring.logger import get_app_logger


# ============================================================================
# LLMResponse -- The standard answer format
# ============================================================================
# Every backend (Ollama, Azure, OpenAI) returns its answer wrapped in
# this same structure. That way the rest of HybridRAG doesn't care
# which backend answered -- it always gets the same fields.
# ============================================================================
@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    text: str              # The actual AI-generated answer
    tokens_in: int         # How many tokens were in the prompt
    tokens_out: int        # How many tokens the AI generated
    model: str             # Which model answered (e.g., "phi4-mini")
    latency_ms: float      # How long the call took in milliseconds


# ============================================================================
# OllamaRouter -- Talks to local Ollama server (offline mode)
# ============================================================================
#
# Ollama runs on your machine at http://localhost:11434. It hosts
# open-source models like Phi-4 and Mistral that work without internet.
#
# This router uses raw httpx because Ollama has a simple REST API
# and doesn't need the openai SDK. No changes from the old version.
#
# INTERNET ACCESS: NONE (localhost only)
# ============================================================================
class OllamaRouter:
    """Route queries to local Ollama server (offline mode)."""

    def __init__(self, config: Config):
        """
        Set up the Ollama router.

        Args:
            config: The HybridRAG configuration object. We read:
                    - config.ollama.base_url (default: http://localhost:11434)
                    - config.ollama.model (default: phi4-mini)
                    - config.ollama.timeout_seconds (default: 120)
        """
        self.config = config
        self.logger = get_app_logger("ollama_router")

        # Base URL for the local Ollama server
        self.base_url = config.ollama.base_url.rstrip("/")

    def is_available(self) -> bool:
        """
        Check if Ollama is running and reachable.

        Returns:
            True if Ollama responds, False otherwise

        How it works:
            Sends a simple GET request to Ollama's root URL.
            If Ollama is running, it responds with "Ollama is running".
            If not, the connection fails and we return False.
        """
        try:
            # -- Network gate check --
            # Even localhost connections go through the gate so they
            # appear in the audit log. The gate allows localhost in
            # both offline and online modes.
            get_gate().check_allowed(self.base_url, "ollama_health", "ollama_router")

            with httpx.Client(timeout=5) as client:
                resp = client.get(self.base_url)
                return resp.status_code == 200
        except NetworkBlockedError:
            return False
        except Exception:
            return False

    def query(self, prompt: str) -> Optional[LLMResponse]:
        """
        Send a prompt to the local Ollama server.

        Args:
            prompt: The complete prompt (context + user question)

        Returns:
            LLMResponse with the answer, or None if the call failed
        """
        start_time = time.time()

        # -- Network gate check --
        try:
            get_gate().check_allowed(
                f"{self.base_url}/api/generate",
                "ollama_query", "ollama_router",
            )
        except NetworkBlockedError as e:
            self.logger.error("ollama_blocked_by_gate", error=str(e))
            return None

        # Build the request body for Ollama's /api/generate endpoint
        payload = {
            "model": self.config.ollama.model,
            "prompt": prompt,
            "stream": False,    # Get the full response at once, not word-by-word
            
        }

        try:
            with httpx.Client(timeout=self.config.ollama.timeout_seconds) as client:
                resp = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()

            data = resp.json()
            response_text = data.get("response", "")
            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)
            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "ollama_query_success",
                model=self.config.ollama.model,
                tokens_in=prompt_eval_count,
                tokens_out=eval_count,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                text=response_text,
                tokens_in=prompt_eval_count,
                tokens_out=eval_count,
                model=self.config.ollama.model,
                latency_ms=latency_ms,
            )

        except httpx.HTTPError as e:
            self.logger.error("ollama_http_error", error=str(e))
            return None
        except Exception as e:
            self.logger.error("ollama_error", error=str(e))
            return None


# ============================================================================
# APIRouter -- Talks to Azure OpenAI or standard OpenAI API
# ============================================================================
#
# THIS IS THE PART THAT CHANGED (February 2026):
#
#   OLD WAY (broke constantly):
#     - Hand-built the URL: base + /openai/deployments/gpt-35-turbo/...
#     - Hand-built the auth header: "api-key: ..." or "Bearer ..."
#     - Result: URL doubling (404), wrong headers (401)
#
#   NEW WAY (using official openai SDK):
#     - AzureOpenAI() client builds URLs automatically
#     - Auth headers are handled internally by the SDK
#     - Same approach your company's own example code uses
#     - Result: it just works
#
# The SDK auto-detects Azure vs standard OpenAI based on which
# client class you use:
#
#   Azure:    AzureOpenAI(azure_endpoint=..., api_key=..., api_version=...)
#   Standard: OpenAI(api_key=...)
#   Home dev: OpenAI(api_key=...)  <-- same as standard, different key
#
# INTERNET ACCESS: YES -- connects to API endpoint
# ============================================================================
class APIRouter:
    """Route queries to Azure OpenAI or standard OpenAI API (online mode)."""

    # -- Fallback defaults (used only if YAML, env vars, and credentials
    #    all fail to provide a value) --
    _DEFAULT_API_VERSION = "2024-02-02"
    _DEFAULT_DEPLOYMENT = "gpt-35-turbo"

    def __init__(self, config: Config, api_key: str, endpoint: str = "",
                 deployment_override: str = "", api_version_override: str = ""):
        """
        Set up the API router using the official openai SDK.

        Args:
            config:   The HybridRAG configuration object
            api_key:  Your API key (from Credential Manager or env var)
            endpoint: Your API base URL (Azure endpoint or OpenAI base)
            deployment_override: Pre-resolved deployment name (from credentials.py).
                If provided, skips the internal _resolve_deployment() chain.
            api_version_override: Pre-resolved API version (from credentials.py).
                If provided, skips the internal _resolve_api_version() chain.

        What happens here:
            1. We figure out if this is Azure or standard OpenAI
            2. We resolve deployment name and API version from config/env
            3. We create the appropriate SDK client
            4. The client handles all URL/header construction from here on

        RESOLUTION ORDER for deployment and api_version:
            0. Override from resolve_credentials() (if provided by LLMRouter)
            1. config/default_config.yaml (api.deployment, api.api_version)
            2. Environment variables (AZURE_OPENAI_DEPLOYMENT, etc.)
            3. Extracted from endpoint URL (if it contains /deployments/xxx)
            4. Hardcoded fallback defaults (last resort)
        """
        self.config = config
        self.api_key = api_key
        self.logger = get_app_logger("api_router")

        # Store the raw endpoint for diagnostics/status reporting
        self.base_endpoint = endpoint.rstrip("/") if endpoint else config.api.endpoint.rstrip("/")

        # -- Auto-detect Azure vs standard OpenAI --
        # If the URL contains "azure" or "aoai" anywhere, it's Azure.
        # Otherwise, it's standard OpenAI (or an OpenAI-compatible service).
        self.is_azure = (
            "azure" in self.base_endpoint.lower()
            or "aoai" in self.base_endpoint.lower()
        )

        # -- Resolve deployment name --
        # If the canonical resolver already extracted deployment from URL
        # or env vars, use that directly (no duplicate resolution).
        if deployment_override:
            self.deployment = deployment_override
        else:
            # Fallback: run the local resolution chain
            self.deployment = self._resolve_deployment(config, self.base_endpoint)

        # -- Resolve API version --
        if api_version_override:
            self.api_version = api_version_override
        else:
            self.api_version = self._resolve_api_version(config, self.base_endpoint)

        # -- Network gate check --
        # Verify the endpoint is allowed before we even create the SDK client.
        # This is an early fail-fast check. The gate also checks on each
        # query() call, but catching it here gives a clearer error message
        # during startup rather than on the first query.
        try:
            get_gate().check_allowed(
                self.base_endpoint, "api_client_init", "api_router",
            )
        except NetworkBlockedError as e:
            self.client = None
            self.logger.error(
                "api_endpoint_blocked_by_gate",
                endpoint=self.base_endpoint,
                error=str(e),
            )
            return

        # -- Check if the openai SDK is available --
        if not OPENAI_SDK_AVAILABLE:
            self.client = None
            self.logger.error(
                "openai_sdk_missing",
                hint="Run: pip install openai",
            )
            return

        # -- Create the appropriate SDK client --
        #
        # WHY TWO DIFFERENT CLIENTS?
        #   Azure and standard OpenAI use different URL formats and auth
        #   methods. The SDK handles this automatically -- you just pick
        #   the right client class and it does the rest.
        #
        # AzureOpenAI client:
        #   - Builds URL: {endpoint}/openai/deployments/{model}/chat/completions
        #   - Sends header: "api-key: your-key"
        #   - Requires: azure_endpoint, api_version, api_key
        #
        # OpenAI client:
        #   - Builds URL: https://api.openai.com/v1/chat/completions
        #   - Sends header: "Authorization: Bearer your-key"
        #   - Requires: api_key (endpoint defaults to api.openai.com)
        #
        try:
            if self.is_azure:
                # -- AZURE CLIENT --
                # This is what your company's example code uses.
                # The SDK extracts the base domain from azure_endpoint and
                # builds the full URL with deployment name and api-version.
                #
                # IMPORTANT: azure_endpoint should be JUST the base URL:
                #   https://your-company.openai.azure.com
                # The SDK appends /openai/deployments/... automatically.
                #
                # If the stored endpoint contains the full path (like
                # .../chat/completions?api-version=...), we strip it down
                # to just the base domain so the SDK doesn't double it.
                clean_endpoint = self._extract_azure_base(self.base_endpoint)

                self.client = AzureOpenAI(
                    azure_endpoint=clean_endpoint,
                    api_key=self.api_key,
                    api_version=self.api_version,
                    # http_client with verify=True ensures SSL works
                    # through enterprise proxy (with pip-system-certs installed)
                    http_client=httpx.Client(verify=True),
                )

                self.logger.info(
                    "api_router_init",
                    provider="azure_openai",
                    endpoint=clean_endpoint,
                    deployment=self.deployment,
                    api_version=self.api_version,
                    sdk="openai_official",
                )

            else:
                # -- STANDARD OPENAI CLIENT --
                # For home development with a personal OpenAI API key,
                # or for any OpenAI-compatible service (OpenRouter, etc).
                #
                # If endpoint is provided and it's not Azure, use it as
                # the base_url (for OpenRouter, local proxies, etc).
                # Otherwise, the SDK defaults to https://api.openai.com/v1
                client_kwargs = {"api_key": self.api_key}

                if self.base_endpoint and "openai.com" not in self.base_endpoint:
                    # Custom endpoint (OpenRouter, Together AI, etc.)
                    client_kwargs["base_url"] = self.base_endpoint

                self.client = OpenAI(**client_kwargs)

                self.logger.info(
                    "api_router_init",
                    provider="openai",
                    sdk="openai_official",
                )

        except Exception as e:
            self.client = None
            self.logger.error("api_router_init_failed", error=str(e))

    @staticmethod
    def _resolve_deployment(config, endpoint_url):
        """
        Resolve Azure deployment name from multiple sources.

        WHY THIS EXISTS:
            Different machines may have different Azure deployment names.
            Instead of hardcoding "gpt-35-turbo", we check multiple sources
            so each machine can configure its own deployment name.

        Resolution order (first non-empty wins):
            1. YAML config: api.deployment in default_config.yaml
            2. Environment variables: AZURE_OPENAI_DEPLOYMENT, etc.
            3. URL extraction: if endpoint contains /deployments/xxx
            4. Fallback: "gpt-35-turbo" (safe default)

        Returns:
            str: The resolved deployment name.
        """
        import re

        # 1. YAML config
        if config.api.deployment:
            return config.api.deployment

        # 2. Environment variables (same aliases as credentials.py)
        for var in [
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_DEPLOYMENT",
            "OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "DEPLOYMENT_NAME",
            "AZURE_CHAT_DEPLOYMENT",
        ]:
            val = os.environ.get(var, "").strip()
            if val:
                return val

        # 3. Extract from URL (e.g., .../deployments/gpt-4o/...)
        if endpoint_url and "/deployments/" in endpoint_url:
            match = re.search(r"/deployments/([^/?]+)", endpoint_url)
            if match:
                return match.group(1)

        # 4. Fallback default
        return APIRouter._DEFAULT_DEPLOYMENT

    @staticmethod
    def _resolve_api_version(config, endpoint_url):
        """
        Resolve Azure API version from multiple sources.

        WHY THIS EXISTS:
            Azure API versions change over time and different tenants
            may require different versions. Instead of hardcoding one
            version, we check multiple sources.

        Resolution order (first non-empty wins):
            1. YAML config: api.api_version in default_config.yaml
            2. Environment variables: AZURE_OPENAI_API_VERSION, etc.
            3. URL extraction: if endpoint contains ?api-version=xxx
            4. Fallback: "2024-02-02" (safe default)

        Returns:
            str: The resolved API version string.
        """
        import re

        # 1. YAML config
        if config.api.api_version:
            return config.api.api_version

        # 2. Environment variables
        for var in [
            "AZURE_OPENAI_API_VERSION",
            "AZURE_API_VERSION",
            "OPENAI_API_VERSION",
            "API_VERSION",
        ]:
            val = os.environ.get(var, "").strip()
            if val:
                return val

        # 3. Extract from URL (e.g., ...?api-version=2024-02-02)
        if endpoint_url and "api-version=" in endpoint_url:
            match = re.search(r"api-version=([^&]+)", endpoint_url)
            if match:
                return match.group(1)

        # 4. Fallback default
        return APIRouter._DEFAULT_API_VERSION

    def _extract_azure_base(self, url: str) -> str:
        """
        Extract just the base domain from an Azure endpoint URL.

        WHY THIS EXISTS:
            Users might store different URL formats:
              - Just the base: https://company.openai.azure.com
              - With deployment path: https://company.openai.azure.com/openai/deployments/...
              - Full URL with query: ...chat/completions?api-version=2024-02-02

            The AzureOpenAI SDK needs ONLY the base domain. If we pass the
            full URL, the SDK will append /openai/deployments/... AGAIN,
            causing the URL doubling that gave us 404 errors before.

            This method strips everything after the domain, no matter what
            format the user stored.

        Examples:
            Input:  https://company.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions
            Output: https://company.openai.azure.com

            Input:  https://company.openai.azure.com
            Output: https://company.openai.azure.com  (unchanged)
        """
        # Find the position right after the domain
        # Look for /openai/ which is where the Azure path starts
        idx = url.lower().find("/openai/")
        if idx > 0:
            return url[:idx]

        # Look for /chat/ in case the URL starts mid-path
        idx = url.lower().find("/chat/")
        if idx > 0:
            return url[:idx]

        # If there's a query string, strip it
        idx = url.find("?")
        if idx > 0:
            return url[:idx]

        # Already a clean base URL
        return url

    def query(self, prompt: str) -> Optional[LLMResponse]:
        """
        Send a prompt to the API and get the AI-generated answer back.

        Args:
            prompt: The complete prompt (context + user question)

        Returns:
            LLMResponse with the answer, or None if the call failed

        How it works:
            1. We call client.chat.completions.create() -- this is the
               standard OpenAI SDK method for chat-based completions.
            2. The SDK handles URL construction, headers, retries.
            3. We parse the response into our standard LLMResponse format.

        The call is IDENTICAL for both Azure and standard OpenAI.
        The only difference is which client we created in __init__.
        """
        if self.client is None:
            self.logger.error(
                "api_client_not_ready",
                hint="openai SDK not installed or client init failed",
            )
            return None

        # -- Network gate check (per-query) --
        # Even though we checked in __init__, the mode could have changed
        # (e.g., user ran rag-mode-offline mid-session). Check again.
        try:
            get_gate().check_allowed(
                self.base_endpoint, "api_query", "api_router",
            )
        except NetworkBlockedError as e:
            self.logger.error("api_query_blocked_by_gate", error=str(e))
            return None

        start_time = time.time()

        # -- Pick the model name --
        # Azure: use the deployment name (gpt-35-turbo)
        # Standard OpenAI: use the model from config (gpt-4o-mini, etc.)
        model_name = self.deployment if self.is_azure else self.config.api.model

        try:
            # -- THE API CALL --
            # This single line replaces all the old httpx URL-building,
            # header-building, and HTTP-sending code. The SDK does it all.
            #
            # For Azure, the SDK sends:
            #   POST https://company.openai.azure.com/openai/deployments/
            #        gpt-35-turbo/chat/completions?api-version=2024-02-02
            #   Header: api-key: your-key
            #
            # For standard OpenAI, the SDK sends:
            #   POST https://api.openai.com/v1/chat/completions
            #   Header: Authorization: Bearer your-key
            #
            # We don't build any of that -- the SDK handles it.

            self.logger.info(
                "api_query_sending",
                provider="azure" if self.is_azure else "openai",
                model=model_name,
            )

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.api.max_tokens,
                temperature=self.config.api.temperature,
            )

            # -- Parse the SDK response --
            # The SDK returns a typed object, not raw JSON.
            # response.choices[0].message.content = the AI's answer
            # response.usage.prompt_tokens = tokens in our prompt
            # response.usage.completion_tokens = tokens the AI generated
            # response.model = which model actually answered

            answer_text = response.choices[0].message.content
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0
            actual_model = response.model or model_name
            latency_ms = (time.time() - start_time) * 1000

            self.logger.info(
                "api_query_success",
                model=actual_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                is_azure=self.is_azure,
            )

            return LLMResponse(
                text=answer_text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=actual_model,
                latency_ms=latency_ms,
            )

        # -- ERROR HANDLING --
        # The openai SDK raises specific exception types for different
        # failures. We catch each one and log a helpful message.

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_name = type(e).__name__
            error_msg = str(e)

            # Classify the error for better troubleshooting
            if "401" in error_msg or "Unauthorized" in error_msg:
                self.logger.error(
                    "api_auth_error",
                    error=error_msg[:500],
                    hint=(
                        "TROUBLESHOOTING 401: "
                        "(1) Is the API key correct and not expired? "
                        "(2) Run rag-store-key to re-enter it. "
                        "(3) Check Azure portal for Cognitive Services User role."
                    ),
                )
            elif "404" in error_msg or "NotFound" in error_msg:
                self.logger.error(
                    "api_not_found",
                    error=error_msg[:500],
                    hint=(
                        "TROUBLESHOOTING 404: "
                        "(1) Is the deployment name correct? Expected: "
                        + self.deployment + ". "
                        "(2) Is the endpoint URL correct? "
                        "(3) Check Azure portal for the exact deployment name."
                    ),
                )
            elif "429" in error_msg or "RateLimit" in error_msg:
                self.logger.error(
                    "api_rate_limited",
                    error=error_msg[:200],
                    hint="Wait 30-60 seconds and retry.",
                )
            elif "SSL" in error_msg or "certificate" in error_msg.lower():
                self.logger.error(
                    "api_ssl_error",
                    error=error_msg[:500],
                    hint=(
                        "TROUBLESHOOTING SSL: "
                        "(1) Is pip-system-certs installed? "
                        "(2) Are you on enterprise LAN (not VPN)? "
                        "(3) Run: pip install pip-system-certs"
                    ),
                )
            elif "Connection" in error_name or "connect" in error_msg.lower():
                self.logger.error(
                    "api_connection_error",
                    error=error_msg[:500],
                    hint="Check VPN/network connection and firewall rules.",
                )
            elif "Timeout" in error_name or "timed out" in error_msg.lower():
                self.logger.error(
                    "api_timeout",
                    error=error_msg[:200],
                    timeout_seconds=self.config.api.timeout_seconds,
                )
            else:
                self.logger.error(
                    "api_error",
                    error_type=error_name,
                    error=error_msg[:500],
                )

            return None

    def get_status(self) -> Dict[str, Any]:
        """
        Return current API router status for diagnostics.

        Used by: rag-cred-status, rag-test-api, rag-status
        """
        status = {
            "provider": "azure_openai" if self.is_azure else "openai",
            "endpoint": self.base_endpoint,
            "api_configured": self.client is not None,
            "sdk_available": OPENAI_SDK_AVAILABLE,
            "sdk": "openai_official",
        }
        if self.is_azure:
            status["deployment"] = self.deployment
            status["api_version"] = self.api_version
            status["clean_endpoint"] = self._extract_azure_base(self.base_endpoint)
        return status


# ============================================================================
# DEPLOYMENT DISCOVERY -- Detects available models on Azure or OpenAI
# ============================================================================
#
# WHY THIS EXISTS:
#   Azure uses a different API endpoint for listing deployments than
#   OpenAI/OpenRouter. Without this, _list_models.py silently returns
#   nothing for Azure endpoints. This module detects the provider type
#   from the stored endpoint and uses the correct discovery path.
#
# AZURE PATH:
#   GET {base}/openai/deployments?api-version={version}
#   Header: api-key: {key}
#   Response: {"value": [{"id": "gpt-4o", "model": "gpt-4o", ...}, ...]}
#
# OPENAI/OPENROUTER PATH:
#   GET {endpoint}/models
#   Header: Authorization: Bearer {key}
#   Response: {"data": [{"id": "gpt-4o", ...}, ...]}
#
# COMPATIBILITY NOTE (openai 1.45.1):
#   We use httpx directly for discovery instead of the openai SDK because:
#   1. The openai 1.45.1 SDK does not have a deployments.list() method
#   2. httpx is already a dependency (used by Ollama path)
#   3. Direct HTTP gives us full control over the request format
#
# INTERNET ACCESS: YES (one GET request per call)
# ============================================================================

# Module-level cache for deployment discovery results
_deployment_cache = None


def _is_azure_endpoint(endpoint):
    """
    Check if an endpoint URL is Azure-style.
    Uses the same detection logic as APIRouter.is_azure.
    """
    if not endpoint:
        return False
    lower = endpoint.lower()
    return "azure" in lower or "aoai" in lower


def get_available_deployments():
    """
    Discover available model deployments from the configured endpoint.

    Detects Azure vs OpenAI from the stored endpoint URL and uses the
    correct API path and auth header format for each.

    Returns:
        list: Deployment name strings. Empty list if endpoint is
              unreachable or credentials are missing. Never raises.
    """
    global _deployment_cache
    if _deployment_cache is not None:
        return list(_deployment_cache)

    # Resolve credentials from the canonical source
    try:
        from ..security.credentials import resolve_credentials
        creds = resolve_credentials()
    except Exception:
        logger.error("[FAIL] Could not resolve credentials for deployment discovery")
        return []

    if not creds.has_key or not creds.has_endpoint:
        logger.error("[FAIL] Credentials incomplete -- need both API key and endpoint")
        return []

    endpoint = creds.endpoint.rstrip("/")
    api_key = creds.api_key

    try:
        if _is_azure_endpoint(endpoint):
            # Azure path: GET {base}/openai/deployments?api-version={version}
            api_version = creds.api_version or "2024-02-02"

            # Strip any existing path to get just the base domain
            # (endpoint might contain /openai/deployments/... already)
            base = endpoint
            idx = base.lower().find("/openai/")
            if idx > 0:
                base = base[:idx]
            idx = base.find("?")
            if idx > 0:
                base = base[:idx]

            url = f"{base}/openai/deployments?api-version={api_version}"

            with httpx.Client(timeout=15, verify=True) as client:
                resp = client.get(
                    url,
                    headers={
                        "api-key": api_key,
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code != 200:
                logger.warning("[WARN] Azure deployment list returned HTTP %s", resp.status_code)
                _deployment_cache = []
                return []

            data = resp.json()
            # Azure returns {"value": [{...}, ...]}
            value_list = data.get("value", [])
            if not isinstance(value_list, list):
                _deployment_cache = []
                return []

            deployments = []
            for item in value_list:
                # Each deployment has an "id" (deployment name) and "model" field
                dep_id = item.get("id", "") or item.get("name", "")
                if dep_id:
                    deployments.append(dep_id)

            _deployment_cache = deployments
            logger.info("[OK] Found %d Azure deployments", len(deployments))
            return list(deployments)

        else:
            # OpenAI/OpenRouter path: GET {endpoint}/models
            url = f"{endpoint}/models"

            with httpx.Client(timeout=15, verify=True) as client:
                resp = client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code != 200:
                logger.warning("[WARN] Model list returned HTTP %s", resp.status_code)
                _deployment_cache = []
                return []

            data = resp.json()

            # Standard OpenAI format: {"data": [{...}, ...]}
            if "data" in data and isinstance(data["data"], list):
                models = [m.get("id", "") for m in data["data"] if m.get("id")]
            elif isinstance(data, list):
                models = [m.get("id", "") for m in data if m.get("id")]
            else:
                _deployment_cache = []
                return []

            _deployment_cache = sorted(models)
            logger.info("[OK] Found %d models", len(models))
            return list(_deployment_cache)

    except Exception as e:
        logger.warning("[WARN] Deployment discovery failed: %s", e)
        _deployment_cache = []
        return []


def refresh_deployments():
    """
    Clear the deployment cache and re-probe the endpoint.

    Returns:
        list: Fresh deployment list from the endpoint.
    """
    global _deployment_cache
    _deployment_cache = None
    result = get_available_deployments()
    logger.info("[OK] Deployment cache refreshed: %d deployments", len(result))
    return result


# ============================================================================
# LLMRouter -- The main switchboard
# ============================================================================
#
# This is the class that the rest of HybridRAG talks to.
# It decides whether to use Ollama (offline) or the API (online)
# based on the mode setting in your config.
#
# Usage:
#   router = LLMRouter(config)
#   answer = router.query("What is the operating frequency?")
#
# The caller never needs to know which backend answered.
# ============================================================================
class LLMRouter:
    """
    Route queries to the appropriate LLM backend.

    Mode selection:
        "offline" --> Ollama (local, free, no internet)
        "online"  --> Azure OpenAI API (company cloud, costs money)
    """

    def __init__(self, config: Config, api_key: Optional[str] = None):
        """
        Initialize the router and set up both backends.

        Args:
            config:  The HybridRAG configuration object
            api_key: Optional explicit API key (overrides all other sources)

        Credential resolution order (tries each in sequence):
            1. Explicit api_key parameter (for testing)
            2. Windows Credential Manager via keyring
            3. AZURE_OPENAI_API_KEY environment variable
            4. AZURE_OPEN_AI_KEY environment variable (company variant)
            5. OPENAI_API_KEY environment variable (home/standard)
        """
        self.config = config
        self.logger = get_app_logger("llm_router")

        # -- Always create the Ollama router (offline mode) --
        # This is lightweight and doesn't need any credentials
        self.ollama = OllamaRouter(config)

        # -- Resolve ALL credentials through the single canonical resolver --
        # This replaces 45 lines of duplicate env-var/keyring logic that
        # previously lived here. Now there is ONE resolver, ONE alias list,
        # ONE keyring schema. If the caller passed an explicit api_key
        # (e.g. from a test), we use that; otherwise resolve_credentials()
        # checks keyring -> env vars -> config in priority order.
        resolved_key = api_key
        resolved_endpoint = ""
        resolved_deployment = ""
        resolved_api_version = ""

        try:
            from ..security.credentials import resolve_credentials
            # Pass config as dict so the resolver can read YAML fallbacks
            config_dict = None
            if hasattr(config, "to_dict"):
                config_dict = config.to_dict()
            elif hasattr(config, "api"):
                config_dict = {
                    "api": {
                        "endpoint": config.api.endpoint or "",
                        "deployment": config.api.deployment or "",
                        "api_version": config.api.api_version or "",
                    }
                }
            creds = resolve_credentials(config_dict)

            if not resolved_key and creds.api_key:
                resolved_key = creds.api_key
            if creds.endpoint:
                resolved_endpoint = creds.endpoint
            if creds.deployment:
                resolved_deployment = creds.deployment
            if creds.api_version:
                resolved_api_version = creds.api_version

            self.logger.info(
                "llm_router_creds_resolved",
                key_source=creds.source_key or "caller",
                endpoint_source=creds.source_endpoint or "config",
            )
        except ImportError:
            # If credentials module can't load, fall through to no-API mode
            self.logger.warning("llm_router_credentials_import_failed")

        # NOTE: We do NOT mutate config.api.endpoint here.
        # The gate was already configured with the correct endpoint by
        # boot_hybridrag() or configure_gate(). Mutating config after
        # gate configuration creates invisible failures where the gate
        # blocks requests to the new endpoint.

        # -- Create the API router (only if we have a key) --
        if resolved_key:
            self.api = APIRouter(
                config, resolved_key, resolved_endpoint,
                deployment_override=resolved_deployment,
                api_version_override=resolved_api_version,
            )
            self.logger.info("llm_router_init", api_mode="enabled")
        else:
            self.api = None
            self.logger.info("llm_router_init", api_mode="disabled_no_key")

    def query(self, prompt: str) -> Optional[LLMResponse]:
        """
        Route a query to the appropriate backend based on config.mode.

        Args:
            prompt: The complete prompt (context + user question)

        Returns:
            LLMResponse with the answer, or None if the call failed
        """
        mode = self.config.mode

        self.logger.info("query_mode", mode=mode)

        if mode == "online":
            if self.api is None:
                self.logger.error(
                    "api_not_configured",
                    hint="Run rag-store-key and rag-store-endpoint first",
                )
                return None
            return self.api.query(prompt)

        else:
            return self.ollama.query(prompt)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of all LLM backends.
        Used by rag-cred-status, rag-test-api, and diagnostics.
        """
        status = {
            "mode": self.config.mode,
            "ollama_available": self.ollama.is_available(),
            "api_configured": self.api is not None,
            "sdk_available": OPENAI_SDK_AVAILABLE,
        }

        if self.api:
            api_status = self.api.get_status()
            status.update({
                "api_provider": api_status["provider"],
                "api_endpoint": api_status["endpoint"],
                "api_sdk": api_status.get("sdk", "unknown"),
            })
            if api_status.get("deployment"):
                status["api_deployment"] = api_status["deployment"]
                status["api_version"] = api_status["api_version"]
                status["api_clean_endpoint"] = api_status.get("clean_endpoint", "")

        return status






