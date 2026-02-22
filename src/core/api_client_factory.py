# ===========================================================================
# HybridRAG v3 -- API CLIENT FACTORY
# ===========================================================================
# FILE: src/core/api_client_factory.py
#
# WHAT THIS IS:
#   The GATE that prevents broken API clients from being created.
#   Before this redesign, the LLMRouter would create an API client
#   even when the endpoint was empty, leading to mysterious runtime
#   failures like "syntax error" or "invalid URL."
#
#   Now, the factory validates EVERYTHING before instantiating:
#     - Endpoint exists and is well-formed
#     - API key exists
#     - Provider is correctly detected
#     - Auth scheme matches provider
#     - Deployment name is present (for Azure)
#     - Final URL is correctly constructed
#
#   If ANY validation fails, you get a clear typed exception that
#   tells you exactly what's wrong and how to fix it.
#
# ANALOGY:
#   Think of an aircraft pre-flight checklist. A pilot doesn't start
#   the engines and hope for the best -- they verify fuel, instruments,
#   control surfaces, and weather BEFORE moving. This factory is the
#   pre-flight checklist for API connections.
#
# HOW IT'S USED:
#   from src.core.api_client_factory import ApiClientFactory
#   from src.security.credentials import resolve_credentials
#
#   creds = resolve_credentials()
#   factory = ApiClientFactory(config_dict)
#   client = factory.build(creds)  # Raises if anything is wrong
#   result = client.chat("What is X?")
#
# DESIGN DECISIONS:
#   - Provider detection uses EXPLICIT config.yaml setting first,
#     then falls back to URL pattern matching. No more silent guessing.
#   - URL construction is done ONCE, in ONE place, with clear logic.
#   - The ApiClient returned is a simple wrapper around HttpClient
#     that knows how to format chat completion requests.
# ===========================================================================

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from src.core.exceptions import (
    EndpointNotConfiguredError,
    ApiKeyNotConfiguredError,
    InvalidEndpointError,
    DeploymentNotConfiguredError,
    ProviderConfigError,
    ApiVersionNotConfiguredError,
    AuthRejectedError,
    DeploymentNotFoundError,
    RateLimitedError,
    ServerError,
    UnexpectedResponseError,
    ConnectionFailedError,       # BUG-01 fix: was missing, caused NameError on API failure
    exception_from_http_status,
)
from src.core.http_client import HttpClient, HttpResponse, create_http_client
from src.security.credentials import ApiCredentials, validate_endpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SUPPORTED PROVIDERS AND AUTH SCHEMES
# ---------------------------------------------------------------------------

# These are the recognized providers. "auto" means detect from URL.
VALID_PROVIDERS = ("azure", "openai", "auto")

# These are the recognized auth schemes.
VALID_AUTH_SCHEMES = ("api_key", "bearer", "auto")

# URL patterns that indicate Azure OpenAI (used by "auto" detection)
# Your company uses "aoai" which is Azure OpenAI abbreviated
AZURE_URL_PATTERNS = [
    "azure",
    ".openai.azure.com",
    "aoai",
    "azure-api",
    "cognitiveservices",
    "azure-cognitive",
]

# Default API version for Azure if not specified anywhere
# IMPORTANT: Must match llm_router.py _DEFAULT_API_VERSION
DEFAULT_AZURE_API_VERSION = "2024-02-02"


# ---------------------------------------------------------------------------
# API CLIENT: The object that actually makes chat completion requests
# ---------------------------------------------------------------------------

@dataclass
class ApiClientConfig:
    """
    Fully resolved, validated configuration for an API client.

    By the time this object exists, everything has been validated:
    the URL works, the auth is correct, and you can make requests.
    """
    provider: str           # "azure" or "openai"
    final_url: str          # Complete URL for chat completions
    auth_header_name: str   # "api-key" or "Authorization"
    auth_header_value: str  # The key or "Bearer {key}"
    deployment: Optional[str] = None
    api_version: Optional[str] = None
    model: Optional[str] = None  # For OpenAI: which model to request


class ApiClient:
    """
    Simple API client for chat completion requests.

    This is a thin wrapper around HttpClient that knows how to:
    1. Format chat completion request bodies
    2. Set the right auth headers
    3. Parse the response
    4. Raise typed exceptions on failure

    Usage:
        result = client.chat("What is a digisonde?", context_chunks=["..."])
    """

    def __init__(self, config: ApiClientConfig, http_client: HttpClient):
        self.config = config
        self.http = http_client

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context_chunks: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> dict:
        """
        Send a chat completion request to the API.

        Args:
            user_message: The user's question.
            system_prompt: Optional system prompt (instructions for the LLM).
            context_chunks: Optional list of retrieved document chunks to
                include as context. These get prepended to the user message.
            max_tokens: Maximum tokens in the response.
            temperature: Randomness (0.0 = deterministic, 1.0 = creative).

        Returns:
            Dict with keys:
                answer (str): The LLM's response text.
                model (str): Which model responded.
                usage (dict): Token counts.
                latency (float): Request time in seconds.
                provider (str): "azure" or "openai".

        Raises:
            AuthRejectedError: 401 -- key is wrong or expired.
            DeploymentNotFoundError: 404 -- deployment name is wrong.
            RateLimitedError: 429 -- too many requests.
            ServerError: 5xx -- server-side failure.
            UnexpectedResponseError: Response wasn't valid JSON.
            ConnectionFailedError: Couldn't reach the server.
        """
        # --- Build messages array ---
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # If we have context chunks from retrieval, include them
        if context_chunks:
            context_text = "\n\n---\n\n".join(context_chunks)
            messages.append({
                "role": "user",
                "content": (
                    f"Use ONLY the following source documents to answer. "
                    f"If the answer is not in the documents, say so.\n\n"
                    f"DOCUMENTS:\n{context_text}\n\n"
                    f"QUESTION: {user_message}"
                ),
            })
        else:
            messages.append({"role": "user", "content": user_message})

        # --- Build request body ---
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # OpenAI requires model in the body; Azure uses deployment in the URL
        if self.config.provider == "openai" and self.config.model:
            body["model"] = self.config.model

        # --- Build headers ---
        headers = {
            self.config.auth_header_name: self.config.auth_header_value,
            "Content-Type": "application/json",
        }

        # --- Send request ---
        logger.info(
            "API request: provider=%s, url=%s",
            self.config.provider,
            self.config.final_url[:80] + "...",
        )

        response = self.http.post(
            url=self.config.final_url,
            headers=headers,
            json_body=body,
        )

        # --- Handle response ---
        if response.is_success:
            data = response.json()
            answer = ""
            model = data.get("model", "unknown")
            usage = data.get("usage", {})

            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    answer = choice["message"]["content"]

            return {
                "answer": answer,
                "model": model,
                "usage": usage,
                "latency": response.latency_seconds,
                "provider": self.config.provider,
            }

        # --- Error handling ---
        if response.error and response.status_code == 0:
            # Network-level failure (already categorized by HttpClient)
            raise ConnectionFailedError(response.error)

        # HTTP error -- raise typed exception
        raise exception_from_http_status(
            response.status_code,
            response.body,
            deployment=self.config.deployment,
        )

    def test_connection(self) -> dict:
        """
        Send a minimal test request to verify the connection works.

        Returns:
            Dict with status, latency, model, and any error info.
        """
        try:
            result = self.chat(
                user_message="Say hello in exactly 3 words.",
                max_tokens=20,
                temperature=0.1,
            )
            return {
                "success": True,
                "latency": result["latency"],
                "model": result["model"],
                "response": result["answer"],
                "provider": result["provider"],
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "fix": getattr(e, "fix_suggestion", None),
            }

    def get_diagnostic_info(self) -> dict:
        """
        Return diagnostic info about this client configuration.
        Safe to print/log -- no secrets exposed.
        """
        return {
            "provider": self.config.provider,
            "final_url": self.config.final_url,
            "auth_header": self.config.auth_header_name,
            "deployment": self.config.deployment,
            "api_version": self.config.api_version,
        }


# ---------------------------------------------------------------------------
# API CLIENT FACTORY: The validation gate
# ---------------------------------------------------------------------------

class ApiClientFactory:
    """
    Factory that validates all configuration and credentials before
    creating an ApiClient. If anything is wrong, you get a clear
    typed exception instead of a silent broken client.

    Usage:
        factory = ApiClientFactory(config_dict)
        client = factory.build(credentials)

    The config_dict comes from config.yaml and should have:
        api:
          provider: "azure"          # or "openai" or "auto"
          auth_scheme: "api_key"     # or "bearer" or "auto"
          deployment: "gpt-35-turbo" # Azure deployment name
          api_version: "2024-02-02"  # Azure API version
          model: "gpt-3.5-turbo"    # OpenAI model name (if provider is openai)
    """

    def __init__(self, config_dict: Optional[dict] = None):
        self.config_dict = config_dict or {}
        self.api_config = self.config_dict.get("api", {})
        if not isinstance(self.api_config, dict):
            self.api_config = {}

    def build(self, credentials: ApiCredentials) -> ApiClient:
        """
        Validate everything and create an ApiClient.

        This is the main entry point. It performs a complete pre-flight
        check and either returns a working client or raises a specific
        exception explaining what's wrong.

        Args:
            credentials: Resolved ApiCredentials from the credentials module.

        Returns:
            ApiClient ready to make requests.

        Raises:
            EndpointNotConfiguredError: No endpoint found.
            ApiKeyNotConfiguredError: No API key found.
            InvalidEndpointError: Endpoint URL is malformed.
            DeploymentNotConfiguredError: Azure needs deployment but none set.
            ProviderConfigError: Invalid provider or auth scheme.
        """
        # --- Step 1: Verify credentials exist ---
        if not credentials.has_endpoint:
            raise EndpointNotConfiguredError()

        if not credentials.has_key:
            raise ApiKeyNotConfiguredError()

        # --- Step 2: Validate endpoint URL ---
        endpoint = validate_endpoint(credentials.endpoint)

        # --- Step 3: Detect provider ---
        provider = self._detect_provider(endpoint)
        logger.info("Provider detected: %s", provider)

        # --- Step 4: Resolve deployment and API version ---
        deployment = self._resolve_deployment(credentials, provider)
        api_version = self._resolve_api_version(credentials, provider)

        # --- Step 5: Build the final URL ---
        final_url = self._build_url(endpoint, provider, deployment, api_version)
        logger.info("Final URL constructed: %s", final_url[:80] + "...")

        # --- Step 6: Determine auth scheme ---
        auth_header_name, auth_header_value = self._resolve_auth(
            credentials.api_key, provider
        )

        # --- Step 7: Create the HTTP client ---
        http_client = create_http_client(self.config_dict)

        # --- Step 8: Build and return the ApiClient ---
        client_config = ApiClientConfig(
            provider=provider,
            final_url=final_url,
            auth_header_name=auth_header_name,
            auth_header_value=auth_header_value,
            deployment=deployment,
            api_version=api_version,
            model=self.api_config.get("model"),
        )

        logger.info(
            "ApiClient created: provider=%s, deployment=%s, auth=%s",
            provider, deployment, auth_header_name,
        )

        return ApiClient(client_config, http_client)

    # -----------------------------------------------------------------------
    # STEP 3: Provider detection
    # -----------------------------------------------------------------------

    def _detect_provider(self, endpoint: str) -> str:
        """
        Detect whether the endpoint is Azure or standard OpenAI.

        Resolution order:
        1. Explicit config.yaml setting (api.provider)
        2. URL pattern matching (auto-detection)

        DESIGN NOTE:
          Explicit config is preferred because URL pattern matching
          is inherently fragile. Enterprise URLs don't always contain
          "azure" in the name. Setting api.provider: "azure" in
          config.yaml eliminates all guessing.
        """
        # Check explicit config first
        explicit_provider = self.api_config.get("provider", "auto").strip().lower()

        if explicit_provider in ("azure", "openai"):
            return explicit_provider

        if explicit_provider != "auto":
            raise ProviderConfigError(
                f"Unrecognized api.provider: '{explicit_provider}'. "
                f"Valid options: azure, openai, auto"
            )

        # Auto-detect from URL patterns
        url_lower = endpoint.lower()
        for pattern in AZURE_URL_PATTERNS:
            if pattern in url_lower:
                logger.info(
                    "Auto-detected Azure from URL pattern: '%s'", pattern
                )
                return "azure"

        # Default to OpenAI if no Azure patterns found
        logger.info("No Azure patterns found in URL -- defaulting to OpenAI")
        return "openai"

    # -----------------------------------------------------------------------
    # STEP 4a: Resolve deployment name
    # -----------------------------------------------------------------------

    def _resolve_deployment(
        self, credentials: ApiCredentials, provider: str
    ) -> Optional[str]:
        """
        Resolve the Azure deployment name.

        Sources (priority order):
        1. Already extracted from URL by credentials module
        2. Config.yaml api.deployment
        3. Environment variable

        For Azure, deployment is REQUIRED. For OpenAI, it's not used.
        """
        if provider != "azure":
            return None

        # Already resolved by credentials module (from URL or env)
        if credentials.deployment:
            return credentials.deployment

        # Check config.yaml
        cfg_deployment = self.api_config.get("deployment", "").strip()
        if cfg_deployment:
            return cfg_deployment

        # Azure requires deployment name -- raise if we can't find it
        raise DeploymentNotConfiguredError(
            "Azure provider requires a deployment name, but none was found. "
            "Set it in config.yaml (api.deployment), environment variable "
            "(AZURE_OPENAI_DEPLOYMENT), or include it in the endpoint URL."
        )

    # -----------------------------------------------------------------------
    # STEP 4b: Resolve API version
    # -----------------------------------------------------------------------

    def _resolve_api_version(
        self, credentials: ApiCredentials, provider: str
    ) -> Optional[str]:
        """
        Resolve the Azure API version.

        For Azure, defaults to 2024-02-02 if not specified.
        For OpenAI, not used.
        """
        if provider != "azure":
            return None

        # Already resolved by credentials module (from URL or env)
        if credentials.api_version:
            return credentials.api_version

        # Check config.yaml
        cfg_version = self.api_config.get("api_version", "").strip()
        if cfg_version:
            return cfg_version

        # Use default -- don't raise, just warn
        logger.info(
            "No API version specified -- using default: %s",
            DEFAULT_AZURE_API_VERSION,
        )
        return DEFAULT_AZURE_API_VERSION

    # -----------------------------------------------------------------------
    # STEP 5: Build the final URL
    # -----------------------------------------------------------------------

    def _build_url(
        self,
        endpoint: str,
        provider: str,
        deployment: Optional[str],
        api_version: Optional[str],
    ) -> str:
        """
        Construct the final chat completions URL.

        This is where the URL doubling bug lived. By centralizing URL
        construction in ONE place with clear logic, we eliminate the
        possibility of double-appending paths.

        LOGIC:
        1. If URL already has /chat/completions -> use as-is
        2. If URL has /deployments/ but no /chat/completions -> append
        3. If URL is just the base -> build the full path

        Azure format:
          {base}/openai/deployments/{deployment}/chat/completions?api-version={ver}

        OpenAI format:
          {base}/v1/chat/completions
        """
        base = endpoint.rstrip("/")

        if provider == "azure":
            # Case 1: URL already has the full path
            if "/chat/completions" in base:
                # Just ensure api-version is present
                if "api-version" not in base and api_version:
                    separator = "&" if "?" in base else "?"
                    return f"{base}{separator}api-version={api_version}"
                return base

            # Case 2: URL has /deployments/ but not /chat/completions
            if "/deployments/" in base:
                url = f"{base}/chat/completions"
                if api_version:
                    url = f"{url}?api-version={api_version}"
                return url

            # Case 3: Just the base URL -- build everything
            url = f"{base}/openai/deployments/{deployment}/chat/completions"
            if api_version:
                url = f"{url}?api-version={api_version}"
            return url

        else:
            # OpenAI format
            if "/chat/completions" in base:
                return base
            if base.endswith("/v1"):
                return f"{base}/chat/completions"
            return f"{base}/v1/chat/completions"

    # -----------------------------------------------------------------------
    # STEP 6: Resolve auth scheme
    # -----------------------------------------------------------------------

    def _resolve_auth(
        self, api_key: str, provider: str
    ) -> tuple:
        """
        Determine the correct auth header name and value.

        Azure uses:    api-key: {key}
        OpenAI uses:   Authorization: Bearer {key}

        Can be overridden by config.yaml api.auth_scheme.

        Returns:
            Tuple of (header_name, header_value)
        """
        explicit_scheme = self.api_config.get("auth_scheme", "auto").strip().lower()

        if explicit_scheme == "api_key" or (explicit_scheme == "auto" and provider == "azure"):
            return ("api-key", api_key)
        elif explicit_scheme == "bearer" or (explicit_scheme == "auto" and provider == "openai"):
            return ("Authorization", f"Bearer {api_key}")
        elif explicit_scheme not in VALID_AUTH_SCHEMES:
            raise ProviderConfigError(
                f"Unrecognized api.auth_scheme: '{explicit_scheme}'. "
                f"Valid options: api_key, bearer, auto"
            )

        # Default fallback
        return ("api-key", api_key)

    # -----------------------------------------------------------------------
    # DIAGNOSTIC: Pre-flight check without building
    # -----------------------------------------------------------------------

    def diagnose(self, credentials: ApiCredentials) -> dict:
        """
        Run the full validation pipeline and return diagnostic info
        without actually creating the client or making any requests.

        Returns a dict with all resolved values and any problems found.
        Useful for rag-debug-url command.
        """
        result = {
            "credentials": credentials.to_diagnostic_dict(),
            "problems": [],
            "provider": None,
            "final_url": None,
            "auth_header": None,
            "deployment": None,
            "api_version": None,
        }

        # Check credentials
        if not credentials.has_endpoint:
            result["problems"].append({
                "code": "CONF-001",
                "message": "No endpoint configured",
                "fix": "Run rag-store-endpoint",
            })
            return result

        if not credentials.has_key:
            result["problems"].append({
                "code": "CONF-002",
                "message": "No API key configured",
                "fix": "Run rag-store-key",
            })

        # Validate endpoint
        try:
            endpoint = validate_endpoint(credentials.endpoint)
        except InvalidEndpointError as e:
            result["problems"].append({
                "code": e.error_code,
                "message": str(e),
                "fix": e.fix_suggestion,
            })
            return result

        # Detect provider
        try:
            provider = self._detect_provider(endpoint)
            result["provider"] = provider
        except ProviderConfigError as e:
            result["problems"].append({
                "code": e.error_code,
                "message": str(e),
                "fix": e.fix_suggestion,
            })
            return result

        # Resolve deployment
        try:
            deployment = self._resolve_deployment(credentials, provider)
            result["deployment"] = deployment
        except DeploymentNotConfiguredError as e:
            result["problems"].append({
                "code": e.error_code,
                "message": str(e),
                "fix": e.fix_suggestion,
            })
            # Don't return -- try to build URL anyway for diagnostic

        # Resolve API version
        api_version = self._resolve_api_version(credentials, provider)
        result["api_version"] = api_version

        # Build URL (best effort)
        try:
            if deployment or provider != "azure":
                final_url = self._build_url(endpoint, provider, deployment, api_version)
                result["final_url"] = final_url

                # Check for problems in the URL
                clean = final_url.replace("https://", "").replace("http://", "")
                if "//" in clean:
                    result["problems"].append({
                        "code": "URL-001",
                        "message": "Double slash in URL path",
                        "fix": "Check endpoint URL for extra slashes",
                    })
                if provider == "azure" and "/v1/chat" in final_url:
                    result["problems"].append({
                        "code": "URL-002",
                        "message": "OpenAI URL format on Azure endpoint",
                        "fix": "Set api.provider: azure in config.yaml",
                    })
        except Exception as e:
            result["problems"].append({
                "code": "URL-ERR",
                "message": f"URL construction failed: {e}",
                "fix": "Check endpoint, deployment, and API version settings",
            })

        # Resolve auth
        if credentials.has_key:
            try:
                auth_name, _ = self._resolve_auth(credentials.api_key, provider)
                result["auth_header"] = auth_name
            except ProviderConfigError as e:
                result["problems"].append({
                    "code": e.error_code,
                    "message": str(e),
                    "fix": e.fix_suggestion,
                })

        return result
