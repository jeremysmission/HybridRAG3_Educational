# ===========================================================================
# HybridRAG v3 -- TYPED EXCEPTIONS
# ===========================================================================
# FILE: src/core/exceptions.py
#
# WHAT THIS IS:
#   Custom error types for HybridRAG. Instead of generic Python errors
#   that say "something went wrong," these tell you EXACTLY what failed
#   and HOW TO FIX IT.
#
# WHY THIS MATTERS:
#   When 10 users share the system and something breaks, you don't want
#   to debug each one manually. Typed exceptions give each error a clear
#   name, message, and fix suggestion that can be shown in the GUI or
#   printed to the console.
#
# ANALOGY:
#   Think of car dashboard warning lights. A generic "CHECK ENGINE" light
#   could mean anything. But "LOW OIL PRESSURE" tells you exactly what's
#   wrong and what to do. These exceptions are like specific warning lights.
#
# HOW IT'S USED:
#   Instead of:  raise Exception("API call failed")
#   We write:    raise AuthRejectedError("401 from Azure. Check API key.")
#
#   The caller catches the specific type and shows the right message:
#     try:
#         result = api_client.query("What is X?")
#     except AuthRejectedError as e:
#         show_user("Authentication failed. Contact admin.")
#     except EndpointNotConfiguredError as e:
#         show_user("API not configured. Run setup first.")
#     except HybridRAGError as e:
#         show_user(f"Error: {e} -- Fix: {e.fix_suggestion}")
#
# DESIGN DECISION:
#   All exceptions inherit from HybridRAGError, which inherits from
#   Python's built-in Exception. This means:
#     - "except HybridRAGError" catches ALL our custom errors
#     - "except AuthRejectedError" catches only auth errors
#     - "except Exception" still catches everything (safety net)
#   This is called an "exception hierarchy" and it's standard practice
#   in production Python applications.
# ===========================================================================

from __future__ import annotations


class HybridRAGError(Exception):
    """
    Base class for all HybridRAG errors.

    Every custom exception below inherits from this, so you can
    catch ALL HybridRAG errors with one line:
        except HybridRAGError as e:

    Attributes:
        fix_suggestion (str | None): Human-readable fix instruction.
        error_code (str | None): Machine-readable code like "CONF-001"
            for logging, dashboards, and audit trails.
    """

    def __init__(self, message, fix_suggestion=None, error_code=None):
        self.fix_suggestion = fix_suggestion
        self.error_code = error_code
        super().__init__(message)

    def to_dict(self):
        """
        Convert to dictionary for JSON logging or GUI display.
        Useful for the flight recorder and audit logs.
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": str(self),
            "fix_suggestion": self.fix_suggestion,
        }


# ---------------------------------------------------------------------------
# CONFIGURATION ERRORS (CONF-xxx)
# These happen before any API call -- something is missing or wrong in setup.
# ---------------------------------------------------------------------------

class EndpointNotConfiguredError(HybridRAGError):
    """
    API endpoint URL is missing or empty.

    WHEN YOU'LL SEE THIS:
      - First time setup, before running rag-store-endpoint
      - If someone clears credentials and forgets to re-set them
      - If config.yaml has api.endpoint: '' (empty string)
    """
    def __init__(self, message=None):
        super().__init__(
            message or "API endpoint is not configured.",
            fix_suggestion="Run 'rag-store-endpoint' to set your Azure endpoint URL.",
            error_code="CONF-001",
        )


class ApiKeyNotConfiguredError(HybridRAGError):
    """
    API key is missing from both keyring and environment variables.

    WHEN YOU'LL SEE THIS:
      - First time setup, before running rag-store-key
      - If keyring loses the stored key (rare but possible after OS update)
      - If environment variable was set in a session that has closed
    """
    def __init__(self, message=None):
        super().__init__(
            message or "API key is not configured.",
            fix_suggestion="Run 'rag-store-key' to store your Azure API key.",
            error_code="CONF-002",
        )


class InvalidEndpointError(HybridRAGError):
    """
    Endpoint URL is present but malformed.

    WHEN YOU'LL SEE THIS:
      - URL missing https:// scheme
      - URL contains smart quotes or hidden Unicode characters
      - URL has double slashes in the path
      - URL has trailing whitespace
      - URL contains spaces
    """
    def __init__(self, message=None, url=None):
        detail = f" Got: '{url}'" if url else ""
        super().__init__(
            message or f"API endpoint URL is malformed.{detail}",
            fix_suggestion=(
                "Check the endpoint URL for typos, smart quotes, or missing "
                "'https://'. Run 'rag-store-endpoint' to re-enter it cleanly."
            ),
            error_code="CONF-003",
        )


class DeploymentNotConfiguredError(HybridRAGError):
    """
    Azure deployment name is needed but not set.

    WHEN YOU'LL SEE THIS:
      - Azure provider selected but no deployment name found
      - Endpoint is just the base URL without /deployments/ in the path
      - No AZURE_OPENAI_DEPLOYMENT environment variable set
    """
    def __init__(self, message=None):
        super().__init__(
            message or "Azure deployment name is not configured.",
            fix_suggestion=(
                "Run 'rag-store-deployment' or set the "
                "AZURE_OPENAI_DEPLOYMENT environment variable."
            ),
            error_code="CONF-004",
        )


class ProviderConfigError(HybridRAGError):
    """
    Provider or auth scheme configuration is invalid.

    WHEN YOU'LL SEE THIS:
      - config.yaml has api.provider set to unrecognized value
      - auth scheme doesn't match provider requirements
    """
    def __init__(self, message=None):
        super().__init__(
            message or "API provider configuration is invalid.",
            fix_suggestion=(
                "Check config.yaml: api.provider should be "
                "'azure', 'openai', or 'auto'."
            ),
            error_code="CONF-005",
        )


class ApiVersionNotConfiguredError(HybridRAGError):
    """
    Azure API version is needed but not set.

    WHEN YOU'LL SEE THIS:
      - Azure provider but no api-version in URL or env vars
    """
    def __init__(self, message=None):
        super().__init__(
            message or "Azure API version is not configured.",
            fix_suggestion=(
                "Run 'rag-store-api-version' or set "
                "AZURE_OPENAI_API_VERSION environment variable. "
                "Common values: 2024-02-01, 2024-06-01"
            ),
            error_code="CONF-006",
        )


# ---------------------------------------------------------------------------
# NETWORK / CONNECTION ERRORS (NET-xxx)
# These happen when trying to reach the API server.
# ---------------------------------------------------------------------------

class ConnectionFailedError(HybridRAGError):
    """
    HTTP request can't reach the server at all.

    WHEN YOU'LL SEE THIS:
      - No network connection
      - VPN disconnected
      - Firewall blocking the request
      - DNS can't resolve the hostname
      - Server is down
    """
    def __init__(self, message=None, host=None):
        detail = f" Host: {host}" if host else ""
        super().__init__(
            message or f"Cannot connect to API server.{detail}",
            fix_suggestion=(
                "Check: (1) Are you on the right network/VPN? "
                "(2) Can you ping the server? "
                "(3) Is a proxy required? Run 'rag-net-check' to diagnose."
            ),
            error_code="NET-001",
        )


class TLSValidationError(HybridRAGError):
    """
    SSL/TLS certificate verification failed.

    WHEN YOU'LL SEE THIS:
      - Corporate proxy intercepting HTTPS (SSL inspection)
      - Self-signed certificate on internal server
      - Expired certificate
      - Wrong CA bundle configured
    """
    def __init__(self, message=None):
        super().__init__(
            message or "SSL/TLS certificate verification failed.",
            fix_suggestion=(
                "This usually means a corporate proxy is intercepting HTTPS. "
                "Run 'rag-ssl-check' to diagnose. "
                "Ask IT for the corporate CA certificate."
            ),
            error_code="NET-002",
        )


class ProxyError(HybridRAGError):
    """
    Proxy-related connection failure.

    WHEN YOU'LL SEE THIS:
      - Proxy server is down or unreachable
      - Proxy requires authentication
      - Wrong proxy address configured
    """
    def __init__(self, message=None):
        super().__init__(
            message or "Proxy connection failed.",
            fix_suggestion=(
                "Run 'rag-proxy-check' to see current proxy settings. "
                "Check HTTP_PROXY and HTTPS_PROXY environment variables."
            ),
            error_code="NET-003",
        )


class TimeoutError(HybridRAGError):
    """
    Request timed out waiting for server response.

    WHEN YOU'LL SEE THIS:
      - Server is slow or overloaded
      - Network latency is too high
      - Request payload is too large
    """
    def __init__(self, message=None, timeout_seconds=None):
        detail = f" (timeout: {timeout_seconds}s)" if timeout_seconds else ""
        super().__init__(
            message or f"API request timed out.{detail}",
            fix_suggestion=(
                "The server didn't respond in time. "
                "Try again, or increase timeout in config.yaml."
            ),
            error_code="NET-004",
        )


# ---------------------------------------------------------------------------
# API / AUTH ERRORS (API-xxx)
# These happen when the server responds but rejects the request.
# ---------------------------------------------------------------------------

class AuthRejectedError(HybridRAGError):
    """
    Server returned 401 Unauthorized.

    WHEN YOU'LL SEE THIS:
      - API key is wrong or expired
      - Wrong auth header format (Bearer vs api-key)
      - Key doesn't have access to this resource
    """
    def __init__(self, message=None, status_code=None):
        code = f" (HTTP {status_code})" if status_code else ""
        super().__init__(
            message or f"Authentication rejected by server.{code}",
            fix_suggestion=(
                "Check: (1) Is the API key correct and not expired? "
                "Run 'rag-show-creds' to verify. "
                "(2) Is the auth scheme correct? Azure uses 'api-key' header, "
                "not 'Authorization: Bearer'."
            ),
            error_code="API-001",
        )


class ForbiddenError(HybridRAGError):
    """
    Server returned 403 Forbidden.

    WHEN YOU'LL SEE THIS:
      - Key doesn't have permission for this deployment
      - IP address is not whitelisted
      - Resource access policy blocks the request
    """
    def __init__(self, message=None):
        super().__init__(
            message or "Access forbidden. Key lacks permission.",
            fix_suggestion=(
                "Your API key may not have access to this deployment. "
                "Check permissions in Azure Portal."
            ),
            error_code="API-002",
        )


class DeploymentNotFoundError(HybridRAGError):
    """
    Server returned 404 Not Found.

    WHEN YOU'LL SEE THIS:
      - Deployment name is wrong
      - API version is not supported
      - URL path is incorrect
    """
    def __init__(self, message=None, deployment=None):
        detail = f" Deployment: '{deployment}'" if deployment else ""
        super().__init__(
            message or f"API endpoint not found (404).{detail}",
            fix_suggestion=(
                "The deployment name or API version may be wrong. "
                "Run 'rag-store-deployment' to set the correct name. "
                "Check Azure Portal > Azure OpenAI > Deployments."
            ),
            error_code="API-003",
        )


class RateLimitedError(HybridRAGError):
    """
    Server returned 429 Too Many Requests.

    WHEN YOU'LL SEE THIS:
      - Too many requests per minute
      - Token quota exceeded
      - Shared key being used by multiple people simultaneously
    """
    def __init__(self, message=None, retry_after=None):
        detail = f" Retry after: {retry_after}s" if retry_after else ""
        super().__init__(
            message or f"Rate limited by API server.{detail}",
            fix_suggestion=(
                "Too many requests. Wait a minute and try again. "
                "If this happens often with 10 users, request a higher quota."
            ),
            error_code="API-004",
        )


class ServerError(HybridRAGError):
    """
    Server returned 5xx error.

    WHEN YOU'LL SEE THIS:
      - Azure/OpenAI service is having problems
      - Temporary server-side failure
    """
    def __init__(self, message=None, status_code=None):
        code = f" (HTTP {status_code})" if status_code else ""
        super().__init__(
            message or f"API server error.{code}",
            fix_suggestion=(
                "The server is having problems. "
                "Wait a few minutes and try again. "
                "If persistent, check Azure service status."
            ),
            error_code="API-005",
        )


class UnexpectedResponseError(HybridRAGError):
    """
    Server responded but the response format is unexpected.

    WHEN YOU'LL SEE THIS:
      - Response is not valid JSON
      - Response is missing expected fields (choices, message, content)
      - Response is HTML (often a proxy error page)
    """
    def __init__(self, message=None):
        super().__init__(
            message or "Unexpected response format from API.",
            fix_suggestion=(
                "The server responded but not with the expected format. "
                "This may indicate a proxy intercept or wrong endpoint. "
                "Run 'rag-debug-url' to verify the URL."
            ),
            error_code="API-006",
        )


# ---------------------------------------------------------------------------
# OLLAMA ERRORS (OLL-xxx)
# ---------------------------------------------------------------------------

class OllamaNotRunningError(HybridRAGError):
    """Ollama service is not running."""
    def __init__(self, message=None):
        super().__init__(
            message or "Ollama is not running.",
            fix_suggestion="Run 'rag-ollama-start' to start the Ollama service.",
            error_code="OLL-001",
        )


class OllamaModelNotFoundError(HybridRAGError):
    """Requested Ollama model is not installed."""
    def __init__(self, message=None, model=None):
        detail = f" Model: '{model}'" if model else ""
        super().__init__(
            message or f"Ollama model not found.{detail}",
            fix_suggestion="Run 'rag-ollama-pull' to download the model.",
            error_code="OLL-002",
        )


# ---------------------------------------------------------------------------
# INDEX ERRORS (IDX-xxx)
# ---------------------------------------------------------------------------

class IndexNotFoundError(HybridRAGError):
    """No index database found."""
    def __init__(self, message=None):
        super().__init__(
            message or "No index database found.",
            fix_suggestion="Run 'rag-index' to index your documents first.",
            error_code="IDX-001",
        )


class IndexCorruptedError(HybridRAGError):
    """Index database is corrupted."""
    def __init__(self, message=None):
        super().__init__(
            message or "Index database appears corrupted.",
            fix_suggestion=(
                "Run 'rag-index-reset' to clear and rebuild the index."
            ),
            error_code="IDX-002",
        )


# ---------------------------------------------------------------------------
# HELPER: Map HTTP status codes to typed exceptions
# ---------------------------------------------------------------------------
# WHY THIS EXISTS:
#   The API client gets back an HTTP status code (401, 404, etc.) and
#   needs to raise the right typed exception. This function does that
#   mapping in one place so you never have to write if/elif chains
#   for status codes anywhere else.
#
# USAGE:
#   from src.core.exceptions import exception_from_http_status
#   raise exception_from_http_status(response.status_code, response.text)
# ---------------------------------------------------------------------------

def exception_from_http_status(status_code, response_body="", deployment=None):
    """
    Convert an HTTP status code into the appropriate typed exception.

    Args:
        status_code: HTTP status code (int)
        response_body: Response body text for extra context
        deployment: Deployment name (for 404 errors)

    Returns:
        A HybridRAGError subclass instance, ready to raise.
    """
    # Truncate response body to avoid huge error messages
    body_preview = response_body[:300] if response_body else ""

    if status_code == 401:
        return AuthRejectedError(
            f"HTTP 401 Unauthorized. Server response: {body_preview}",
            status_code=401,
        )
    elif status_code == 403:
        return ForbiddenError(
            f"HTTP 403 Forbidden. Server response: {body_preview}"
        )
    elif status_code == 404:
        return DeploymentNotFoundError(
            f"HTTP 404 Not Found. Server response: {body_preview}",
            deployment=deployment,
        )
    elif status_code == 429:
        return RateLimitedError(
            f"HTTP 429 Rate Limited. Server response: {body_preview}"
        )
    elif 500 <= status_code < 600:
        return ServerError(
            f"HTTP {status_code} Server Error. Response: {body_preview}",
            status_code=status_code,
        )
    else:
        return HybridRAGError(
            f"HTTP {status_code}. Response: {body_preview}",
            error_code=f"API-{status_code}",
        )
