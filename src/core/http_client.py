# ===========================================================================
# HybridRAG v3 -- CENTRALIZED HTTP CLIENT
# ===========================================================================
# FILE: src/core/http_client.py
#
# WHAT THIS IS:
#   The ONE AND ONLY place that creates HTTP connections. Every module
#   that needs to talk to the network goes through this. No module
#   should ever create its own urllib/requests/httpx session directly.
#
# WHY THIS MATTERS:
#   In a corporate/production environment, HTTP connections need:
#     - Proxy configuration (corporate proxy servers)
#     - TLS/SSL certificate handling (corporate CA certs)
#     - Timeout enforcement (no hanging requests)
#     - Retry logic (transient failures)
#     - Audit logging (every request logged for compliance)
#
#   Before this redesign, proxy settings were scattered across
#   start_hybridrag.ps1, environment variables, and inline code.
#   A change in one place wouldn't affect the others. Now everything
#   is centralized here.
#
# DESIGN DECISIONS:
#   - Uses Python's built-in urllib.request (no extra dependency)
#     instead of httpx or requests. This keeps the "zero magic"
#     dependency philosophy -- one less thing to install and debug.
#   - Falls back gracefully if SSL context can't be created
#   - Logs every request (URL + method + status, never the body or key)
#   - Configurable via config.yaml, not hardcoded
#
# NETWORK KILL SWITCH:
#   Set HYBRIDRAG_OFFLINE=1 to block ALL outbound HTTP requests.
#   This is the master network kill switch for offline environments.
# ===========================================================================

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CONFIGURATION DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class HttpClientConfig:
    """
    Configuration for the centralized HTTP client.

    Can be populated from config.yaml or set programmatically.

    Attributes:
        timeout_seconds: Max time to wait for a response.
        max_retries: Number of retries for transient failures (5xx, timeout).
        retry_delay_seconds: Wait between retries (doubles each retry).
        ca_bundle_path: Path to custom CA certificate bundle (for corp proxies).
        verify_ssl: Whether to verify SSL certificates. ALWAYS True in production.
        user_agent: User-Agent header for requests.
        offline_mode: If True, block all HTTP requests.
    """
    timeout_seconds: int = 30
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    ca_bundle_path: Optional[str] = None
    verify_ssl: bool = True
    user_agent: str = "HybridRAG/3.0"
    offline_mode: bool = False


# ---------------------------------------------------------------------------
# RESPONSE DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class HttpResponse:
    """
    Structured HTTP response.

    This is what every HTTP call returns, giving the caller a consistent
    interface regardless of whether we use urllib, requests, or httpx
    under the hood.

    Attributes:
        status_code: HTTP status code (200, 401, 404, etc.)
        body: Response body as string.
        headers: Response headers as dict.
        latency_seconds: How long the request took.
        error: Error message if the request failed at the network level.
        is_success: True if status_code is 2xx.
    """
    status_code: int = 0
    body: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    latency_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        """Parse body as JSON. Returns empty dict on failure."""
        try:
            return json.loads(self.body)
        except (json.JSONDecodeError, TypeError):
            return {}


# ---------------------------------------------------------------------------
# CENTRALIZED HTTP CLIENT
# ---------------------------------------------------------------------------

class HttpClient:
    """
    Centralized HTTP client for all HybridRAG network operations.

    Usage:
        client = HttpClient(config)
        response = client.post(url, headers=headers, json_body=payload)
        if response.is_success:
            data = response.json()

    All proxy, TLS, timeout, and retry logic is handled internally.
    """

    def __init__(self, config: Optional[HttpClientConfig] = None):
        self.config = config or HttpClientConfig()

        # Legacy kill switch check (backward compatible)
        # The centralized NetworkGate now handles all access control,
        # but we still respect HYBRIDRAG_OFFLINE for backward compatibility
        # and as a secondary enforcement layer (enterprise-in-depth).
        if os.environ.get("HYBRIDRAG_OFFLINE", "").strip() in ("1", "true", "yes"):
            self.config.offline_mode = True
            logger.info("NETWORK KILL SWITCH: HYBRIDRAG_OFFLINE is set -- all HTTP blocked")

        # Build SSL context once (reused for all requests)
        self._ssl_context = self._build_ssl_context()

    def _build_ssl_context(self) -> ssl.SSLContext:
        """
        Build the SSL context used for all HTTPS requests.

        WHY THIS IS CENTRALIZED:
          Corporate environments often have proxy servers that intercept
          HTTPS traffic using their own CA certificate. Python's default
          SSL context won't trust that certificate, causing
          CERTIFICATE_VERIFY_FAILED errors.

          By centralizing SSL context creation, we can:
          1. Load the corporate CA bundle if configured
          2. Fall back to the OS trust store
          3. Provide clear error messages when SSL fails
        """
        ctx = ssl.create_default_context()

        if self.config.ca_bundle_path:
            # Load custom CA certificate (for corporate proxy SSL inspection)
            if os.path.exists(self.config.ca_bundle_path):
                ctx.load_verify_locations(self.config.ca_bundle_path)
                logger.info("Loaded custom CA bundle: %s", self.config.ca_bundle_path)
            else:
                logger.warning(
                    "CA bundle not found: %s -- using system defaults",
                    self.config.ca_bundle_path,
                )
        else:
            # Check common environment variables for CA bundle
            for env_var in ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"]:
                ca_path = os.environ.get(env_var, "")
                if ca_path and os.path.exists(ca_path):
                    ctx.load_verify_locations(ca_path)
                    logger.info("Loaded CA bundle from env %s: %s", env_var, ca_path)
                    break

        if not self.config.verify_ssl:
            # DANGER: Disabling SSL verification. Only for debugging.
            # This would NEVER pass a enterprise audit.
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            logger.warning("SSL VERIFICATION DISABLED -- not suitable for production")

        return ctx

    def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[dict] = None,
        raw_body: Optional[bytes] = None,
    ) -> HttpResponse:
        """
        Send an HTTP POST request.

        Args:
            url: Full URL to send to.
            headers: HTTP headers dict.
            json_body: Dict to serialize as JSON body.
            raw_body: Pre-encoded bytes body (overrides json_body).

        Returns:
            HttpResponse with status, body, latency, etc.
        """
        return self._request("POST", url, headers, json_body, raw_body)

    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> HttpResponse:
        """Send an HTTP GET request."""
        return self._request("GET", url, headers)

    def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_body: Optional[dict] = None,
        raw_body: Optional[bytes] = None,
    ) -> HttpResponse:
        """
        Internal method that handles the actual HTTP request with
        retry logic, timeout enforcement, and error classification.
        """
        # Import exceptions here to avoid circular imports
        from src.core.exceptions import (
            ConnectionFailedError,
            TLSValidationError,
            ProxyError,
        )

        # --- Kill switch check (legacy, backward compatible) ---
        if self.config.offline_mode:
            logger.warning("HTTP blocked by offline mode: %s %s", method, url)
            return HttpResponse(
                status_code=0,
                error="Network requests blocked. HYBRIDRAG_OFFLINE is enabled.",
            )

        # --- Network gate check (centralized access control) ---
        try:
            from src.core.network_gate import get_gate
            get_gate().check_allowed(url, "http_request", "http_client")
        except ImportError:
            # Gate module not available (shouldn't happen, but fail-open
            # here because the legacy kill switch above already caught
            # the truly offline case)
            pass
        except Exception as gate_error:
            logger.warning("HTTP blocked by network gate: %s %s -- %s", method, url, gate_error)
            return HttpResponse(
                status_code=0,
                error=f"Network gate blocked: {gate_error}",
            )

        # --- Prepare request ---
        all_headers = {"User-Agent": self.config.user_agent}
        if headers:
            all_headers.update(headers)

        if json_body is not None and raw_body is None:
            raw_body = json.dumps(json_body).encode("utf-8")
            if "Content-Type" not in all_headers:
                all_headers["Content-Type"] = "application/json"

        # --- Retry loop ---
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(1, self.config.max_retries + 2):  # +2 because range is exclusive
            start_time = time.time()

            try:
                req = urllib.request.Request(
                    url,
                    data=raw_body,
                    headers=all_headers,
                    method=method,
                )

                response = urllib.request.urlopen(
                    req,
                    context=self._ssl_context,
                    timeout=self.config.timeout_seconds,
                )

                latency = time.time() - start_time
                body = response.read().decode("utf-8")

                # Audit log: URL + status + latency (never log body or keys)
                logger.info(
                    "HTTP %s %s -> %d (%.2fs)",
                    method, _mask_url(url), response.status, latency,
                )

                return HttpResponse(
                    status_code=response.status,
                    body=body,
                    headers=dict(response.headers),
                    latency_seconds=latency,
                )

            except urllib.error.HTTPError as e:
                # Server responded with an error status code
                latency = time.time() - start_time
                error_body = ""
                try:
                    error_body = e.read().decode("utf-8")
                except Exception:
                    pass

                logger.warning(
                    "HTTP %s %s -> %d (%.2fs) attempt %d",
                    method, _mask_url(url), e.code, latency, attempt,
                )

                # Retry on 5xx or 429 (server errors / rate limit)
                if e.code in (429, 500, 502, 503, 504) and attempt <= self.config.max_retries:
                    # Check for Retry-After header
                    retry_after = e.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else delay
                    logger.info("Retrying in %.1fs...", wait)
                    time.sleep(wait)
                    delay *= 2  # Exponential backoff
                    continue

                # Non-retryable error -- return immediately
                return HttpResponse(
                    status_code=e.code,
                    body=error_body,
                    headers=dict(e.headers) if e.headers else {},
                    latency_seconds=latency,
                    error=f"HTTP {e.code}: {e.reason}",
                )

            except urllib.error.URLError as e:
                latency = time.time() - start_time
                reason = str(e.reason) if hasattr(e, "reason") else str(e)

                logger.error(
                    "HTTP %s %s -> CONNECTION FAILED (%.2fs) attempt %d: %s",
                    method, _mask_url(url), latency, attempt, reason,
                )

                # Classify the error
                reason_lower = reason.lower()
                if "ssl" in reason_lower or "certificate" in reason_lower:
                    last_error = TLSValidationError(
                        f"SSL/TLS error connecting to {_mask_url(url)}: {reason}"
                    )
                elif "proxy" in reason_lower:
                    last_error = ProxyError(
                        f"Proxy error connecting to {_mask_url(url)}: {reason}"
                    )
                else:
                    last_error = ConnectionFailedError(
                        f"Cannot connect to {_mask_url(url)}: {reason}"
                    )

                # Retry on connection errors
                if attempt <= self.config.max_retries:
                    logger.info("Retrying in %.1fs...", delay)
                    time.sleep(delay)
                    delay *= 2
                    continue

                # All retries exhausted
                return HttpResponse(
                    status_code=0,
                    latency_seconds=latency,
                    error=str(last_error),
                )

            except Exception as e:
                latency = time.time() - start_time
                logger.error(
                    "HTTP %s %s -> UNEXPECTED ERROR (%.2fs): %s",
                    method, _mask_url(url), latency, e,
                )
                return HttpResponse(
                    status_code=0,
                    latency_seconds=latency,
                    error=f"Unexpected error: {str(e)[:200]}",
                )

        # Should never reach here, but just in case
        return HttpResponse(status_code=0, error="All retries exhausted")


# ---------------------------------------------------------------------------
# HELPER: Mask URL for logging (hide API keys in query params)
# ---------------------------------------------------------------------------

def _mask_url(url):
    """
    Remove any API key from URL query parameters before logging.
    Example: ...?api-version=2024&api-key=abc123 -> ...?api-version=2024&api-key=****
    """
    import re
    return re.sub(
        r'(api[-_]?key=)[^&]+',
        r'\1****',
        url,
        flags=re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# FACTORY: Create configured HttpClient from config dict
# ---------------------------------------------------------------------------

def create_http_client(config_dict=None) -> HttpClient:
    """
    Create an HttpClient from a config dictionary (e.g., from config.yaml).

    Args:
        config_dict: Optional dict with keys like:
            http.timeout, http.max_retries, http.ca_bundle, http.verify_ssl

    Returns:
        Configured HttpClient instance.
    """
    cfg = HttpClientConfig()

    if config_dict:
        http_cfg = config_dict.get("http", {})
        if isinstance(http_cfg, dict):
            if "timeout" in http_cfg:
                cfg.timeout_seconds = int(http_cfg["timeout"])
            if "max_retries" in http_cfg:
                cfg.max_retries = int(http_cfg["max_retries"])
            if "retry_delay" in http_cfg:
                cfg.retry_delay_seconds = float(http_cfg["retry_delay"])
            if "ca_bundle" in http_cfg:
                cfg.ca_bundle_path = http_cfg["ca_bundle"]
            if "verify_ssl" in http_cfg:
                cfg.verify_ssl = bool(http_cfg["verify_ssl"])

    return HttpClient(cfg)
