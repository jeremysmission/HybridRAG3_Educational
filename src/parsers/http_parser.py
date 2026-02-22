# ============================================================================
# HybridRAG -- HTTP/Intranet Page Parser (src/parsers/http_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Fetches a web page from an HTTP(S) URL and extracts readable text.
#   Designed for intranet documentation: SharePoint pages, Confluence,
#   internal wikis, and any HTML-based documentation system.
#
# HOW IT WORKS:
#   1. Checks NetworkGate (respects offline/online/admin modes)
#   2. Fetches the URL via the centralized HTTP client (urllib)
#   3. Strips HTML tags and extracts text via html_parser.py
#   4. Returns text + diagnostic details
#
# NETWORK ACCESS: YES -- fetches the specified URL.
#   Gated by NetworkGate. In offline mode, all fetches are blocked.
#   In online mode, only the configured API endpoint is allowed.
#   In admin mode, any URL is allowed.
#
# DESIGN DECISIONS:
#   - Uses stdlib urllib (no httpx/requests dependency)
#   - Respects NetworkGate for access control
#   - Handles common encodings (UTF-8, Latin-1, Windows-1252)
#   - Max content size cap to prevent memory issues
#   - Timeout enforcement (no hanging on slow pages)
#
# USAGE:
#   This parser is NOT registered in the extension-based registry
#   (URLs don't have file extensions). Instead, it is called directly:
#
#     from src.parsers.http_parser import HttpParser
#     parser = HttpParser()
#     text, details = parser.fetch_and_parse("https://intranet.company.com/docs/page")
# ============================================================================

from __future__ import annotations

import logging
import re
import ssl
import time
import urllib.request
import urllib.error
from typing import Tuple, Dict, Any, Optional

from .html_parser import extract_text_from_html

logger = logging.getLogger(__name__)

# Maximum content size to download (10 MB)
_MAX_CONTENT_BYTES = 10 * 1024 * 1024

# Default timeout for HTTP requests (seconds)
_DEFAULT_TIMEOUT = 30

# Content types we can parse
_PARSEABLE_TYPES = frozenset({
    "text/html", "text/plain", "text/xml",
    "application/xhtml+xml", "application/xml",
})


class HttpParser:
    """
    Fetches and parses web pages from HTTP(S) URLs.

    Respects the NetworkGate for access control. In offline mode,
    all fetches are blocked. In online/admin mode, fetches are
    allowed to permitted endpoints.

    Usage:
        parser = HttpParser()
        text, details = parser.fetch_and_parse(url)
        if details.get("error"):
            handle_error(details["error"])
    """

    def __init__(
        self,
        timeout: int = _DEFAULT_TIMEOUT,
        max_bytes: int = _MAX_CONTENT_BYTES,
        user_agent: str = "HybridRAG/3.0",
        verify_ssl: bool = True,
    ):
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._user_agent = user_agent
        self._verify_ssl = verify_ssl

    def fetch_and_parse(
        self,
        url: str,
        purpose: str = "http_parser",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Fetch a URL and extract text content.

        Args:
            url:     The HTTP(S) URL to fetch.
            purpose: Audit label for NetworkGate logging.

        Returns:
            (text, details) tuple. On failure, text is empty and
            details["error"] contains the error message.
        """
        details: Dict[str, Any] = {
            "parser": "HttpParser",
            "url": url,
            "purpose": purpose,
        }

        start = time.time()

        # --- Step 1: Network gate check ---
        try:
            from src.core.network_gate import get_gate
            gate = get_gate()
            gate.check_allowed(url, purpose, "http_parser")
        except ImportError:
            # NetworkGate not available -- allow (development mode)
            logger.warning("NetworkGate not available, allowing fetch")
        except Exception as e:
            details["error"] = f"Network blocked: {e}"
            details["latency_ms"] = (time.time() - start) * 1000
            return "", details

        # --- Step 2: Fetch the URL ---
        try:
            html, fetch_details = self._fetch_url(url)
            details.update(fetch_details)
        except Exception as e:
            details["error"] = f"Fetch failed: {type(e).__name__}: {e}"
            details["latency_ms"] = (time.time() - start) * 1000
            return "", details

        if not html:
            details["latency_ms"] = (time.time() - start) * 1000
            return "", details

        # --- Step 3: Parse HTML to text ---
        content_type = details.get("content_type", "text/html")
        if "text/plain" in content_type:
            # Plain text -- no HTML parsing needed
            text = html
            details["parse_method"] = "plain_text"
        else:
            text, parse_details = extract_text_from_html(html)
            details["html_parse"] = parse_details
            details["parse_method"] = "html_strip"

        details["text_length"] = len(text)
        details["latency_ms"] = (time.time() - start) * 1000

        return text, details

    def _fetch_url(self, url: str) -> Tuple[str, Dict[str, Any]]:
        """
        Fetch raw content from a URL using stdlib urllib.

        Returns:
            (content_string, fetch_details)
        """
        details: Dict[str, Any] = {}

        # Build SSL context
        ssl_ctx: Optional[ssl.SSLContext] = None
        if url.startswith("https"):
            ssl_ctx = ssl.create_default_context()
            if not self._verify_ssl:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE

        # Build request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "text/html, text/plain, application/xhtml+xml, */*",
                "Accept-Encoding": "identity",
            },
        )

        # Execute request
        response = urllib.request.urlopen(
            req,
            timeout=self._timeout,
            context=ssl_ctx,
        )

        details["status_code"] = response.status
        details["content_type"] = response.headers.get(
            "Content-Type", "text/html"
        )

        # Check content type
        ct = details["content_type"].split(";")[0].strip().lower()
        if ct not in _PARSEABLE_TYPES:
            details["error"] = f"Unsupported content type: {ct}"
            return "", details

        # Read with size limit
        raw = response.read(self._max_bytes)
        details["content_bytes"] = len(raw)

        # Detect encoding
        encoding = self._detect_encoding(
            details["content_type"], raw[:1024]
        )
        details["encoding"] = encoding

        try:
            content = raw.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            content = raw.decode("utf-8", errors="replace")
            details["encoding_fallback"] = "utf-8"

        return content, details

    def _detect_encoding(
        self, content_type: str, sample: bytes
    ) -> str:
        """
        Detect character encoding from Content-Type header and content.

        Priority:
          1. charset in Content-Type header
          2. meta charset in HTML head
          3. BOM detection
          4. Default to UTF-8
        """
        # Check Content-Type header
        ct_lower = content_type.lower()
        match = re.search(r'charset\s*=\s*([^\s;]+)', ct_lower)
        if match:
            return match.group(1).strip('"\'')

        # Check HTML meta tag
        try:
            head = sample.decode("ascii", errors="ignore")
            meta = re.search(
                r'<meta[^>]+charset\s*=\s*["\']?([^"\'\s>]+)',
                head, re.IGNORECASE,
            )
            if meta:
                return meta.group(1)
        except Exception:
            pass

        # BOM detection
        if sample[:3] == b'\xef\xbb\xbf':
            return "utf-8"
        if sample[:2] in (b'\xff\xfe', b'\xfe\xff'):
            return "utf-16"

        return "utf-8"
