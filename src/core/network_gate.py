# ===========================================================================
# HybridRAG v3 -- NETWORK GATE (Centralized Network Access Control)
# ===========================================================================
# FILE: src/core/network_gate.py
#
# WHAT THIS IS:
#   The ONE AND ONLY gatekeeper for all outbound network connections.
#   Every module that touches the network MUST check with this gate
#   before connecting. If the gate says no, the connection is blocked.
#
# WHY THIS MATTERS:
#   Before this redesign, network access was scattered:
#     - llm_router.py used raw httpx (no kill switch)
#     - http_client.py had its own HYBRIDRAG_OFFLINE check
#     - boot.py used urllib directly (no kill switch)
#     - fault_analysis.py used httpx directly (no kill switch)
#     - embedder.py had HF_HUB_OFFLINE (separate mechanism)
#
#   Result: 5 out of 8 network paths bypassed the kill switch.
#   A corporate proxy is the first enterprise layer, but enterprise-in-depth
#   means the APPLICATION ITSELF must enforce network policy too.
#
# NETWORK MODES:
#   offline  - Localhost only (Ollama). Zero internet traffic.
#              This is the DEFAULT mode. Safe for restricted environments.
#
#   online   - Localhost + configured API endpoint(s) only.
#              The API endpoint must be explicitly configured in YAML
#              or via rag-store-endpoint. No other destinations allowed.
#
#   admin    - Unrestricted. For maintenance only (pip install, model
#              downloads, updates). Must be explicitly activated by the
#              user. Logs a warning on every connection.
#
# HOW TO USE (for other modules):
#   from src.core.network_gate import get_gate
#   gate = get_gate()
#   gate.check_allowed("https://mycompany.openai.azure.com/...", "api_query")
#
#   If blocked, raises NetworkBlockedError.
#   If allowed, returns silently and logs the access.
#
# AUDIT TRAIL:
#   Every connection attempt (allowed AND denied) is logged with:
#     - timestamp, URL, purpose, mode, result (ALLOW/DENY)
#   This creates an auditable record for security review.
#
# INTERNET ACCESS: NONE (this module is the gatekeeper, not a consumer)
# ===========================================================================

from __future__ import annotations

import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NETWORK MODES
# ---------------------------------------------------------------------------

class NetworkMode(Enum):
    """
    The three allowed network access levels.

    Think of these like access authorization levels:
      OFFLINE = badge gets you into the building (localhost) only
      ONLINE  = badge gets you into the building + one approved external site
      ADMIN   = badge gets you anywhere (maintenance mode)
    """
    OFFLINE = "offline"     # Localhost only (Ollama)
    ONLINE = "online"       # Localhost + configured API endpoint
    ADMIN = "admin"         # Unrestricted (maintenance only)


# ---------------------------------------------------------------------------
# LOCALHOST PATTERNS (always allowed in OFFLINE and ONLINE modes)
# ---------------------------------------------------------------------------
# These are all the ways "localhost" can appear in a URL.
# Ollama runs on localhost:11434, so these must always be reachable
# when the system is in offline or online mode.
# ---------------------------------------------------------------------------

_LOCALHOST_HOSTS = frozenset([
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
])


# ---------------------------------------------------------------------------
# EXCEPTIONS
# ---------------------------------------------------------------------------

class NetworkBlockedError(Exception):
    """
    Raised when a network connection is denied by the gate.

    Attributes:
        url: The URL that was blocked.
        mode: The current network mode.
        reason: Human-readable explanation of why it was blocked.
    """
    def __init__(self, url: str, mode: str, reason: str):
        self.url = url
        self.mode = mode
        self.reason = reason
        super().__init__(
            f"NETWORK BLOCKED [{mode}]: {reason}\n"
            f"  URL: {url}\n"
            f"  To allow this connection, check your network mode and "
            f"allowed endpoints in config/default_config.yaml"
        )


# ---------------------------------------------------------------------------
# AUDIT LOG ENTRY
# ---------------------------------------------------------------------------

@dataclass
class NetworkAuditEntry:
    """
    One entry in the network access audit trail.

    Every connection attempt (allowed or denied) creates one of these.
    The audit trail can be exported for security review.
    """
    timestamp: float            # time.time() when the check happened
    url: str                    # The full URL that was checked
    host: str                   # Extracted hostname
    purpose: str                # Why the connection was made (e.g., "api_query")
    mode: str                   # Network mode at time of check
    allowed: bool               # True = connection was allowed
    reason: str                 # Why it was allowed/denied
    caller: str = ""            # Which module requested the connection

    def to_log_line(self) -> str:
        """Format as a single-line log entry for the audit trail."""
        result = "ALLOW" if self.allowed else "DENY"
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
        return (
            f"[{ts}] [{result}] [{self.mode}] "
            f"host={self.host} purpose={self.purpose} "
            f"caller={self.caller} reason={self.reason}"
        )


# ---------------------------------------------------------------------------
# NETWORK GATE (Singleton)
# ---------------------------------------------------------------------------

class NetworkGate:
    """
    Centralized network access control gate.

    DESIGN DECISIONS:
      - Singleton: All modules share one gate instance so policy is
        consistent across the entire application.
      - Fail-closed: If the mode is unknown or the gate isn't configured,
        ALL connections are blocked. Better to break functionality than
        leak data.
      - Audit everything: Both allowed and denied connections are logged.
        In a security review, you want to see BOTH.
      - URL-level control: We check hostnames, not IP addresses. This
        is practical because DNS resolution happens after our check.
        The corporate firewall handles IP-level blocking.
    """

    def __init__(self):
        """Initialize the gate in OFFLINE mode (safest default)."""
        self._mode = NetworkMode.OFFLINE
        self._allowed_hosts: List[str] = []
        self._allowed_prefixes: List[str] = []
        self._audit_log: List[NetworkAuditEntry] = []
        self._max_audit_entries = 1000  # Rolling buffer to prevent memory leak

    # -- MODE MANAGEMENT --

    @property
    def mode(self) -> NetworkMode:
        """Current network mode."""
        return self._mode

    @property
    def mode_name(self) -> str:
        """Current mode as a string (for logging/display)."""
        return self._mode.value

    def configure(
        self,
        mode: str,
        api_endpoint: str = "",
        allowed_prefixes: Optional[List[str]] = None,
    ) -> None:
        """
        Configure the gate with the current mode and allowed endpoints.

        Called during startup (boot.py or start_hybridrag.ps1 equivalent).
        Should be called ONCE at startup. Can be reconfigured if the user
        switches modes via rag-mode-online / rag-mode-offline.

        Args:
            mode: "offline", "online", or "admin"
            api_endpoint: The configured API endpoint URL (for online mode)
            allowed_prefixes: Additional URL prefixes to allow
        """
        # Resolve mode string to enum
        mode_lower = mode.strip().lower()

        # -- Legacy environment variable override (enterprise-in-depth) --
        # If HYBRIDRAG_OFFLINE is set, force offline mode regardless of
        # what the config says. This consolidates the kill switch that
        # was previously duplicated in http_client.py into the single
        # authoritative gate. One place, one check, one audit trail.
        if os.environ.get("HYBRIDRAG_OFFLINE", "").strip() in ("1", "true", "yes"):
            mode_lower = "offline"
            logger.info(
                "NETWORK GATE: HYBRIDRAG_OFFLINE env var set -- forcing offline mode"
            )

        if mode_lower == "online":
            self._mode = NetworkMode.ONLINE
        elif mode_lower == "admin":
            self._mode = NetworkMode.ADMIN
            logger.warning(
                "NETWORK GATE: ADMIN mode active -- all outbound connections allowed. "
                "Switch back to offline/online when maintenance is complete."
            )
        else:
            # Default to OFFLINE for any unrecognized mode
            self._mode = NetworkMode.OFFLINE

        # Build the allowed hosts list from the API endpoint
        self._allowed_hosts = []
        self._allowed_prefixes = list(allowed_prefixes or [])

        if api_endpoint:
            try:
                parsed = urlparse(api_endpoint)
                if parsed.hostname:
                    self._allowed_hosts.append(parsed.hostname.lower())
            except Exception:
                pass

        # NOTE: We intentionally do NOT add hosts from allowed_prefixes
        # to _allowed_hosts. Prefixes give path-level control:
        #   "https://proxy.corp.com/api/" allows /api/* but NOT /admin/*
        # Adding the host to _allowed_hosts would allow ALL paths on
        # that host, defeating the purpose of prefix-level control.

        logger.info(
            "NETWORK GATE configured: mode=%s, allowed_hosts=%s",
            self._mode.value,
            self._allowed_hosts or ["(localhost only)"],
        )

    # -- ACCESS CONTROL --

    def check_allowed(
        self,
        url: str,
        purpose: str = "unknown",
        caller: str = "",
    ) -> None:
        """
        Check if a URL is allowed under the current network policy.

        MUST be called before ANY outbound network connection.

        Args:
            url: The full URL being connected to.
            purpose: Why the connection is needed (for audit log).
                     Examples: "api_query", "ollama_check", "model_download"
            caller: Which module is making the request (for audit log).
                    Examples: "llm_router", "boot", "fault_analysis"

        Raises:
            NetworkBlockedError: If the connection is not allowed.

        Returns:
            None (silently) if the connection is allowed.
        """
        # Parse the URL to extract the hostname
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            scheme = (parsed.scheme or "").lower()
        except Exception:
            host = ""
            scheme = ""

        # -- ADMIN mode: allow everything (with warning) --
        if self._mode == NetworkMode.ADMIN:
            self._log_access(url, host, purpose, caller, True, "admin_mode")
            logger.warning(
                "NETWORK GATE [ADMIN]: Allowing %s to %s (purpose: %s)",
                caller or "unknown", host, purpose,
            )
            return

        # -- Reject non-HTTP(S) schemes --
        # Only http and https are valid for our use case.
        # ftp://, file://, data://, etc. are always blocked.
        # (Admin mode is exempt -- it needs ftp/etc for maintenance.)
        if scheme and scheme not in ("http", "https"):
            self._log_access(url, host, purpose, caller, False, f"blocked_scheme_{scheme}")
            raise NetworkBlockedError(
                url=url,
                mode=self._mode.value,
                reason=f"Only http:// and https:// are allowed. Got: {scheme}://",
            )

        # -- Check if it's localhost --
        is_localhost = host in _LOCALHOST_HOSTS

        # -- OFFLINE mode: localhost only --
        if self._mode == NetworkMode.OFFLINE:
            if is_localhost:
                self._log_access(url, host, purpose, caller, True, "localhost_allowed")
                return
            else:
                self._log_access(url, host, purpose, caller, False, "offline_blocks_internet")
                raise NetworkBlockedError(
                    url=url,
                    mode="offline",
                    reason=(
                        f"Offline mode blocks all internet access. "
                        f"Host '{host}' is not localhost. "
                        f"Switch to online mode with: rag-mode-online"
                    ),
                )

        # -- ONLINE mode: localhost + configured endpoints --
        if self._mode == NetworkMode.ONLINE:
            if is_localhost:
                self._log_access(url, host, purpose, caller, True, "localhost_allowed")
                return

            # Check against allowed hosts (extracted from API endpoint config)
            if host in self._allowed_hosts:
                self._log_access(url, host, purpose, caller, True, "allowed_host")
                return

            # Check against allowed URL prefixes
            for prefix in self._allowed_prefixes:
                if url.lower().startswith(prefix.lower()):
                    self._log_access(url, host, purpose, caller, True, "allowed_prefix")
                    return

            # Not in any allowlist
            self._log_access(url, host, purpose, caller, False, "host_not_in_allowlist")
            raise NetworkBlockedError(
                url=url,
                mode="online",
                reason=(
                    f"Host '{host}' is not in the allowed endpoints list. "
                    f"Allowed: {', '.join(self._allowed_hosts) or '(none configured)'}. "
                    f"Configure api.endpoint in config/default_config.yaml or use "
                    f"api.allowed_endpoint_prefixes to add more allowed destinations."
                ),
            )

        # -- Unknown mode: fail closed --
        self._log_access(url, host, purpose, caller, False, "unknown_mode")
        raise NetworkBlockedError(
            url=url,
            mode=str(self._mode),
            reason="Unknown network mode -- blocking all connections (fail-closed).",
        )

    def is_allowed(self, url: str) -> bool:
        """
        Non-raising version of check_allowed.

        Returns True if the URL would be allowed, False if it would be blocked.
        Does NOT log to the audit trail (use check_allowed for that).
        Useful for conditional logic where you want to check without raising.
        """
        try:
            # Temporarily suppress logging
            old_log = self._audit_log
            self._audit_log = []
            self.check_allowed(url, purpose="is_allowed_check")
            self._audit_log = old_log
            return True
        except NetworkBlockedError:
            self._audit_log = old_log
            return False

    # -- AUDIT LOG --

    def _log_access(
        self,
        url: str,
        host: str,
        purpose: str,
        caller: str,
        allowed: bool,
        reason: str,
    ) -> None:
        """Record an access attempt in the audit log."""
        entry = NetworkAuditEntry(
            timestamp=time.time(),
            url=url,
            host=host,
            purpose=purpose,
            mode=self._mode.value,
            allowed=allowed,
            reason=reason,
            caller=caller,
        )

        self._audit_log.append(entry)

        # Rolling buffer -- drop oldest entries if we exceed the max
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

        # Log to Python logger
        if allowed:
            logger.debug("NET %s: %s -> %s (%s)", "ALLOW", caller, host, reason)
        else:
            logger.warning("NET %s: %s -> %s (%s)", "DENY", caller, host, reason)

    def get_audit_log(self, last_n: int = 50) -> List[NetworkAuditEntry]:
        """Get the most recent audit log entries."""
        return self._audit_log[-last_n:]

    def get_audit_summary(self) -> dict:
        """
        Get a summary of network access for diagnostic display.

        Returns dict with:
            mode: Current mode name
            total_checks: Total number of checks performed
            allowed: Number of allowed connections
            denied: Number of denied connections
            allowed_hosts: List of configured allowed hosts
            unique_hosts_contacted: Set of hosts that were actually contacted
        """
        allowed_count = sum(1 for e in self._audit_log if e.allowed)
        denied_count = sum(1 for e in self._audit_log if not e.allowed)
        unique_hosts = set(e.host for e in self._audit_log if e.allowed)

        return {
            "mode": self._mode.value,
            "total_checks": len(self._audit_log),
            "allowed": allowed_count,
            "denied": denied_count,
            "allowed_hosts": list(self._allowed_hosts),
            "unique_hosts_contacted": sorted(unique_hosts),
        }

    def clear_audit_log(self) -> int:
        """Clear the audit log. Returns number of entries cleared."""
        count = len(self._audit_log)
        self._audit_log.clear()
        return count

    # -- STATUS / DISPLAY --

    def status_report(self) -> str:
        """
        Human-readable status report for rag-status or diagnostics.

        Shows current mode, allowed destinations, and recent activity.
        """
        lines = []
        lines.append(f"  Network Mode: {self._mode.value.upper()}")
        lines.append(f"  Allowed Hosts: {', '.join(self._allowed_hosts) or '(localhost only)'}")

        if self._allowed_prefixes:
            lines.append(f"  Allowed Prefixes: {', '.join(self._allowed_prefixes)}")

        summary = self.get_audit_summary()
        lines.append(
            f"  Connections: {summary['allowed']} allowed, "
            f"{summary['denied']} denied"
        )

        if summary["unique_hosts_contacted"]:
            lines.append(
                f"  Hosts Contacted: {', '.join(summary['unique_hosts_contacted'])}"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SINGLETON ACCESS
# ---------------------------------------------------------------------------
# WHY A SINGLETON:
#   Every module needs to check the SAME gate with the SAME policy.
#   If each module created its own gate, they could have different modes
#   or different allowlists, defeating the purpose of centralized control.
#
# The gate is created at import time in OFFLINE mode (safest default).
# It gets configured during startup (boot.py or start_hybridrag.ps1).
# ---------------------------------------------------------------------------

_gate_instance: Optional[NetworkGate] = None


def get_gate() -> NetworkGate:
    """
    Get the singleton NetworkGate instance.

    Creates it on first call (in OFFLINE mode).
    All subsequent calls return the same instance.

    Usage:
        from src.core.network_gate import get_gate
        gate = get_gate()
        gate.check_allowed("http://localhost:11434", "ollama_check", "boot")
    """
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = NetworkGate()
    return _gate_instance


def configure_gate(
    mode: str,
    api_endpoint: str = "",
    allowed_prefixes: Optional[List[str]] = None,
) -> NetworkGate:
    """
    Configure the singleton gate. Called once at startup.

    This is the main entry point for boot.py or start_hybridrag.ps1.

    Args:
        mode: "offline", "online", or "admin"
        api_endpoint: The configured API endpoint URL
        allowed_prefixes: Additional URL prefixes to allow

    Returns:
        The configured gate instance.
    """
    gate = get_gate()
    gate.configure(mode, api_endpoint, allowed_prefixes)
    return gate


def reset_gate() -> None:
    """
    Reset the gate to a fresh OFFLINE state.

    Only used in testing. In production, the gate lives for the
    entire process lifetime.
    """
    global _gate_instance
    _gate_instance = None
