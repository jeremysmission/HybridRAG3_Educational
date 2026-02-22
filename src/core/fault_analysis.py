# ============================================================================
# HybridRAG3 -- Fault Analysis System (src/monitoring/fault_analysis.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   This is the "immune system" for HybridRAG3. Just like your body detects
#   infections before you feel sick, this module detects and classifies system
#   problems before the user notices something is wrong.
#
#   It has four major components:
#
#     1. SEVERITY LEVELS (SEV-1 through SEV-4):
#        How bad is the problem? SEV-1 = total system failure.
#        SEV-4 = minor cosmetic issue.
#
#     2. ERROR TAXONOMY (11 classes):
#        What kind of problem is it? Network failure? Corrupted index?
#        Wrong credentials? Each class has its own troubleshooting steps.
#
#     3. FLIGHT RECORDER:
#        A "black box" like an airplane has. Records the last N events
#        so when something goes wrong, you can rewind and see exactly
#        what happened leading up to the failure.
#
#     4. GOLDEN PROBES:
#        Automated health checks that run on a schedule. "Can I reach
#        Ollama? Can the embedder load? Is the index readable?" If a
#        probe fails, it fires an alert BEFORE the user hits the problem.
#
# DESIGN DECISIONS:
#   - No external dependencies beyond Python stdlib + structlog
#   - All data stored in-memory (flight recorder) and JSON files (fault log)
#   - Circular buffer for flight recorder to prevent memory leaks
#   - Thread-safe for concurrent queries
#   - No internet access (fully offline capable)
#   - No em-dashes, emojis, or non-ASCII chars (PS 5.1 safe)
#
# INTERNET ACCESS: NONE -- this module never touches the network
#
# FILE ACCESS:
#   - READS: config files, index database (for health probes)
#   - WRITES: logs/fault_analysis.jsonl (append-only fault log)
#            logs/flight_recorder.jsonl (rotating event log)
#
# ALTERNATIVES CONSIDERED:
#   - Sentry/Datadog: requires internet, cloud account, costs money.
#     Our approach: zero dependencies, fully offline, production-safe.
#   - Python logging module alone: no structured data, no severity
#     classification, no flight recorder. Our approach: structured
#     JSON events with full context.
#   - SQLite for fault log: overkill for append-only log. JSONL is
#     simpler, greppable, and doesn't need a database driver.
# ============================================================================

from __future__ import annotations

import os
import json
import time
import threading
import traceback
from enum import IntEnum, Enum
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple


# ============================================================================
# SECTION 1: SEVERITY LEVELS (SEV-1 through SEV-4)
# ============================================================================
#
# WHY SEVERITY LEVELS?
#   Not all problems are equal. A typo in a log message is not the same
#   as a total database corruption. Severity levels let you:
#     - Page the right person (SEV-1 = wake someone up, SEV-4 = weekly review)
#     - Prioritize fixes (SEV-1 blocks everything, SEV-4 can wait)
#     - Track trends (are SEV-2s increasing? Something systemic is wrong)
#
# MAPPING TO enterprise STANDARDS:
#   SEV-1 maps to industry standard "Critical" and industry security framework "Critical"
#   SEV-2 maps to industry standard "High" and industry "High"
#   SEV-3 maps to industry standard "Medium" and industry "Medium"
#   SEV-4 maps to industry standard "Low" and industry "Low"
# ============================================================================

class Severity(IntEnum):
    """
    Severity levels for fault classification.

    Uses IntEnum so severities can be compared with < > operators:
      if fault.severity <= Severity.SEV_2:
          # This is bad -- take action immediately
    """

    SEV_1 = 1
    # CRITICAL -- Total system failure or data loss.
    # Examples:
    #   - SQLite database corrupted (all indexed data lost)
    #   - Embedding model fails to load (no queries possible)
    #   - Security credential exposure detected
    #   - PII sanitization bypass detected
    # Response: Immediate halt. Block all queries until resolved.
    # Industry equivalent: Mission-critical system down.
    # Auto-action: Stop accepting queries, log full diagnostics,
    #   alert via GUI popup if running.

    SEV_2 = 2
    # HIGH -- Major feature broken, but system partially functional.
    # Examples:
    #   - Ollama not responding (offline mode broken, online may work)
    #   - API returns 401 (online mode broken, offline may work)
    #   - Index is stale (>7 days since last successful index)
    #   - Retriever returns zero results for known-good queries
    # Response: Degrade gracefully. Switch to backup mode if available.
    # Industry equivalent: Primary sensor down, backup operational.
    # Auto-action: Log, attempt automatic recovery, switch modes.

    SEV_3 = 3
    # MEDIUM -- Feature degraded but usable.
    # Examples:
    #   - API response slower than 10 seconds (timeout approaching)
    #   - Reranker disabled due to missing model (results less precise)
    #   - File parsing failures on specific file types (partial index)
    #   - Memory usage above 80% threshold
    # Response: Continue operating, schedule fix.
    # Industry equivalent: System operational with reduced capability.
    # Auto-action: Log, add to daily report, suggest optimization.

    SEV_4 = 4
    # LOW -- Cosmetic or informational issues.
    # Examples:
    #   - Config value using default (not explicitly set)
    #   - Deprecated function called (works but should be updated)
    #   - Log directory doesn't exist (auto-created)
    #   - Minor version mismatch in dependency
    # Response: Track in weekly review.
    # Industry equivalent: Maintenance item, no mission impact.
    # Auto-action: Log only, include in weekly summary.


# Human-readable descriptions for each severity level
SEVERITY_DESCRIPTIONS = {
    Severity.SEV_1: "CRITICAL -- System failure or data loss. Immediate action required.",
    Severity.SEV_2: "HIGH -- Major feature broken. Graceful degradation active.",
    Severity.SEV_3: "MEDIUM -- Feature degraded but usable. Scheduled fix needed.",
    Severity.SEV_4: "LOW -- Cosmetic or informational. Weekly review item.",
}


# ============================================================================
# SECTION 2: ERROR TAXONOMY (11 classes)
# ============================================================================
#
# WHY AN ERROR TAXONOMY?
#   "Something went wrong" is useless for troubleshooting. Classifying
#   errors into categories lets you:
#     - Route to the right fix (network error? check VPN. Auth error? check key.)
#     - Track patterns (50 network errors this week? infrastructure problem.)
#     - Automate recovery (known error class -> known fix -> auto-retry)
#
# THE 11 CLASSES:
#   These were chosen based on actual HybridRAG3 failure modes observed
#   during the first 4 weeks of development. Each class maps to specific
#   troubleshooting steps that the developer can follow.
# ============================================================================

class ErrorClass(str, Enum):
    """
    The 11-class error taxonomy for HybridRAG3.

    Each class represents a distinct category of failure with its own
    troubleshooting playbook.
    """

    # -- Class 1: NETWORK_ERROR -----------------------------------------------
    # Connection failures, DNS resolution, proxy issues, firewall blocks.
    # Typical cause: VPN disconnected, corporate proxy blocking, WiFi down.
    # Troubleshooting: Check VPN, ping endpoint, verify proxy settings.
    NETWORK_ERROR = "NETWORK_ERROR"

    # -- Class 2: AUTH_ERROR --------------------------------------------------
    # Authentication/authorization failures. 401, 403, invalid API key.
    # Typical cause: Wrong API key, expired credentials, missing RBAC role.
    # Troubleshooting: Run rag-cred-status, check Azure portal, re-store key.
    AUTH_ERROR = "AUTH_ERROR"

    # -- Class 3: API_ERROR ---------------------------------------------------
    # API-level failures that aren't auth or network. 429, 500, bad request.
    # Typical cause: Rate limiting, model overloaded, malformed request body.
    # Troubleshooting: Check rate limits, verify model name, inspect payload.
    API_ERROR = "API_ERROR"

    # -- Class 4: INDEX_ERROR -------------------------------------------------
    # Database corruption, missing index files, schema mismatch, memmap errors.
    # Typical cause: Interrupted indexing run, disk full, version upgrade.
    # Troubleshooting: Run index diagnostics, check disk space, rebuild index.
    INDEX_ERROR = "INDEX_ERROR"

    # -- Class 5: EMBEDDING_ERROR ---------------------------------------------
    # Embedding model failures. Model won't load, OOM, wrong dimensions.
    # Typical cause: Corrupted model cache, insufficient RAM, wrong model name.
    # Troubleshooting: Clear .model_cache, check RAM, verify model in config.
    EMBEDDING_ERROR = "EMBEDDING_ERROR"

    # -- Class 6: PARSE_ERROR -------------------------------------------------
    # File parsing failures. Corrupted PDF, unsupported format, encoding error.
    # Typical cause: Corrupted file, DRM-protected PDF, non-UTF-8 encoding.
    # Troubleshooting: Open file manually, check file integrity, try fallback parser.
    PARSE_ERROR = "PARSE_ERROR"

    # -- Class 7: CONFIG_ERROR ------------------------------------------------
    # Configuration errors. Missing required field, invalid value, type mismatch.
    # Typical cause: Manual YAML edit with typo, missing file, wrong path.
    # Troubleshooting: Validate YAML syntax, check default_config.yaml, diff configs.
    CONFIG_ERROR = "CONFIG_ERROR"

    # -- Class 8: RESOURCE_ERROR ----------------------------------------------
    # Resource exhaustion. Out of memory, disk full, too many open files.
    # Typical cause: Large indexing run, memory leak, temp files not cleaned.
    # Troubleshooting: Check RAM/disk usage, run cleanup, reduce batch size.
    RESOURCE_ERROR = "RESOURCE_ERROR"

    # -- Class 9: SECURITY_ERROR ----------------------------------------------
    # Security violations. PII detected in output, credential exposure,
    # unauthorized network access attempt, audit log tampering.
    # Typical cause: PII sanitizer missed a pattern, log file permissions wrong.
    # Troubleshooting: Review PII patterns, check file permissions, audit logs.
    SECURITY_ERROR = "SECURITY_ERROR"

    # -- Class 10: RETRIEVAL_ERROR --------------------------------------------
    # Search/retrieval failures. Zero results for known queries, score anomalies,
    # reranker failures, hybrid search mismatch.
    # Typical cause: Empty index, wrong min_score threshold, stale embeddings.
    # Troubleshooting: Run golden queries, check min_score, verify index populated.
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"

    # -- Class 11: INTERNAL_ERROR ---------------------------------------------
    # Catch-all for unexpected failures. Unhandled exceptions, assertion errors,
    # logic bugs, type errors.
    # Typical cause: Code bug, unexpected input, version incompatibility.
    # Troubleshooting: Check stack trace, reproduce with minimal input, file bug.
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Mapping from error class to default severity
# This is the "first guess" severity -- specific faults can override this
ERROR_CLASS_DEFAULT_SEVERITY = {
    ErrorClass.NETWORK_ERROR:    Severity.SEV_2,
    ErrorClass.AUTH_ERROR:       Severity.SEV_2,
    ErrorClass.API_ERROR:        Severity.SEV_2,
    ErrorClass.INDEX_ERROR:      Severity.SEV_1,
    ErrorClass.EMBEDDING_ERROR:  Severity.SEV_1,
    ErrorClass.PARSE_ERROR:      Severity.SEV_3,
    ErrorClass.CONFIG_ERROR:     Severity.SEV_2,
    ErrorClass.RESOURCE_ERROR:   Severity.SEV_2,
    ErrorClass.SECURITY_ERROR:   Severity.SEV_1,
    ErrorClass.RETRIEVAL_ERROR:  Severity.SEV_3,
    ErrorClass.INTERNAL_ERROR:   Severity.SEV_2,
}

# Troubleshooting playbooks for each error class
# These are the FIRST steps to try for each type of error
TROUBLESHOOTING_PLAYBOOKS = {
    ErrorClass.NETWORK_ERROR: [
        "1. Check VPN connection: is it connected?",
        "2. Test basic connectivity: ping 8.8.8.8",
        "3. Check proxy settings: echo $env:HTTP_PROXY",
        "4. Verify endpoint is reachable: curl <endpoint_url>",
        "5. Check corporate firewall: is the endpoint whitelisted?",
        "6. Try switching to offline mode as a workaround",
    ],
    ErrorClass.AUTH_ERROR: [
        "1. Run: python -m hybridrag3.tools.cred_status",
        "2. Verify API key is not expired in Azure portal",
        "3. Re-store credentials: python -m hybridrag3.tools.store_key",
        "4. Check for 'Cognitive Services User' RBAC role in Azure",
        "5. Verify endpoint URL matches Azure portal 'Keys and Endpoint' page",
        "6. Try a fresh API key from the portal",
    ],
    ErrorClass.API_ERROR: [
        "1. Check if it's a rate limit (429): wait 60 seconds and retry",
        "2. Verify model/deployment name matches Azure portal",
        "3. Check API version string (should be 2024-02-02 for Azure)",
        "4. Test with a minimal prompt to rule out payload issues",
        "5. Check Azure service health status page",
    ],
    ErrorClass.INDEX_ERROR: [
        "1. Check disk space: Get-Volume D",
        "2. Run index diagnostics: python -m hybridrag3.tools.diag_index",
        "3. Check if index files exist in the index/ directory",
        "4. Try rebuilding: python -m hybridrag3.tools.rebuild_index",
        "5. Check for interrupted indexing runs (partial writes)",
    ],
    ErrorClass.EMBEDDING_ERROR: [
        "1. Check available RAM: Get-Process python | Select WorkingSet",
        "2. Clear model cache: Remove-Item .model_cache -Recurse -Force",
        "3. Re-download model: will auto-download on next run",
        "4. Verify model name in config matches a real model",
        "5. Check PyTorch/sentence-transformers version compatibility",
    ],
    ErrorClass.PARSE_ERROR: [
        "1. Try opening the file manually -- is it corrupted?",
        "2. Check file permissions: Get-Acl <filepath>",
        "3. Check encoding: file might not be UTF-8",
        "4. For PDFs: is it DRM-protected or scanned (needs OCR)?",
        "5. File will be skipped -- check logs for the specific error",
    ],
    ErrorClass.CONFIG_ERROR: [
        "1. Validate YAML syntax: python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"",
        "2. Compare with default_config.yaml for missing fields",
        "3. Check for invisible characters (smart quotes, em-dashes)",
        "4. Ensure all paths exist and are accessible",
    ],
    ErrorClass.RESOURCE_ERROR: [
        "1. Check RAM: Get-Process python | Select WorkingSet",
        "2. Check disk: Get-Volume D",
        "3. Run cleanup: python -m hybridrag3.tools.cleanup",
        "4. Reduce chunk batch size in config",
        "5. Close other applications to free RAM",
    ],
    ErrorClass.SECURITY_ERROR: [
        "1. STOP -- do not continue until reviewed",
        "2. Check audit log for the specific violation",
        "3. If PII exposure: identify what was exposed and to where",
        "4. If credential exposure: rotate the affected key immediately",
        "5. Review security config settings",
        "6. Document the incident for compliance records",
    ],
    ErrorClass.RETRIEVAL_ERROR: [
        "1. Run golden probe queries (known-good test queries)",
        "2. Check min_score threshold -- might be too high",
        "3. Verify index is populated: check chunk count",
        "4. Test with a very broad query to confirm retrieval works",
        "5. Check if embeddings were generated with the correct model",
    ],
    ErrorClass.INTERNAL_ERROR: [
        "1. Read the full stack trace in the fault log",
        "2. Try to reproduce with a minimal input",
        "3. Check if a recent code change caused the regression",
        "4. Search GitHub issues for the error message",
        "5. If persistent, file a detailed bug report",
    ],
}


# ============================================================================
# SECTION 3: FAULT EVENT DATA STRUCTURE
# ============================================================================
#
# A FaultEvent is a single recorded problem. It contains everything
# needed to understand, classify, and troubleshoot the issue.
# ============================================================================

@dataclass
class FaultEvent:
    """
    A single fault event recorded by the system.

    This is the core data structure -- every detected problem becomes
    one of these. It gets stored in the fault log and can be queried
    for trend analysis.
    """

    # When did it happen? ISO 8601 format for machine parsing.
    timestamp: str

    # How bad is it? SEV-1 (critical) through SEV-4 (informational)
    severity: int

    # What category of problem? One of the 11 ErrorClass values.
    error_class: str

    # Short human-readable summary (one line).
    # Example: "Ollama connection refused on localhost:11434"
    summary: str

    # Full technical details (stack trace, error message, context).
    details: str

    # Which module/component reported this fault?
    # Example: "llm_router.OllamaRouter", "indexer.Indexer"
    source_module: str

    # Optional: the function or method where the fault occurred.
    source_function: str = ""

    # Optional: recovery action taken automatically (if any).
    # Example: "Switched to online mode", "Retried 3 times"
    auto_recovery: str = ""

    # Optional: what the user was doing when this happened.
    # Example: "query: 'What is the operating frequency?'"
    user_context: str = ""

    # Optional: troubleshooting steps to try.
    # Auto-populated from TROUBLESHOOTING_PLAYBOOKS.
    troubleshooting: List[str] = field(default_factory=list)

    # Optional: related fault IDs (for cascading failures).
    # Example: A network error causing an API error.
    related_faults: List[str] = field(default_factory=list)

    # Unique ID for this fault event (auto-generated).
    fault_id: str = ""

    # Was this fault resolved? Updated later when fixed.
    resolved: bool = False
    resolved_at: str = ""
    resolution_notes: str = ""

    def __post_init__(self):
        """Auto-generate fault_id if not provided."""
        if not self.fault_id:
            # Format: FAULT-YYYYMMDD-HHMMSS-XXXX (where XXXX is a counter)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            # Use thread ID as a simple uniqueness suffix
            tid = threading.current_thread().ident or 0
            self.fault_id = f"FAULT-{ts}-{tid % 10000:04d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to a JSON string (one line, for JSONL files)."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaultEvent":
        """Reconstruct a FaultEvent from a dictionary."""
        return cls(**data)

    @property
    def severity_label(self) -> str:
        """Human-readable severity label."""
        try:
            return SEVERITY_DESCRIPTIONS[Severity(self.severity)]
        except (ValueError, KeyError):
            return f"UNKNOWN SEVERITY: {self.severity}"


# ============================================================================
# SECTION 4: FAULT CLASSIFIER
# ============================================================================
#
# Takes a raw exception or error message and classifies it into the
# appropriate error class and severity level.
#
# WHY?
#   Raw exceptions like "ConnectionRefusedError" or "401 Unauthorized"
#   are meaningless to most users. The classifier translates them into
#   actionable categories with troubleshooting steps.
# ============================================================================

class FaultClassifier:
    """
    Classifies raw exceptions into error classes and severity levels.

    Uses pattern matching on exception types and error messages to
    determine the most appropriate classification.
    """

    # Pattern rules: (condition_function, error_class, severity_override)
    # The first matching rule wins.
    # severity_override=None means use the default for that error class.
    CLASSIFICATION_RULES: List[Tuple[Callable, ErrorClass, Optional[Severity]]] = []

    @classmethod
    def _init_rules(cls):
        """
        Initialize classification rules on first use.

        WHY LAZY INIT?
          We define rules as (function, class, severity) tuples.
          The functions are lambdas that check exception type and message.
          Lazy init avoids circular import issues.
        """
        if cls.CLASSIFICATION_RULES:
            return  # Already initialized

        # Each rule is: (check_function, error_class, severity_override)
        # check_function takes (exception_type_name: str, error_msg: str)
        # and returns True if this rule matches.

        cls.CLASSIFICATION_RULES = [
            # -- Security errors (check first -- highest priority) ----------
            (
                lambda t, m: "PII" in m.upper() or "credential" in m.lower()
                or "exposure" in m.lower(),
                ErrorClass.SECURITY_ERROR,
                Severity.SEV_1,
            ),
            (
                lambda t, m: "unauthorized access" in m.lower()
                or "audit" in m.lower() and "tamper" in m.lower(),
                ErrorClass.SECURITY_ERROR,
                Severity.SEV_1,
            ),

            # -- Auth errors -----------------------------------------------
            (
                lambda t, m: "401" in m or "Unauthorized" in m
                or "AuthenticationError" in t,
                ErrorClass.AUTH_ERROR,
                Severity.SEV_2,
            ),
            (
                lambda t, m: "403" in m or "Forbidden" in m
                or "PermissionError" in t and "api" in m.lower(),
                ErrorClass.AUTH_ERROR,
                Severity.SEV_2,
            ),

            # -- Network errors --------------------------------------------
            (
                lambda t, m: "ConnectionRefused" in t or "ConnectionError" in t
                or "connection refused" in m.lower(),
                ErrorClass.NETWORK_ERROR,
                None,  # Use default severity
            ),
            (
                lambda t, m: "DNS" in m.upper() or "resolve" in m.lower()
                or "NameResolutionError" in t,
                ErrorClass.NETWORK_ERROR,
                None,
            ),
            (
                lambda t, m: "SSL" in m or "certificate" in m.lower()
                or "SSLError" in t,
                ErrorClass.NETWORK_ERROR,
                Severity.SEV_2,
            ),
            (
                lambda t, m: "TimeoutError" in t or "timed out" in m.lower()
                or "Timeout" in t,
                ErrorClass.NETWORK_ERROR,
                Severity.SEV_3,
            ),

            # -- API errors ------------------------------------------------
            (
                lambda t, m: "429" in m or "RateLimit" in m
                or "rate limit" in m.lower(),
                ErrorClass.API_ERROR,
                Severity.SEV_3,
            ),
            (
                lambda t, m: "404" in m and ("api" in m.lower()
                or "deployment" in m.lower() or "model" in m.lower()),
                ErrorClass.API_ERROR,
                Severity.SEV_2,
            ),
            (
                lambda t, m: "500" in m or "502" in m or "503" in m
                or "InternalServerError" in t,
                ErrorClass.API_ERROR,
                Severity.SEV_2,
            ),
            (
                lambda t, m: "BadRequest" in t or "400" in m
                and "api" in m.lower(),
                ErrorClass.API_ERROR,
                Severity.SEV_3,
            ),

            # -- Index errors ----------------------------------------------
            (
                lambda t, m: "sqlite" in m.lower() or "database" in m.lower()
                or "OperationalError" in t,
                ErrorClass.INDEX_ERROR,
                Severity.SEV_1,
            ),
            (
                lambda t, m: "memmap" in m.lower() or "mmap" in m.lower()
                or "numpy" in m.lower() and "error" in m.lower(),
                ErrorClass.INDEX_ERROR,
                Severity.SEV_2,
            ),

            # -- Embedding errors ------------------------------------------
            (
                lambda t, m: "SentenceTransformer" in m
                or "embedding" in m.lower() and "model" in m.lower(),
                ErrorClass.EMBEDDING_ERROR,
                Severity.SEV_1,
            ),
            (
                lambda t, m: "CUDA" in m or "OutOfMemory" in m
                or "out of memory" in m.lower(),
                ErrorClass.RESOURCE_ERROR,
                Severity.SEV_2,
            ),

            # -- Parse errors ----------------------------------------------
            (
                lambda t, m: "PDFSyntaxError" in t or "pdf" in m.lower()
                and ("corrupt" in m.lower() or "parse" in m.lower()),
                ErrorClass.PARSE_ERROR,
                Severity.SEV_3,
            ),
            (
                lambda t, m: "UnicodeDecodeError" in t or "encoding" in m.lower(),
                ErrorClass.PARSE_ERROR,
                Severity.SEV_4,
            ),

            # -- Config errors ---------------------------------------------
            (
                lambda t, m: "YAMLError" in t or "yaml" in m.lower()
                and ("parse" in m.lower() or "syntax" in m.lower()),
                ErrorClass.CONFIG_ERROR,
                Severity.SEV_2,
            ),
            (
                lambda t, m: "config" in m.lower()
                and ("missing" in m.lower() or "not found" in m.lower()),
                ErrorClass.CONFIG_ERROR,
                Severity.SEV_2,
            ),

            # -- Resource errors -------------------------------------------
            (
                lambda t, m: "MemoryError" in t or "memory" in m.lower()
                and "error" in m.lower(),
                ErrorClass.RESOURCE_ERROR,
                Severity.SEV_2,
            ),
            (
                lambda t, m: "disk" in m.lower() and "full" in m.lower()
                or "No space left" in m,
                ErrorClass.RESOURCE_ERROR,
                Severity.SEV_1,
            ),

            # -- Retrieval errors ------------------------------------------
            (
                lambda t, m: "retriev" in m.lower()
                and ("zero" in m.lower() or "empty" in m.lower()
                     or "no results" in m.lower()),
                ErrorClass.RETRIEVAL_ERROR,
                Severity.SEV_3,
            ),
        ]

    @classmethod
    def classify(
        cls,
        exception: Optional[Exception] = None,
        error_type: str = "",
        error_message: str = "",
        context: str = "",
    ) -> Tuple[ErrorClass, Severity]:
        """
        Classify an error into an ErrorClass and Severity.

        Args:
            exception:     The actual Python exception (if available)
            error_type:    The exception type name as a string
            error_message: The error message text
            context:       Additional context (what was happening)

        Returns:
            (ErrorClass, Severity) tuple

        HOW IT WORKS:
            1. Extract the exception type name and message
            2. Walk through classification rules in order
            3. First matching rule determines the class and severity
            4. If no rule matches, default to INTERNAL_ERROR / SEV-2
        """
        cls._init_rules()

        # Extract type name and message from the exception object
        if exception is not None:
            error_type = error_type or type(exception).__name__
            error_message = error_message or str(exception)

        # Combine message with context for broader pattern matching
        full_message = f"{error_message} {context}"

        # Walk through rules -- first match wins
        for check_fn, error_class, severity_override in cls.CLASSIFICATION_RULES:
            try:
                if check_fn(error_type, full_message):
                    severity = severity_override or ERROR_CLASS_DEFAULT_SEVERITY[error_class]
                    return error_class, severity
            except Exception:
                # A broken classification rule should never crash the classifier
                continue

        # No rule matched -- default to INTERNAL_ERROR
        return ErrorClass.INTERNAL_ERROR, Severity.SEV_2


# ============================================================================
# SECTION 5: FLIGHT RECORDER
# ============================================================================
#
# WHAT IS A FLIGHT RECORDER?
#   Like an airplane's "black box," it continuously records system events.
#   When something goes wrong, you can rewind and see exactly what happened
#   in the seconds/minutes leading up to the failure.
#
# HOW IT WORKS:
#   - Uses a circular buffer (deque with maxlen)
#   - New events push out old events when the buffer is full
#   - Thread-safe for concurrent access
#   - Events include: queries, responses, config changes, errors, probes
#
# WHY CIRCULAR BUFFER?
#   If we stored EVERY event forever, memory would grow without bound.
#   A circular buffer keeps the last N events (default 1000) and
#   automatically discards older ones. This gives you enough context
#   to debug any recent failure without memory leaks.
#
# MEMORY USAGE:
#   ~1000 events * ~500 bytes/event = ~500 KB. Negligible.
# ============================================================================

@dataclass
class FlightRecorderEvent:
    """
    A single event in the flight recorder trace.

    These are lightweight breadcrumbs -- not full fault events.
    Think of them as "flight data recorder points" that create
    a timeline of system activity.
    """
    timestamp: str                      # ISO 8601
    event_type: str                     # query, response, error, probe, config, startup, shutdown
    module: str                         # Which module generated this event
    summary: str                        # One-line description
    data: Dict[str, Any] = field(      # Optional structured data
        default_factory=dict
    )
    duration_ms: float = 0.0            # How long the operation took (if applicable)


class FlightRecorder:
    """
    In-memory circular buffer that records system events for debugging.

    Usage:
        recorder = FlightRecorder(max_events=1000)
        recorder.record("query", "query_engine", "User asked about freq",
                        data={"query": "operating frequency"})

        # Later, when something goes wrong:
        recent = recorder.get_recent(50)  # Last 50 events
        trace = recorder.get_trace_around(fault_time, window_seconds=30)
    """

    def __init__(self, max_events: int = 1000, log_dir: str = "logs"):
        """
        Initialize the flight recorder.

        Args:
            max_events: Maximum events to keep in memory.
                        Older events are automatically discarded.
            log_dir:    Directory for periodic flush to disk.
        """
        self._buffer: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._flush_file = self._log_dir / "flight_recorder.jsonl"
        self._event_count = 0

    def record(
        self,
        event_type: str,
        module: str,
        summary: str,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> FlightRecorderEvent:
        """
        Record a new event.

        Args:
            event_type:  Category of event (query, response, error, etc.)
            module:      Which module is recording this
            summary:     One-line human-readable description
            data:        Optional structured data (dict)
            duration_ms: How long the operation took (0 if N/A)

        Returns:
            The recorded FlightRecorderEvent

        Thread-safe: uses a lock to prevent concurrent write corruption.
        """
        event = FlightRecorderEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            module=module,
            summary=summary,
            data=data or {},
            duration_ms=duration_ms,
        )

        with self._lock:
            self._buffer.append(event)
            self._event_count += 1

            # Periodic flush to disk every 100 events
            # WHY: If the process crashes, we don't lose everything.
            # The in-memory buffer gives speed; the disk file gives durability.
            if self._event_count % 100 == 0:
                self._flush_to_disk()

        return event

    def get_recent(self, n: int = 50) -> List[FlightRecorderEvent]:
        """
        Get the N most recent events.

        Args:
            n: Number of events to retrieve (default 50)

        Returns:
            List of events, newest first
        """
        with self._lock:
            events = list(self._buffer)

        # Return newest first
        return list(reversed(events[-n:]))

    def get_trace_around(
        self,
        center_time: str,
        window_seconds: float = 30.0,
    ) -> List[FlightRecorderEvent]:
        """
        Get events within a time window around a specific moment.

        This is the "rewind" function -- when a fault occurs at time T,
        call this to see everything that happened from T-30s to T+30s.

        Args:
            center_time:    ISO 8601 timestamp (the moment of interest)
            window_seconds: How many seconds before/after to include

        Returns:
            List of events within the window, oldest first
        """
        try:
            center_dt = datetime.fromisoformat(center_time.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return self.get_recent(50)  # Fallback if timestamp is bad

        results = []
        with self._lock:
            for event in self._buffer:
                try:
                    event_dt = datetime.fromisoformat(
                        event.timestamp.replace("Z", "+00:00")
                    )
                    diff = abs((event_dt - center_dt).total_seconds())
                    if diff <= window_seconds:
                        results.append(event)
                except (ValueError, AttributeError):
                    continue

        return results

    def get_events_by_type(self, event_type: str) -> List[FlightRecorderEvent]:
        """Get all events of a specific type (e.g., 'error')."""
        with self._lock:
            return [e for e in self._buffer if e.event_type == event_type]

    def clear(self):
        """Clear all events from the buffer. Useful for testing."""
        with self._lock:
            self._buffer.clear()
            self._event_count = 0

    def _flush_to_disk(self):
        """
        Write recent events to the JSONL log file.

        JSONL = JSON Lines: one JSON object per line. This format is:
          - Append-only (safe for concurrent writes)
          - Greppable (search with grep/findstr)
          - Parseable (read line by line)

        Called automatically every 100 events. Also call manually
        before shutdown to ensure nothing is lost.
        """
        try:
            # Only flush the last 100 events (avoid re-writing old ones)
            recent = list(self._buffer)[-100:]
            with open(self._flush_file, "a", encoding="utf-8") as f:
                for event in recent:
                    line = json.dumps(asdict(event), default=str)
                    f.write(line + "\n")
        except Exception:
            pass  # Don't crash on flush failure -- flight recorder is best-effort

    def flush(self):
        """Public method to force a flush to disk."""
        with self._lock:
            self._flush_to_disk()

    @property
    def size(self) -> int:
        """Current number of events in the buffer."""
        return len(self._buffer)


# ============================================================================
# SECTION 6: GOLDEN PROBES
# ============================================================================
#
# WHAT ARE GOLDEN PROBES?
#   Automated health checks that test specific system capabilities.
#   "Golden" because they use known-good inputs with expected outputs.
#   If the probe fails, something is broken -- even if no user has
#   noticed yet.
#
# WHY?
#   Proactive detection beats reactive debugging. Instead of waiting
#   for a user to report "queries aren't working," a golden probe
#   catches the issue within minutes of it happening.
#
# PROBE TYPES:
#   - Connectivity probes: Can we reach Ollama? Can we reach the API?
#   - Index probes: Is the database readable? Are embeddings valid?
#   - Retrieval probes: Do known-good queries return expected results?
#   - Resource probes: Is RAM/disk within acceptable limits?
#   - Security probes: Are audit logs writable? Are permissions correct?
# ============================================================================

@dataclass
class ProbeResult:
    """
    Result of a single golden probe execution.

    probe_name: Which probe ran (e.g., "ollama_connectivity")
    passed:     Did the probe succeed?
    message:    Human-readable result message
    latency_ms: How long the probe took to execute
    details:    Additional structured data about the result
    """
    probe_name: str
    passed: bool
    message: str
    latency_ms: float
    severity: int = Severity.SEV_4    # Severity if probe failed
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class GoldenProbes:
    """
    Suite of automated health check probes for HybridRAG3.

    Usage:
        probes = GoldenProbes(config, vector_store, embedder)
        results = probes.run_all()

        for result in results:
            if not result.passed:
                print(f"[FAIL] {result.probe_name}: {result.message}")
    """

    def __init__(
        self,
        config: Any,
        vector_store: Any = None,
        embedder: Any = None,
        llm_router: Any = None,
    ):
        """
        Initialize the probe suite.

        Args:
            config:       HybridRAG3 config object
            vector_store: The SQLite/memmap vector store (optional)
            embedder:     The SentenceTransformer embedder (optional)
            llm_router:   The LLM router (optional)

        WHY ARE DEPENDENCIES OPTIONAL?
          Not all probes need all components. You should be able to run
          connectivity probes even if the embedder hasn't loaded yet.
        """
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_router = llm_router

    def run_all(self) -> List[ProbeResult]:
        """
        Run all applicable probes and return results.

        Probes are run in order of priority:
          1. Configuration probes (cheapest, catch basic issues)
          2. Resource probes (check system health)
          3. Connectivity probes (check network/services)
          4. Index probes (check data integrity)
          5. Retrieval probes (check end-to-end functionality)
        """
        from .golden_probe_checks import (
            check_config_valid,
            check_disk_space,
            check_memory_usage,
            check_ollama_connectivity,
            check_api_connectivity,
            check_index_readable,
            check_index_not_empty,
            check_embedder_loaded,
            check_embedding_dimensions,
            check_golden_query,
        )

        results = []

        # Group 1: Configuration
        results.append(check_config_valid(self.config))

        # Group 2: Resources
        results.append(check_disk_space(self.config))
        results.append(check_memory_usage())

        # Group 3: Connectivity
        results.append(check_ollama_connectivity(self.config))
        if self.llm_router and hasattr(self.llm_router, 'api') and self.llm_router.api:
            results.append(check_api_connectivity(self.config))

        # Group 4: Index integrity
        if self.vector_store:
            results.append(check_index_readable(self.vector_store))
            results.append(check_index_not_empty(self.vector_store))

        # Group 5: Embedding
        if self.embedder:
            results.append(check_embedder_loaded(self.embedder))
            results.append(check_embedding_dimensions(self.embedder))

        # Group 6: Retrieval (only if all prerequisites are available)
        if self.vector_store and self.embedder:
            results.append(check_golden_query(
                self.vector_store, self.embedder, self.config,
            ))

        return results


# ============================================================================
# SECTION 7: FAULT ANALYSIS ENGINE (ties everything together)
# ============================================================================
#
# This is the main orchestrator that combines:
#   - Fault classification
#   - Flight recording
#   - Golden probes
#   - Fault logging
#   - Trend analysis
# ============================================================================

class FaultAnalysisEngine:
    """
    The main fault analysis system for HybridRAG3.

    This is the single entry point that the rest of the system uses
    to report errors, run health checks, and query fault history.

    Usage:
        engine = FaultAnalysisEngine(config)

        # Report a fault
        engine.report_fault(exception=e, source_module="llm_router",
                           user_context="query: 'operating frequency'")

        # Run health checks
        results = engine.run_probes()

        # Get recent faults
        recent = engine.get_recent_faults(n=10)

        # Get fault summary
        summary = engine.get_summary()
    """

    def __init__(
        self,
        config: Any,
        log_dir: str = "logs",
        max_flight_events: int = 1000,
        vector_store: Any = None,
        embedder: Any = None,
        llm_router: Any = None,
    ):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.flight_recorder = FlightRecorder(
            max_events=max_flight_events, log_dir=log_dir
        )
        self.golden_probes = GoldenProbes(
            config, vector_store, embedder, llm_router
        )
        self.classifier = FaultClassifier()

        # Fault storage
        self._fault_log_file = self.log_dir / "fault_analysis.jsonl"
        self._faults: List[FaultEvent] = []
        self._lock = threading.Lock()

        # Record startup
        self.flight_recorder.record(
            "startup", "fault_analysis", "FaultAnalysisEngine initialized"
        )

    def report_fault(
        self,
        exception: Optional[Exception] = None,
        error_type: str = "",
        error_message: str = "",
        source_module: str = "",
        source_function: str = "",
        user_context: str = "",
        auto_recovery: str = "",
        severity_override: Optional[int] = None,
    ) -> FaultEvent:
        """
        Report a fault to the analysis engine.

        This is the main method other modules call when something goes wrong.
        It classifies the error, creates a FaultEvent, records it in the
        flight recorder, writes to the fault log, and returns the event.

        Args:
            exception:         The Python exception (if available)
            error_type:        Exception type name (if no exception object)
            error_message:     Error message text (if no exception object)
            source_module:     Which module reported this (e.g., "llm_router")
            source_function:   Which function (e.g., "query")
            user_context:      What the user was doing (e.g., query text)
            auto_recovery:     What recovery was attempted
            severity_override: Override the auto-restricted severity

        Returns:
            The created FaultEvent (for further processing or display)
        """
        # Step 1: Classify the error
        error_class, severity = self.classifier.classify(
            exception=exception,
            error_type=error_type,
            error_message=error_message,
            context=user_context,
        )

        # Apply severity override if provided
        if severity_override is not None:
            severity = Severity(severity_override)

        # Step 2: Get the stack trace (if available)
        details = ""
        if exception is not None:
            details = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        else:
            details = error_message

        # Step 3: Get troubleshooting steps
        playbook = TROUBLESHOOTING_PLAYBOOKS.get(error_class, [])

        # Step 4: Create the FaultEvent
        fault = FaultEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=int(severity),
            error_class=error_class.value,
            summary=f"[{error_class.value}] {error_message[:200]}",
            details=details[:5000],  # Cap at 5KB to prevent bloat
            source_module=source_module,
            source_function=source_function,
            auto_recovery=auto_recovery,
            user_context=user_context[:500],  # Cap user context
            troubleshooting=playbook,
        )

        # Step 5: Store it
        with self._lock:
            self._faults.append(fault)

        # Step 6: Record in flight recorder
        self.flight_recorder.record(
            event_type="fault",
            module=source_module,
            summary=fault.summary,
            data={
                "fault_id": fault.fault_id,
                "severity": int(severity),
                "error_class": error_class.value,
            },
        )

        # Step 7: Write to fault log file (append-only JSONL)
        self._write_fault_to_log(fault)

        return fault

    def run_probes(self) -> List[ProbeResult]:
        """
        Run all golden probes and record results.

        Returns:
            List of ProbeResult objects

        Failures are automatically recorded as flight recorder events.
        """
        results = self.golden_probes.run_all()

        for result in results:
            self.flight_recorder.record(
                event_type="probe",
                module="golden_probes",
                summary=f"[{'PASS' if result.passed else 'FAIL'}] {result.probe_name}: {result.message}",
                data=result.details,
                duration_ms=result.latency_ms,
            )

            # Auto-report failed probes as faults
            if not result.passed:
                self.report_fault(
                    error_message=result.message,
                    source_module="golden_probes",
                    source_function=result.probe_name,
                    severity_override=result.severity,
                )

        return results

    def get_recent_faults(self, n: int = 10) -> List[FaultEvent]:
        """Get the N most recent fault events."""
        with self._lock:
            return list(reversed(self._faults[-n:]))

    def get_faults_by_severity(self, severity: Severity) -> List[FaultEvent]:
        """Get all faults of a specific severity level."""
        with self._lock:
            return [f for f in self._faults if f.severity == int(severity)]

    def get_faults_by_class(self, error_class: ErrorClass) -> List[FaultEvent]:
        """Get all faults of a specific error class."""
        with self._lock:
            return [f for f in self._faults if f.error_class == error_class.value]

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all recorded faults.

        Returns a dict with:
          - total_faults: Total number of faults recorded
          - by_severity: Count per severity level
          - by_class: Count per error class
          - unresolved: Count of unresolved faults
          - most_recent: The most recent fault (if any)
          - flight_recorder_size: Events in the flight recorder
        """
        with self._lock:
            faults = list(self._faults)

        by_severity = {}
        for sev in Severity:
            count = sum(1 for f in faults if f.severity == int(sev))
            if count > 0:
                by_severity[f"SEV_{sev.value}"] = count

        by_class = {}
        for ec in ErrorClass:
            count = sum(1 for f in faults if f.error_class == ec.value)
            if count > 0:
                by_class[ec.value] = count

        unresolved = sum(1 for f in faults if not f.resolved)

        return {
            "total_faults": len(faults),
            "by_severity": by_severity,
            "by_class": by_class,
            "unresolved": unresolved,
            "most_recent": faults[-1].to_dict() if faults else None,
            "flight_recorder_size": self.flight_recorder.size,
        }

    def get_flight_trace(
        self,
        fault_id: Optional[str] = None,
        window_seconds: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """
        Get the flight recorder trace around a specific fault.

        Args:
            fault_id:        The fault to investigate (uses most recent if None)
            window_seconds:  Time window around the fault

        Returns:
            List of flight recorder events as dicts
        """
        # Find the fault
        fault = None
        if fault_id:
            with self._lock:
                for f in self._faults:
                    if f.fault_id == fault_id:
                        fault = f
                        break

        if fault is None and self._faults:
            fault = self._faults[-1]

        if fault is None:
            return []

        events = self.flight_recorder.get_trace_around(
            fault.timestamp, window_seconds
        )

        return [asdict(e) for e in events]

    def resolve_fault(
        self, fault_id: str, resolution_notes: str = ""
    ) -> bool:
        """
        Mark a fault as resolved.

        Args:
            fault_id:         The fault to resolve
            resolution_notes: What fixed it

        Returns:
            True if the fault was found and resolved, False otherwise
        """
        with self._lock:
            for fault in self._faults:
                if fault.fault_id == fault_id:
                    fault.resolved = True
                    fault.resolved_at = datetime.now(timezone.utc).isoformat()
                    fault.resolution_notes = resolution_notes
                    self._write_fault_to_log(fault)
                    return True
        return False

    def _write_fault_to_log(self, fault: FaultEvent):
        """Append a fault event to the JSONL log file."""
        try:
            with open(self._fault_log_file, "a", encoding="utf-8") as f:
                f.write(fault.to_json() + "\n")
        except Exception:
            pass  # Don't crash on log write failure

    def shutdown(self):
        """
        Clean shutdown -- flush all buffers to disk.

        Call this before the application exits to ensure no events are lost.
        """
        self.flight_recorder.record(
            "shutdown", "fault_analysis", "FaultAnalysisEngine shutting down"
        )
        self.flight_recorder.flush()


# ============================================================================
# SECTION 8: CONVENIENCE FUNCTIONS
# ============================================================================
# These make it easy to use the fault analysis system from any module
# without needing to import the full class hierarchy.
# ============================================================================

# Module-level singleton (initialized on first use)
_engine: Optional[FaultAnalysisEngine] = None
_engine_lock = threading.Lock()


def init_fault_analysis(
    config: Any,
    log_dir: str = "logs",
    vector_store: Any = None,
    embedder: Any = None,
    llm_router: Any = None,
) -> FaultAnalysisEngine:
    """
    Initialize the global fault analysis engine.

    Call this once during application startup. After this, use
    report_fault() and run_health_check() from anywhere.
    """
    global _engine
    with _engine_lock:
        _engine = FaultAnalysisEngine(
            config=config,
            log_dir=log_dir,
            vector_store=vector_store,
            embedder=embedder,
            llm_router=llm_router,
        )
    return _engine


def get_engine() -> Optional[FaultAnalysisEngine]:
    """Get the global fault analysis engine (None if not initialized)."""
    return _engine


def report_fault(**kwargs) -> Optional[FaultEvent]:
    """
    Report a fault to the global engine.

    Convenience wrapper -- call from any module:
        from src.monitoring.fault_analysis import report_fault
        report_fault(exception=e, source_module="my_module")
    """
    if _engine is not None:
        return _engine.report_fault(**kwargs)
    return None


def record_event(event_type: str, module: str, summary: str, **kwargs):
    """
    Record a flight recorder event.

    Convenience wrapper:
        from src.monitoring.fault_analysis import record_event
        record_event("query", "query_engine", "User query processed",
                     data={"query": user_query}, duration_ms=elapsed)
    """
    if _engine is not None:
        _engine.flight_recorder.record(event_type, module, summary, **kwargs)


def run_health_check() -> Optional[List[ProbeResult]]:
    """
    Run all golden probes.

    Convenience wrapper:
        from src.monitoring.fault_analysis import run_health_check
        results = run_health_check()
    """
    if _engine is not None:
        return _engine.run_probes()
    return None
