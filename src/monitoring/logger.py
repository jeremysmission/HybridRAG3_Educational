# ============================================================================
# HybridRAG -- Structured Logger (src/monitoring/logger.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Sets up the logging system for the entire application. Every important
#   event (file indexed, query executed, error occurred, cost incurred) gets
#   recorded in structured log files.
#
# WHY "STRUCTURED" LOGGING?
#   Normal logging: "Indexed file manual.pdf in 2.3 seconds"
#   Structured logging: {"event": "file_indexed", "file": "manual.pdf", "seconds": 2.3}
#
#   Structured logs are machine-readable JSON. This means you can:
#   - Search logs with tools like jq, grep, or Python scripts
#   - Build dashboards showing indexing speed over time
#   - Prove to auditors exactly what happened and when
#   - Debug problems by filtering for specific events
#
# LOG FILE TYPES:
#   The system creates separate log files for different purposes:
#   - app_YYYY-MM-DD.log:   General application events (indexing, queries)
#   - error_YYYY-MM-DD.log: Errors and failures (things that went wrong)
#   - audit_YYYY-MM-DD.log: Security-relevant events (who did what when)
#   - cost_YYYY-MM-DD.log:  API cost tracking (online mode token usage)
#
# LOG ENTRY BUILDERS:
#   The bottom of this file has "builder" classes that create consistent
#   log entries. Instead of each piece of code formatting its own log
#   messages differently, everyone uses these builders for uniformity.
#
# HOW TO USE (from other code):
#   from src.monitoring.logger import get_app_logger
#   logger = get_app_logger("my_module")
#   logger.info("something_happened", key="value", count=42)
#
# DEPENDENCIES:
#   - structlog: A structured logging library that outputs JSON
#   - Python's built-in logging module (structlog builds on top of it)
#
# INTERNET ACCESS: None -- writes to local files only
# ============================================================================

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from datetime import datetime


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================

class LoggerSetup:
    """Initialize and configure structlog for HybridRAG"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._configured = False

    def setup(self) -> None:
        """Configure structlog with timestamped log files"""
        if self._configured:
            return

        # Configure standard logging first (for third-party libraries)
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.WARNING,  # Only warnings and above to console
        )

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        self._configured = True

    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a named logger bound to a log file"""
        self.setup()
        return structlog.get_logger(name)

    def get_file_logger(self, name: str, log_type: str = "app") -> structlog.BoundLogger:
        """
        Get a logger that writes to a specific log file.
        log_type: "app", "error", "audit", "cost"
        """
        self.setup()
        logger = structlog.get_logger(name)

        # Add file handler for this logger
        log_file = self.log_dir / f"{log_type}_{self._get_date_str()}.log"
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(message)s")
        )

        # Get the underlying Python logger and add handler
        py_logger = logging.getLogger(name)
        py_logger.addHandler(handler)
        py_logger.setLevel(logging.DEBUG)

        return logger

    @staticmethod
    def _get_date_str() -> str:
        """Get current date as YYYY-MM-DD string"""
        return datetime.now().strftime("%Y-%m-%d")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_logger_setup: Optional[LoggerSetup] = None


def initialize_logging(log_dir: str = "logs") -> LoggerSetup:
    """Initialize logging (call once at app startup)"""
    global _logger_setup
    if _logger_setup is None:
        _logger_setup = LoggerSetup(log_dir)
        _logger_setup.setup()
    return _logger_setup


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger (auto-initializes if needed)"""
    if _logger_setup is None:
        initialize_logging()
    return _logger_setup.get_logger(name)


def get_app_logger(name: str = "app") -> structlog.BoundLogger:
    """Get app logger (writes to app_YYYY-MM-DD.log)"""
    if _logger_setup is None:
        initialize_logging()
    return _logger_setup.get_file_logger(name, "app")


def get_error_logger(name: str = "error") -> structlog.BoundLogger:
    """Get error logger (writes to error_YYYY-MM-DD.log)"""
    if _logger_setup is None:
        initialize_logging()
    return _logger_setup.get_file_logger(name, "error")


def get_audit_logger(name: str = "audit") -> structlog.BoundLogger:
    """Get audit logger (writes to audit_YYYY-MM-DD.log)"""
    if _logger_setup is None:
        initialize_logging()
    return _logger_setup.get_file_logger(name, "audit")


def get_cost_logger(name: str = "cost") -> structlog.BoundLogger:
    """Get cost logger (writes to cost_YYYY-MM-DD.log)"""
    if _logger_setup is None:
        initialize_logging()
    return _logger_setup.get_file_logger(name, "cost")


# ============================================================================
# LOG ENTRY BUILDERS (for consistent structured data)
# ============================================================================

class AuditLogEntry:
    """Builder for structured audit log entries"""

    @staticmethod
    def build(
        action: str,
        user: str,
        mode: str,
        details: Optional[Dict[str, Any]] = None,
        ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a structured audit log entry"""
        return {
            "action": action,
            "user": user,
            "mode": mode,
            "details": details or {},
            "ip": ip,
            "timestamp": datetime.now().isoformat(),
        }


class CostLogEntry:
    """Builder for structured cost log entries"""

    @staticmethod
    def build(
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Build a structured cost log entry"""
        return {
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 4),
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.now().isoformat(),
        }


class QueryLogEntry:
    """Builder for structured query log entries"""

    @staticmethod
    def build(
        query: str,
        mode: str,
        chunks_retrieved: int,
        latency_ms: float,
        cost_usd: float = 0.0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a structured query log entry"""
        return {
            "query": query,
            "mode": mode,
            "chunks_retrieved": chunks_retrieved,
            "latency_ms": round(latency_ms, 2),
            "cost_usd": round(cost_usd, 4),
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
