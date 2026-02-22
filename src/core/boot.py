# ===========================================================================
# HybridRAG v3 -- BOOT PIPELINE
# ===========================================================================
# FILE: src/core/boot.py
#
# WHAT THIS IS:
#   The single entry point that starts HybridRAG. It runs every
#   validation step in the correct order and either produces a fully
#   working system or tells you exactly what's broken.
#
# WHY THIS MATTERS:
#   Before this redesign, startup was scattered across multiple files:
#     - start_hybridrag.ps1 set env vars
#     - config.py loaded the YAML
#     - llm_router.py built clients (sometimes with missing credentials)
#     - various modules validated things at different times
#
#   This meant "it works if you do things in the right order" -- but
#   if you forgot a step, you'd get mysterious failures much later.
#
#   Now there's ONE pipeline that runs ALL checks upfront:
#     1. Load config
#     2. Resolve credentials
#     3. Validate config + credentials together
#     4. Construct services (API client, Ollama client, etc.)
#     5. Return a ready-to-use HybridRAG instance
#
# ANALOGY:
#   Like a car's startup sequence: turn key -> check battery -> fuel pump
#   prime -> engine crank -> oil pressure check -> ready to drive.
#   Each step must pass before the next one runs.
#
# USAGE:
#   from src.core.boot import boot_hybridrag
#
#   rag = boot_hybridrag()  # Validates everything
#   # rag.query("What is X?", mode="api")
#   # rag.query("What is X?", mode="offline")
#
# DESIGN DECISIONS:
#   - Returns a HybridRAGInstance that holds all initialized services
#   - Does NOT crash on missing API credentials -- just marks online
#     mode as unavailable (offline mode still works)
#   - Logs every step so you can see exactly what happened during boot
#   - boot_hybridrag() is the ONLY function that creates services
# ===========================================================================

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class BootResult:
    """
    Result of the boot pipeline.

    Attributes:
        success: True if at least offline mode is available.
        online_available: True if API client was created successfully.
        offline_available: True if Ollama is configured.
        api_client: The ApiClient instance, or None if not available.
        config: The loaded configuration dictionary.
        credentials: Resolved credentials (with masked key).
        warnings: Non-fatal issues found during boot.
        errors: Fatal issues that prevented a mode from starting.
    """
    boot_timestamp: str = ""
    success: bool = False
    online_available: bool = False
    offline_available: bool = False
    api_client: Optional[Any] = None  # ApiClient instance
    config: Dict[str, Any] = field(default_factory=dict)
    credentials: Optional[Any] = None  # ApiCredentials instance
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable boot summary for console/GUI display."""
        lines = []
        lines.append("=" * 50)
        lines.append("  HYBRIDRAG BOOT STATUS")
        lines.append("=" * 50)
        lines.append(f"  Overall:  {'READY' if self.success else 'FAILED'}")
        lines.append(f"  Online:   {'AVAILABLE' if self.online_available else 'NOT AVAILABLE'}")
        lines.append(f"  Offline:  {'AVAILABLE' if self.offline_available else 'NOT AVAILABLE'}")

        if self.warnings:
            lines.append("")
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    [!] {w}")

        if self.errors:
            lines.append("")
            lines.append("  ERRORS:")
            for e in self.errors:
                lines.append(f"    [X] {e}")

        lines.append("=" * 50)
        return "\n".join(lines)


def load_config(config_path=None) -> dict:
    """
    Load configuration from YAML file.

    Search order:
      1. Explicit path argument
      2. config/default_config.yaml (relative to project root)
      3. config.yaml (relative to project root)

    Returns:
        Dict of configuration values.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed -- using empty config")
        return {}

    # Determine project root (two levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent

    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    search_paths.extend([
        project_root / "config" / "default_config.yaml",
        project_root / "config.yaml",
    ])

    for path in search_paths:
        if path.exists():
            logger.info("Loading config from: %s", path)
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return config

    logger.warning("No config file found -- using defaults")
    return {}


def boot_hybridrag(config_path=None) -> BootResult:
    """
    Run the complete boot pipeline.

    Steps:
      1. Load configuration from YAML
      2. Resolve credentials from keyring/env/config
      3. Attempt to build API client (online mode)
      4. Check Ollama availability (offline mode)
      5. Return BootResult with status of everything

    This function NEVER crashes. It catches all exceptions and
    records them in BootResult.errors so the caller can decide
    what to do.

    Args:
        config_path: Optional path to config YAML file.

    Returns:
        BootResult with all status information.
    """
    result = BootResult(
        boot_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # === STEP 1: Load Configuration ===
    logger.info("BOOT Step 1: Loading configuration...")
    try:
        config = load_config(config_path)
        result.config = config
        logger.info("BOOT Step 1: Configuration loaded")
    except Exception as e:
        result.errors.append(f"Config load failed: {e}")
        logger.error("BOOT Step 1 FAILED: %s", e)
        return result

    # === STEP 2: Resolve Credentials ===
    logger.info("BOOT Step 2: Resolving credentials...")
    try:
        from src.security.credentials import resolve_credentials
        creds = resolve_credentials(config)
        result.credentials = creds

        if creds.has_endpoint:
            logger.info("BOOT Step 2: Endpoint found (source: %s)", creds.source_endpoint)
        else:
            result.warnings.append("No API endpoint configured -- online mode unavailable")

        if creds.has_key:
            logger.info("BOOT Step 2: API key found (source: %s)", creds.source_key)
        else:
            result.warnings.append("No API key configured -- online mode unavailable")

    except Exception as e:
        result.warnings.append(f"Credential resolution failed: {e}")
        logger.warning("BOOT Step 2 WARNING: %s", e)

    # === STEP 2.5: Configure Network Gate ===
    # The gate must be configured BEFORE any network calls (Steps 3-4).
    # It reads the mode from config and the endpoint from credentials
    # to build the access control policy.
    logger.info("BOOT Step 2.5: Configuring network gate...")
    try:
        from src.core.network_gate import configure_gate

        # Determine the mode and endpoint for the gate
        boot_mode = config.get("mode", "offline") if isinstance(config, dict) else "offline"
        boot_endpoint = ""
        if result.credentials and result.credentials.endpoint:
            boot_endpoint = result.credentials.endpoint
        elif isinstance(config, dict):
            boot_endpoint = config.get("api", {}).get("endpoint", "")

        # Get allowed_endpoint_prefixes from config if available
        allowed_prefixes = []
        if isinstance(config, dict):
            allowed_prefixes = config.get("api", {}).get("allowed_endpoint_prefixes", [])

        # Validate endpoint URL format before using it.
        # Catches "openai.azure.com" (missing https://) with a clear
        # message instead of a 30-second mystery timeout.
        if boot_endpoint and not (
            boot_endpoint.startswith("http://") or
            boot_endpoint.startswith("https://")):
            result.warnings.append(
                f"Invalid endpoint format: {boot_endpoint}. "
                f"Expected http:// or https://")
            boot_endpoint = ""

        gate = configure_gate(
            mode=boot_mode,
            api_endpoint=boot_endpoint,
            allowed_prefixes=allowed_prefixes,
        )
        logger.info(
            "BOOT Step 2.5: Network gate configured (mode=%s, endpoint=%s)",
            boot_mode, boot_endpoint[:50] if boot_endpoint else "(none)",
        )
    except Exception as e:
        # If the gate fails to configure, we continue with it in OFFLINE
        # mode (the safe default). This is fail-closed behavior.
        result.warnings.append(f"Network gate configuration failed: {e}")
        logger.warning("BOOT Step 2.5: Gate config failed, defaulting to OFFLINE: %s", e)

    # === STEP 3: Build API Client (Online Mode) ===
    logger.info("BOOT Step 3: Building API client...")
    if result.credentials and result.credentials.is_online_ready:
        try:
            from src.core.api_client_factory import ApiClientFactory
            factory = ApiClientFactory(config)
            client = factory.build(result.credentials)
            result.api_client = client
            result.online_available = True
            logger.info("BOOT Step 3: API client created successfully")

            # Log diagnostic info (safe -- no secrets)
            diag = client.get_diagnostic_info()
            logger.info("BOOT Step 3: Provider=%s, Auth=%s", diag["provider"], diag["auth_header"])

        except Exception as e:
            error_msg = str(e)
            fix = getattr(e, "fix_suggestion", None)
            result.errors.append(f"API client creation failed: {error_msg}")
            if fix:
                result.errors.append(f"  Fix: {fix}")
            logger.error("BOOT Step 3 FAILED: %s", e)
    else:
        result.warnings.append("Skipping API client -- credentials incomplete")
        logger.info("BOOT Step 3: Skipped (credentials incomplete)")

    # === STEP 4: Check Ollama (non-blocking) ===
    # Ollama is localhost -- responds in <50ms when running. Run the check
    # in a daemon thread with a short join timeout so boot is not blocked
    # for 3s if Ollama is down or slow to respond.
    logger.info("BOOT Step 4: Checking Ollama (non-blocking)...")

    def _check_ollama():
        try:
            import urllib.request
            import urllib.error
            from src.core.network_gate import get_gate
            ollama_host = config.get("ollama", {}).get(
                "host", "http://localhost:11434",
            )
            get_gate().check_allowed(
                f"{ollama_host}/api/tags", "ollama_boot_check", "boot",
            )
            req = urllib.request.Request(
                f"{ollama_host}/api/tags", method="GET",
            )
            response = urllib.request.urlopen(req, timeout=3)
            if response.status == 200:
                result.offline_available = True
                logger.info("BOOT Step 4: Ollama is running")
            else:
                result.warnings.append(
                    "Ollama responded but with unexpected status"
                )
        except Exception:
            result.warnings.append(
                "Ollama is not running -- offline mode unavailable"
            )
            logger.info("BOOT Step 4: Ollama not reachable")

    ollama_thread = threading.Thread(target=_check_ollama, daemon=True)
    ollama_thread.start()
    ollama_thread.join(timeout=0.5)

    if ollama_thread.is_alive():
        # Ollama slow or down -- don't block boot any longer.
        # Status bar will verify availability within 30s.
        result.offline_available = True
        logger.info("BOOT Step 4: Ollama check timed out, assuming available")

    # === FINAL: Determine overall success ===
    result.success = result.online_available or result.offline_available

    if not result.success:
        result.errors.append(
            "Neither online nor offline mode is available. "
            "Run 'rag-status' for diagnostics."
        )

    logger.info("BOOT Complete: success=%s, online=%s, offline=%s",
                result.success, result.online_available, result.offline_available)

    return result
