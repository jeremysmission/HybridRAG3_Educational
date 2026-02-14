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
#   Like a car's startup sequence: turn key → check battery → fuel pump
#   prime → engine crank → oil pressure check → ready to drive.
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
from dataclasses import dataclass, field
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
    result = BootResult()

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

    # === STEP 4: Check Ollama (Offline Mode) ===
    logger.info("BOOT Step 4: Checking Ollama...")
    try:
        import urllib.request
        ollama_host = config.get("ollama", {}).get("host", "http://localhost:11434")
        req = urllib.request.Request(
            f"{ollama_host}/api/tags",
            method="GET",
        )
        response = urllib.request.urlopen(req, timeout=3)
        if response.status == 200:
            result.offline_available = True
            logger.info("BOOT Step 4: Ollama is running")
        else:
            result.warnings.append("Ollama responded but with unexpected status")
    except urllib.error.URLError:
        result.warnings.append("Ollama is not running -- offline mode unavailable")
        logger.info("BOOT Step 4: Ollama not reachable")
    except Exception as e:
        result.warnings.append(f"Ollama check failed: {e}")
        logger.info("BOOT Step 4: Ollama check error: %s", e)

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
