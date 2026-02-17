# ===========================================================================
# HybridRAG v3 -- CANONICAL CREDENTIALS MODULE
# ===========================================================================
# FILE: src/security/credentials.py
#
# WHAT THIS IS:
#   The ONE AND ONLY place that reads API credentials. No other module
#   should ever read env vars or keyring directly. Everything goes
#   through this module.
#
# WHY THIS MATTERS:
#   Before this redesign, credentials were read in multiple places:
#     - config.py checked HYBRIDRAG_API_ENDPOINT
#     - llm_router.py checked AZURE_OPENAI_ENDPOINT then OPENAI_API_ENDPOINT
#     - start_hybridrag.ps1 set yet another variable name
#   This meant you could set the right variable in one place and the
#   wrong one in another, leading to "it works sometimes" bugs.
#
#   Now there's ONE resolution order, ONE set of aliases, ONE place
#   to debug credential problems.
#
# RESOLUTION ORDER (first match wins):
#   1. Windows Credential Manager (via keyring) -- most secure
#   2. Environment variables -- useful for CI/CD or session overrides
#   3. Config file -- least preferred for secrets, but OK for endpoints
#
# ACCEPTED ENVIRONMENT VARIABLE ALIASES:
#   API Key:
#     - HYBRIDRAG_API_KEY (canonical)
#     - AZURE_OPENAI_API_KEY
#     - OPENAI_API_KEY
#
#   Endpoint:
#     - HYBRIDRAG_API_ENDPOINT (canonical)
#     - AZURE_OPENAI_ENDPOINT
#     - OPENAI_API_ENDPOINT
#     - AZURE_OPENAI_BASE_URL
#     - OPENAI_BASE_URL
#
#   Deployment Name:
#     - AZURE_OPENAI_DEPLOYMENT (canonical)
#     - AZURE_DEPLOYMENT
#     - OPENAI_DEPLOYMENT
#     - AZURE_OPENAI_DEPLOYMENT_NAME
#     - DEPLOYMENT_NAME
#     - AZURE_CHAT_DEPLOYMENT
#
#   API Version:
#     - AZURE_OPENAI_API_VERSION (canonical)
#     - AZURE_API_VERSION
#     - OPENAI_API_VERSION
#     - API_VERSION
#
# DESIGN DECISIONS:
#   - Returns a dataclass (ApiCredentials) not a dict, so you get
#     autocomplete and type checking in your editor
#   - Records WHERE each credential came from (keyring, env, config)
#     so diagnostic tools can show "Key loaded from: keyring" in logs
#   - Never logs or prints the actual key value -- only masked previews
#   - Validates endpoint URL format before returning it
# ===========================================================================

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATA CLASS: Structured container for resolved credentials
# ---------------------------------------------------------------------------
# WHY A DATACLASS:
#   A dict would work but gives no autocomplete and no type hints.
#   A dataclass gives you creds.endpoint, creds.api_key, etc. with
#   full editor support. It's also immutable-friendly (frozen=False
#   here because we build it incrementally).
# ---------------------------------------------------------------------------

@dataclass
class ApiCredentials:
    """
    Container for resolved API credentials.

    Attributes:
        api_key: The API key string, or None if not found.
        endpoint: The API endpoint URL, or None if not found.
        deployment: Azure deployment name, or None.
        api_version: Azure API version, or None.
        source_key: Where the key came from ("keyring", "env:VAR_NAME", "config").
        source_endpoint: Where the endpoint came from.
        source_deployment: Where deployment came from.
        source_api_version: Where api_version came from.
    """
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    api_version: Optional[str] = None
    source_key: Optional[str] = None
    source_endpoint: Optional[str] = None
    source_deployment: Optional[str] = None
    source_api_version: Optional[str] = None

    @property
    def has_key(self) -> bool:
        """True if an API key was found."""
        return bool(self.api_key)

    @property
    def has_endpoint(self) -> bool:
        """True if an endpoint URL was found."""
        return bool(self.endpoint)

    @property
    def is_online_ready(self) -> bool:
        """True if both key and endpoint are present -- minimum for online mode."""
        return self.has_key and self.has_endpoint

    @property
    def key_preview(self) -> str:
        """
        Masked preview of the API key for logging/display.
        Shows first 4 and last 4 characters only.
        NEVER log the full key.
        """
        if not self.api_key:
            return "(not set)"
        if len(self.api_key) <= 8:
            return "****"
        return self.api_key[:4] + "..." + self.api_key[-4:]

    def to_diagnostic_dict(self) -> dict:
        """
        Safe-to-log dictionary for diagnostic output.
        API key is always masked.
        """
        return {
            "endpoint": self.endpoint or "(not set)",
            "api_key": self.key_preview,
            "deployment": self.deployment or "(not set)",
            "api_version": self.api_version or "(not set)",
            "source_key": self.source_key or "(not found)",
            "source_endpoint": self.source_endpoint or "(not found)",
            "source_deployment": self.source_deployment or "(not found)",
            "source_api_version": self.source_api_version or "(not found)",
            "online_ready": self.is_online_ready,
        }


# ---------------------------------------------------------------------------
# ENV VAR ALIASES: All accepted names for each credential
# ---------------------------------------------------------------------------
# WHY ALIASES:
#   Different tools and docs use different names. Azure docs say
#   AZURE_OPENAI_API_KEY, OpenAI docs say OPENAI_API_KEY, and we
#   defined HYBRIDRAG_API_KEY. Instead of forcing users to remember
#   one name, we accept all of them and resolve in priority order.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PUBLIC CONSTANTS -- importable by any module that needs canonical names.
# These are the SINGLE SOURCE OF TRUTH for credential naming.
#
# If you need keyring names or env var lists in another file, import these:
#   from src.security.credentials import KEYRING_SERVICE, KEY_ENV_ALIASES
#
# DO NOT hardcode keyring service/key names anywhere else.
# DO NOT duplicate these env var lists anywhere else.
# ---------------------------------------------------------------------------

KEY_ENV_ALIASES = [
    "HYBRIDRAG_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPEN_AI_KEY",         # Company variant (also used by LLMRouter)
    "OPENAI_API_KEY",
]

ENDPOINT_ENV_ALIASES = [
    "HYBRIDRAG_API_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_ENDPOINT",
    "AZURE_OPENAI_BASE_URL",
    "OPENAI_BASE_URL",
]

DEPLOYMENT_ENV_ALIASES = [
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_DEPLOYMENT",
    "OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "DEPLOYMENT_NAME",
    "AZURE_CHAT_DEPLOYMENT",
]

API_VERSION_ENV_ALIASES = [
    "AZURE_OPENAI_API_VERSION",
    "AZURE_API_VERSION",
    "OPENAI_API_VERSION",
    "API_VERSION",
]

# Keyring service and key names (PUBLIC -- import these, don't hardcode)
KEYRING_SERVICE = "hybridrag"
KEYRING_KEY_NAME = "azure_api_key"
KEYRING_ENDPOINT_NAME = "azure_endpoint"

# Backward-compatible aliases (underscore versions still work)
_KEY_ENV_ALIASES = KEY_ENV_ALIASES
_ENDPOINT_ENV_ALIASES = ENDPOINT_ENV_ALIASES
_DEPLOYMENT_ENV_ALIASES = DEPLOYMENT_ENV_ALIASES
_API_VERSION_ENV_ALIASES = API_VERSION_ENV_ALIASES
_KEYRING_SERVICE = KEYRING_SERVICE
_KEYRING_KEY_NAME = KEYRING_KEY_NAME
_KEYRING_ENDPOINT_NAME = KEYRING_ENDPOINT_NAME


# ---------------------------------------------------------------------------
# HELPER: Read first matching env var from a list of aliases
# ---------------------------------------------------------------------------

def _resolve_env_var(aliases):
    """
    Check each alias in order, return (value, var_name) for the first
    one that is set and non-empty. Returns (None, None) if none found.
    """
    for var_name in aliases:
        value = os.environ.get(var_name, "").strip()
        if value:
            return value, var_name
    return None, None


# ---------------------------------------------------------------------------
# HELPER: Read from keyring safely
# ---------------------------------------------------------------------------

def _read_keyring(key_name):
    """
    Read a value from Windows Credential Manager via keyring.
    Returns None if keyring is not available or key not found.

    WHY TRY/EXCEPT:
      keyring may not be installed (ImportError),
      or the backend may not be available on this OS (keyring.errors.*),
      or the credential simply doesn't exist (returns None).
      We handle all cases gracefully.
    """
    try:
        import keyring
        value = keyring.get_password(_KEYRING_SERVICE, key_name)
        if value and value.strip():
            return value.strip()
    except ImportError:
        logger.debug("keyring module not installed -- skipping keyring lookup")
    except Exception as e:
        logger.debug("keyring read failed for '%s': %s", key_name, e)
    return None


# ---------------------------------------------------------------------------
# ENDPOINT VALIDATION
# ---------------------------------------------------------------------------

# Characters that should NEVER appear in a URL -- these are common
# copy-paste artifacts from Word, Outlook, Teams, etc.
_BAD_URL_CHARS = re.compile(r'[\u201c\u201d\u2018\u2019\u2013\u2014\u00a0\ufeff]')


def validate_endpoint(url):
    """
    Validate and clean an endpoint URL.

    Checks:
      - Not empty
      - Starts with https:// (or http:// if explicitly allowed)
      - No smart quotes or hidden Unicode characters
      - No spaces
      - No double slashes in the path (after scheme)
      - Strips trailing slashes for consistency

    Args:
        url: The endpoint URL string to validate.

    Returns:
        Cleaned URL string.

    Raises:
        InvalidEndpointError: If the URL is malformed.
    """
    # Import here to avoid circular import
    from src.core.exceptions import InvalidEndpointError

    if not url or not url.strip():
        raise InvalidEndpointError("Endpoint URL is empty.", url=url)

    url = url.strip()

    # Check for smart quotes and other bad characters
    bad_chars = _BAD_URL_CHARS.findall(url)
    if bad_chars:
        chars_display = ", ".join(repr(c) for c in bad_chars)
        raise InvalidEndpointError(
            f"Endpoint URL contains invalid characters: {chars_display}. "
            "This usually happens from copy-pasting from Word or a chat app.",
            url=url,
        )

    # Check for spaces
    if " " in url:
        raise InvalidEndpointError(
            "Endpoint URL contains spaces.",
            url=url,
        )

    # Check scheme
    if not url.startswith("https://") and not url.startswith("http://"):
        raise InvalidEndpointError(
            f"Endpoint URL must start with https:// -- got: '{url[:30]}...'",
            url=url,
        )

    # Check for double slashes in path (after scheme)
    scheme_end = url.index("://") + 3
    path_part = url[scheme_end:]
    if "//" in path_part:
        raise InvalidEndpointError(
            "Endpoint URL has double slashes in the path. "
            "This usually means a URL was concatenated incorrectly.",
            url=url,
        )

    # Strip trailing slash for consistency
    url = url.rstrip("/")

    return url


# ---------------------------------------------------------------------------
# MAIN FUNCTION: Resolve all credentials
# ---------------------------------------------------------------------------

def resolve_credentials(config_dict=None):
    """
    Resolve API credentials from all sources.

    Resolution order (first match wins):
      1. Windows Credential Manager (keyring)
      2. Environment variables (checks all aliases)
      3. Config dictionary (if provided)

    Args:
        config_dict: Optional dict from config.yaml with keys like
            api.key, api.endpoint, api.deployment, api.api_version

    Returns:
        ApiCredentials dataclass with all resolved values.

    NOTE: This function does NOT raise exceptions for missing credentials.
    It returns whatever it finds and lets the caller (ApiClientFactory)
    decide whether that's enough to proceed.
    """
    creds = ApiCredentials()

    # --- Resolve API Key ---
    # Priority 1: Keyring
    key_from_keyring = _read_keyring(_KEYRING_KEY_NAME)
    if key_from_keyring:
        creds.api_key = key_from_keyring
        creds.source_key = "keyring"
        logger.debug("API key loaded from keyring")
    else:
        # Priority 2: Environment variables
        key_val, key_var = _resolve_env_var(_KEY_ENV_ALIASES)
        if key_val:
            creds.api_key = key_val
            creds.source_key = f"env:{key_var}"
            logger.debug("API key loaded from env: %s", key_var)
        elif config_dict:
            # Priority 3: Config file
            cfg_key = _nested_get(config_dict, "api", "key")
            if cfg_key:
                creds.api_key = cfg_key
                creds.source_key = "config"
                logger.debug("API key loaded from config file")

    # --- Resolve Endpoint ---
    # Priority 1: Keyring
    endpoint_from_keyring = _read_keyring(_KEYRING_ENDPOINT_NAME)
    if endpoint_from_keyring:
        creds.endpoint = endpoint_from_keyring
        creds.source_endpoint = "keyring"
        logger.debug("Endpoint loaded from keyring")
    else:
        # Priority 2: Environment variables
        ep_val, ep_var = _resolve_env_var(_ENDPOINT_ENV_ALIASES)
        if ep_val:
            creds.endpoint = ep_val
            creds.source_endpoint = f"env:{ep_var}"
            logger.debug("Endpoint loaded from env: %s", ep_var)
        elif config_dict:
            # Priority 3: Config file
            cfg_ep = _nested_get(config_dict, "api", "endpoint")
            if cfg_ep:
                creds.endpoint = cfg_ep
                creds.source_endpoint = "config"
                logger.debug("Endpoint loaded from config file")

    # --- Resolve Deployment Name ---
    # Check URL first (may contain /deployments/name)
    if creds.endpoint and "/deployments/" in creds.endpoint:
        match = re.search(r"/deployments/([^/?]+)", creds.endpoint)
        if match:
            creds.deployment = match.group(1)
            creds.source_deployment = "extracted_from_url"
            logger.debug("Deployment extracted from URL: %s", creds.deployment)

    # If not in URL, check env vars
    if not creds.deployment:
        dep_val, dep_var = _resolve_env_var(_DEPLOYMENT_ENV_ALIASES)
        if dep_val:
            creds.deployment = dep_val
            creds.source_deployment = f"env:{dep_var}"
            logger.debug("Deployment loaded from env: %s", dep_var)
        elif config_dict:
            cfg_dep = _nested_get(config_dict, "api", "deployment")
            if cfg_dep:
                creds.deployment = cfg_dep
                creds.source_deployment = "config"

    # --- Resolve API Version ---
    # Check URL first (may contain ?api-version=xxx)
    if creds.endpoint and "api-version=" in creds.endpoint:
        match = re.search(r"api-version=([^&]+)", creds.endpoint)
        if match:
            creds.api_version = match.group(1)
            creds.source_api_version = "extracted_from_url"

    if not creds.api_version:
        ver_val, ver_var = _resolve_env_var(_API_VERSION_ENV_ALIASES)
        if ver_val:
            creds.api_version = ver_val
            creds.source_api_version = f"env:{ver_var}"
        elif config_dict:
            cfg_ver = _nested_get(config_dict, "api", "api_version")
            if cfg_ver:
                creds.api_version = cfg_ver
                creds.source_api_version = "config"

    # --- Validate endpoint if present ---
    if creds.endpoint:
        try:
            creds.endpoint = validate_endpoint(creds.endpoint)
        except Exception as e:
            logger.warning("Endpoint validation failed: %s", e)
            # Don't clear endpoint -- let the caller decide what to do
            # The factory will re-validate and raise if needed

    return creds


# ---------------------------------------------------------------------------
# STORE FUNCTIONS: Write credentials to keyring
# ---------------------------------------------------------------------------

def store_api_key(key):
    """Store API key in Windows Credential Manager."""
    import keyring
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_KEY_NAME, key)
    logger.info("API key stored in keyring")


def store_endpoint(endpoint):
    """Store API endpoint in Windows Credential Manager."""
    import keyring
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_ENDPOINT_NAME, endpoint)
    logger.info("Endpoint stored in keyring")


def clear_credentials():
    """Remove all stored credentials from keyring."""
    import keyring
    for name in [_KEYRING_KEY_NAME, _KEYRING_ENDPOINT_NAME]:
        try:
            keyring.delete_password(_KEYRING_SERVICE, name)
        except Exception:
            pass
    logger.info("All credentials cleared from keyring")


# ---------------------------------------------------------------------------
# HELPER: Safe nested dict access
# ---------------------------------------------------------------------------

def _nested_get(d, *keys):
    """
    Safely get a nested value from a dictionary.
    _nested_get({"api": {"key": "abc"}}, "api", "key") returns "abc"
    _nested_get({"api": {}}, "api", "key") returns None
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return None
    return d if d else None


# ---------------------------------------------------------------------------
# CONVENIENCE: Quick access functions (backward compatible)
# ---------------------------------------------------------------------------
# These exist so existing code that calls get_api_key() still works
# without needing to switch to the full resolve_credentials() pattern.
# New code should use resolve_credentials() instead.
# ---------------------------------------------------------------------------

def get_api_key():
    """Get API key from keyring or env vars. Returns None if not found."""
    creds = resolve_credentials()
    return creds.api_key


def get_api_endpoint():
    """Get API endpoint from keyring or env vars. Returns None if not found."""
    creds = resolve_credentials()
    return creds.endpoint


# ---------------------------------------------------------------------------
# STATUS FUNCTION: Used by _check_creds.py and rag-mode-online
# ---------------------------------------------------------------------------
# WHY THIS EXISTS:
#   rag-mode-online needs to check if credentials are stored BEFORE
#   switching modes. This function returns a simple dict that the
#   PowerShell wrapper can parse to decide whether to proceed.
#
#   Example return value:
#     {
#         'api_key_set': True,
#         'api_endpoint_set': True,
#         'api_key_source': 'keyring',
#         'api_endpoint_source': 'keyring',
#     }
# ---------------------------------------------------------------------------

def credential_status():
    """
    Check what credentials are currently stored/available.

    Returns:
        dict with keys:
            api_key_set (bool): True if an API key was found anywhere
            api_endpoint_set (bool): True if an endpoint was found anywhere
            api_key_source (str): Where the key came from ('keyring', 'env:VAR', 'config', 'none')
            api_endpoint_source (str): Where the endpoint came from
    """
    creds = resolve_credentials()
    return {
        'api_key_set': creds.has_key,
        'api_endpoint_set': creds.has_endpoint,
        'api_key_source': creds.source_key or 'none',
        'api_endpoint_source': creds.source_endpoint or 'none',
    }


# ---------------------------------------------------------------------------
# CLI HANDLER: Called by rag-store-key, rag-store-endpoint, etc.
# ---------------------------------------------------------------------------
# WHY THIS EXISTS:
#   api_mode_commands.ps1 calls:
#     python -m src.security.credentials store
#     python -m src.security.credentials endpoint
#     python -m src.security.credentials status
#     python -m src.security.credentials delete
#
#   Without this __main__ block, those commands do nothing.
#   This was a known bug from February 2026 that caused rag-store-key
#   to silently exit without prompting for a key.
#
# SECURITY NOTE:
#   The key is read via getpass (hidden input on Windows).
#   It is NEVER echoed to the screen or written to any file.
#   It goes straight from your keyboard to Windows Credential Manager.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import getpass

    # Determine which subcommand was requested
    command = sys.argv[1] if len(sys.argv) > 1 else "status"

    if command == "store":
        # rag-store-key calls this
        # getpass hides the input so the key never appears on screen
        print("Enter your API key (input is hidden):")
        try:
            key = getpass.getpass(prompt="API Key: ")
        except EOFError:
            # Handle case where input is piped or redirected
            key = input("API Key: ")

        if not key or not key.strip():
            print("ERROR: No key entered. Nothing stored.")
            sys.exit(1)

        key = key.strip()
        store_api_key(key)
        print("API key stored in Windows Credential Manager.")
        print("Key preview: " + key[:4] + "..." + key[-4:] if len(key) > 8 else "****")

    elif command == "endpoint":
        # rag-store-endpoint calls this
        print("Enter your API endpoint URL:")
        try:
            endpoint = input("Endpoint: ")
        except EOFError:
            endpoint = ""

        if not endpoint or not endpoint.strip():
            print("ERROR: No endpoint entered. Nothing stored.")
            sys.exit(1)

        endpoint = endpoint.strip()

        # Validate before storing
        try:
            endpoint = validate_endpoint(endpoint)
        except Exception as e:
            print(f"ERROR: Invalid endpoint URL: {e}")
            sys.exit(1)

        store_endpoint(endpoint)
        print(f"Endpoint stored: {endpoint}")

    elif command == "status":
        # rag-cred-status calls this
        status = credential_status()
        creds = resolve_credentials()

        print("")
        print("  Credential Status:")
        print("  ------------------")
        print(f"  API Key:    {'STORED' if status['api_key_set'] else 'NOT SET'}"
              f"  (source: {status['api_key_source']})")
        if status['api_key_set']:
            print(f"  Key preview: {creds.key_preview}")
        print(f"  Endpoint:   {'STORED' if status['api_endpoint_set'] else 'NOT SET'}"
              f"  (source: {status['api_endpoint_source']})")
        if status['api_endpoint_set']:
            print(f"  Endpoint:   {creds.endpoint}")
        if creds.deployment:
            print(f"  Deployment: {creds.deployment} (source: {creds.source_deployment})")
        if creds.api_version:
            print(f"  API Ver:    {creds.api_version} (source: {creds.source_api_version})")
        print(f"  Online ready: {creds.is_online_ready}")
        print("")

    elif command == "delete":
        # rag-cred-delete calls this
        clear_credentials()
        print("All credentials removed from Windows Credential Manager.")

    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m src.security.credentials [store|endpoint|status|delete]")
        sys.exit(1)
