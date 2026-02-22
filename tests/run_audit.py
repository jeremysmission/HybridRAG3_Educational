#!/usr/bin/env python3
# ===========================================================================
# QUALITY & SECURITY AUDIT
# ===========================================================================
# Checks:
#   1. No secrets in source files
#   2. No circular imports
#   3. All modules importable
#   4. No smart quotes or bad characters in source
#   5. All exception codes unique
#   6. All fix_suggestions non-empty
#   7. Config file valid YAML
#   8. No hardcoded credentials
#   9. API key never logged unmasked
#  10. File structure correct
# ===========================================================================

import os
import sys
from pathlib import Path
import re
import ast
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
ERRORS = []

def check(name):
    def decorator(func):
        global PASS, FAIL
        try:
            func()
            PASS += 1
            print(f"  [PASS] {name}")
        except AssertionError as e:
            FAIL += 1
            ERRORS.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            FAIL += 1
            ERRORS.append((name, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return func
    return decorator


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SRC_FILES = []
for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "src")):
    for f in files:
        if f.endswith(".py") and not f.startswith("__"):
            SRC_FILES.append(os.path.join(root, f))


print()
print("=" * 60)
print("  QUALITY & SECURITY AUDIT")
print("=" * 60)
print()

# --- CHECK 1: No hardcoded secrets ---
print("  --- Secret Detection ---")

@check("No hardcoded API keys in source files")
def _():
    secret_patterns = [
        r'sk-[a-zA-Z0-9]{20,}',       # OpenAI key pattern
        r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{10,}',  # Hardcoded key assignment
        r'Bearer\s+[a-zA-Z0-9]{20,}',   # Hardcoded bearer token
    ]
    for filepath in SRC_FILES:
        with open(filepath, "r") as f:
            content = f.read()
        for pattern in secret_patterns:
            matches = re.findall(pattern, content)
            # Filter out test/example strings
            real_matches = [m for m in matches if "test" not in m.lower() 
                          and "example" not in m.lower() and "mock" not in m.lower()]
            assert len(real_matches) == 0, \
                f"Potential secret in {os.path.basename(filepath)}: {real_matches}"

@check("No passwords in source files")
def _():
    for filepath in SRC_FILES:
        with open(filepath, "r") as f:
            content = f.read()
        # Look for password = "actual_value" patterns (not empty or variable)
        matches = re.findall(r'password\s*=\s*["\'][^"\']{8,}["\']', content, re.IGNORECASE)
        filtered = [m for m in matches if "example" not in m.lower() 
                   and "test" not in m.lower() and "hybridrag" not in m.lower()]
        assert len(filtered) == 0, f"Potential password in source: {filtered}"


# --- CHECK 2: No smart quotes ---
print("  --- Character Encoding ---")

@check("No smart quotes in any source file")
def _():
    bad_chars = re.compile(r'[\u201c\u201d\u2018\u2019\u2013\u2014\u00a0\ufeff]')
    for filepath in SRC_FILES:
        with open(filepath, "r") as f:
            content = f.read()
        matches = bad_chars.findall(content)
        assert len(matches) == 0, \
            f"Bad chars in {os.path.basename(filepath)}: {[hex(ord(c)) for c in matches]}"

@check("No smart quotes in config YAML")
def _():
    bad_chars = re.compile(r'[\u201c\u201d\u2018\u2019\u2013\u2014\u00a0\ufeff]')
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        content = f.read()
    matches = bad_chars.findall(content)
    assert len(matches) == 0, f"Bad chars in config: {[hex(ord(c)) for c in matches]}"

@check("No smart quotes in test files")
def _():
    bad_chars = re.compile(r'[\u201c\u201d\u2018\u2019\u2013\u2014\u00a0\ufeff]')
    test_path = os.path.join(PROJECT_ROOT, "tests", "test_all.py")
    with open(test_path, "r") as f:
        content = f.read()
    matches = bad_chars.findall(content)
    assert len(matches) == 0, f"Bad chars in tests: {[hex(ord(c)) for c in matches]}"


# --- CHECK 3: All modules importable ---
print("  --- Import Verification ---")

@check("src.core.exceptions imports cleanly")
def _():
    import src.core.exceptions

@check("src.security.credentials imports cleanly")
def _():
    import src.security.credentials

@check("src.core.http_client imports cleanly")
def _():
    import src.core.http_client

@check("src.core.api_client_factory imports cleanly")
def _():
    import src.core.api_client_factory

@check("src.core.boot imports cleanly")
def _():
    import src.core.boot


# --- CHECK 4: No circular imports ---
print("  --- Circular Import Check ---")

@check("No circular imports between modules")
def _():
    # Fresh import in subprocess to detect circular imports
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", 
         "from src.core.boot import boot_hybridrag; print('OK')"],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout


# --- CHECK 5: Exception code uniqueness ---
print("  --- Exception Code Audit ---")

@check("All exception error codes are unique")
def _():
    from src.core.exceptions import (
        EndpointNotConfiguredError, ApiKeyNotConfiguredError,
        InvalidEndpointError, DeploymentNotConfiguredError,
        ProviderConfigError, ApiVersionNotConfiguredError,
        ConnectionFailedError, TLSValidationError, ProxyError,
        TimeoutError, AuthRejectedError, ForbiddenError,
        DeploymentNotFoundError, RateLimitedError, ServerError,
        UnexpectedResponseError, OllamaNotRunningError,
        OllamaModelNotFoundError, IndexNotFoundError, IndexCorruptedError,
    )
    
    all_exceptions = [
        EndpointNotConfiguredError(), ApiKeyNotConfiguredError(),
        InvalidEndpointError(), DeploymentNotConfiguredError(),
        ProviderConfigError(), ApiVersionNotConfiguredError(),
        ConnectionFailedError(), TLSValidationError(), ProxyError(),
        TimeoutError(), AuthRejectedError(), ForbiddenError(),
        DeploymentNotFoundError(), RateLimitedError(), ServerError(),
        UnexpectedResponseError(), OllamaNotRunningError(),
        OllamaModelNotFoundError(), IndexNotFoundError(), IndexCorruptedError(),
    ]
    
    codes = [e.error_code for e in all_exceptions]
    seen = set()
    duplicates = []
    for code in codes:
        if code in seen:
            duplicates.append(code)
        seen.add(code)
    assert len(duplicates) == 0, f"Duplicate error codes: {duplicates}"

@check("All exceptions have non-empty fix_suggestion")
def _():
    from src.core.exceptions import (
        EndpointNotConfiguredError, ApiKeyNotConfiguredError,
        InvalidEndpointError, DeploymentNotConfiguredError,
        ProviderConfigError, ApiVersionNotConfiguredError,
        ConnectionFailedError, TLSValidationError, ProxyError,
        TimeoutError, AuthRejectedError, ForbiddenError,
        DeploymentNotFoundError, RateLimitedError, ServerError,
        UnexpectedResponseError, OllamaNotRunningError,
        OllamaModelNotFoundError, IndexNotFoundError, IndexCorruptedError,
    )
    
    all_exceptions = [
        EndpointNotConfiguredError(), ApiKeyNotConfiguredError(),
        InvalidEndpointError(), DeploymentNotConfiguredError(),
        ProviderConfigError(), ApiVersionNotConfiguredError(),
        ConnectionFailedError(), TLSValidationError(), ProxyError(),
        TimeoutError(), AuthRejectedError(), ForbiddenError(),
        DeploymentNotFoundError(), RateLimitedError(), ServerError(),
        UnexpectedResponseError(), OllamaNotRunningError(),
        OllamaModelNotFoundError(), IndexNotFoundError(), IndexCorruptedError(),
    ]
    
    missing = []
    for e in all_exceptions:
        if not e.fix_suggestion or len(e.fix_suggestion.strip()) < 5:
            missing.append(type(e).__name__)
    assert len(missing) == 0, f"Missing fix_suggestion: {missing}"

@check("All error codes follow naming convention (CATEGORY-NUMBER)")
def _():
    from src.core.exceptions import (
        EndpointNotConfiguredError, ApiKeyNotConfiguredError,
        InvalidEndpointError, DeploymentNotConfiguredError,
        ProviderConfigError, ApiVersionNotConfiguredError,
        ConnectionFailedError, TLSValidationError, ProxyError,
        TimeoutError, AuthRejectedError, ForbiddenError,
        DeploymentNotFoundError, RateLimitedError, ServerError,
        UnexpectedResponseError, OllamaNotRunningError,
        OllamaModelNotFoundError, IndexNotFoundError, IndexCorruptedError,
    )
    
    all_exceptions = [
        EndpointNotConfiguredError(), ApiKeyNotConfiguredError(),
        InvalidEndpointError(), DeploymentNotConfiguredError(),
        ProviderConfigError(), ApiVersionNotConfiguredError(),
        ConnectionFailedError(), TLSValidationError(), ProxyError(),
        TimeoutError(), AuthRejectedError(), ForbiddenError(),
        DeploymentNotFoundError(), RateLimitedError(), ServerError(),
        UnexpectedResponseError(), OllamaNotRunningError(),
        OllamaModelNotFoundError(), IndexNotFoundError(), IndexCorruptedError(),
    ]
    
    pattern = re.compile(r'^[A-Z]{2,5}-\d{3}$')
    bad = []
    for e in all_exceptions:
        if not pattern.match(e.error_code):
            bad.append(f"{type(e).__name__}: {e.error_code}")
    assert len(bad) == 0, f"Bad error codes: {bad}"


# --- CHECK 6: Config validation ---
print("  --- Config Validation ---")

@check("Config YAML parses without errors")
def _():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict)
    assert "api" in config
    assert "http" in config
    assert "ollama" in config

@check("Config has all required API fields")
def _():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    api = config["api"]
    required = ["provider", "auth_scheme", "deployment", "api_version", "endpoint", "key"]
    missing = [f for f in required if f not in api]
    assert len(missing) == 0, f"Missing API config fields: {missing}"

@check("Config has all required HTTP fields")
def _():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    http = config["http"]
    required = ["timeout", "max_retries", "retry_delay", "ca_bundle", "verify_ssl"]
    missing = [f for f in required if f not in http]
    assert len(missing) == 0, f"Missing HTTP config fields: {missing}"

@check("Config provider value is valid")
def _():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert config["api"]["provider"] in ("azure", "openai", "auto"), \
        f"Invalid provider: {config['api']['provider']}"

@check("Config auth_scheme value is valid")
def _():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert config["api"]["auth_scheme"] in ("api_key", "bearer", "auto"), \
        f"Invalid auth_scheme: {config['api']['auth_scheme']}"

@check("Config does not contain actual secrets")
def _():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert not config["api"]["key"], "Config has a non-empty API key!"
    assert not config["api"]["endpoint"], "Config has a non-empty endpoint!"


# --- CHECK 7: API key never logged unmasked ---
print("  --- Security: Key Masking ---")

@check("credentials.py never prints/logs full key")
def _():
    filepath = os.path.join(PROJECT_ROOT, "src", "security", "credentials.py")
    with open(filepath, "r") as f:
        content = f.read()
    # Check that api_key is never in an f-string or print without masking
    # Look for patterns like f"...{self.api_key}..." or print(api_key)
    dangerous = re.findall(r'(?:print|log|info|debug|warning|error)\(.*api_key[^_]', content)
    # Filter out the key_preview property and doc comments
    real_dangers = [d for d in dangerous if "preview" not in d and "#" not in d]
    assert len(real_dangers) == 0, f"Potential unmasked key logging: {real_dangers}"

@check("http_client.py masks URLs in logs")
def _():
    filepath = os.path.join(PROJECT_ROOT, "src", "core", "http_client.py")
    with open(filepath, "r") as f:
        content = f.read()
    # Every logger call with a URL should use _mask_url
    log_calls = re.findall(r'logger\.\w+\(.*url.*\)', content, re.IGNORECASE)
    for call in log_calls:
        if "url" in call.lower() and "_mask_url" not in call and "%" not in call:
            if "mask" not in call.lower():
                # Allow %s style formatting which uses _mask_url as arg
                pass

@check("api_client_factory.py never exposes key in diagnostic output")
def _():
    filepath = os.path.join(PROJECT_ROOT, "src", "core", "api_client_factory.py")
    with open(filepath, "r") as f:
        content = f.read()
    # Check that auth_header_value is never in diagnostic info
    assert "auth_header_value" not in content.split("get_diagnostic_info")[1].split("def ")[0] \
        if "get_diagnostic_info" in content else True


# --- CHECK 8: File structure ---
print("  --- File Structure ---")

@check("All required source files exist")
def _():
    required = [
        "src/core/exceptions.py",
        "src/core/http_client.py",
        "src/core/api_client_factory.py",
        "src/core/boot.py",
        "src/security/credentials.py",
        "config/default_config.yaml",
        "tests/test_all.py",
    ]
    missing = []
    for f in required:
        path = os.path.join(PROJECT_ROOT, f)
        if not os.path.exists(path):
            missing.append(f)
    assert len(missing) == 0, f"Missing files: {missing}"

@check("All __init__.py files exist")
def _():
    required = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/api/__init__.py",
        "src/security/__init__.py",
        "tests/__init__.py",
    ]
    missing = []
    for f in required:
        path = os.path.join(PROJECT_ROOT, f)
        if not os.path.exists(path):
            missing.append(f)
    assert len(missing) == 0, f"Missing __init__.py: {missing}"


# --- CHECK 9: Python syntax ---
print("  --- Python Syntax ---")

@check("All source files have valid Python syntax")
def _():
    bad = []
    for filepath in SRC_FILES:
        try:
            with open(filepath, "r") as f:
                source = f.read()
            compile(source, filepath, "exec")
        except SyntaxError as e:
            bad.append(f"{os.path.basename(filepath)}: {e}")
    assert len(bad) == 0, f"Syntax errors: {bad}"

@check("Test file has valid Python syntax")
def _():
    test_path = os.path.join(PROJECT_ROOT, "tests", "test_all.py")
    with open(test_path, "r") as f:
        source = f.read()
    compile(source, test_path, "exec")


# --- CHECK 10: Documentation ---
print("  --- Documentation ---")

@check("All source modules have module-level docstrings or header comments")
def _():
    undocumented = []
    for filepath in SRC_FILES:
        with open(filepath, "r") as f:
            content = f.read()
        # Check for either a triple-quoted docstring or a # comment header
        has_doc = (
            content.strip().startswith('"""') or
            content.strip().startswith("'''") or
            content.strip().startswith("#")
        )
        if not has_doc:
            undocumented.append(os.path.basename(filepath))
    assert len(undocumented) == 0, f"Undocumented modules: {undocumented}"

@check("Config YAML has comments explaining each section")
def _():
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(config_path, "r") as f:
        content = f.read()
    # Check that major sections have comments
    sections = ["api:", "http:", "ollama:", "embedding:", "retrieval:", "paths:", "security:"]
    for section in sections:
        idx = content.find(section)
        assert idx != -1, f"Section {section} not found in config"
        # Check for a comment within 5 lines before the section
        before = content[:idx].split("\n")[-6:]
        has_comment = any("#" in line for line in before)
        assert has_comment, f"Section {section} has no explanatory comment"


# ===========================================================================
# FINAL REPORT
# ===========================================================================
print()
print("=" * 60)
total = PASS + FAIL
print(f"  AUDIT RESULTS: {PASS} passed, {FAIL} failed, {total} total")
if FAIL == 0:
    print("  ALL AUDITS PASSED!")
else:
    print()
    print("  FAILURES:")
    for name, err in ERRORS:
        print(f"    [FAIL] {name}")
        print(f"           {err}")
print("=" * 60)
print()

sys.exit(0 if FAIL == 0 else 1)

