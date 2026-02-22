# ============================================================================
# HybridRAG v3 - API Mode and Profile Commands (api_mode_commands.ps1)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Provides PowerShell functions for managing API mode and performance:
#     rag-store-key       Store your API key in Windows Credential Manager
#     rag-store-endpoint  Store your company API endpoint URL
#     rag-cred-status     Check what credentials are stored
#     rag-cred-delete     Remove stored credentials
#     rag-mode-online     Switch to online API mode
#     rag-mode-offline    Switch to offline AI mode (Ollama, Phi-4, Mistral, etc.)
#     rag-models          Show all available AI models (offline + online)
#     rag-test-api        Quick test that the API connection works
#     rag-profile         View/switch performance profile
#
# TECHNICAL NOTE:
#   All Python logic lives in the scripts/ folder as separate .py files.
#   PowerShell only calls "python scripts\_something.py" with no inline
#   Python code. This prevents here-string indentation bugs and keeps
#   PS functions clean.
#
# SECURITY NOTES:
#   API keys stored via Windows Credential Manager (DPAPI encrypted).
#   Keys tied to YOUR Windows login. Other users cannot read them.
#   Keys never appear in config files, logs, or git.
#   HuggingFace remains blocked in ALL modes.
#   Network Gate enforces which endpoints are reachable per mode.
#
# INTERNET ACCESS:
#   rag-store-key/endpoint: NO internet (local Credential Manager only)
#   rag-test-api: YES (makes one HTTP request to your API endpoint)
#   rag-mode-online: NO requests (just changes config.mode in YAML)
#   rag-mode-offline: NO internet (queries localhost Ollama only)
#   rag-models: NO internet (queries localhost Ollama only)
#
# CHANGE LOG:
#   2026-02-15: Removed HYBRIDRAG_NETWORK_KILL_SWITCH references
#   2026-02-16: Redesigned mode switching with model selection
#               Added rag-models command
#               Updated labels: "offline AI mode" instead of "Ollama mode"
# ============================================================================


# ============================================================================
# CREDENTIAL MANAGEMENT COMMANDS
# ============================================================================


function rag-store-key {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  HybridRAG v3 - Store API Key" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your key will be encrypted in Windows Credential Manager."
    Write-Host "It will NOT appear in any file, log, or config."
    Write-Host ""
    python -m src.security.credentials store
}


function rag-store-endpoint {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  HybridRAG v3 - Store API Endpoint" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Enter your company internal GPT API endpoint URL."
    Write-Host "Example: https://your-company.com/v1/chat/completions"
    Write-Host ""
    python -m src.security.credentials endpoint
}


function rag-cred-status {
    python -m src.security.credentials status
}


function rag-cred-delete {
    Write-Host ""
    $confirm = Read-Host "Delete stored API key and endpoint? (yes/no)"
    if ($confirm -eq "yes") {
        python -m src.security.credentials delete
        Write-Host "Credentials removed." -ForegroundColor Green
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
}


# ============================================================================
# MODE SWITCHING COMMANDS
# ============================================================================


function rag-mode-online {
    <#
    .SYNOPSIS
    Switch to online API mode.

    WHAT HAPPENS:
      1. Checks that API key and endpoint are configured
      2. Sets mode to "online" in config/default_config.yaml
      3. Shows current API model and endpoint
      4. Queries now route to cloud API

    CREDENTIALS:
      API key resolved via 3-layer system (keyring -> env -> config)
      managed by src/security/credentials.py. This function does NOT
      modify credentials -- it only checks they exist.
    #>
    Write-Host ""
    Write-Host "Checking credentials..." -ForegroundColor Cyan

    $status = python "$PROJECT_ROOT\scripts\_check_creds.py" 2>$null

    $hasKey = $status | Select-String "KEY:True"
    $hasEndpoint = $status | Select-String "ENDPOINT:True"

    if (-not $hasKey) {
        Write-Host "  ERROR: No API key found!" -ForegroundColor Red
        Write-Host "  Run: rag-store-key" -ForegroundColor Yellow
        Write-Host ""
        return
    }

    if (-not $hasEndpoint) {
        Write-Host "  WARNING: No custom endpoint set." -ForegroundColor Yellow
        Write-Host "  Using default: api.openai.com" -ForegroundColor Yellow
        Write-Host "  To set your company endpoint: rag-store-endpoint" -ForegroundColor Yellow
        Write-Host ""
    }

    python "$PROJECT_ROOT\scripts\_set_online.py"

    Write-Host ""
    Write-Host "  HF_HUB_OFFLINE:       $env:HF_HUB_OFFLINE (still locked)" -ForegroundColor Green
    Write-Host "  TRANSFORMERS_OFFLINE:  $env:TRANSFORMERS_OFFLINE (still locked)" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Network Gate status:" -ForegroundColor Green
    python -m src.tools.net_status
    Write-Host ""
    Write-Host "  Online mode active. Queries now route to cloud API." -ForegroundColor Cyan
    Write-Host "  To switch back: rag-mode-offline" -ForegroundColor DarkGray
    Write-Host ""
}


function rag-mode-offline {
    <#
    .SYNOPSIS
    Switch to offline AI mode (local models via Ollama).

    WHAT HAPPENS:
      1. Detects all installed local AI models (Phi-4, Mistral, etc.)
      2. Prompts you to choose which model to use
      3. Sets mode to "offline" in config/default_config.yaml
      4. Queries now route to your local model via Ollama

    MODELS ARE AUTO-DETECTED:
      Any model you've pulled with 'ollama pull' appears in the list.
      Add models:    ollama pull mistral
      Remove models: ollama rm phi4-mini
    #>
    python "$PROJECT_ROOT\scripts\_set_offline.py"

    Write-Host ""
    Write-Host "  Network Gate status:" -ForegroundColor Yellow
    python -m src.tools.net_status
    Write-Host ""
    Write-Host "  Make sure Ollama is running: ollama serve" -ForegroundColor DarkGray
    Write-Host ""
}


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================


function rag-models {
    <#
    .SYNOPSIS
    Show all available AI models (offline and online).

    DISPLAYS:
      - All installed offline models (auto-detected from Ollama)
      - Currently configured online API model
      - Which model is active based on current mode
      - How to add, remove, or switch models

    INTERNET ACCESS: NONE (Ollama query is localhost only)
    #>
    python "$PROJECT_ROOT\scripts\_list_models.py"
}


# ============================================================================
# API CONNECTIVITY TEST
# ============================================================================


function rag-test-api {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  HybridRAG v3 - API Connectivity Test" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Sending test prompt to API endpoint..." -ForegroundColor Cyan
    Write-Host ""

    python "$PROJECT_ROOT\scripts\_test_api.py"

    Write-Host ""
}


# ============================================================================
# PERFORMANCE PROFILE SWITCHING
# ============================================================================


function rag-profile {
    param(
        [Parameter(Position=0)]
        [ValidateSet("laptop_safe", "desktop_power", "server_max", "status")]
        [string]$Profile = "status"
    )

    if ($Profile -eq "status") {
        Write-Host ""
        Write-Host "Current performance profile:" -ForegroundColor Cyan

        python "$PROJECT_ROOT\scripts\_profile_status.py"

        Write-Host ""
        Write-Host "  Switch with: rag-profile laptop_safe" -ForegroundColor DarkGray
        Write-Host "               rag-profile desktop_power" -ForegroundColor DarkGray
        Write-Host "               rag-profile server_max" -ForegroundColor DarkGray
        Write-Host ""
        return
    }

    Write-Host ""
    Write-Host "Switching to profile: $Profile" -ForegroundColor Cyan

    python "$PROJECT_ROOT\scripts\_profile_switch.py" $Profile

    Write-Host ""
    Write-Host "  Profile applied. Re-index to use new batch settings." -ForegroundColor Green
    Write-Host ""
}


# ============================================================================
# MODEL SELECTION WIZARD
# ============================================================================

function rag-set-model {
    <#
    .SYNOPSIS
    3-step model selection wizard: Mode -> Objective -> Pick
    #>
    Write-Host ""
    Write-Host "Starting model selection wizard..." -ForegroundColor Cyan

    python "$PROJECT_ROOT\scripts\_set_model.py"
}


# ============================================================================
# STARTUP MESSAGE
# ============================================================================

Write-Host ""
Write-Host "API + Profile Commands loaded:" -ForegroundColor Cyan
Write-Host "  rag-set-model       Model selection wizard (recommended)" -ForegroundColor Yellow
Write-Host "  rag-store-key       Store API key (encrypted)" -ForegroundColor DarkGray
Write-Host "  rag-store-endpoint  Store custom API endpoint" -ForegroundColor DarkGray
Write-Host "  rag-cred-status     Check credential status" -ForegroundColor DarkGray
Write-Host "  rag-cred-delete     Remove stored credentials" -ForegroundColor DarkGray
Write-Host "  rag-mode-online     Switch to online API mode (direct)" -ForegroundColor DarkGray
Write-Host "  rag-mode-offline    Switch to offline AI mode (direct)" -ForegroundColor DarkGray
Write-Host "  rag-models          Show all available AI models" -ForegroundColor DarkGray
Write-Host "  rag-test-api        Test API connectivity" -ForegroundColor DarkGray
Write-Host "  rag-profile         View/switch performance profile" -ForegroundColor DarkGray
Write-Host ""
