# ============================================================================
# HybridRAG v3 -- Work Laptop Model Setup (setup_work_models.ps1)
# ============================================================================
# PURPOSE:
#   Install Python dependencies and pull all WORK_ONLY Ollama models.
#   Run this ONCE on the work laptop to prepare for offline validation.
#
# PREREQUISITES:
#   - Python 3.10+ installed and on PATH
#   - Ollama installed and running (ollama serve)
#   - Internet access (for pip install and model downloads)
#
# USAGE:
#   .\setup_work_models.ps1
#
# INTERNET ACCESS: YES (pip install + ollama pull)
# ============================================================================

$ErrorActionPreference = 'Stop'

function Write-Status {
    param([string]$Tag, [string]$Message)
    $timestamp = Get-Date -Format 'HH:mm:ss'
    Write-Host "[$timestamp] [$Tag] $Message"
}

# ---------------------------------------------------------------------------
# Step 1: Verify prerequisites
# ---------------------------------------------------------------------------

Write-Status 'INFO' 'Checking prerequisites...'

# Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Status 'OK' "Python found: $pyVersion"
} catch {
    Write-Status 'FAIL' 'Python not found on PATH. Install Python 3.10+ first.'
    exit 1
}

# Check Ollama
try {
    $ollamaVersion = ollama --version 2>&1
    Write-Status 'OK' "Ollama found: $ollamaVersion"
} catch {
    Write-Status 'FAIL' 'Ollama not found on PATH. Install from https://ollama.com'
    exit 1
}

# Check Ollama is running
try {
    $ollamaList = ollama list 2>&1
    Write-Status 'OK' 'Ollama service is running'
} catch {
    Write-Status 'WARN' 'Ollama may not be running. Starting ollama serve...'
    Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies
# ---------------------------------------------------------------------------

Write-Status 'INFO' 'Installing Python dependencies...'

$pipPackages = @(
    'sentence-transformers',
    'numpy',
    'keyring',
    'httpx'
)

foreach ($pkg in $pipPackages) {
    Write-Status 'INFO' "Installing $pkg..."
    pip install $pkg --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Status 'OK' "$pkg installed"
    } else {
        Write-Status 'WARN' "$pkg install may have issues (exit code: $LASTEXITCODE)"
    }
}

# ---------------------------------------------------------------------------
# Step 3: Pull WORK_ONLY Ollama models
# ---------------------------------------------------------------------------

Write-Status 'INFO' 'Pulling WORK_ONLY Ollama models...'

# Models verified against Ollama library 2026-02-21
$models = @(
    @{ Tag = 'phi4-mini';                   Size = '2.3 GB';  Note = 'Primary for 7/9 profiles (MIT, Microsoft/USA)' },
    @{ Tag = 'mistral:7b';                  Size = '4.1 GB';  Note = 'Alt for eng/sys/fe/cyber (Apache 2.0, Mistral/France)' },
    @{ Tag = 'phi4:14b-q4_K_M';            Size = '9.1 GB';  Note = 'Logistics primary, CAD alt (MIT, Microsoft/USA)' },
    @{ Tag = 'gemma3:4b';                   Size = '3.3 GB';  Note = 'PM fast summarization (Apache 2.0, Google/USA)' },
    @{ Tag = 'mistral-nemo:12b';            Size = '7.1 GB';  Note = 'Upgrade for sw/eng/sys/cyber/gen (Apache 2.0, Mistral+NVIDIA, 128K ctx)' }
)

$totalSize = 0
$successCount = 0

foreach ($model in $models) {
    Write-Status 'INFO' "Pulling $($model.Tag) (~$($model.Size))... $($model.Note)"
    ollama pull $model.Tag 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Status 'OK' "$($model.Tag) pulled successfully"
        $successCount++
    } else {
        Write-Status 'FAIL' "$($model.Tag) pull failed (exit code: $LASTEXITCODE)"
    }
}

Write-Status 'INFO' "Model pull complete: $successCount/$($models.Count) succeeded"

# ---------------------------------------------------------------------------
# Step 4: Verify all models are available
# ---------------------------------------------------------------------------

Write-Status 'INFO' 'Verifying installed models...'

$installedRaw = ollama list 2>&1
Write-Host ''
Write-Host '  Installed models:'
Write-Host '  -----------------'
Write-Host $installedRaw
Write-Host ''

foreach ($model in $models) {
    $tag = $model.Tag.Split(':')[0]
    if ($installedRaw -match $tag) {
        Write-Status 'OK' "$($model.Tag) is installed"
    } else {
        Write-Status 'FAIL' "$($model.Tag) NOT found in ollama list"
    }
}

# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------

Write-Host ''
Write-Host '  ============================================'
Write-Host '  Setup Complete'
Write-Host '  ============================================'
Write-Host "  Models pulled: $successCount / $($models.Count)"
Write-Host '  Estimated disk usage: ~26 GB for all 5 models'
Write-Host ''
Write-Host '  Next steps:'
Write-Host '    1. Run validate_offline_models.py to test each model'
Write-Host '    2. Run validate_online_api.py to test Azure API access'
Write-Host '  ============================================'
Write-Host ''

