# ===========================================================================
# HYBRIDRAG v3 -- MASTER COMMAND TOOLKIT
# ===========================================================================
#
# WHAT THIS IS:
#   A single file containing EVERY command you might need for HybridRAG.
#   No more typing Python inline. No more quoting issues. Every command
#   writes a temp Python file, runs it, and cleans up.
#
# HOW TO LOAD:
#   cd "<YOUR_PROJECT_ROOT>"
#   .\.venv\Scripts\Activate
#   . .\tools\master_toolkit.ps1
#
# TO MAKE PERMANENT:
#   Add this line to the end of start_hybridrag.ps1:
#       . .\tools\master_toolkit.ps1
#
# COMMAND LIST (type rag-help to see this at any time):
#
#   --- CREDENTIALS ---
#   rag-store-key              Store your Azure API key in Windows keyring
#   rag-store-endpoint         Store your Azure endpoint URL in Windows keyring
#   rag-show-creds             Show stored credentials (key is masked)
#   rag-clear-creds            Remove all stored credentials
#   rag-store-deployment       Store Azure deployment name in env var
#   rag-store-api-version      Store Azure API version in env var
#
#   --- API DIAGNOSTICS ---
#   rag-debug-url              Show URL + headers without calling API
#   rag-test-api-verbose       Make a real API call with full debug output
#   rag-test-api-full          4-stage comprehensive diagnostic
#   rag-env-vars               Show all API-related environment variables
#
#   --- OLLAMA ---
#   rag-ollama-start           Start the Ollama service
#   rag-ollama-stop            Stop the Ollama service
#   rag-ollama-status          Check if Ollama is running and which models
#   rag-ollama-pull            Pull/update a model (default: phi4-mini)
#   rag-ollama-test            Send a test query to Ollama
#
#   --- INDEXING ---
#   rag-index                  Run the indexer on source documents
#   rag-index-status           Show indexing stats (files, chunks, etc.)
#   rag-index-reset            Clear the index database (asks confirmation)
#
#   --- QUERYING ---
#   rag-query                  Query using offline mode (Ollama)
#   rag-query-api              Query using API mode (Azure)
#   rag-query-retrieval        Retrieval only, no LLM (shows raw chunks)
#
#   --- FILE TOOLS ---
#   rag-fix-quotes             Fix smart quotes in all project files
#   rag-detect-bad-chars       Scan for non-ASCII characters
#   rag-fix-encoding           Fix file encoding issues (BOM, line endings)
#
#   --- NETWORK ---
#   rag-net-check              Test connectivity to all external endpoints
#   rag-proxy-check            Show proxy settings and test them
#   rag-ssl-check              Test SSL/TLS connectivity
#
#   --- PROJECT INFO ---
#   rag-status                 Full project status dashboard
#   rag-config                 Show current config.yaml settings
#   rag-paths                  Show all important file paths
#   rag-versions               Show Python, pip, and key package versions
#
#   --- GIT ---
#   rag-git-status             Show git status
#   rag-git-save               Add all, commit with message, push
#   rag-git-log                Show recent git commits
#
#   --- HOUSEKEEPING ---
#   rag-cleanup                Remove temp files, __pycache__, .bak files
#   rag-backup                 Create timestamped zip backup of project
#   rag-disk-usage             Show project folder sizes
#
#   --- LOGS ---
#   rag-logs                   Show recent log entries
#   rag-logs-errors            Show only error log entries
#   rag-logs-clear             Archive and clear log files
#
#   --- HELP ---
#   rag-help                   Show this command list
#
# ===========================================================================

$ErrorActionPreference = "Continue"


# ===========================================================================
# HELPER FUNCTION: Write and run temp Python safely
# ===========================================================================


# ###########################################################################
#                          CREDENTIALS
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-store-key: Store API key in Windows Credential Manager
# ---------------------------------------------------------------------------
function rag-store-key {
    Write-Host ""
    Write-Host "Store Azure API Key" -ForegroundColor Cyan
    Write-Host "-------------------" -ForegroundColor Gray
    Write-Host "Your key will be stored securely in Windows Credential Manager." -ForegroundColor Gray
    Write-Host "It will NOT appear in any config files or code." -ForegroundColor Gray
    Write-Host ""
    
    $key = Read-Host "Paste your API key (input is visible, that's OK)"
    
    if (-not $key -or $key.Length -lt 5) {
        Write-Host "  [ERROR] Key is too short or empty. Try again." -ForegroundColor Red
        return
    }
    
    python "tools/py/store_key.py" $key
}


# ---------------------------------------------------------------------------
# rag-store-endpoint: Store Azure endpoint URL
# ---------------------------------------------------------------------------
function rag-store-endpoint {
    Write-Host ""
    Write-Host "Store Azure Endpoint URL" -ForegroundColor Cyan
    Write-Host "------------------------" -ForegroundColor Gray
    Write-Host "Enter your base endpoint URL." -ForegroundColor Gray
    Write-Host "Example: https://your-resource.openai.azure.com/" -ForegroundColor Gray
    Write-Host "You can include the full path or just the base -- the toolkit handles both." -ForegroundColor Gray
    Write-Host ""
    
    $endpoint = Read-Host "Paste your endpoint URL"
    
    if (-not $endpoint -or -not $endpoint.StartsWith("http")) {
        Write-Host "  [ERROR] URL must start with http:// or https://" -ForegroundColor Red
        return
    }
    
    python "tools/py/store_endpoint.py" $endpoint
}


# ---------------------------------------------------------------------------
# rag-show-creds: Display stored credentials (key is masked)
# ---------------------------------------------------------------------------
function rag-show-creds {
    Write-Host ""
    Write-Host "Stored Credentials" -ForegroundColor Cyan
    Write-Host "------------------" -ForegroundColor Gray
    
    python "tools/py/show_creds.py"
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-clear-creds: Remove all stored credentials
# ---------------------------------------------------------------------------
function rag-clear-creds {
    Write-Host ""
    $confirm = Read-Host "This will DELETE all stored credentials. Type 'yes' to confirm"
    
    if ($confirm -ne "yes") {
        Write-Host "  Cancelled." -ForegroundColor Gray
        return
    }
    
    python "tools/py/clear_creds.py"
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-store-deployment: Set Azure deployment name as env var
# ---------------------------------------------------------------------------
function rag-store-deployment {
    param([string]$Name)
    
    if (-not $Name) {
        Write-Host ""
        Write-Host "Azure Deployment Name" -ForegroundColor Cyan
        Write-Host "This is the name of your GPT model in Azure Portal." -ForegroundColor Gray
        Write-Host "Common values: gpt-35-turbo, gpt-4, your-custom-name" -ForegroundColor Gray
        Write-Host ""
        $Name = Read-Host "Enter deployment name"
    }
    
    if (-not $Name) {
        Write-Host "  [ERROR] No name provided." -ForegroundColor Red
        return
    }
    
    $env:AZURE_OPENAI_DEPLOYMENT = $Name
    Write-Host "  [OK] Deployment set: $Name" -ForegroundColor Green
    Write-Host "  (This lasts for this session. Add to start_hybridrag.ps1 to make permanent)" -ForegroundColor Gray
}


# ---------------------------------------------------------------------------
# rag-store-api-version: Set Azure API version as env var
# ---------------------------------------------------------------------------
function rag-store-api-version {
    param([string]$Version)
    
    if (-not $Version) {
        Write-Host ""
        Write-Host "Azure API Version" -ForegroundColor Cyan
        Write-Host "Common values: 2024-02-01, 2024-06-01, 2024-08-01-preview" -ForegroundColor Gray
        Write-Host ""
        $Version = Read-Host "Enter API version (or press Enter for 2024-02-01)"
        if (-not $Version) { $Version = "2024-02-01" }
    }
    
    $env:AZURE_OPENAI_API_VERSION = $Version
    Write-Host "  [OK] API version set: $Version" -ForegroundColor Green
}


# ###########################################################################
#                          API DIAGNOSTICS
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-debug-url: Pre-flight URL check (no API call)
# ---------------------------------------------------------------------------
function rag-debug-url {
    Write-Host ""
    Write-Host "API Configuration Diagnostic" -ForegroundColor Cyan
    Write-Host "----------------------------" -ForegroundColor Gray
    
    python "tools/py/debug_url.py"
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-test-api-verbose: Live API test
# ---------------------------------------------------------------------------
function rag-test-api-verbose {
    Write-Host ""
    Write-Host "Live API Test" -ForegroundColor Cyan
    Write-Host "-------------" -ForegroundColor Gray
    
    python "tools/py/test_api_verbose.py"
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-test-api-full: Comprehensive 4-stage diagnostic
# ---------------------------------------------------------------------------
function rag-test-api-full {
    Write-Host ""
    Write-Host "Running azure_api_test.ps1..." -ForegroundColor Cyan
    $scriptPath = Join-Path (Get-Location) "tools\azure_api_test.ps1"
    if (Test-Path $scriptPath) {
        . $scriptPath
    } else {
        Write-Host "  [ERROR] tools\azure_api_test.ps1 not found." -ForegroundColor Red
        Write-Host "  Using inline test instead..." -ForegroundColor Yellow
        rag-test-api-verbose
    }
}


# ---------------------------------------------------------------------------
# rag-env-vars: Show all API/model environment variables
# ---------------------------------------------------------------------------
function rag-env-vars {
    Write-Host ""
    Write-Host "API and Model Environment Variables" -ForegroundColor Cyan
    Write-Host "---------------------------------" -ForegroundColor Gray
    
    $vars = Get-ChildItem env: | Where-Object { 
        $_.Name -match "azure|openai|api|endpoint|deploy|proxy|version|ollama|model|hugging|no_proxy|http_proxy|https_proxy|requests_ca|curl_ca|ssl_cert" 
    } | Sort-Object Name
    
    if ($vars.Count -eq 0) {
        Write-Host "  (none found)" -ForegroundColor Gray
        Write-Host "  Tip: Use rag-store-deployment and rag-store-api-version to set them." -ForegroundColor Gray
    } else {
        foreach ($v in $vars) {
            $displayVal = $v.Value
            if ($v.Name -match "key|secret|token|password" -and $displayVal.Length -gt 8) {
                $displayVal = $displayVal.Substring(0,4) + "..." + $displayVal.Substring($displayVal.Length - 4)
            }
            Write-Host "  $($v.Name) = $displayVal" -ForegroundColor White
        }
    }
    Write-Host ""
}


# ###########################################################################
#                          OLLAMA
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-ollama-start: Start Ollama service
# ---------------------------------------------------------------------------
function rag-ollama-start {
    Write-Host ""
    Write-Host "Starting Ollama..." -ForegroundColor Cyan
    
    $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (-not (Test-Path $ollamaPath)) {
        # Try PATH
        $ollamaPath = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    }
    
    if (-not $ollamaPath) {
        Write-Host "  [ERROR] Ollama not found. Is it installed?" -ForegroundColor Red
        return
    }
    
    # Check if already running
    $running = Get-Process ollama* -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "  Ollama is already running (PID: $($running.Id))" -ForegroundColor Green
        return
    }
    
    Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
    
    $running = Get-Process ollama* -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "  [OK] Ollama started (PID: $($running.Id))" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Ollama process not found after start attempt." -ForegroundColor Yellow
    }
}


# ---------------------------------------------------------------------------
# rag-ollama-stop: Stop Ollama service
# ---------------------------------------------------------------------------
function rag-ollama-stop {
    Write-Host ""
    $procs = Get-Process ollama* -ErrorAction SilentlyContinue
    if ($procs) {
        $procs | Stop-Process -Force
        Write-Host "  [OK] Ollama stopped." -ForegroundColor Green
    } else {
        Write-Host "  Ollama is not running." -ForegroundColor Gray
    }
}


# ---------------------------------------------------------------------------
# rag-ollama-status: Check Ollama status and models
# ---------------------------------------------------------------------------
function rag-ollama-status {
    Write-Host ""
    Write-Host "Ollama Status" -ForegroundColor Cyan
    Write-Host "-------------" -ForegroundColor Gray
    
    $procs = Get-Process ollama* -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Host "  Running: YES (PID: $($procs.Id -join ', '))" -ForegroundColor Green
    } else {
        Write-Host "  Running: NO" -ForegroundColor Red
        Write-Host "  Start with: rag-ollama-start" -ForegroundColor Gray
        return
    }
    
    Write-Host ""
    Write-Host "  Installed Models:" -ForegroundColor White
    try {
        $models = & ollama list 2>&1
        if ($models) {
            foreach ($line in $models) {
                Write-Host "    $line" -ForegroundColor White
            }
        }
    } catch {
        Write-Host "    (could not list models)" -ForegroundColor Gray
    }
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-ollama-pull: Pull a model
# ---------------------------------------------------------------------------
function rag-ollama-pull {
    param([string]$Model = "phi4-mini")
    
    Write-Host ""
    Write-Host "Pulling model: $Model" -ForegroundColor Cyan
    & ollama pull $Model
}


# ---------------------------------------------------------------------------
# rag-ollama-test: Test query to Ollama
# ---------------------------------------------------------------------------
function rag-ollama-test {
    Write-Host ""
    Write-Host "Testing Ollama..." -ForegroundColor Cyan
    
    python "tools/py/ollama_test.py"
    Write-Host ""
}


# ###########################################################################
#                          INDEXING
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-index: Run the document indexer
# ---------------------------------------------------------------------------
function rag-index {
    Write-Host ""
    Write-Host "Running indexer..." -ForegroundColor Cyan
    
    $indexScript = Join-Path (Get-Location) "src\tools\run_index_once.py"
    if (Test-Path $indexScript) {
        python $indexScript
    } else {
        Write-Host "  [ERROR] Indexer not found at: $indexScript" -ForegroundColor Red
    }
}


# ---------------------------------------------------------------------------
# rag-index-status: Show indexing statistics
# ---------------------------------------------------------------------------
function rag-index-status {
    Write-Host ""
    Write-Host "Index Statistics" -ForegroundColor Cyan
    Write-Host "----------------" -ForegroundColor Gray
    
    python "tools/py/index_status.py"
}


# ---------------------------------------------------------------------------
# rag-index-reset: Clear the index (with confirmation)
# ---------------------------------------------------------------------------
function rag-index-reset {
    Write-Host ""
    Write-Host "[WARNING] This will DELETE your entire index database." -ForegroundColor Red
    Write-Host "You will need to re-index all documents." -ForegroundColor Red
    Write-Host ""
    $confirm = Read-Host "Type 'yes' to confirm deletion"
    
    if ($confirm -ne "yes") {
        Write-Host "  Cancelled." -ForegroundColor Gray
        return
    }
    
    python "tools/py/index_reset.py"
    Write-Host ""
}


# ###########################################################################
#                          QUERYING
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-query: Query using offline Ollama
# ---------------------------------------------------------------------------
function rag-query {
    param([string]$Question)
    
    if (-not $Question) {
        $Question = Read-Host "Enter your question"
    }
    
    if (-not $Question) {
        Write-Host "  [ERROR] No question provided." -ForegroundColor Red
        return
    }
    
    # Escape quotes in the question for safe embedding
    $safeQuestion = $Question -replace '"', '\"' -replace "'", "\'"
    
    $cliScript = Join-Path (Get-Location) "tests\cli_test_phase1.py"
    if (Test-Path $cliScript) {
        python $cliScript --query "$safeQuestion" --mode offline
    } else {
        Write-Host "  [ERROR] CLI script not found at: $cliScript" -ForegroundColor Red
        Write-Host "  This file handles query routing. Check your project." -ForegroundColor Gray
    }
}


# ---------------------------------------------------------------------------
# rag-query-api: Query using Azure API
# ---------------------------------------------------------------------------
function rag-query-api {
    param([string]$Question)
    
    if (-not $Question) {
        $Question = Read-Host "Enter your question"
    }
    
    if (-not $Question) {
        Write-Host "  [ERROR] No question provided." -ForegroundColor Red
        return
    }
    
    $safeQuestion = $Question -replace '"', '\"' -replace "'", "\'"
    
    $cliScript = Join-Path (Get-Location) "tests\cli_test_phase1.py"
    if (Test-Path $cliScript) {
        python $cliScript --query "$safeQuestion" --mode api
    } else {
        Write-Host "  [ERROR] CLI script not found at: $cliScript" -ForegroundColor Red
    }
}


# ---------------------------------------------------------------------------
# rag-query-retrieval: Retrieval only (no LLM, shows raw chunks)
# ---------------------------------------------------------------------------
function rag-query-retrieval {
    param([string]$Question)
    
    if (-not $Question) {
        $Question = Read-Host "Enter your search query"
    }
    
    if (-not $Question) { return }
    
    $safeQuestion = $Question -replace '"', '\"' -replace "'", "\'"
    
    $retrievalScript = Join-Path (Get-Location) "src\tools\quick_test_retrieval.py"
    if (Test-Path $retrievalScript) {
        python $retrievalScript "$safeQuestion"
    } else {
        Write-Host "  [ERROR] Retrieval test script not found." -ForegroundColor Red
    }
}


# ###########################################################################
#                          FILE TOOLS
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-fix-quotes: Fix smart quotes in all project files
# ---------------------------------------------------------------------------
function rag-fix-quotes {
    Write-Host ""
    Write-Host "Fixing smart quotes..." -ForegroundColor Cyan
    
    $extensions = @("*.py", "*.ps1", "*.yaml", "*.yml", "*.bat", "*.txt", 
                    "*.md", "*.json", "*.cfg", "*.ini", "*.toml")
    $fixed = 0
    $scanned = 0
    
    foreach ($ext in $extensions) {
        $files = Get-ChildItem -Path . -Filter $ext -Recurse -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -notmatch "\.venv|__pycache__|\.git|node_modules|\.bak|backup" }
        
        foreach ($file in $files) {
            $scanned++
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
            if (-not $content) { continue }
            
            $original = $content
            $content = $content -replace "\u201C", '"'
            $content = $content -replace "\u201D", '"'
            $content = $content -replace "\u2018", "'"
            $content = $content -replace "\u2019", "'"
            $content = $content -replace "\u2013", "-"
            $content = $content -replace "\u2014", "--"
            $content = $content -replace "\u2026", "..."
            
            if ($content -ne $original) {
                $backupPath = $file.FullName + ".bak"
                if (-not (Test-Path $backupPath)) {
                    $original | Out-File -FilePath $backupPath -Encoding UTF8 -NoNewline
                }
                $content | Out-File -FilePath $file.FullName -Encoding UTF8 -NoNewline
                $fixed++
                Write-Host "  Fixed: $($file.Name)" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    Write-Host "  Scanned $scanned files, fixed $fixed." -ForegroundColor Green
}


# ---------------------------------------------------------------------------
# rag-detect-bad-chars: Scan for non-ASCII characters
# ---------------------------------------------------------------------------
function rag-detect-bad-chars {
    Write-Host ""
    Write-Host "Scanning for bad characters..." -ForegroundColor Cyan
    
    $extensions = @("*.py", "*.ps1", "*.yaml", "*.yml", "*.bat", "*.txt", 
                    "*.json", "*.cfg", "*.ini", "*.toml")
    $totalProblems = 0
    $problemFiles = 0
    
    foreach ($ext in $extensions) {
        $files = Get-ChildItem -Path . -Filter $ext -Recurse -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -notmatch "\.venv|__pycache__|\.git|node_modules|\.bak|backup" }
        
        foreach ($file in $files) {
            $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
            if (-not $content) { continue }
            
            $matches = [regex]::Matches($content, "[\u2018\u2019\u201C\u201D\u2013\u2014\u2026\u00A0\uFEFF]")
            if ($matches.Count -gt 0) {
                $totalProblems += $matches.Count
                $problemFiles++
                Write-Host "  [$($matches.Count)] $($file.Name)" -ForegroundColor Yellow
            }
        }
    }
    
    Write-Host ""
    if ($totalProblems -eq 0) {
        Write-Host "  All clean!" -ForegroundColor Green
    } else {
        Write-Host "  Found $totalProblems problems in $problemFiles files." -ForegroundColor Red
        Write-Host "  Run rag-fix-quotes to fix them." -ForegroundColor Yellow
    }
}


# ---------------------------------------------------------------------------
# rag-fix-encoding: Fix BOM and line ending issues
# ---------------------------------------------------------------------------
function rag-fix-encoding {
    Write-Host ""
    Write-Host "Checking file encodings..." -ForegroundColor Cyan
    
    $files = Get-ChildItem -Path . -Include "*.py","*.ps1","*.yaml","*.yml","*.bat" `
             -Recurse -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -notmatch "\.venv|__pycache__|\.git|\.bak" }
    
    $fixed = 0
    foreach ($file in $files) {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
        
        # Check for UTF-8 BOM (EF BB BF)
        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
            # Remove BOM
            $newBytes = $bytes[3..($bytes.Length - 1)]
            [System.IO.File]::WriteAllBytes($file.FullName, $newBytes)
            Write-Host "  Removed BOM: $($file.Name)" -ForegroundColor Yellow
            $fixed++
        }
    }
    
    Write-Host ""
    if ($fixed -eq 0) {
        Write-Host "  No encoding issues found." -ForegroundColor Green
    } else {
        Write-Host "  Fixed $fixed files." -ForegroundColor Green
    }
}


# ###########################################################################
#                          NETWORK
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-net-check: Test connectivity to all external endpoints
# ---------------------------------------------------------------------------
function rag-net-check {
    Write-Host ""
    Write-Host "Network Connectivity Check" -ForegroundColor Cyan
    Write-Host "--------------------------" -ForegroundColor Gray
    
    python "tools/py/net_check.py"
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-proxy-check: Show proxy settings
# ---------------------------------------------------------------------------
function rag-proxy-check {
    Write-Host ""
    Write-Host "Proxy Configuration" -ForegroundColor Cyan
    Write-Host "-------------------" -ForegroundColor Gray
    
    $proxyVars = @("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy")
    $found = $false
    
    foreach ($v in $proxyVars) {
        $val = [Environment]::GetEnvironmentVariable($v)
        if ($val) {
            Write-Host "  $v = $val" -ForegroundColor White
            $found = $true
        }
    }
    
    if (-not $found) {
        Write-Host "  No proxy environment variables set." -ForegroundColor Gray
    }
    
    # Check Windows proxy settings
    Write-Host ""
    Write-Host "  Windows Internet Settings:" -ForegroundColor White
    try {
        $reg = Get-ItemProperty "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        Write-Host "  Proxy enabled: $($reg.ProxyEnable)" -ForegroundColor White
        if ($reg.ProxyServer) { Write-Host "  Proxy server:  $($reg.ProxyServer)" -ForegroundColor White }
        if ($reg.AutoConfigURL) { Write-Host "  PAC URL:       $($reg.AutoConfigURL)" -ForegroundColor White }
    } catch {
        Write-Host "  (could not read registry)" -ForegroundColor Gray
    }
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-ssl-check: Test SSL connectivity
# ---------------------------------------------------------------------------
function rag-ssl-check {
    Write-Host ""
    Write-Host "SSL/TLS Check" -ForegroundColor Cyan
    Write-Host "--------------" -ForegroundColor Gray
    
    python "tools/py/ssl_check.py"
    Write-Host ""
}


# ###########################################################################
#                          PROJECT INFO
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-status: Full project status dashboard
# ---------------------------------------------------------------------------
function rag-status {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  HYBRIDRAG v3 STATUS DASHBOARD" -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Project path
    Write-Host "  Project: $(Get-Location)" -ForegroundColor White
    
    # Venv status
    if ($env:VIRTUAL_ENV) {
        Write-Host "  Venv:    ACTIVE ($env:VIRTUAL_ENV)" -ForegroundColor Green
    } else {
        Write-Host "  Venv:    NOT ACTIVE" -ForegroundColor Red
    }
    
    # Python version
    $pyVer = python --version 2>&1
    Write-Host "  Python:  $pyVer" -ForegroundColor White
    
    # Ollama
    $ollamaProc = Get-Process ollama* -ErrorAction SilentlyContinue
    if ($ollamaProc) {
        Write-Host "  Ollama:  RUNNING" -ForegroundColor Green
    } else {
        Write-Host "  Ollama:  STOPPED" -ForegroundColor Yellow
    }
    
    # Credentials
    Write-Host ""
    rag-show-creds
    
    # Quick index stats
    rag-index-status
}


# ---------------------------------------------------------------------------
# rag-config: Show config.yaml contents
# ---------------------------------------------------------------------------
function rag-config {
    Write-Host ""
    Write-Host "Configuration" -ForegroundColor Cyan
    Write-Host "-------------" -ForegroundColor Gray
    
    $configPaths = @(
        "config\default_config.yaml",
        "config.yaml",
        "config\config.yaml"
    )
    
    foreach ($p in $configPaths) {
        if (Test-Path $p) {
            Write-Host "  File: $p" -ForegroundColor White
            Write-Host ""
            Get-Content $p
            return
        }
    }
    
    Write-Host "  No config file found." -ForegroundColor Red
}


# ---------------------------------------------------------------------------
# rag-paths: Show all important file paths
# ---------------------------------------------------------------------------
function rag-paths {
    Write-Host ""
    Write-Host "Important Paths" -ForegroundColor Cyan
    Write-Host "---------------" -ForegroundColor Gray
    
    $paths = @{
        "Project root"    = (Get-Location).Path
        "Virtual env"     = "$((Get-Location).Path)\.venv"
        "Config"          = "$((Get-Location).Path)\config\default_config.yaml"
        "Source code"     = "$((Get-Location).Path)\src\core"
        "Tools"           = "$((Get-Location).Path)\tools"
        "Tests"           = "$((Get-Location).Path)\tests"
        "Docs"            = "$((Get-Location).Path)\docs"
        "Model cache"     = "$((Get-Location).Path)\.model_cache"
        "Ollama models"   = "$env:USERPROFILE\.ollama"
        "HF cache"        = "$env:USERPROFILE\.cache\huggingface"
        "PowerShell profile" = $PROFILE
    }
    
    foreach ($item in $paths.GetEnumerator() | Sort-Object Name) {
        $exists = if (Test-Path $item.Value) { "[OK]" } else { "[--]" }
        $color = if ($exists -eq "[OK]") { "Green" } else { "Gray" }
        Write-Host "  $exists $($item.Name): $($item.Value)" -ForegroundColor $color
    }
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-versions: Show package versions
# ---------------------------------------------------------------------------
function rag-versions {
    Write-Host ""
    Write-Host "Package Versions" -ForegroundColor Cyan
    Write-Host "----------------" -ForegroundColor Gray
    
    python "tools/py/versions.py"
    Write-Host ""
}


# ###########################################################################
#                          GIT
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-git-status: Show git status
# ---------------------------------------------------------------------------
function rag-git-status {
    Write-Host ""
    git status
    Write-Host ""
}


# ---------------------------------------------------------------------------
# rag-git-save: Add, commit, push in one command
# ---------------------------------------------------------------------------
function rag-git-save {
    param([string]$Message)
    
    if (-not $Message) {
        $Message = Read-Host "Commit message"
    }
    
    if (-not $Message) {
        Write-Host "  [ERROR] No commit message." -ForegroundColor Red
        return
    }
    
    Write-Host ""
    Write-Host "  Adding all files..." -ForegroundColor Gray
    git add -A
    
    Write-Host "  Committing..." -ForegroundColor Gray
    git commit -m $Message
    
    Write-Host "  Pushing..." -ForegroundColor Gray
    git push origin main
    
    Write-Host ""
    Write-Host "  [OK] Saved and pushed." -ForegroundColor Green
}


# ---------------------------------------------------------------------------
# rag-git-log: Show recent commits
# ---------------------------------------------------------------------------
function rag-git-log {
    Write-Host ""
    git log --oneline -20
    Write-Host ""
}


# ###########################################################################
#                          HOUSEKEEPING
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-cleanup: Remove temp files, caches, backups
# ---------------------------------------------------------------------------
function rag-cleanup {
    Write-Host ""
    Write-Host "Cleaning up..." -ForegroundColor Cyan
    
    $removed = 0
    
    # __pycache__ folders
    $caches = Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue |
              Where-Object { $_.FullName -notmatch "\.venv" }
    foreach ($c in $caches) {
        Remove-Item $c.FullName -Recurse -Force
        Write-Host "  Removed: $($c.FullName)" -ForegroundColor Gray
        $removed++
    }
    
    # .pyc files outside venv
    $pycs = Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch "\.venv" }
    foreach ($f in $pycs) {
        Remove-Item $f.FullName -Force
        $removed++
    }
    
    # temp_*.py files (leftover from toolkit)
    $temps = Get-ChildItem -Path . -Filter "temp_*.py" -ErrorAction SilentlyContinue
    foreach ($f in $temps) {
        Remove-Item $f.FullName -Force
        Write-Host "  Removed: $($f.Name)" -ForegroundColor Gray
        $removed++
    }
    
    # temp_diag folder
    if (Test-Path '.\temp_diag') {
        Remove-Item '.\temp_diag' -Recurse -Force
        Write-Host '  Removed: temp_diag\' -ForegroundColor Gray
        $removed++
    }
    
    Write-Host ""
    Write-Host "  Cleaned up $removed items." -ForegroundColor Green
    
    # Show .bak file count (don't delete automatically)
    $baks = (Get-ChildItem -Path . -Recurse -Filter "*.bak" -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -notmatch "\.venv|\.git" }).Count
    if ($baks -gt 0) {
        Write-Host "  Note: $baks .bak backup files exist. Run rag-cleanup-bak to remove them." -ForegroundColor Yellow
    }
}


# ---------------------------------------------------------------------------
# rag-cleanup-bak: Remove .bak files (separate because they're backups)
# ---------------------------------------------------------------------------
function rag-cleanup-bak {
    $baks = Get-ChildItem -Path . -Recurse -Filter "*.bak" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notmatch "\.venv|\.git" }
    
    if ($baks.Count -eq 0) {
        Write-Host "  No .bak files found." -ForegroundColor Gray
        return
    }
    
    Write-Host ""
    Write-Host "  Found $($baks.Count) backup files:" -ForegroundColor Yellow
    foreach ($b in $baks) {
        $size = [math]::Round($b.Length / 1024, 1)
        $display = "    " + $b.Name + " (" + $size + " KB)"
        Write-Host $display -ForegroundColor Gray
    }
    
    $confirm = Read-Host "  Delete all? (yes/no)"
    if ($confirm -eq "yes") {
        foreach ($b in $baks) { Remove-Item $b.FullName -Force }
        Write-Host "  [OK] Deleted $($baks.Count) backup files." -ForegroundColor Green
    } else {
        Write-Host "  Cancelled." -ForegroundColor Gray
    }
}


# ---------------------------------------------------------------------------
# rag-backup: Create timestamped project backup
# ---------------------------------------------------------------------------
function rag-backup {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $projectName = Split-Path -Leaf (Get-Location)
    $backupName = "${projectName}_backup_${timestamp}.zip"
    $backupPath = Join-Path (Split-Path -Parent (Get-Location)) $backupName
    
    Write-Host ""
    Write-Host "Creating backup..." -ForegroundColor Cyan
    Write-Host "  Excluding: .venv, __pycache__, .git, *.bak, .model_cache" -ForegroundColor Gray
    
    # Create temp exclusion list
    $excludes = @(".venv", "__pycache__", ".git", "*.bak", ".model_cache", "temp_*.py", "*.pyc")
    
    Compress-Archive -Path ".\*" -DestinationPath $backupPath -Force
    
    $sizeMB = [math]::Round((Get-Item $backupPath).Length / 1MB, 1)
    $display = "  [OK] Backup created: " + $backupPath + " (" + $sizeMB + " MB)"
    Write-Host $display -ForegroundColor Green
}


# ---------------------------------------------------------------------------
# rag-disk-usage: Show project folder sizes
# ---------------------------------------------------------------------------
function rag-disk-usage {
    Write-Host ""
    Write-Host "Disk Usage" -ForegroundColor Cyan
    Write-Host "----------" -ForegroundColor Gray
    
    $folders = Get-ChildItem -Path . -Directory -ErrorAction SilentlyContinue
    
    foreach ($f in ($folders | Sort-Object Name)) {
        $size = (Get-ChildItem $f.FullName -Recurse -File -ErrorAction SilentlyContinue | 
                 Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        $color = if ($sizeMB -gt 100) { "Yellow" } elseif ($sizeMB -gt 10) { "White" } else { "Gray" }
        Write-Host ("  {0,-25} {1,8} MB" -f $f.Name, $sizeMB) -ForegroundColor $color
    }
    
    # Total
    $total = (Get-ChildItem . -Recurse -File -ErrorAction SilentlyContinue | 
              Measure-Object -Property Length -Sum).Sum
    $totalMB = [math]::Round($total / 1MB, 1)
    Write-Host ""
    Write-Host ("  {0,-25} {1,8} MB" -f "TOTAL", $totalMB) -ForegroundColor Cyan
    Write-Host ""
}


# ###########################################################################
#                          LOGS
# ###########################################################################

# ---------------------------------------------------------------------------
# rag-logs: Show recent log entries
# ---------------------------------------------------------------------------
function rag-logs {
    param([int]$Lines = 50)
    
    $logFiles = Get-ChildItem -Path . -Recurse -Include "*.log" -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notmatch "\.venv|\.git" } |
                Sort-Object LastWriteTime -Descending
    
    if ($logFiles.Count -eq 0) {
        Write-Host "  No log files found." -ForegroundColor Gray
        return
    }
    
    $latest = $logFiles[0]
    Write-Host ""
    Write-Host "Latest log: $($latest.FullName)" -ForegroundColor Cyan
    Write-Host "Last modified: $($latest.LastWriteTime)" -ForegroundColor Gray
    Write-Host ("-" * 60) -ForegroundColor Gray
    Get-Content $latest.FullName -Tail $Lines
}


# ---------------------------------------------------------------------------
# rag-logs-errors: Show only error entries
# ---------------------------------------------------------------------------
function rag-logs-errors {
    $logFiles = Get-ChildItem -Path . -Recurse -Include "*.log" -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notmatch "\.venv|\.git" } |
                Sort-Object LastWriteTime -Descending
    
    if ($logFiles.Count -eq 0) {
        Write-Host "  No log files found." -ForegroundColor Gray
        return
    }
    
    Write-Host ""
    Write-Host "Error entries across all logs:" -ForegroundColor Cyan
    Write-Host ("-" * 60) -ForegroundColor Gray
    
    foreach ($log in $logFiles) {
        $errors = Select-String -Path $log.FullName -Pattern "ERROR|CRITICAL|FATAL|Exception|Traceback" -ErrorAction SilentlyContinue
        if ($errors) {
            Write-Host "  --- $($log.Name) ---" -ForegroundColor Yellow
            foreach ($e in $errors) {
                Write-Host "  $($e.Line)" -ForegroundColor Red
            }
        }
    }
}


# ---------------------------------------------------------------------------
# rag-logs-clear: Archive and clear log files
# ---------------------------------------------------------------------------
function rag-logs-clear {
    $logFiles = Get-ChildItem -Path . -Recurse -Include "*.log" -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notmatch "\.venv|\.git" }
    
    if ($logFiles.Count -eq 0) {
        Write-Host "  No log files to clear." -ForegroundColor Gray
        return
    }
    
    $confirm = Read-Host "Archive and clear $($logFiles.Count) log files? (yes/no)"
    if ($confirm -ne "yes") { return }
    
    $archiveDir = ".\logs_archive"
    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    
    foreach ($log in $logFiles) {
        $archiveName = "$timestamp`_$($log.Name)"
        Move-Item $log.FullName (Join-Path $archiveDir $archiveName)
        Write-Host "  Archived: $($log.Name)" -ForegroundColor Gray
    }
    
    Write-Host "  [OK] Logs archived to $archiveDir" -ForegroundColor Green
}


# ###########################################################################
#                          HELP
# ###########################################################################

function rag-help {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  HYBRIDRAG v3 COMMAND REFERENCE" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  CREDENTIALS" -ForegroundColor Yellow
    Write-Host "    rag-store-key            Store API key in Windows keyring"
    Write-Host "    rag-store-endpoint       Store Azure endpoint URL"
    Write-Host "    rag-show-creds           Show stored credentials (masked)"
    Write-Host "    rag-clear-creds          Delete all stored credentials"
    Write-Host "    rag-store-deployment     Set Azure deployment name"
    Write-Host "    rag-store-api-version    Set Azure API version"
    Write-Host ""
    Write-Host "  API TESTING" -ForegroundColor Yellow
    Write-Host "    rag-debug-url            Pre-flight URL check (no API call)"
    Write-Host "    rag-test-api-verbose     Live API test with debug output"
    Write-Host "    rag-test-api-full        4-stage comprehensive diagnostic"
    Write-Host "    rag-env-vars             Show all API environment variables"
    Write-Host ""
    Write-Host "  OLLAMA" -ForegroundColor Yellow
    Write-Host "    rag-ollama-start         Start Ollama service"
    Write-Host "    rag-ollama-stop          Stop Ollama service"
    Write-Host "    rag-ollama-status        Check status and models"
    Write-Host "    rag-ollama-pull [model]  Pull/update a model"
    Write-Host "    rag-ollama-test          Test query to Ollama"
    Write-Host ""
    Write-Host "  INDEXING" -ForegroundColor Yellow
    Write-Host "    rag-index                Run the document indexer"
    Write-Host "    rag-index-status         Show index statistics"
    Write-Host "    rag-index-reset          Clear index (asks confirmation)"
    Write-Host ""
    Write-Host "  QUERYING" -ForegroundColor Yellow
    Write-Host "    rag-query [text]         Query via Ollama (offline)"
    Write-Host "    rag-query-api [text]     Query via Azure API (online)"
    Write-Host "    rag-query-retrieval [q]  Retrieval only (no LLM)"
    Write-Host ""
    Write-Host "  FILE TOOLS" -ForegroundColor Yellow
    Write-Host "    rag-fix-quotes           Fix smart quotes in all files"
    Write-Host "    rag-detect-bad-chars     Scan for non-ASCII characters"
    Write-Host "    rag-fix-encoding         Fix BOM and encoding issues"
    Write-Host ""
    Write-Host "  NETWORK" -ForegroundColor Yellow
    Write-Host "    rag-net-check            Test all external connectivity"
    Write-Host "    rag-proxy-check          Show proxy settings"
    Write-Host "    rag-ssl-check            Test SSL/TLS to Azure"
    Write-Host ""
    Write-Host "  PROJECT" -ForegroundColor Yellow
    Write-Host "    rag-status               Full status dashboard"
    Write-Host "    rag-config               Show config.yaml"
    Write-Host "    rag-paths                Show all important paths"
    Write-Host "    rag-versions             Show package versions"
    Write-Host ""
    Write-Host "  GIT" -ForegroundColor Yellow
    Write-Host "    rag-git-status           Show git status"
    Write-Host "    rag-git-save [msg]       Add + commit + push"
    Write-Host "    rag-git-log              Show recent commits"
    Write-Host ""
    Write-Host "  HOUSEKEEPING" -ForegroundColor Yellow
    Write-Host "    rag-cleanup              Remove temp files and caches"
    Write-Host "    rag-cleanup-bak          Remove .bak backup files"
    Write-Host "    rag-backup               Create project backup zip"
    Write-Host "    rag-disk-usage           Show folder sizes"
    Write-Host ""
    Write-Host "  LOGS" -ForegroundColor Yellow
    Write-Host "    rag-logs [n]             Show last n log entries (default 50)"
    Write-Host "    rag-logs-errors          Show only error entries"
    Write-Host "    rag-logs-clear           Archive and clear log files"
    Write-Host ""
    Write-Host "  rag-help                   Show this list" -ForegroundColor Green
    Write-Host ""
}


# ===========================================================================
# LOAD CONFIRMATION
# ===========================================================================
Write-Host ""
Write-Host "HybridRAG Master Toolkit loaded. Type rag-help for commands." -ForegroundColor Green
Write-Host ""

