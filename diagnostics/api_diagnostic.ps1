# HybridRAG Upstream API Path/Auth Diagnostic (PowerShell 5.1 compatible)
# Save as: diagnostics\api_diagnostic.ps1
# Run: powershell.exe -ExecutionPolicy Bypass -File ".\diagnostics\api_diagnostic.ps1"
#
# Outputs (in repo root):
#  - api_diag_report.txt
#  - api_diag_results.json

$ErrorActionPreference = "Continue"

# ====== CHANGE THIS ONE LINE ONLY (if needed) ======
$BaseHost = "https://aiml-aoai-api.gcl.mycompany.com"
# ===================================================

# Optional Azure hints (only used for Azure-style probes). Leave as env-driven.
$AzureDeployment = $env:HYBRIDRAG_AZURE_DEPLOYMENT
$AzureApiVersion = $env:HYBRIDRAG_AZURE_API_VERSION
if (-not $AzureApiVersion) { $AzureApiVersion = "2024-02-01" }

# Token discovery (env first)
$Token = $env:HYBRIDRAG_API_KEY
if (-not $Token) { $Token = $env:OPENAI_API_KEY }
if (-not $Token) { $Token = $env:AZURE_OPENAI_API_KEY }

# Prompt once if not found
if (-not $Token) {
  Write-Host ""
  Write-Host "No token found in env (HYBRIDRAG_API_KEY / OPENAI_API_KEY / AZURE_OPENAI_API_KEY)."
  $Token = Read-Host "Paste token now (or press Enter to run unauth-only probes)"
}

function Normalize-Base([string]$b) {
  if ($null -eq $b) { return "" }
  $b = $b.Trim()
  if ($b.EndsWith("/")) { $b = $b.Substring(0, $b.Length-1) }
  return $b
}

$BaseHost = Normalize-Base $BaseHost

# Candidate paths (models + chat)
$ModelPaths = @(
  "/v1/models",
  "/openai/v1/models",
  "/aoai/v1/models",
  "/api/v1/models",
  "/api/openai/v1/models",
  "/api/aoai/v1/models",
  "/openai/models",
  "/models"
)

$ChatPaths = @(
  "/v1/chat/completions",
  "/openai/v1/chat/completions",
  "/aoai/v1/chat/completions",
  "/api/openai/v1/chat/completions",
  "/api/aoai/v1/chat/completions",
  "/chat/completions",
  "/openai/chat/completions"
)

# Azure-style chat paths (only if deployment known)
$AzureChatPaths = @()
if ($AzureDeployment) {
  $AzureChatPaths += "/openai/deployments/$AzureDeployment/chat/completions?api-version=$AzureApiVersion"
  $AzureChatPaths += "/aoai/deployments/$AzureDeployment/chat/completions?api-version=$AzureApiVersion"
  $AzureChatPaths += "/api/openai/deployments/$AzureDeployment/chat/completions?api-version=$AzureApiVersion"
}

# Auth strategies to try
$AuthStrategies = @()
$AuthStrategies += @{ name="no_auth"; headers=@{} }

if ($Token) {
  $AuthStrategies += @{ name="bearer"; headers=@{ "Authorization"="Bearer $Token" } }
  $AuthStrategies += @{ name="api_key"; headers=@{ "api-key"="$Token" } }
  $AuthStrategies += @{ name="subscription_key"; headers=@{ "Ocp-Apim-Subscription-Key"="$Token" } }
}

function Invoke-Probe {
  param(
    [string]$Method,
    [string]$Url,
    [hashtable]$Headers,
    [string]$BodyJson
  )

  $result = New-Object PSObject -Property @{
    method     = $Method
    url        = $Url
    status     = $null
    statusText = $null
    activityId = $null
    snippet    = $null
    ok         = $false
    error      = $null
  }

  try {
    $hdrs = @{}
    foreach ($k in $Headers.Keys) { $hdrs[$k] = $Headers[$k] }
    if (-not $hdrs.ContainsKey("Accept")) { $hdrs["Accept"] = "application/json" }

    $resp = $null
    if ($BodyJson) {
      if (-not $hdrs.ContainsKey("Content-Type")) { $hdrs["Content-Type"] = "application/json" }
      $resp = Invoke-WebRequest -Method $Method -Uri $Url -Headers $hdrs -Body $BodyJson -UseBasicParsing -TimeoutSec 20
    } else {
      $resp = Invoke-WebRequest -Method $Method -Uri $Url -Headers $hdrs -UseBasicParsing -TimeoutSec 20
    }

    $result.status = [int]$resp.StatusCode
    $result.statusText = $resp.StatusDescription

    $txt = $resp.Content
    if ($txt) {
      if ($txt.Length -gt 400) { $txt = $txt.Substring(0,400) + "..." }
      $result.snippet = $txt
      if ($txt -match "(?i)activityId[^A-Za-z0-9\-]*([A-Za-z0-9\-]{8,})") { $result.activityId = $Matches[1] }
    }

    if ($result.status -ge 200 -and $result.status -lt 300) { $result.ok = $true }
  }
  catch {
    $ex = $_.Exception
    $result.error = $ex.Message

    try {
      $r = $ex.Response
      if ($r) {
        $result.status = [int]$r.StatusCode
        $result.statusText = $r.StatusDescription

        $stream = $r.GetResponseStream()
        if ($stream) {
          $reader = New-Object System.IO.StreamReader($stream)
          $txt = $reader.ReadToEnd()
          if ($txt) {
            if ($txt.Length -gt 400) { $txt = $txt.Substring(0,400) + "..." }
            $result.snippet = $txt
            if ($txt -match "(?i)activityId[^A-Za-z0-9\-]*([A-Za-z0-9\-]{8,})") { $result.activityId = $Matches[1] }
          }
        }
      }
    } catch {}
  }

  return $result
}

function Get-StatusScore([int]$s) {
  if ($s -ge 200 -and $s -lt 300) { return 100 }
  if ($s -eq 401 -or $s -eq 403) { return 90 }   # route exists; auth needed/wrong
  if ($s -eq 429) { return 80 }
  if ($s -eq 400 -or $s -eq 415) { return 70 }   # route exists; body/ctype mismatch
  if ($s -eq 404) { return 10 }
  if ($s -ge 500 -and $s -lt 600) { return 40 }  # gateway error; still informative
  return 20
}

# Build minimal chat body
$ChatBodyObj = @{
  model = "gpt-4o-mini"
  messages = @(@{ role="user"; content="OK" })
  max_tokens = 5
}
$ChatBody = $ChatBodyObj | ConvertTo-Json -Depth 6

$Results = @()

Write-Host ""
Write-Host "=== HybridRAG API Diagnostic (PS 5.1) ==="
Write-Host ("BaseHost: " + $BaseHost)
Write-Host ("Token present: " + [bool]$Token)
if ($AzureDeployment) {
  Write-Host ("AzureDeployment (env): " + $AzureDeployment)
  Write-Host ("AzureApiVersion: " + $AzureApiVersion)
}

foreach ($auth in $AuthStrategies) {
  $authName = $auth.name
  $hdrs = $auth.headers

  Write-Host ""
  Write-Host ("-- Auth strategy: " + $authName + " --")

  foreach ($p in $ModelPaths) {
    $url = "$BaseHost$p"
    Write-Host ("  Probing: GET $url") -ForegroundColor Gray
    $Results += (Invoke-Probe -Method "GET" -Url $url -Headers $hdrs -BodyJson $null)
  }

  foreach ($p in $ChatPaths) {
    $url = "$BaseHost$p"
    Write-Host ("  Probing: OPTIONS $url") -ForegroundColor Gray
    $Results += (Invoke-Probe -Method "OPTIONS" -Url $url -Headers $hdrs -BodyJson $null)
    Write-Host ("  Probing: POST $url") -ForegroundColor Gray
    $Results += (Invoke-Probe -Method "POST" -Url $url -Headers $hdrs -BodyJson $ChatBody)
  }

  foreach ($p in $AzureChatPaths) {
    $url = "$BaseHost$p"
    Write-Host ("  Probing: POST $url") -ForegroundColor Gray
    $Results += (Invoke-Probe -Method "POST" -Url $url -Headers $hdrs -BodyJson $ChatBody)
  }
}

# Rank results
$Ranked = $Results |
  Where-Object { $null -ne $_.status } |
  Select-Object method, url, status, statusText, activityId, snippet, error,
    @{n="score"; e={ Get-StatusScore $_.status }} |
  Sort-Object @{Expression="score";Descending=$true}, @{Expression="status";Ascending=$true}

$Top = $Ranked | Select-Object -First 25
$best = $Ranked | Select-Object -First 1

$txtPath = Join-Path (Get-Location) "api_diag_report.txt"
$jsonPath = Join-Path (Get-Location) "api_diag_results.json"

$sb = New-Object System.Text.StringBuilder
$null = $sb.AppendLine("=== HybridRAG API Diagnostic Report (PS 5.1) ===")
$null = $sb.AppendLine("Timestamp: " + (Get-Date).ToString("s"))
$null = $sb.AppendLine("BaseHost: " + $BaseHost)
$null = $sb.AppendLine("Token present: " + [bool]$Token)

$azDepOut = "<none>"
if ($AzureDeployment) { $azDepOut = $AzureDeployment }

$null = $sb.AppendLine("AzureDeployment: " + $azDepOut)
$null = $sb.AppendLine("AzureApiVersion: " + $AzureApiVersion)
$null = $sb.AppendLine("")
$null = $sb.AppendLine("Top candidate endpoints (highest likelihood first):")
$null = $sb.AppendLine("--------------------------------------------------")

foreach ($r in $Top) {
  $null = $sb.AppendLine(("{0} {1}  =>  {2} {3}" -f $r.method, $r.url, $r.status, $r.statusText))
  if ($r.activityId) { $null = $sb.AppendLine("  activityId: " + $r.activityId) }
  if ($r.snippet) { $null = $sb.AppendLine("  snippet: " + (($r.snippet -replace "\s+"," ").Trim())) }
  if ($r.error) { $null = $sb.AppendLine("  error: " + $r.error) }
  $null = $sb.AppendLine("")
}

$null = $sb.AppendLine("=== Next Move (auto guidance) ===")
if (-not $best) {
  $null = $sb.AppendLine("No HTTP responses captured. Likely DNS/TLS/proxy blocked any connection.")
  $null = $sb.AppendLine("Next: test base host in browser from service machine and verify proxy/TLS.")
} else {
  $null = $sb.AppendLine(("Best signal: {0} {1} => {2} {3}" -f $best.method, $best.url, $best.status, $best.statusText))

  if ($best.status -ge 200 -and $best.status -lt 300) {
    $null = $sb.AppendLine("[OK] Working route found with the tested auth scheme.")
    $null = $sb.AppendLine("Next: configure RAG to use this PATH (not hardcoded /v1/chat/completions).")
  } elseif ($best.status -eq 401 -or $best.status -eq 403) {
    $null = $sb.AppendLine("[OK] Route likely exists. Auth missing or incorrect for this gateway.")
    $null = $sb.AppendLine("Next: align RAG outbound header scheme with what produced 401/403 here (Bearer vs api-key vs subscription-key).")
  } elseif ($best.status -eq 404) {
    $null = $sb.AppendLine("[FAIL] 404 means the path does not exist on this host.")
    $null = $sb.AppendLine("Next: choose a non-404 endpoint from the Top list and make it your configured chat_path.")
  } elseif ($best.status -ge 500 -and $best.status -lt 600) {
    $null = $sb.AppendLine("[WARN] 5xx from gateway (ActivityId may be present). Often means route exists but gateway errors without proper auth/routing.")
    $null = $sb.AppendLine("Next: try the same URL with correct auth; if still 5xx, give ActivityId to gateway team.")
  } else {
    $null = $sb.AppendLine("Signal suggests route exists but request shape/headers are wrong.")
    $null = $sb.AppendLine("Next: use best-scoring chat endpoint and align your client with its requirements.")
  }

  $null = $sb.AppendLine("")
  $null = $sb.AppendLine("Practical tip:")
  $null = $sb.AppendLine("- Store ONLY the base host (e.g. https://aiml-aoai-api.gcl.mycompany.com).")
  $null = $sb.AppendLine("- Configure a separate chat_path (gateway prefix) instead of hardcoding /v1/chat/completions.")
}

$sb.ToString() | Out-File -FilePath $txtPath -Encoding UTF8
$Results | ConvertTo-Json -Depth 6 | Out-File -FilePath $jsonPath -Encoding UTF8

Write-Host ""
Write-Host "DONE." -ForegroundColor Green
Write-Host ("Report: " + $txtPath)
Write-Host ("JSON  : " + $jsonPath)
Write-Host ""
Write-Host "Open api_diag_report.txt and read the 'Top candidate endpoints' and 'Best signal' lines."

