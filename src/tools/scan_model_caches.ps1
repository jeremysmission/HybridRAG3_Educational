# ============================================================================
# HybridRAG Model Cache Scanner
# ============================================================================
# PURPOSE: Find ALL embedding model caches on your machine.
#   The all-MiniLM-L6-v2 model is ~80MB. You only need ONE copy.
#   This script finds every copy so you can decide which to keep.
#
# HOW IT WORKS:
#   1. Searches common cache locations (HuggingFace, torch, pip, etc.)
#   2. Searches your project folders for model files
#   3. Reports size of each cache found
#   4. Tells you which ones are safe to delete
#
# RUN: .\scan_model_caches.ps1
# ============================================================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Model Cache Scanner" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# These are the standard places where Python ML libraries
# download and cache models on Windows.
# ============================================================
$searchPaths = @(
    # --- HuggingFace default cache ---
    # When you use sentence-transformers or transformers library,
    # models download here by default.
    "$env:USERPROFILE\.cache\huggingface",

    # --- Torch hub cache ---
    # PyTorch sometimes caches model weights here.
    "$env:USERPROFILE\.cache\torch",

    # --- Sentence-transformers cache ---
    # Some versions cache in their own folder.
    "$env:USERPROFILE\.cache\sentence_transformers",

    # --- Generic model cache names in project folders ---
    # These are custom cache folders HybridRAG might use.
    "{PROJECT_ROOT}\model_cache",
    "{PROJECT_ROOT}\.model_cache",
    "{PROJECT_ROOT}\.hf_cache",
    "{PROJECT_ROOT}\.torch_cache",
    "{PROJECT_ROOT}\src\core\model_cache",

    # --- Old C drive project location ---
    "C:\Users\jerem\OneDrive\Desktop\AI Project\HybridRAG\model_cache",
    "C:\Users\jerem\OneDrive\Desktop\AI Project\HybridRAG\.model_cache",
    "C:\Users\jerem\OneDrive\Desktop\AI Project\HybridRAG\.hf_cache",

    # --- ONNX cache (if using ONNX runtime for embeddings) ---
    "$env:USERPROFILE\.cache\onnx",

    # --- Pip cache (can grow huge over time) ---
    "$env:LOCALAPPDATA\pip\cache"
)

Write-Host "Scanning known cache locations..." -ForegroundColor Yellow
Write-Host ""

$totalSize = 0
$foundCaches = @()

foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        # Calculate folder size
        $size = (Get-ChildItem $path -Recurse -Force -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($size / 1MB, 1)
        $totalSize += $size

        # Count files
        $fileCount = (Get-ChildItem $path -Recurse -File -Force -ErrorAction SilentlyContinue).Count

        $foundCaches += [PSCustomObject]@{
            Path = $path
            SizeMB = $sizeMB
            Files = $fileCount
        }

        # Color code by size
        if ($sizeMB -gt 100) {
            $color = "Red"
        } elseif ($sizeMB -gt 30) {
            $color = "Yellow"
        } else {
            $color = "Green"
        }

        Write-Host "  FOUND  $path" -ForegroundColor $color
        Write-Host "         Size: $sizeMB MB  |  Files: $fileCount" -ForegroundColor Gray
        Write-Host ""
    }
}

# ============================================================
# Also search for model files by name anywhere in project folders
# ============================================================
Write-Host "Searching for model files in project folders..." -ForegroundColor Yellow
Write-Host ""

$modelPatterns = @(
    "*.onnx",           # ONNX model weights (this is what MiniLM uses)
    "*.bin",            # PyTorch model weights
    "*.safetensors",    # Newer safe model format
    "tokenizer.json",   # Tokenizer file (part of model)
    "config.json"       # Model config (many false positives, but useful to check)
)

$projectRoots = @(
    "{PROJECT_ROOT}",
    "C:\Users\jerem\OneDrive\Desktop\AI Project\HybridRAG"
)

foreach ($root in $projectRoots) {
    if (Test-Path $root) {
        foreach ($pattern in $modelPatterns) {
            Get-ChildItem $root -Recurse -Filter $pattern -Force -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notlike "*\.venv*" -and $_.FullName -notlike "*node_modules*" } |
                ForEach-Object {
                    $sizeMB = [math]::Round($_.Length / 1MB, 1)
                    if ($sizeMB -gt 1) {  # Only report files > 1MB to reduce noise
                        Write-Host "  MODEL FILE  $($_.FullName)" -ForegroundColor Magenta
                        Write-Host "              Size: $sizeMB MB" -ForegroundColor Gray
                        Write-Host ""
                    }
                }
        }
    }
}

# ============================================================
# Summary and recommendations
# ============================================================
$totalMB = [math]::Round($totalSize / 1MB, 1)
$totalGB = [math]::Round($totalSize / 1GB, 2)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Caches found: $($foundCaches.Count)" -ForegroundColor White
Write-Host "  Total size:   $totalMB MB ($totalGB GB)" -ForegroundColor White
Write-Host ""

Write-Host "RECOMMENDATIONS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  KEEP:" -ForegroundColor Green
Write-Host "    - {PROJECT_ROOT}\model_cache (or wherever your active project loads from)" -ForegroundColor White
Write-Host "    - $env:USERPROFILE\.cache\huggingface (shared by all Python ML projects)" -ForegroundColor White
Write-Host ""
Write-Host "  SAFE TO DELETE:" -ForegroundColor Red
Write-Host "    - Any cache in the OLD C:\...\HybridRAG\ folder (you moved to D:)" -ForegroundColor White
Write-Host "    - Duplicate model_cache / .model_cache / .hf_cache if you have multiple" -ForegroundColor White
Write-Host "    - pip cache ($env:LOCALAPPDATA\pip\cache) if disk space is tight" -ForegroundColor White
Write-Host "      (pip will just re-download packages next time you install)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  DO NOT DELETE:" -ForegroundColor Red
Write-Host "    - The one model_cache your active project uses (check config.yaml" -ForegroundColor White
Write-Host "      for 'model_cache_dir' to see which path the code actually reads)" -ForegroundColor White
Write-Host ""
Write-Host "  To check which cache your code uses:" -ForegroundColor Yellow
Write-Host '    Select-String "model_cache\|cache_dir\|cache_folder" {PROJECT_ROOT}\src\core\embedder.py' -ForegroundColor Gray
Write-Host ""
