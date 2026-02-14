# ============================================================================
# open_diagnostic.ps1 â€” Review and run the HybridRAG diagnostic tool
# ============================================================================
# HOW: . .\open_diagnostic.ps1   (from your HybridRAG PowerShell session)
# ============================================================================

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  HybridRAG Diagnostic Tool" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Open all diagnostic files in Notepad for review
$files = @(
    "hybridrag_diagnostic.py",
    "src\diagnostic\__init__.py",
    "src\diagnostic\health_tests.py",
    "src\diagnostic\component_tests.py",
    "src\diagnostic\perf_benchmarks.py",
    "src\diagnostic\report.py"
)

foreach ($f in $files) {
    $path = Join-Path $PSScriptRoot $f
    if (Test-Path $path) {
        $lines = (Get-Content $path).Count
        Write-Host "  Opening: $f ($lines lines)" -ForegroundColor DarkGray
        Start-Process notepad.exe $path
    } else {
        Write-Host "  MISSING: $f" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "  HOW TO RUN" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activate first:   . .\start_hybridrag.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "  Quick check:      python hybridrag_diagnostic.py" -ForegroundColor White
Write-Host "  Detailed:         python hybridrag_diagnostic.py -v" -ForegroundColor White
Write-Host "  Show bug fixes:   python hybridrag_diagnostic.py --fix-preview -v" -ForegroundColor White
Write-Host "  With embedding:   python hybridrag_diagnostic.py --test-embed" -ForegroundColor White
Write-Host "  Parse a file:     python hybridrag_diagnostic.py --test-parse file.pdf" -ForegroundColor White
Write-Host "  Live query:       python hybridrag_diagnostic.py --test-embed --test-query `"freq range`"" -ForegroundColor White
Write-Host "  Save JSON:        python hybridrag_diagnostic.py --json-file report.json" -ForegroundColor White
Write-Host "  Perf only:        python hybridrag_diagnostic.py --perf-only --benchmark-iters 5" -ForegroundColor White
Write-Host ""

