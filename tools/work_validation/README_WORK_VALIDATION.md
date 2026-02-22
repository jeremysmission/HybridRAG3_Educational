# Work Laptop Validation -- Quick Start

**Date**: February 2026
**Hardware target**: 64 GB RAM, 12 GB NVIDIA VRAM

---

## Prerequisites

1. Python 3.10+ installed and on PATH
2. Ollama installed from your software store or approved by IT
3. Enterprise VPN connected (for online API tests)

---

## How to Get This Package to the Work Laptop

**Do NOT use git clone or git pull on the work machine.**
Browser download only -- no git credentials on work hardware.

1. On your **home PC**: push latest changes to GitHub
2. On the **work laptop**: open browser
3. Go to your personal GitHub repo (private)
4. Navigate to Releases or the `releases/` folder
5. Download `work_validation_transfer.zip`
6. Extract the zip to your Downloads folder
7. Copy the `work_validation/` folder into your HybridRAG3 project directory

---

## Step 1: Check Dependencies (Run First)

Open PowerShell and navigate to the work_validation folder:

```powershell
cd work_validation
```

```powershell
python check_dependencies.py
```

This checks Python version, PyPI reachability, Ollama, and key packages.

**If PyPI is reachable** (install packages from the internet):

```powershell
python check_dependencies.py --install
```

**If PyPI is blocked** (install from included wheels bundle):

```powershell
python check_dependencies.py --wheels
```

Fix any [FAIL] items before continuing. [WARN] items are informational.

---

## Step 2: Pull Ollama Models (One Time)

```powershell
.\setup_work_models.ps1
```

This will:
- Pull 4 Ollama models (~23 GB total download)
- Verify all models are installed

**Estimated time**: 15-30 minutes (depends on download speed)

**If Ollama is not installed**: Check your work software store first.
If not available there, request IT approval to install from https://ollama.com

---

## Step 3: Run Offline Validation

```powershell
python validate_offline_models.py --log offline_results.log
```

This will:
- Test each model against 5 work profiles (Engineer, PM, Logistics, CAD, SysAdmin)
- Send a test query per profile and check response quality
- Log [OK]/[FAIL]/[WARN] for each model-profile combination

**What to look for**:
- All primary models should show [OK]
- Alt models should show [OK] or [WARN] (keyword matching is approximate)
- [FAIL] means the model did not respond or gave empty output

**Estimated time**: 5-15 minutes (models load on first query)

---

## Step 4: Run Online API Validation

```powershell
python validate_online_api.py --log online_results.log
```

This will:
- Resolve API credentials from Windows Credential Manager or env vars
- Probe the Azure endpoint for available models
- Test confirmed deployments (GPT-3.5 Turbo, GPT-4)
- Probe for optional deployments (GPT-4o, etc.)
- Test online/offline mode switching

**If you get a 401 error**: The script prints detailed diagnostics:
- The exact URL attempted
- Which auth header was used
- Where the API key came from
- Suggested troubleshooting steps

**Override endpoint** if auto-detection fails:

```powershell
python validate_online_api.py --endpoint https://your-endpoint.openai.azure.com --log online_results.log
```

---

## Step 5: Review Results

Check the log files for [FAIL] entries:

```powershell
Select-String -Path offline_results.log -Pattern '\[FAIL\]'
```

```powershell
Select-String -Path online_results.log -Pattern '\[FAIL\]'
```

---

## Troubleshooting

### Ollama not running

```powershell
ollama serve
```

### Model not found

```powershell
ollama pull phi4-mini
```

```powershell
ollama pull mistral:7b
```

```powershell
ollama pull phi4:14b-q4_K_M
```

```powershell
ollama pull gemma3:4b
```

### API key not set

```powershell
$env:AZURE_OPENAI_API_KEY = 'your-key-here'
```

```powershell
$env:AZURE_OPENAI_ENDPOINT = 'https://your-endpoint.openai.azure.com'
```

### Enterprise SSL/proxy issues

If you see SSL certificate errors, ask IT for the enterprise root CA:

```powershell
$env:SSL_CERT_FILE = 'C:\path\to\enterprise-ca-bundle.crt'
```

```powershell
$env:REQUESTS_CA_BUNDLE = 'C:\path\to\enterprise-ca-bundle.crt'
```

### PyPI blocked -- use wheels bundle

If pip cannot reach PyPI, use the offline wheels:

```powershell
python check_dependencies.py --wheels
```

If wheels/ folder is missing, run `build_wheels_bundle.py` on your home PC
first, then copy the wheels/ folder into this directory.

---

## Model Summary

| Model              | Size   | Profiles Using It           |
|--------------------|--------|------------------------------|
| phi4-mini           | 5.2 GB | Primary: eng, pm, draft, sys |
| mistral:7b     | 5.2 GB | Alt: eng, sys (reasoning)    |
| phi4:14b-q4_K_M    | 9.1 GB | Primary: log; Alt: draft, eng|
| gemma3:4b          | 3.3 GB | Alt: pm (fast summarization) |

Total disk: ~23 GB
