# Hallucination Guard -- Installation Guide

**Date:** 2026-02-16
**SIM Test:** 149 PASS / 0 FAIL / 1 WARN
**Files:** 24 total (22 Python, 1 PowerShell, 1 YAML)

---

## What Changed From the Other Session's Delivery

1. **config.py REDESIGNED** -- HallucinationGuardConfig moved to its own file (`guard_config.py`). config.py gets a 27-line addition (import + field + properties + load line) instead of a full 637-line replacement.

2. **reranker_top_n = 12** -- Confirmed by the developer for demo speed.

3. **Test file SPLIT** -- 687-line test split into Part 1 (386 lines) and Part 2 (429 lines).

4. **YAML built from the developer's actual** -- Only added hallucination_guard section and changed reranker_top_n. No other values touched.

---

## Installation Steps (one command per block)

### Step 1: Create the hallucination guard directory

```powershell
New-Item -ItemType Directory -Path "{PROJECT_ROOT}\src\core\hallucination_guard" -Force
```

### Step 2: Extract the zip

```powershell
Expand-Archive -Path "$HOME\Downloads\hallucination_guard_delivery.zip" -DestinationPath "$HOME\Downloads\hg_delivery" -Force
```

### Step 3: Copy guard package (11 files)

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\src\core\hallucination_guard\*" "{PROJECT_ROOT}\src\core\hallucination_guard\" -Recurse -Force
```

### Step 4: Copy NEW file -- guard_config.py

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\src\core\guard_config.py" "{PROJECT_ROOT}\src\core\guard_config.py"
```

### Step 5: Copy NEW file -- grounded_query_engine.py

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\src\core\grounded_query_engine.py" "{PROJECT_ROOT}\src\core\grounded_query_engine.py"
```

### Step 6: Copy NEW file -- feature_registry.py

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\src\core\feature_registry.py" "{PROJECT_ROOT}\src\core\feature_registry.py"
```

### Step 7: Copy NEW file -- guard_diagnostic.py

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\src\diagnostic\guard_diagnostic.py" "{PROJECT_ROOT}\src\diagnostic\guard_diagnostic.py"
```

### Step 8: Copy NEW file -- rag-features.ps1

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\tools\rag-features.ps1" "{PROJECT_ROOT}\tools\rag-features.ps1"
```

### Step 9: REPLACE config.py (redesigned with guard integration)

```powershell
Copy-Item "{PROJECT_ROOT}\src\core\config.py" "{PROJECT_ROOT}\src\core\config.py.bak"
```

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\src\core\config.py" "{PROJECT_ROOT}\src\core\config.py" -Force
```

### Step 10: REPLACE default_config.yaml (added guard section + reranker=12)

```powershell
Copy-Item "{PROJECT_ROOT}\config\default_config.yaml" "{PROJECT_ROOT}\config\default_config.yaml.bak"
```

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\config\default_config.yaml" "{PROJECT_ROOT}\config\default_config.yaml" -Force
```

### Step 11: Copy tests

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\tests\virtual_test_guard_part1.py" "{PROJECT_ROOT}\tests\"
```

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\tests\virtual_test_guard_part2.py" "{PROJECT_ROOT}\tests\"
```

### Step 12: Copy LimitlessApp files

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\limitless\fact_extractor.py" "{KNOWLEDGE_BASE}\LimitlessApp\src\fact_extractor.py"
```

```powershell
Copy-Item "$HOME\Downloads\hg_delivery\delivery\limitless\claim_verifier.py" "{KNOWLEDGE_BASE}\LimitlessApp\src\claim_verifier.py"
```

### Step 13: Verify the feature list

```powershell
cd {PROJECT_ROOT}
```

```powershell
.\tools\rag-features.ps1 list
```

---

## First-Time NLI Model Download (one-time, home PC only)

The hallucination guard uses a ~440MB NLI model. Download it once on your home PC:

```powershell
$env:HYBRIDRAG_ADMIN_MODE = "1"
```

```powershell
python -c "from sentence_transformers import CrossEncoder; m = CrossEncoder('cross-encoder/nli-deberta-v3-base'); print('[OK] Model downloaded')"
```

```powershell
$env:HYBRIDRAG_ADMIN_MODE = ""
```

For the work laptop: copy the HuggingFace cache folder from home.

---

## File Inventory

| File | Lines | Location | Status |
|------|-------|----------|--------|
| guard_config.py | 78 | src/core/ | NEW |
| config.py | 616 | src/core/ | REDESIGNED |
| grounded_query_engine.py | 425 | src/core/ | NEW |
| feature_registry.py | 497 | src/core/ | NEW |
| hallucination_guard/__init__.py | 115 | src/core/ | NEW |
| hallucination_guard/__main__.py | 19 | src/core/ | NEW |
| hallucination_guard/claim_extractor.py | 217 | src/core/ | NEW |
| hallucination_guard/dual_path.py | 207 | src/core/ | NEW |
| hallucination_guard/golden_probes.py | 496 | src/core/ | NEW |
| hallucination_guard/guard_types.py | 327 | src/core/ | NEW |
| hallucination_guard/hallucination_guard.py | 358 | src/core/ | NEW |
| hallucination_guard/nli_verifier.py | 468 | src/core/ | NEW |
| hallucination_guard/prompt_hardener.py | 218 | src/core/ | NEW |
| hallucination_guard/response_scoring.py | 336 | src/core/ | NEW |
| hallucination_guard/self_test.py | 196 | src/core/ | NEW |
| hallucination_guard/startup_bit.py | 209 | src/core/ | NEW |
| guard_diagnostic.py | 494 | src/diagnostic/ | NEW |
| rag-features.ps1 | 355 | tools/ | NEW |
| default_config.yaml | -- | config/ | UPDATED |
| virtual_test_guard_part1.py | 386 | tests/ | NEW |
| virtual_test_guard_part2.py | 429 | tests/ | NEW |
| virtual_test_limitless_verifier.py | 280 | tests/ | NEW |
| fact_extractor.py | 268 | LimitlessApp/src/ | NEW |
| claim_verifier.py | 279 | LimitlessApp/src/ | NEW |

---

## Known Issue

**config.py at 616 lines** -- the developer's original was already 589 (pre-existing, before this change). The guard integration adds 27 lines (import, field, 3 properties, load line, changelog entry). The HallucinationGuardConfig dataclass itself lives in guard_config.py (78 lines) to minimize the addition. A future session could split more dataclasses out to get config.py under 500.
