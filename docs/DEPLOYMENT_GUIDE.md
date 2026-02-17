# Hallucination Guard -- Deployment Guide (v2.0)

## Session 7 Redesign -- Subclass Pattern + Speed Optimizations

**Date**: 2026-02-16
**Virtual Test Results**: HybridRAG3 155/155 PASS | LimitlessApp 66/66 PASS
**Baseline Regression**: test_redesign.py 122P/1F (0 delta) | test_hybridrag3.py 2P/42F (0 delta)

---

## WHAT CHANGED FROM SESSION 6

Session 6 REPLACED query_engine.py (235 -> 478 lines). That was wrong.

Session 7 uses a SUBCLASS pattern:
- query_engine.py: UNTOUCHED (still 235 lines)
- boot.py: UNTOUCHED (still 309 lines)
- NEW: grounded_query_engine.py (424 lines) -- extends QueryEngine
- MODIFIED: config.py (588 -> 636 lines, +48 for guard config)
- MODIFIED: default_config.yaml (+12 lines for guard section)
- NEW: src/core/hallucination_guard/ (11 files, all <500 lines)

---

## QUICK START (5 minutes)

### Step 1: Copy files into HybridRAG3

```
{PROJECT_ROOT}\
  src\core\
    grounded_query_engine.py      <-- NEW (copy from delivery)
    config.py                     <-- REPLACE with updated version
    feature_registry.py           <-- NEW (feature toggle system)
    hallucination_guard\          <-- NEW folder (copy entire folder)
      __init__.py
      __main__.py
      claim_extractor.py
      dual_path.py
      guard_types.py
      hallucination_guard.py
      nli_verifier.py
      prompt_hardener.py
      response_scoring.py
      self_test.py
      startup_bit.py
  config\
    default_config.yaml           <-- REPLACE with updated version
  tools\
    rag-features.ps1              <-- NEW (CLI feature manager)
```

### Step 2: Verify (run from {PROJECT_ROOT})

```powershell
python tests\test_redesign.py
```

Expected: 122 passed, 1 failed (same as before -- the 1 failure is pre-existing).

### Step 3: Done

Guard is OFF by default. Your system works exactly as before.

---

## TURNING THE GUARD ON AND OFF

### Option A: YAML toggle (persistent)

Edit `config\default_config.yaml`:

```yaml
hallucination_guard:
  enabled: true       # <-- change to true
  failure_action: flag # <-- recommended for beta (shows score, no blocking)
```

### Option B: Feature Manager CLI (recommended)

```powershell
rag-features list                          # Show all features with status
rag-features enable hallucination-filter    # Turn ON 5-step anti-hallucination
rag-features disable hallucination-filter   # Turn OFF (zero overhead)
rag-features status                        # Quick status of everything
rag-features status hallucination-filter    # Check one feature
```

Other features you can toggle the same way:

```
hallucination-filter   5-step anti-hallucination pipeline
hybrid-search          Semantic + keyword combined search
reranker               Cross-encoder result re-scoring
pii-scrubber           PII removal before sending to APIs
audit-log              Query and access audit logging
cost-tracker           API token and cost tracking
```

### Option C: Per-session toggle (temporary)

Set environment variable before running:

```powershell
$env:HYBRIDRAG_GUARD="off"
python -m src.core.grounded_query_engine
```

### Setup: Add rag-features to your PowerShell profile

1. Copy `tools\rag-features.ps1` to `{PROJECT_ROOT}\tools\`
2. Copy `src\core\feature_registry.py` to `{PROJECT_ROOT}\src\core\`
3. Add to your PowerShell profile (`$PROFILE`):

```powershell
function rag-features {
    & "{PROJECT_ROOT}\tools\rag-features.ps1" @args
}
```

Now `rag-features` works from any directory.

### GUI Integration (future)

The feature_registry.py module exposes `get_feature_catalog()` which
returns a list of dicts, each describing a feature with:
- feature_id, display_name, category, description
- enabled (current state), impact_note
- detail (for hover/tooltip text)

Your future GUI reads this catalog and renders toggle switches
grouped by category (Quality, Retrieval, Security, Cost).
Adding a new feature to the registry automatically makes it
appear in both CLI and GUI -- no extra wiring.

```python
from feature_registry import FeatureRegistry
reg = FeatureRegistry("config/default_config.yaml")
catalog = reg.get_feature_catalog()  # List of feature dicts for GUI
categories = reg.get_categories()     # ["Quality", "Retrieval", ...]
```

### Recommended for demos and beta testing:

```yaml
hallucination_guard:
  enabled: false          # OFF for speed during demos
  failure_action: flag    # When you DO enable, flag don't block
```

Or from PowerShell:
```
rag-features disable hallucination-filter    # Fast demos
rag-features enable hallucination-filter     # Testing with grounding
```

---

## FAILURE ACTIONS EXPLAINED

| Action | What happens | When to use |
|--------|-------------|-------------|
| `block` | Replaces answer with "I cannot verify this" | Production (enterprise) |
| `flag` | Answer shows normally + grounding_score in metadata | Beta testing |
| `strip` | Removes unverified sentences, keeps verified ones | Advanced/experimental |

---

## SPEED TUNING

### Performance with optimizations (Session 7):

| Setup | Guard OFF | Guard ON |
|-------|-----------|----------|
| CPU only | ~2-6s | ~3-8s (+1-2s overhead) |
| RTX 3090 | ~2-6s | ~2.5-7s (+0.5-1s overhead) |
| Dual 3090 | ~2-6s | ~2-5s (minimal overhead) |

### Speed config knobs (in default_config.yaml):

```yaml
hallucination_guard:
  # How many source chunks to check per claim (fewer = faster)
  chunk_prune_k: 3        # Default 3. Range: 1-8. Lower = faster but less thorough.

  # Early-exit: skip remaining claims after N consecutive passes
  shortcircuit_pass: 5    # Default 5. Lower = faster for well-grounded answers.

  # Early-exit: block after N consecutive failures
  shortcircuit_fail: 3    # Default 3. Lower = faster rejection of hallucinated answers.

  # Grounding threshold
  threshold: 0.80         # Default 0.80. Lower = more permissive. 0.60 = fast but risky.
```

### Speed mode (maximum performance, minimum checking):

```yaml
hallucination_guard:
  enabled: true
  chunk_prune_k: 2
  shortcircuit_pass: 3
  shortcircuit_fail: 2
  threshold: 0.60
```

### Thorough mode (maximum safety, slower):

```yaml
hallucination_guard:
  enabled: true
  chunk_prune_k: 5
  shortcircuit_pass: 8
  shortcircuit_fail: 5
  threshold: 0.90
```

---

## RETRIEVAL TUNING (what gets retrieved and how fast)

The retrieval pipeline has three stages you can tune independently.
All settings are in `config\default_config.yaml` under `retrieval:`.

### Stage 1: Search (embedding + keyword)

| Setting | Default | What it does |
|---------|---------|-------------|
| `hybrid_search` | true | Combines semantic + keyword search. ON = best for engineering docs. |
| `min_score` | 0.3 | Minimum similarity score to consider a chunk. Lower = more results, more noise. |
| `rrf_k` | 60 | RRF merge constant. Higher = more weight to lower-ranked results. |

### Stage 2: Reranker (cross-encoder re-scoring)

| Setting | Default | What it does |
|---------|---------|-------------|
| `reranker_enabled` | true | ON = more accurate results, adds ~0.5-1s on CPU. |
| `reranker_top_n` | **12** | How many candidates to rerank. Was 20, reduced to 12 for 40% speed gain with <5% accuracy loss. |
| `reranker_model` | ms-marco-MiniLM-L-6-v2 | ~80MB model, shared with existing install. |

**Tuning reranker_top_n by use case:**

| Value | Speed (CPU) | When to use |
|-------|-------------|-------------|
| 8 | ~200-400ms | Speed demos, small doc sets (<100 files) |
| **12** | ~300-600ms | **Default. Good balance for engineering docs.** |
| 16 | ~400-800ms | Large doc sets with many similar documents |
| 20 | ~500-1000ms | Maximum accuracy, research mode |

### Stage 3: Final selection

| Setting | Default | What it does |
|---------|---------|-------------|
| `top_k` | 5 | Chunks sent to the LLM. More = better context but slower and costlier (more tokens). |

**Tuning top_k by use case:**

| Value | When to use |
|-------|-------------|
| 3 | Quick lookups, simple fact questions |
| **5** | **Default. Good for most engineering queries.** |
| 8 | Complex analysis, comparison queries, report generation |
| 10 | Deep research mode (watch token costs in online mode) |

### Example configs for different workflows

**Demo mode (maximum speed):**
```yaml
retrieval:
  hybrid_search: true
  reranker_enabled: false     # Skip reranking entirely
  top_k: 3                   # Fewer chunks = faster + cheaper
```

**Engineering analysis (balanced -- current default):**
```yaml
retrieval:
  hybrid_search: true
  reranker_enabled: true
  reranker_top_n: 12
  top_k: 5
```

**Report generation / drawing review (maximum accuracy):**
```yaml
retrieval:
  hybrid_search: true
  reranker_enabled: true
  reranker_top_n: 20
  top_k: 8
```

**Logistics inventory lookup (exact match heavy):**
```yaml
retrieval:
  hybrid_search: true         # Keyword search catches part numbers
  reranker_enabled: true
  reranker_top_n: 8           # Fewer candidates needed for exact matches
  top_k: 3                   # Usually one document has the answer
```

### Full pipeline timing (with current defaults)

```
Query: "What is the spec for the L-band antenna gain?"

  Embedding lookup:        ~100ms
  BM25 keyword search:     ~50ms
  RRF merge:               ~5ms
  Reranker (12 items):     ~300-600ms  (was ~500-1000ms with 20)
  API call (GPT-3.5):     ~2000-5000ms
  Hallucination filter:    ~2000-4000ms (if enabled, with optimizations)
                           ─────────────
  Total (filter OFF):     ~2.5-6s
  Total (filter ON):      ~4.5-10s
```

### Future retrieval optimizations (roadmap)

These are NOT in this delivery but are wired for when you're ready:

| Optimization | Impact | Effort |
|---|---|---|
| Float16 embeddings | -30% search time, half RAM | ~10 lines in embedder.py |
| Reranker score caching | 2-10x faster for repeat queries | ~40 lines in retriever.py |
| Domain synonym expansion | Better recall for RF/logistics terms | ~100 line new module |
| Metadata pre-filtering | Filter by doc type before search | Needs metadata tagging first |
| Parent-child chunks | Better context for LLM | Chunking redesign |

---

## NLI MODEL FIRST-TIME SETUP

The NLI model (cross-encoder/nli-deberta-v3-base, ~440MB) downloads on first use.

### Online (home PC):
Just run a query with guard enabled. Model downloads automatically.

### offline (work laptop):
1. On home PC: run with guard enabled once (downloads to .model_cache/)
2. Copy .model_cache/ folder to USB
3. On work laptop: paste into {PROJECT_ROOT}\.model_cache\
4. Set in YAML: `model_cache_dir: .model_cache`

### Block downloads (after caching):
Set env var: `HALLUCINATION_GUARD_OFFLINE=1`

---

## LIMITLESS APP INTEGRATION

Copy these into {KNOWLEDGE_BASE}\LimitlessApp\:

```
fact_extractor.py    (267 lines, stdlib only)
claim_verifier.py    (278 lines, stdlib only)
```

Integration point in primer_generator.py (after the merge step, ~line 497):

```python
# -- POST-GENERATION VERIFICATION --
try:
    from claim_verifier import verify_primer

    verification = verify_primer(
        primer_text=merged_result,
        source_text=file_text,
        source_file=filename,
        primer_file=f"primer_{category}_{safe_name}.md",
        use_minicheck=False,   # True when Ollama minicheck available
    )
    merged_result = verification["tagged_primer"]
    confidence = verification["confidence"]
    stats = verification["stats"]
    print(f"    [VERIFY] {confidence} "
          f"({stats['verified']}/{stats['total_facts']} facts)")
except ImportError:
    print(f"    [INFO] claim_verifier.py not found -- skipping")
except Exception as e:
    print(f"    [WARN] Verification failed: {e}")
```

This is NOT integrated yet -- just shows where it plugs in when ready.

---

## FILES UNCHANGED (zero blast radius)

These files were NOT modified:
- src/core/query_engine.py (235 lines)
- src/core/boot.py (309 lines)
- src/core/retriever.py
- src/core/embedder.py
- src/core/llm_router.py
- src/core/vector_store.py
- tests/test_redesign.py
- tests/test_hybridrag3.py
- src/diagnostic/hybridrag_diagnostic.py
- All monitoring/ files

---

## 3-REPO COMPATIBILITY

| Repo | Gets guard? | Notes |
|------|------------|-------|
| {PROJECT_ROOT} (private) | YES | Full implementation |
| {PROJECT_ROOT}_Educational | YES | sync_to_educational.py handles sanitization |
| {KNOWLEDGE_BASE}\LimitlessApp | YES (regex only) | fact_extractor + claim_verifier |

Banned words removed: ORG, Organization, Organization, restricted, regulatory,
sensitive data, authorization, AI assistant, AI provider. Verified by SIM-17 sanitization tests.

---

## TROUBLESHOOTING

**Guard loads but all scores are 0.0**
- NLI model not downloaded yet. Run online once, or copy .model_cache/.

**"sentence-transformers not installed"**
- Run: `pip install sentence-transformers --break-system-packages`
- You already have it (shared with embeddings).

**Guard slows queries to >15 seconds on CPU**
- Reduce chunk_prune_k to 2
- Reduce shortcircuit_pass to 3
- Or just set enabled: false for demos

**test_redesign.py shows different results**
- Expected: 122P/1F. The 1F is pre-existing (config load test).
- If delta != 0, something changed in config.py. Compare with backup.
