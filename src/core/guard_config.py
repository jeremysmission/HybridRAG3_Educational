# ============================================================================
# HybridRAG -- Hallucination Guard Configuration (src/core/guard_config.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Defines the settings for the post-generation hallucination filter.
#   When enabled, every LLM response is verified against source documents
#   using NLI (Natural Language Inference) before being shown to the user.
#
# WHY THIS IS A SEPARATE FILE:
#   config.py was already 589 lines. Adding 48 more lines for the guard
#   config would push it over the 500-line limit. Instead, we put the
#   guard dataclass here and import it into config.py -- same pattern as
#   how a growing codebase splits related configs into submodules.
#
# HOW IT CONNECTS:
#   config.py imports HallucinationGuardConfig from this file.
#   load_config() in config.py parses the "hallucination_guard" section
#   of default_config.yaml into this dataclass.
#   grounded_query_engine.py reads these settings via config.hallucination_guard.
#
# NETWORK ACCESS:
#   NONE from this file. The NLI model download happens in nli_verifier.py,
#   not here. This file only defines settings.
#
# CHANGES:
#   2026-02-16: Created. Split from config.py to maintain 500-line limit.
# ============================================================================

from dataclasses import dataclass


@dataclass
class HallucinationGuardConfig:
    """
    Post-generation hallucination verification settings.

    When enabled, the GroundedQueryEngine verifies every LLM claim
    against source chunks using NLI (Natural Language Inference).

    The NLI model runs LOCALLY after a one-time download (~440MB).
    For offline deployment, copy .model_cache/ to the target machine.

    PERFORMANCE: Adds ~2-5 seconds per query on CPU, <1s on GPU.
    TOGGLE: Set enabled=false to disable with zero overhead.
    """
    # --- Core settings ---
    enabled: bool = False             # Off by default until tested
    threshold: float = 0.80           # Min grounding score (0.0-1.0)
    failure_action: str = "block"     # block | flag | strip

    # --- NLI model ---
    # cross-encoder/nli-deberta-v3-base is ~440MB, runs on CPU or GPU.
    # Downloaded once via sentence-transformers, cached in model_cache_dir.
    # For offline work: copy the cache folder from a connected machine.
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    model_cache_dir: str = ".model_cache"

    # --- Dual-path consensus (optional, 2x cost) ---
    # Sends the same query to two LLMs and compares answers.
    # Only useful when you have both Ollama and API available.
    enable_dual_path: bool = False

    # --- Speed optimizations (software-side, no GPU needed) ---
    # These reduce NLI calls without meaningful accuracy loss.
    #
    # chunk_prune_k: Instead of checking a claim against all 8 retrieved
    #   chunks, only check the top 3 most relevant. 3x faster, <2% loss.
    #
    # shortcircuit_pass: If N consecutive claims are SUPPORTED, skip the
    #   rest and mark the response as grounded. Saves time on good answers.
    #
    # shortcircuit_fail: If N consecutive claims are UNSUPPORTED, block
    #   immediately. Fails fast on hallucinated responses.
    chunk_prune_k: int = 3            # Top N chunks per claim (3 vs 8 = 3x faster)
    shortcircuit_pass: int = 5        # Early-pass after N consecutive supported
    shortcircuit_fail: int = 3        # Early-fail after N consecutive unsupported
