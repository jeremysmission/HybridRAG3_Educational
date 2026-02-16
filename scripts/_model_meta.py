#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Model Metadata Helper (scripts/_model_meta.py)
# ============================================================================
#
# WHAT THIS DOES (plain English):
#   Provides model information for both offline and online AI models
#   with DUAL RANKING:
#
#     ENG = Engineering rank (code, math, structured data, STEM)
#     GEN = General knowledge rank (broad reasoning, world knowledge, writing)
#
#   This lets you see at a glance:
#     - Which model to use for engineering/RAG work (sort by ENG)
#     - Which model to use for general questions (sort by GEN)
#
# THREE-LAYER MODEL DETECTION:
#   1. API /models endpoint (OpenRouter, OpenAI)
#   2. Built-in knowledge base (Azure behind proxy, corporate firewall)
#   3. "No information available" (truly unknown models)
#
# INTERNET ACCESS:
#   Offline functions: NONE
#   Online fetch: YES (one GET to /models) -- fails gracefully
#   Knowledge base lookup: NONE (all data is built-in)
# ============================================================================

import re
import subprocess


# ============================================================================
# USE-CASE DEFINITIONS
# ============================================================================
# Each use case is a weighted blend of the two core scores (tier_eng, tier_gen).
# This lets us rank models for any role without storing 7 scores per model.
#
# Why weighted blends work:
#   - Software Engineering is ~90% code/math (eng) + 10% reasoning (gen)
#   - Program Management is ~25% structure (eng) + 75% writing/knowledge (gen)
#   - Each role falls somewhere on the eng<->gen spectrum
#
# key: short code used internally
# letter: what the user types at the prompt
# label: display name
# desc: one-line description shown in the menu
# eng_w / gen_w: weights (must sum to 1.0)
# ============================================================================

USE_CASES = {
    "sw":    {"letter": "S", "label": "Software Engineering",
              "desc": "Code generation, debugging, code review, algorithms",
              "eng_w": 0.90, "gen_w": 0.10},

    "eng":   {"letter": "E", "label": "Engineering / STEM",
              "desc": "Math, structured data, technical analysis, RAG queries",
              "eng_w": 0.80, "gen_w": 0.20},

    "sys":   {"letter": "Y", "label": "Systems Administration",
              "desc": "Scripts, configs, troubleshooting, security, networking",
              "eng_w": 0.70, "gen_w": 0.30},

    "draft": {"letter": "D", "label": "Drafting / AutoCAD",
              "desc": "Technical specs, structured output, precision documents",
              "eng_w": 0.75, "gen_w": 0.25},

    "log":   {"letter": "L", "label": "Logistics Analyst",
              "desc": "Data analysis, optimization, supply chain, spreadsheets",
              "eng_w": 0.60, "gen_w": 0.40},

    "pm":    {"letter": "P", "label": "Program Management",
              "desc": "Documentation, scheduling, reporting, communication",
              "eng_w": 0.25, "gen_w": 0.75},

    "gen":   {"letter": "G", "label": "General AI",
              "desc": "World knowledge, creative writing, broad reasoning",
              "eng_w": 0.10, "gen_w": 0.90},
}

# Reverse lookup: letter -> use case key  (e.g. "S" -> "sw")
_LETTER_TO_UC = {uc["letter"]: k for k, uc in USE_CASES.items()}


def use_case_score(tier_eng, tier_gen, uc_key):
    """
    Compute a model's rank for a specific use case.

    Args:
        tier_eng: Engineering/STEM score (0-100)
        tier_gen: General knowledge score (0-100)
        uc_key:   Use case key (e.g. "sw", "eng", "pm")

    Returns:
        int: Blended score (0-100), higher = better for that use case
    """
    uc = USE_CASES.get(uc_key)
    if not uc:
        return tier_eng  # fallback
    return int(tier_eng * uc["eng_w"] + tier_gen * uc["gen_w"])


# ============================================================================
# BUILT-IN KNOWLEDGE BASE
# ============================================================================
# Dual ranking system:
#   tier_eng: Engineering/STEM (code, math, structured data, reasoning)
#   tier_gen: General knowledge (world knowledge, writing, broad reasoning)
#
# Scale: 0-100 (higher = better)
#   90-100: Frontier
#   70-89:  Strong
#   50-69:  Good
#   30-49:  Basic
#   10-29:  Legacy
#
# Sources: Official model cards, MMLU/HumanEval/MATH benchmarks,
#          Chatbot Arena, community testing (HuggingFace, Reddit).
# ============================================================================

KNOWN_MODELS = {
    # ---- OpenAI GPT Family ----
    #                                 ctx        $/1M in   $/1M out  eng  gen  family     note
    "gpt-4.1":          {"ctx": 1047576, "price_in": 2.0,    "price_out": 8.0,    "tier_eng": 97, "tier_gen": 96, "family": "OpenAI",    "note": "Latest GPT-4 series, 1M ctx"},
    "gpt-4.1-mini":     {"ctx": 1047576, "price_in": 0.40,   "price_out": 1.60,   "tier_eng": 82, "tier_gen": 80, "family": "OpenAI",    "note": "Cost-efficient GPT-4.1"},
    "gpt-4.1-nano":     {"ctx": 1047576, "price_in": 0.10,   "price_out": 0.40,   "tier_eng": 65, "tier_gen": 62, "family": "OpenAI",    "note": "Fastest GPT-4.1"},
    "gpt-4o":           {"ctx": 128000,  "price_in": 2.50,   "price_out": 10.0,   "tier_eng": 93, "tier_gen": 95, "family": "OpenAI",    "note": "Flagship multimodal, 128K ctx"},
    "gpt-4o-mini":      {"ctx": 128000,  "price_in": 0.15,   "price_out": 0.60,   "tier_eng": 72, "tier_gen": 75, "family": "OpenAI",    "note": "Cost-efficient GPT-4o"},
    "gpt-4-turbo":      {"ctx": 128000,  "price_in": 10.0,   "price_out": 30.0,   "tier_eng": 88, "tier_gen": 90, "family": "OpenAI",    "note": "Previous flagship, 128K ctx"},
    "gpt-4":            {"ctx": 8192,    "price_in": 30.0,   "price_out": 60.0,   "tier_eng": 82, "tier_gen": 85, "family": "OpenAI",    "note": "Original GPT-4, 8K ctx"},
    "gpt-3.5-turbo":    {"ctx": 16385,   "price_in": 0.50,   "price_out": 1.50,   "tier_eng": 50, "tier_gen": 58, "family": "OpenAI",    "note": "Fast and cheap, 16K ctx"},
    "o1":               {"ctx": 200000,  "price_in": 15.0,   "price_out": 60.0,   "tier_eng": 98, "tier_gen": 92, "family": "OpenAI",    "note": "Reasoning model, 200K ctx"},
    "o1-mini":          {"ctx": 128000,  "price_in": 3.0,    "price_out": 12.0,   "tier_eng": 88, "tier_gen": 78, "family": "OpenAI",    "note": "Smaller reasoning model"},
    "o3":               {"ctx": 200000,  "price_in": 2.0,    "price_out": 8.0,    "tier_eng": 98, "tier_gen": 93, "family": "OpenAI",    "note": "Latest reasoning model"},
    "o3-mini":          {"ctx": 200000,  "price_in": 1.10,   "price_out": 4.40,   "tier_eng": 89, "tier_gen": 80, "family": "OpenAI",    "note": "Efficient reasoning"},
    "o4-mini":          {"ctx": 200000,  "price_in": 1.10,   "price_out": 4.40,   "tier_eng": 91, "tier_gen": 82, "family": "OpenAI",    "note": "Latest efficient reasoning"},

    # ---- Anthropic Claude Family ----
    "claude-3.7-sonnet":  {"ctx": 200000, "price_in": 3.0,   "price_out": 15.0,   "tier_eng": 94, "tier_gen": 95, "family": "Anthropic", "note": "Extended thinking, 200K ctx"},
    "claude-3.5-sonnet":  {"ctx": 200000, "price_in": 3.0,   "price_out": 15.0,   "tier_eng": 92, "tier_gen": 93, "family": "Anthropic", "note": "Strong all-around, 200K ctx"},
    "claude-3.5-haiku":   {"ctx": 200000, "price_in": 0.80,  "price_out": 4.0,    "tier_eng": 75, "tier_gen": 78, "family": "Anthropic", "note": "Fast and affordable, 200K ctx"},
    "claude-3-haiku":     {"ctx": 200000, "price_in": 0.25,  "price_out": 1.25,   "tier_eng": 60, "tier_gen": 65, "family": "Anthropic", "note": "Previous fast model"},
    "claude-3-opus":      {"ctx": 200000, "price_in": 15.0,  "price_out": 75.0,   "tier_eng": 90, "tier_gen": 92, "family": "Anthropic", "note": "Most capable Claude 3"},

    # ---- Meta Llama Family ----
    # Llama: strong general knowledge (MMLU), slightly behind Qwen on STEM
    "meta-llama/llama-3.3-70b-instruct":  {"ctx": 131072, "price_in": 0.10, "price_out": 0.25, "tier_eng": 80, "tier_gen": 84, "family": "Meta", "note": "70B, 128K ctx"},
    "meta-llama/llama-3.1-405b-instruct": {"ctx": 131072, "price_in": 1.00, "price_out": 1.00, "tier_eng": 88, "tier_gen": 92, "family": "Meta", "note": "405B, largest open model"},
    "meta-llama/llama-3.1-70b-instruct":  {"ctx": 131072, "price_in": 0.10, "price_out": 0.25, "tier_eng": 78, "tier_gen": 82, "family": "Meta", "note": "70B, open weights"},
    "meta-llama/llama-3.1-8b-instruct":   {"ctx": 131072, "price_in": 0.02, "price_out": 0.05, "tier_eng": 55, "tier_gen": 62, "family": "Meta", "note": "8B, MMLU 77.5"},
    "meta-llama/llama-3-8b-instruct":     {"ctx": 8192,   "price_in": 0.03, "price_out": 0.06, "tier_eng": 52, "tier_gen": 60, "family": "Meta", "note": "8B, 8K ctx"},
    "meta-llama/llama-3-70b-instruct":    {"ctx": 8192,   "price_in": 0.20, "price_out": 0.20, "tier_eng": 76, "tier_gen": 80, "family": "Meta", "note": "70B, 8K ctx"},

    # ---- Mistral Family ----
    "mistralai/mistral-large":           {"ctx": 128000, "price_in": 2.0,  "price_out": 6.0,  "tier_eng": 82, "tier_gen": 85, "family": "Mistral", "note": "Flagship Mistral"},
    "mistralai/mistral-small":           {"ctx": 32000,  "price_in": 0.20, "price_out": 0.60, "tier_eng": 62, "tier_gen": 65, "family": "Mistral", "note": "Efficient Mistral"},
    "mistralai/mistral-7b-instruct":     {"ctx": 32768,  "price_in": 0.03, "price_out": 0.06, "tier_eng": 48, "tier_gen": 55, "family": "Mistral", "note": "7B, open weights"},
    "mistralai/mixtral-8x7b-instruct":   {"ctx": 32768,  "price_in": 0.06, "price_out": 0.06, "tier_eng": 65, "tier_gen": 68, "family": "Mistral", "note": "MoE 8x7B"},

    # ---- DeepSeek Family ----
    # DeepSeek R1: exceptional at STEM reasoning, decent general
    "deepseek/deepseek-r1":    {"ctx": 65536, "price_in": 0.55, "price_out": 2.19, "tier_eng": 93, "tier_gen": 82, "family": "DeepSeek", "note": "Reasoning, 64K ctx"},
    "deepseek/deepseek-chat":  {"ctx": 65536, "price_in": 0.14, "price_out": 0.28, "tier_eng": 75, "tier_gen": 72, "family": "DeepSeek", "note": "General chat"},
    "deepseek/deepseek-v3":    {"ctx": 65536, "price_in": 0.14, "price_out": 0.28, "tier_eng": 82, "tier_gen": 78, "family": "DeepSeek", "note": "Latest base model"},

    # ---- Google Gemini Family ----
    "google/gemini-2.5-pro":        {"ctx": 1048576, "price_in": 1.25, "price_out": 10.0, "tier_eng": 95, "tier_gen": 96, "family": "Google", "note": "1M ctx, thinking"},
    "google/gemini-2.5-flash":      {"ctx": 1048576, "price_in": 0.15, "price_out": 0.60, "tier_eng": 78, "tier_gen": 80, "family": "Google", "note": "1M ctx, fast"},
    "google/gemini-2.0-flash-001":  {"ctx": 1048576, "price_in": 0.10, "price_out": 0.40, "tier_eng": 72, "tier_gen": 75, "family": "Google", "note": "1M ctx, efficient"},

    # ---- Qwen Family ----
    # Qwen: best-in-class STEM at every param count, weaker general knowledge
    "qwen/qwen-2.5-72b-instruct": {"ctx": 131072, "price_in": 0.15, "price_out": 0.15, "tier_eng": 86, "tier_gen": 78, "family": "Qwen", "note": "72B, MATH 83.1"},
    "qwen/qwen-2.5-32b-instruct": {"ctx": 131072, "price_in": 0.06, "price_out": 0.06, "tier_eng": 80, "tier_gen": 72, "family": "Qwen", "note": "32B, good value"},
    "qwen/qwen3-235b-a22b":       {"ctx": 131072, "price_in": 0.20, "price_out": 0.60, "tier_eng": 92, "tier_gen": 85, "family": "Qwen", "note": "MoE 235B/22B active"},

    # ---- Microsoft Phi Family ----
    "microsoft/phi-4":           {"ctx": 16384, "price_in": 0.03, "price_out": 0.06, "tier_eng": 65, "tier_gen": 58, "family": "Microsoft", "note": "14B, strong for size"},

    # ---- xAI Grok Family ----
    "x-ai/grok-3-beta":      {"ctx": 131072, "price_in": 3.0,  "price_out": 15.0, "tier_eng": 90, "tier_gen": 93, "family": "xAI", "note": "Flagship Grok"},
    "x-ai/grok-3-mini-beta": {"ctx": 131072, "price_in": 0.30, "price_out": 0.50, "tier_eng": 75, "tier_gen": 78, "family": "xAI", "note": "Efficient Grok"},

    # ---- Amazon Nova Family ----
    "amazon/nova-pro-v1":   {"ctx": 300000, "price_in": 0.80, "price_out": 3.20, "tier_eng": 68, "tier_gen": 75, "family": "Amazon", "note": "300K ctx, multimodal"},
    "amazon/nova-lite-v1":  {"ctx": 300000, "price_in": 0.06, "price_out": 0.24, "tier_eng": 55, "tier_gen": 60, "family": "Amazon", "note": "300K ctx, fast"},
    "amazon/nova-micro-v1": {"ctx": 128000, "price_in": 0.035,"price_out": 0.14, "tier_eng": 42, "tier_gen": 50, "family": "Amazon", "note": "Text only, cheap"},
}

# Azure deployment name -> knowledge base key
AZURE_ALIASES = {
    "gpt-35-turbo": "gpt-3.5-turbo", "gpt-35-turbo-16k": "gpt-3.5-turbo",
    "gpt-4-turbo": "gpt-4-turbo", "gpt-4-32k": "gpt-4",
    "gpt-4o": "gpt-4o", "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1", "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "o1": "o1", "o1-mini": "o1-mini", "o3": "o3",
    "o3-mini": "o3-mini", "o4-mini": "o4-mini",
}


def lookup_known_model(model_id):
    """
    Look up a model in the built-in knowledge base.
    Tries: exact -> strip prefix -> Azure alias -> fuzzy.
    Returns dict with specs, or None if not found.
    """
    if model_id in KNOWN_MODELS:
        return KNOWN_MODELS[model_id]

    short = model_id.split("/", 1)[-1] if "/" in model_id else model_id

    if short in KNOWN_MODELS:
        return KNOWN_MODELS[short]
    for kid, specs in KNOWN_MODELS.items():
        ks = kid.split("/", 1)[-1] if "/" in kid else kid
        if ks == short:
            return specs

    if short in AZURE_ALIASES:
        canonical = AZURE_ALIASES[short]
        if canonical in KNOWN_MODELS:
            return KNOWN_MODELS[canonical]

    clean = re.sub(r'[:@].*$', '', short.lower())
    clean = re.sub(r'-\d{6,}$', '', clean)
    for kid, specs in KNOWN_MODELS.items():
        kc = re.sub(r'[:@].*$', '', kid.lower())
        kc = re.sub(r'-\d{6,}$', '', kc)
        kc = kc.split("/")[-1] if "/" in kc else kc
        if clean == kc:
            return specs

    return None


def get_known_model_info(model_id):
    """Get info for a single model from knowledge base. Returns dict or None."""
    info = lookup_known_model(model_id)
    if info:
        return {
            "id": model_id, "ctx": info["ctx"],
            "price_in": info["price_in"], "price_out": info["price_out"],
            "tier_eng": info["tier_eng"], "tier_gen": info["tier_gen"],
            "family": info.get("family", "Unknown"),
            "note": info.get("note", ""), "source": "knowledge_base",
        }
    return None


# ============================================================================
# OFFLINE: Parse model specs from name + size
# ============================================================================

# Benchmark-based scores for OFFLINE (local) model families.
# These reflect published benchmark results, not opinions.
#
# Key benchmarks used:
#   ENG: HumanEval, MATH, LiveCodeBench, structured output quality
#   GEN: MMLU, ARC, HellaSwag, TruthfulQA, world knowledge tests
#
# Format: (pattern, eng_bonus, gen_bonus)
# Searched top-to-bottom; first match wins.

_OFFLINE_FAMILY_SCORES = [
    # Qwen 3: thinking mode, best STEM at every size
    ("qwen3",    38, 30),
    # Qwen 2.5: HumanEval 84.8, MATH 75.5 (7B); but MMLU 74.2 vs Llama's 77.5
    ("qwen2.5",  35, 25),
    # DeepSeek: exceptional reasoning, decent general
    ("deepseek", 34, 28),
    # Llama 3: MMLU 77.5 (7B class leader), HumanEval 80.5, MATH 69.9
    ("llama3",   28, 33), ("llama-3", 28, 33),
    # Phi-4: punches above weight on STEM for its 14B size
    ("phi-4",    30, 24), ("phi4", 30, 24),
    # Gemma 3: competitive all-around
    ("gemma",    26, 28),
    # Mixtral MoE: good efficiency and capability
    ("mixtral",  25, 27),
    # Qwen 2 (older): still decent STEM
    ("qwen2",    22, 20), ("qwen", 15, 14),
    # Mistral 7B: solid baseline
    ("mistral",  20, 23),
    # Phi-3 and older Phi
    ("phi-3",    22, 18), ("phi3", 22, 18), ("phi", 18, 16),
    # Code-specialized models
    ("code",     24, 15),
    # Llama 2: a generation behind on everything
    ("llama2",   12, 15), ("llama-2", 12, 15),
]


def parse_model_specs(name, size_str):
    """
    Extract params, quant, RAM/VRAM, and dual tier from Ollama model name.

    Returns dict with: params, quant, ram_gb, vram_gb, tier_eng, tier_gen
    """
    name_lower = name.lower()

    # ---- Parameter count ----
    param_match = re.search(r'(\d+)b', name_lower)
    params = int(param_match.group(1)) if param_match else _est_params(size_str)

    # ---- Quantization ----
    quant = "Q4"  # Ollama default
    for q, label in [("q8","Q8"),("q6","Q6"),("q5","Q5"),("q4","Q4"),
                      ("q3","Q3"),("q2","Q2"),("fp16","FP16"),("f16","FP16")]:
        if q in name_lower:
            quant = label
            break

    # ---- Hardware estimates ----
    ram_gb, vram_gb = _est_memory(params, quant)

    # ---- Dual tier ----
    tier_eng, tier_gen = _dual_tier(name_lower, params, quant)

    return {"params": params, "quant": quant, "ram_gb": ram_gb,
            "vram_gb": vram_gb, "tier_eng": tier_eng, "tier_gen": tier_gen}


def _est_params(size_str):
    """Estimate param count from file size. Q4: ~0.55 GB per 1B params."""
    try:
        parts = size_str.strip().split()
        val = float(parts[0])
        unit = parts[1].upper()
        gb = val if unit == "GB" else val / 1024
        est = round(gb / 0.55)
        snaps = [1,2,3,7,8,13,14,32,34,70,72]
        return min(snaps, key=lambda x: abs(x - est))
    except Exception:
        return 7


def _est_memory(params, quant):
    """Estimate RAM (CPU) and VRAM (GPU) in GB."""
    bpp = {"Q2":0.3,"Q3":0.4,"Q4":0.55,"Q5":0.65,"Q6":0.75,"Q8":1.0,"FP16":2.0}
    base = params * bpp.get(quant, 0.55)
    return round(base + 2.0, 1), round(base + 1.5, 1)


def _dual_tier(name_lower, params, quant):
    """
    Assign dual capability tiers based on benchmarks.

    tier_eng: Engineering/STEM score
    tier_gen: General knowledge score

    Components:
      1. Family bonus (from benchmark data)
      2. Parameter count score (bigger = more capable)
      3. Quantization bonus (higher quant = better output quality)
    """
    # Family scores
    eng_bonus = 12
    gen_bonus = 12
    for pat, eb, gb in _OFFLINE_FAMILY_SCORES:
        if pat in name_lower:
            eng_bonus = eb
            gen_bonus = gb
            break

    # Parameter count score
    if params >= 70:
        ps = 50
    elif params >= 32:
        ps = 40
    elif params >= 13:
        ps = 30
    elif params >= 7:
        ps = 20
    elif params >= 3:
        ps = 10
    else:
        ps = 5

    # Quantization bonus (applies equally to both)
    qb = {"Q2": 0, "Q3": 0, "Q4": 0, "Q5": 2, "Q6": 3, "Q8": 4, "FP16": 5}
    quant_bonus = qb.get(quant, 0)

    return eng_bonus + ps + quant_bonus, gen_bonus + ps + quant_bonus


# ============================================================================
# OFFLINE: Get models with specs
# ============================================================================

def get_offline_models_with_specs():
    """Get installed Ollama models sorted by tier_eng (best first)."""
    try:
        result = subprocess.run(["ollama","list"], capture_output=True,
                                text=True, timeout=10)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []
        models = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                name = parts[0]
                size_str = "unknown"
                for i, p in enumerate(parts):
                    if p in ("GB","MB","KB") and i > 0:
                        size_str = parts[i-1] + " " + p
                        break
                specs = parse_model_specs(name, size_str)
                models.append({"name": name, "size": size_str, **specs})
        models.sort(key=lambda m: m["tier_eng"], reverse=True)
        return models
    except Exception:
        return []


# ============================================================================
# ONLINE: Name-pattern scorer for models NOT in knowledge base
# ============================================================================
# When an online model isn't in KNOWN_MODELS, we use the model ID string
# to estimate whether it leans Engineering/STEM or General Knowledge.
#
# Same logic as the offline family scores, applied to API model IDs.
# Provider prefixes like "openai/" are stripped before matching.
#
# Format: (pattern, eng_offset, gen_offset)
#   Positive offset = stronger in that domain vs the base price score
#   Negative offset = weaker in that domain
# ============================================================================

_ONLINE_FAMILY_PATTERNS = [
    # Code-specialized models: strong ENG, weak GEN
    ("coder",      +12, -8),
    ("code",       +10, -6),
    ("codestral",  +12, -8),

    # Math-specialized: strong ENG, weak GEN
    ("math",       +12, -10),

    # Reasoning models: strong ENG, moderate GEN
    ("-r1",        +8,  -3),
    ("reason",     +6,  -2),

    # Qwen family: STEM-heavy training
    ("qwen",       +6,  -4),

    # DeepSeek: strong reasoning
    ("deepseek",   +5,  -2),

    # Llama family: balanced, slight GEN advantage
    ("llama",      -2,  +3),

    # Gemma: balanced, slight GEN advantage
    ("gemma",      -1,  +2),

    # Mistral/Mixtral: balanced, slight GEN advantage
    ("mistral",    +0,  +2),
    ("mixtral",    +2,  +1),

    # Claude: balanced, slight GEN advantage
    ("claude",     +0,  +2),

    # GPT: balanced to slight GEN advantage
    ("gpt",        +0,  +2),

    # Phi: STEM-heavy
    ("phi",        +5,  -3),

    # Vision/multimodal: broader training gives GEN advantage
    ("vision",     -5,  +6),
    ("-vl",        -5,  +6),

    # Chat/instruct: slight GEN advantage
    ("chat",       -1,  +2),
    ("instruct",   +0,  +1),

    # Creative/writing models
    ("creative",   -5,  +8),
    ("writing",    -5,  +8),
    ("story",      -8,  +10),

    # RP (role-play) models: GEN-heavy
    ("rp",         -8,  +6),
    ("roleplay",   -8,  +6),
]


def _estimate_online_dual_tier(model_id, base_score):
    """
    Estimate ENG and GEN tiers for an online model not in the knowledge base.

    Uses the model ID string to detect family patterns and apply
    directional offsets from the base price-derived score.

    Args:
        model_id: Full model ID (e.g., "qwen/qwen2.5-coder-32b-instruct")
        base_score: Price-derived base tier (0-95)

    Returns:
        (tier_eng, tier_gen) tuple
    """
    name_lower = model_id.lower()
    # Strip provider prefix for matching
    if "/" in name_lower:
        name_lower = name_lower.split("/", 1)[-1]

    eng_offset = 0
    gen_offset = 0

    # Apply ALL matching patterns (not just first match)
    # A model like "qwen2.5-coder-32b-instruct" should get both
    # the "qwen" bonus and the "coder" bonus
    for pattern, e_off, g_off in _ONLINE_FAMILY_PATTERNS:
        if pattern in name_lower:
            eng_offset += e_off
            gen_offset += g_off

    tier_eng = max(5, min(99, base_score + eng_offset))
    tier_gen = max(5, min(99, base_score + gen_offset))

    return tier_eng, tier_gen


# ============================================================================
# ONLINE: Fetch + knowledge base fallback
# ============================================================================

def fetch_online_models_with_meta(endpoint, api_key):
    """
    Fetch models with metadata. Three-layer fallback.
    Returns (models_by_provider, total_count).
    Each model has tier_eng and tier_gen.
    """
    if not api_key or not endpoint:
        return {}, 0

    api_models = _try_api(endpoint, api_key)

    if api_models:
        by_provider = {}
        for m in api_models:
            mid = m.get("id", "")
            if not mid:
                continue
            provider = mid.split("/")[0] if "/" in mid else "other"
            ctx = m.get("context_length", 0) or 0
            pricing = m.get("pricing", {}) or {}
            try:
                pi = float(pricing.get("prompt", "0") or "0") * 1_000_000
            except (ValueError, TypeError):
                pi = 0.0
            try:
                po = float(pricing.get("completion", "0") or "0") * 1_000_000
            except (ValueError, TypeError):
                po = 0.0

            # Try knowledge base for tier scores
            kb = lookup_known_model(mid)
            if kb:
                tier_eng = kb["tier_eng"]
                tier_gen = kb["tier_gen"]
                if ctx == 0:
                    ctx = kb["ctx"]
                if pi == 0:
                    pi = kb["price_in"]
                if po == 0:
                    po = kb["price_out"]
            else:
                # Not in KB -- estimate from pricing + name patterns
                base = int(min(po * 2 + pi + (ctx / 10000), 95))
                tier_eng, tier_gen = _estimate_online_dual_tier(mid, base)

            info = {"id": mid, "ctx": ctx, "price_in": round(pi, 2),
                    "price_out": round(po, 2), "tier_eng": tier_eng,
                    "tier_gen": tier_gen, "source": "api"}

            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(info)

        for p in by_provider:
            by_provider[p].sort(key=lambda m: m["tier_eng"], reverse=True)

        total = sum(len(v) for v in by_provider.values())
        return by_provider, total

    return {}, 0


def _try_api(endpoint, api_key):
    """Try GET /models. Returns list or None. Never crashes."""
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                endpoint.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_context_length(ctx):
    """128000 -> '128K', 1048576 -> '1M', 0 -> '---'."""
    if ctx == 0:
        return "---"
    if ctx >= 1_000_000:
        return f"{ctx / 1_000_000:.0f}M"
    if ctx >= 1000:
        return f"{ctx // 1000}K"
    return str(ctx)

def format_price(price_per_million):
    """3.0 -> '$3.00', 0 -> '---'."""
    if price_per_million == 0:
        return "---"
    if price_per_million < 0.01:
        return f"${price_per_million:.4f}"
    return f"${price_per_million:.2f}"
