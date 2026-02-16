#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Model Selection Wizard (scripts/_set_model.py)
# ============================================================================
#
# FLOW:
#   [auto-detect ALL models -- Ollama + API -- before any prompts]
#   Prompt 1: "Which mode?"    -> O = Offline / A = Online API
#             Shows detected models in each mode so you know what's there
#   Prompt 2: "Which use case?" -> S/E/Y/D/L/P/G (7 roles)
#             Only the selected use case's rank column is shown
#   Prompt 3: "Which model?"   -> Pick by number, search, or M = show more
#
# USE CASES (ranked by weighted blend of Engineering + General scores):
#   S = Software Engineering   (90% eng, 10% gen)
#   E = Engineering / STEM     (80% eng, 20% gen)
#   P = Program Management     (25% eng, 75% gen)
#   Y = Systems Administration (70% eng, 30% gen)
#   L = Logistics Analyst      (60% eng, 40% gen)
#   D = Drafting / AutoCAD     (75% eng, 25% gen)
#   G = General AI             (10% eng, 90% gen)
#
# COLUMNS:
#   Offline:  #, Model, Size, RAM, VRAM, Rank, Price/Question=Free
#   Online:   #, Model, Ctx, Rank, In/1M, Out/1M, Price/Question
#
# COMMAND: rag-set-model
# ============================================================================

import os
import sys
import yaml

sys.path.insert(0, os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."))
sys.path.insert(0, os.path.join(
    os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."), "scripts"
))

from _model_meta import (
    get_offline_models_with_specs,
    fetch_online_models_with_meta,
    get_known_model_info,
    format_context_length,
    format_price,
    USE_CASES,
    _LETTER_TO_UC,
    use_case_score,
)

TI = 1000   # tokens in per Q&A
TO = 500    # tokens out per Q&A
QB = 1      # per-question pricing
W = 92      # display width
PAGE = 20   # default page size


def est_cost(pi, po):
    if pi == 0 and po == 0: return None
    return ((TI * pi) + (TO * po)) / 1_000_000 * QB

def fmt_cost(c):
    if c is None: return "---"
    if c < 0.0001: return "<$0.001"
    if c < 0.10:   return f"${c:.4f}"
    if c < 100.0:  return f"${c:.2f}"
    return f"${c:.0f}"

def ask(prompt):
    try:
        sys.stdout.flush()
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Cancelled -- no changes made.\n")
        return None

def banner(text):
    print()
    print(f"  +{'-' * (W - 2)}+")
    print(f"  |{text:^{W - 2}}|")
    print(f"  +{'-' * (W - 2)}+")
    print()

def divider():
    print(f"  {'.' * W}")


def _resolve_creds():
    api_key = endpoint = None
    try:
        from src.security.credentials import get_api_key, get_api_endpoint
        api_key = get_api_key()
        endpoint = get_api_endpoint()
    except ImportError:
        pass
    if not api_key:
        for v in ["HYBRIDRAG_API_KEY", "AZURE_OPENAI_API_KEY",
                   "AZURE_OPEN_AI_KEY", "OPENAI_API_KEY"]:
            val = os.environ.get(v, "").strip()
            if val: api_key = val; break
    if not endpoint:
        for v in ["HYBRIDRAG_API_ENDPOINT", "AZURE_OPENAI_ENDPOINT",
                   "OPENAI_API_ENDPOINT", "OPENAI_BASE_URL"]:
            val = os.environ.get(v, "").strip()
            if val: endpoint = val; break
    if not endpoint:
        try:
            with open("config/default_config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            endpoint = cfg.get("api", {}).get("endpoint", "")
        except Exception:
            pass
    return api_key, endpoint


def _short_name(full_name):
    """qwen2.5:7b-instruct-q5_K_M -> Qwen2.5-7B"""
    name = full_name.split(":")[0] if ":latest" in full_name else full_name
    name = name.replace(":", " ").replace("-", " ")
    parts = name.split()
    family = parts[0].capitalize() if parts else full_name
    size = ""
    for p in parts:
        if p.lower().endswith("b") and p[:-1].replace(".", "").isdigit():
            size = f"-{p.upper()}"
            break
    return f"{family}{size}"


# ============================================================================
# AUTO-DETECT (runs before any prompts)
# ============================================================================

def scan_all():
    results = {
        "offline_models": None, "offline_names": [],
        "online_models": None, "online_endpoint": None,
        "online_count": 0, "online_status": "none",
    }

    offline = get_offline_models_with_specs()
    if offline:
        results["offline_models"] = offline
        results["offline_names"] = [_short_name(m["name"]) for m in offline]

    api_key, endpoint = _resolve_creds()
    results["online_endpoint"] = endpoint

    if not api_key:
        results["online_status"] = "no_key"
    elif not endpoint:
        results["online_status"] = "no_endpoint"
    else:
        by_provider, total = fetch_online_models_with_meta(endpoint, api_key)
        if by_provider and total > 0:
            flat = [m for models in by_provider.values() for m in models]
            results["online_models"] = flat
            results["online_count"] = total
            results["online_status"] = "ok"
        else:
            results["online_models"] = "fallback"
            results["online_status"] = "fallback"

    return results


# ============================================================================
# PROMPT 1: Which mode?
# ============================================================================

def prompt_mode(scan):
    banner("MODEL SELECTION WIZARD")

    print("  Scanning for available models ...")
    print()

    off = scan["offline_models"]
    off_names = scan["offline_names"]
    has_off = off is not None and len(off) > 0

    on_status = scan["online_status"]
    on_count = scan["online_count"]
    on_ep = scan["online_endpoint"] or "(not configured)"
    has_on = on_status in ("ok", "fallback")

    print("  Which mode do you want to run?")
    print()

    if has_off:
        nlist = ", ".join(off_names[:6])
        if len(off_names) > 6:
            nlist += f" +{len(off_names) - 6} more"
        print(f"      O  =  Offline AI  ({len(off)} model(s) detected)")
        print(f"            Local via Ollama -- private, no internet, uses your GPU")
        print(f"            Available: {nlist}")
    else:
        print(f"      O  =  Offline AI  (no models detected)")
        print(f"            Install: https://ollama.com   then: ollama pull llama3")
    print()

    if on_status == "ok":
        print(f"      A  =  Online AI   ({on_count} model(s) available)")
        print(f"            Via API -- faster, more powerful, costs per query")
        print(f"            Endpoint: {on_ep}")
    elif on_status == "fallback":
        print(f"      A  =  Online AI   (endpoint configured)")
        print(f"            Model list unavailable -- will use built-in library")
        print(f"            Endpoint: {on_ep}")
    elif on_status == "no_key":
        print(f"      A  =  Online AI   (not configured -- run: rag-store-key)")
    else:
        print(f"      A  =  Online AI   (not configured -- run: rag-store-endpoint)")
    print()

    if not has_off and not has_on:
        print("  [!] Neither mode is ready. Set up Ollama or an API key first.")
        print()
        return None

    while True:
        r = ask("  >>> Enter O or A:  ")
        if r is None: return None
        ru = r.upper()
        if ru in ("O", "OFFLINE"):
            if not has_off:
                print("\n      No offline models available. Install Ollama first.\n")
                continue
            return "offline"
        if ru in ("A", "ONLINE", "API"):
            if not has_on:
                print(f"\n      Online not configured. Run: rag-store-key\n")
                continue
            return "online"
        print("\n      Type O for Offline or A for Online API.\n")


# ============================================================================
# PROMPT 2: Which use case?
# ============================================================================

def prompt_use_case():
    divider()
    print()
    print("  What will you primarily use this AI for?")
    print()

    # Display order (matches user preference)
    order = ["sw", "eng", "pm", "sys", "log", "draft", "gen"]
    for key in order:
        uc = USE_CASES[key]
        print(f"      {uc['letter']}  =  {uc['label']}")
        print(f"            {uc['desc']}")
        print()

    valid_letters = [USE_CASES[k]["letter"] for k in order]
    hint = "/".join(valid_letters)

    while True:
        r = ask(f"  >>> Enter {hint}:  ")
        if r is None: return None
        ru = r.upper()
        if ru in _LETTER_TO_UC:
            return _LETTER_TO_UC[ru]
        # Also accept full words
        for key in order:
            uc = USE_CASES[key]
            if ru == key.upper() or ru in uc["label"].upper():
                return key
        print(f"\n      Please enter one of: {hint}\n")


# ============================================================================
# PROMPT 3a: Pick offline model
# ============================================================================

def prompt_pick_offline(models, uc_key, current_model):
    uc = USE_CASES[uc_key]
    label = uc["label"]

    # Compute use-case score for each model and sort
    for m in models:
        m["_score"] = use_case_score(m["tier_eng"], m["tier_gen"], uc_key)
    models.sort(key=lambda m: m["_score"], reverse=True)

    divider()
    print()
    print(f"  Models ranked for: {label}")
    print()

    print(f"  {'#':>4}   {'Model':<32}  {'Size':>6}"
          f"   {'RAM':>5}  {'VRAM':>5}"
          f"   {'Rank':>4}  {'Price/':>8}")
    print(f"  {'':>4}   {'':32}  {'':6}"
          f"   {'':5}  {'':5}"
          f"   {'':4}  {'Question':>8}")
    print(f"  {'----':>4}   {'--------------------------------':<32}  {'------':>6}"
          f"   {'-----':>5}  {'-----':>5}"
          f"   {'----':>4}  {'--------':>8}")

    for i, m in enumerate(models, 1):
        tag = "  << active" if m["name"] == current_model else ""
        name = m["name"][:30]
        ram = f"{m['ram_gb']:.0f}GB"
        vram = f"{m['vram_gb']:.0f}GB"
        print(f"  {i:>4}   {name:<32}  {m['size']:>6}"
              f"   {ram:>5}  {vram:>5}"
              f"   {m['_score']:>4}  {'Free':>8}{tag}")

    print()
    print(f"  RAM  = memory for CPU inference (slower)")
    print(f"  VRAM = GPU memory for full offload (fastest)")

    if len(models) == 1:
        print(f"\n  Only one model -- auto-selecting: {models[0]['name']}")
        return models[0]["name"]

    print()
    while True:
        r = ask(f"  >>> Enter model number [1-{len(models)}]:  ")
        if r is None: return None
        try:
            idx = int(r)
            if 1 <= idx <= len(models):
                return models[idx - 1]["name"]
        except ValueError:
            pass
        print(f"      Please enter a number between 1 and {len(models)}.\n")


# ============================================================================
# PROMPT 3b: Pick online model (API list)
# ============================================================================

def _display_online_table(models, uc_key, current_model, start, count):
    """Print a page of online models. Returns how many were shown."""
    shown = 0
    end = min(start + count, len(models))
    for i in range(start, end):
        m = models[i]
        tag = " <<" if m["id"] == current_model else ""
        name = m["id"][:34]
        ctx = format_context_length(m["ctx"])
        pi = format_price(m["price_in"])
        po = format_price(m["price_out"])
        cost = est_cost(m["price_in"], m["price_out"])
        fc = fmt_cost(cost)
        print(f"  {i+1:>4}   {name:<34}  {ctx:>5}"
              f"   {m['_score']:>4}"
              f"   {pi:>7}  {po:>7}  {fc:>8}{tag}")
        shown += 1
    return shown


def prompt_pick_online(models, uc_key, current_model):
    uc = USE_CASES[uc_key]
    label = uc["label"]

    for m in models:
        m["_score"] = use_case_score(m["tier_eng"], m["tier_gen"], uc_key)
    models.sort(key=lambda m: m["_score"], reverse=True)

    # Bump current to top
    for i, m in enumerate(models):
        if m["id"] == current_model and i > 0:
            models.insert(0, models.pop(i))
            break

    total = len(models)
    showing_all = total <= PAGE

    divider()
    print()
    print(f"  {total} models ranked for: {label}")
    print()

    # Header
    print(f"  {'#':>4}   {'Model':<34}  {'Ctx':>5}"
          f"   {'Rank':>4}"
          f"   {'In/1M':>7}  {'Out/1M':>7}  {'Price/':>8}")
    print(f"  {'':>4}   {'':34}  {'':5}"
          f"   {'':4}"
          f"   {'':7}  {'':7}  {'Question':>8}")
    print(f"  {'----':>4}   {'----------------------------------':<34}  {'-----':>5}"
          f"   {'----':>4}"
          f"   {'-------':>7}  {'-------':>7}  {'--------':>8}")

    if showing_all:
        _display_online_table(models, uc_key, current_model, 0, total)
    else:
        _display_online_table(models, uc_key, current_model, 0, PAGE)
        print()
        print(f"         Showing top {PAGE} of {total}.")

    print()
    print(f"  Price/Question = estimated cost per question")
    print(f"                   ({TI} tokens in + {TO} tokens out)")
    print()

    if not showing_all:
        print(f"  Enter a number to select, type to search, or M = show all {total}")
    else:
        print(f"  Enter a number to select, or type to search")
    print()

    expanded = showing_all

    while True:
        r = ask(f"  >>> Enter number [1-{total}], search, or M:  ")
        if r is None: return None
        if not r: return current_model if current_model else None

        # Show all
        if r.upper() == "M" and not expanded:
            expanded = True
            print()
            print(f"  Showing all {total} models:")
            print()
            print(f"  {'#':>4}   {'Model':<34}  {'Ctx':>5}"
                  f"   {'Rank':>4}"
                  f"   {'In/1M':>7}  {'Out/1M':>7}  {'Price/':>8}")
            print(f"  {'':>4}   {'':34}  {'':5}"
                  f"   {'':4}"
                  f"   {'':7}  {'':7}  {'Question':>8}")
            print(f"  {'----':>4}   {'----------------------------------':<34}  {'-----':>5}"
                  f"   {'----':>4}"
                  f"   {'-------':>7}  {'-------':>7}  {'--------':>8}")
            _display_online_table(models, uc_key, current_model, 0, total)
            print()
            continue

        # Number
        try:
            idx = int(r)
            if 1 <= idx <= total:
                return models[idx - 1]["id"]
            print(f"      Number must be between 1 and {total}.\n")
            continue
        except ValueError:
            pass

        # Search
        hits = [m for m in models if r.lower() in m["id"].lower()]
        if not hits:
            print(f"      No models matching '{r}'.\n")
            continue

        if len(hits) == 1:
            m = hits[0]
            cost = est_cost(m["price_in"], m["price_out"])
            fc = fmt_cost(cost)
            print(f"\n      Found: {m['id']}")
            print(f"      Rank={m['_score']}  Price/Question={fc}\n")
            c = ask("      Use this model? (y/n):  ")
            if c and c.lower() in ("y", "yes"): return m["id"]
            print()
            continue

        s = min(10, len(hits))
        print(f"\n      {len(hits)} matches:\n")
        for i, m in enumerate(hits[:s], 1):
            cost = est_cost(m["price_in"], m["price_out"])
            fc = fmt_cost(cost)
            tag = " <<" if m["id"] == current_model else ""
            print(f"        [{i}]  {m['id'][:42]}   Rank={m['_score']}  {fc}{tag}")
        if len(hits) > s:
            print(f"        ... +{len(hits) - s} more")
        print()

        pick = ask(f"      Enter number [1-{s}]:  ")
        if pick is None: return None
        try:
            pi = int(pick)
            if 1 <= pi <= s: return hits[pi - 1]["id"]
        except ValueError:
            pass
        print()


# ============================================================================
# PROMPT 3c: Pick online (fallback -- proxy blocked API)
# ============================================================================

def prompt_pick_fallback(uc_key, current_model, endpoint):
    uc = USE_CASES[uc_key]
    label = uc["label"]

    divider()
    print()

    kb = get_known_model_info(current_model)
    if kb:
        score = use_case_score(kb["tier_eng"], kb["tier_gen"], uc_key)
        ctx = format_context_length(kb["ctx"])
        pi = format_price(kb["price_in"])
        po = format_price(kb["price_out"])
        cost = est_cost(kb["price_in"], kb["price_out"])
        fc = fmt_cost(cost)
        print(f"  Current model:  {current_model}")
        print()
        print(f"      Family:       {kb['family']}")
        print(f"      Context:      {ctx}")
        print(f"      Pricing:      {pi} in  /  {po} out  per 1M tokens")
        print(f"      Price/Question:   {fc}")
        print(f"      {label} Rank:  {score}")
        if kb["note"]:
            print(f"      Note:         {kb['note']}")
    else:
        print(f"  Current model:  {current_model}")
        print(f"      (Not in model library)")

    print()
    print(f"  Type a model name to switch (e.g. gpt-4o, gpt-4.1, o3)")
    print(f"  Or press Enter to keep the current model.")
    print()

    r = ask("  >>> Enter model name:  ")
    if r is None: return None
    if not r: return current_model

    info = get_known_model_info(r)
    if info:
        score = use_case_score(info["tier_eng"], info["tier_gen"], uc_key)
        ctx = format_context_length(info["ctx"])
        cost = est_cost(info["price_in"], info["price_out"])
        fc = fmt_cost(cost)
        print(f"\n      {r}:")
        print(f"      {info['family']}   {ctx} ctx   Rank={score}   Price/Question={fc}")
    else:
        print(f"\n      {r}: Not in model library (may still work)")

    print()
    c = ask("      Use this model? (y/n):  ")
    if c and c.lower() in ("y", "yes"): return r
    return current_model


# ============================================================================
# MAIN
# ============================================================================

def main():
    try:
        with open("config/default_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("  [FAIL] config/default_config.yaml not found")
        sys.exit(1)

    # Auto-detect everything
    scan = scan_all()

    # Prompt 1: mode
    mode = prompt_mode(scan)
    if mode is None: return

    # Prompt 2: use case
    uc_key = prompt_use_case()
    if uc_key is None: return

    # Prompt 3: pick model
    if mode == "offline":
        current = cfg.get("ollama", {}).get("model", "llama3")
        chosen = prompt_pick_offline(scan["offline_models"], uc_key, current)
    else:
        current = cfg.get("api", {}).get("model", "gpt-3.5-turbo")
        if scan["online_models"] == "fallback":
            chosen = prompt_pick_fallback(
                uc_key, current, scan["online_endpoint"])
        else:
            chosen = prompt_pick_online(scan["online_models"], uc_key, current)

    if chosen is None: return

    # Save
    cfg["mode"] = mode
    if mode == "offline":
        if "ollama" not in cfg: cfg["ollama"] = {}
        cfg["ollama"]["model"] = chosen
    else:
        if "api" not in cfg: cfg["api"] = {}
        cfg["api"]["model"] = chosen

    with open("config/default_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Confirmation
    uc = USE_CASES[uc_key]
    banner("SELECTION COMPLETE")

    if mode == "offline":
        from _model_meta import parse_model_specs
        specs = parse_model_specs(chosen, "0 GB")
        score = use_case_score(specs["tier_eng"], specs["tier_gen"], uc_key)
        print(f"      Mode:       Offline (local)")
        print(f"      Model:      {chosen}")
        print(f"      Use case:   {uc['label']}")
        print(f"      Rank:       {score}")
        print(f"      Cost:       Free")
    else:
        info = get_known_model_info(chosen)
        print(f"      Mode:       Online (API)")
        print(f"      Model:      {chosen}")
        print(f"      Use case:   {uc['label']}")
        if info:
            score = use_case_score(info["tier_eng"], info["tier_gen"], uc_key)
            ctx = format_context_length(info["ctx"])
            cost = est_cost(info["price_in"], info["price_out"])
            fc = fmt_cost(cost)
            print(f"      Rank:       {score}")
            print(f"      Context:    {ctx}")
            print(f"      Price/Question: {fc}")

    print()
    print(f"      Saved to: config/default_config.yaml")
    print()


if __name__ == "__main__":
    main()
