#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Set Online API Mode (scripts/_set_online.py)
# ============================================================================
#
# Shows ranked models with dual ENG/GEN scores, pricing, and context.
# Falls back to knowledge base if API /models unavailable (Azure/proxy).
# INTERNET ACCESS: YES (one GET to /models, fails gracefully)
# ============================================================================

import os, sys, yaml
sys.path.insert(0, os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."))
sys.path.insert(0, os.path.join(
    os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."), "scripts"))

from _model_meta import (
    fetch_online_models_with_meta, get_known_model_info,
    format_context_length, format_price,
)


def resolve_api_credentials():
    api_key = endpoint = None
    try:
        from src.security.credentials import get_api_key, get_api_endpoint
        api_key = get_api_key()
        endpoint = get_api_endpoint()
    except ImportError:
        pass
    if not api_key:
        for v in ["HYBRIDRAG_API_KEY","AZURE_OPENAI_API_KEY",
                   "AZURE_OPEN_AI_KEY","OPENAI_API_KEY"]:
            val = os.environ.get(v, "").strip()
            if val: api_key = val; break
    if not endpoint:
        for v in ["HYBRIDRAG_API_ENDPOINT","AZURE_OPENAI_ENDPOINT",
                   "OPENAI_API_ENDPOINT","OPENAI_BASE_URL"]:
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


def prompt_model_choice_rich(by_provider, current_model, page_size=20):
    """Show ranked models with dual tier + pricing. Supports search."""
    all_models = []
    for models in by_provider.values():
        all_models.extend(models)
    all_models.sort(key=lambda m: m["tier_eng"], reverse=True)

    display = list(all_models)
    for i, m in enumerate(display):
        if m["id"] == current_model and i > 0:
            display.insert(0, display.pop(i))
            break

    total = len(display)
    show = min(page_size, total)

    print()
    print(f"  Online API Models ({total} available):")
    print("  ENG = Engineering/STEM   GEN = General Knowledge")
    print("  " + "-" * 78)
    print(f"    {'#':>4}  {'Model':<36} {'Ctx':>5} {'In/1M':>7} {'Out/1M':>7} {'ENG':>3} {'GEN':>3}")
    print("  " + "-" * 78)

    for i in range(show):
        m = display[i]
        marker = " <<" if m["id"] == current_model else ""
        name = m["id"][:34]
        ctx = format_context_length(m["ctx"])
        pi = format_price(m["price_in"])
        po = format_price(m["price_out"])
        print(f"    {i+1:>4}. {name:<36} {ctx:>5} {pi:>7} {po:>7}"
              f" {m['tier_eng']:>3} {m['tier_gen']:>3}{marker}")

    if total > page_size:
        print(f"    ... and {total - page_size} more")

    print()
    print(f"  Number [1-{total}] | Search (e.g. 'llama','gpt') | Enter = keep current")
    print()

    while True:
        try:
            raw = input("  Select: ").strip()
            if not raw:
                return current_model if current_model else None

            try:
                idx = int(raw)
                if 1 <= idx <= total:
                    return display[idx - 1]["id"]
                print(f"  Number must be 1-{total}")
                continue
            except ValueError:
                pass

            matches = [m for m in all_models if raw.lower() in m["id"].lower()]
            if not matches:
                print(f"  No models matching '{raw}'"); continue

            if len(matches) == 1:
                m = matches[0]
                ctx = format_context_length(m["ctx"])
                print(f"  Found: {m['id']}  ({ctx} ctx, ENG:{m['tier_eng']} GEN:{m['tier_gen']})")
                c = input("  Use this model? (y/n): ").strip().lower()
                if c in ("y","yes",""):
                    return m["id"]
                continue

            print(f"  Found {len(matches)} matches:")
            s = min(15, len(matches))
            for i, m in enumerate(matches[:s], 1):
                ctx = format_context_length(m["ctx"])
                marker = " <<" if m["id"] == current_model else ""
                print(f"    [{i}] {m['id'][:42]}  {ctx} ENG:{m['tier_eng']} GEN:{m['tier_gen']}{marker}")
            if len(matches) > s:
                print(f"    ... +{len(matches)-s} more")
            pick = input(f"  Select [1-{s}]: ").strip()
            try:
                pi = int(pick)
                if 1 <= pi <= s:
                    return matches[pi-1]["id"]
            except ValueError:
                pass

        except KeyboardInterrupt:
            print("\n  Cancelled."); return None
        except EOFError:
            return current_model if current_model else None


def prompt_model_choice_fallback(current_model, endpoint):
    """Fallback when API unavailable. Shows KB info, allows typing model name."""
    print()
    print("  Could not query model list from API.")
    print(f"  (Endpoint: {endpoint})")
    print()

    kb = get_known_model_info(current_model)
    if kb:
        ctx = format_context_length(kb["ctx"])
        pi = format_price(kb["price_in"])
        po = format_price(kb["price_out"])
        print(f"  Current model: {current_model}")
        print(f"    Family:  {kb['family']}   Context: {ctx}")
        print(f"    Pricing: {pi} in / {po} out per 1M tokens")
        print(f"    ENG: {kb['tier_eng']}   GEN: {kb['tier_gen']}")
        if kb["note"]:
            print(f"    Note:    {kb['note']}")
    else:
        print(f"  Current model: {current_model}")
        print(f"    No information available")

    print()
    print("  Type model name to switch, or Enter to keep current:")
    print()

    try:
        raw = input("  New model (or Enter): ").strip()
        if not raw:
            return current_model

        info = get_known_model_info(raw)
        if info:
            ctx = format_context_length(info["ctx"])
            print(f"\n  {raw}: {info['family']}, {ctx} ctx, "
                  f"ENG:{info['tier_eng']} GEN:{info['tier_gen']}")
        else:
            print(f"\n  {raw}: No information available (may still work)")

        c = input("  Use this model? (y/n): ").strip().lower()
        if c in ("y","yes",""):
            return raw
        return current_model
    except KeyboardInterrupt:
        print("\n  Cancelled."); return None
    except EOFError:
        return current_model


def main():
    try:
        with open("config/default_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("  [FAIL] config/default_config.yaml not found"); sys.exit(1)

    current_model = cfg.get("api", {}).get("model", "gpt-3.5-turbo")
    current_endpoint = cfg.get("api", {}).get("endpoint", "")

    api_key, endpoint = resolve_api_credentials()
    if not endpoint:
        endpoint = current_endpoint

    if not api_key:
        print("\n  [FAIL] No API key found. Run: rag-store-key")
        cfg["mode"] = "online"
        with open("config/default_config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        return

    if not endpoint:
        print("\n  [FAIL] No endpoint. Run: rag-store-endpoint"); sys.exit(1)

    print()
    print(f"  Querying models from: {endpoint}")

    by_provider, total = fetch_online_models_with_meta(endpoint, api_key)

    if by_provider and total > 0:
        chosen = prompt_model_choice_rich(by_provider, current_model)
    else:
        chosen = prompt_model_choice_fallback(current_model, endpoint)

    if chosen is None:
        return

    cfg["mode"] = "online"
    if "api" not in cfg:
        cfg["api"] = {}
    cfg["api"]["model"] = chosen

    with open("config/default_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print()
    print(f"  [OK] Mode set to: online")
    print(f"       Model: {chosen}")

    info = get_known_model_info(chosen)
    if info:
        ctx = format_context_length(info["ctx"])
        pi = format_price(info["price_in"])
        po = format_price(info["price_out"])
        print(f"       ENG: {info['tier_eng']}  GEN: {info['tier_gen']}  Context: {ctx}")
        print(f"       Pricing: {pi} in / {po} out per 1M tokens")

    print(f"       Endpoint: {endpoint}")


if __name__ == "__main__":
    main()
