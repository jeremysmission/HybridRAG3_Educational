#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Set Offline AI Mode (scripts/_set_offline.py)
# ============================================================================
#
# Shows ranked models with dual ENG/GEN scores and hardware specs.
# INTERNET ACCESS: NONE (localhost:11434 only)
# ============================================================================

import os, sys, yaml
sys.path.insert(0, os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."))
sys.path.insert(0, os.path.join(
    os.environ.get("HYBRIDRAG_PROJECT_ROOT", "."), "scripts"))

from _model_meta import get_offline_models_with_specs


def prompt_model_choice(models, current_model):
    """Show ranked models with hardware specs and dual tier. Returns name or None."""
    print()
    print("  Offline AI Models Available:")
    print("  ENG = Engineering/STEM   GEN = General Knowledge")
    print("  " + "-" * 75)
    print(f"    {'#':>3}  {'Model':<38} {'Size':>6}  {'RAM':>5} {'VRAM':>5}  {'ENG':>3} {'GEN':>3}")
    print("  " + "-" * 75)

    for rank, m in enumerate(models, 1):
        marker = "  << current" if m["name"] == current_model else ""
        name = m["name"][:36]
        ram = f"{m['ram_gb']:.0f}GB"
        vram = f"{m['vram_gb']:.0f}GB"
        print(f"    {rank:>3}. {name:<38} {m['size']:>6}  {ram:>5} {vram:>5}"
              f"  {m['tier_eng']:>3} {m['tier_gen']:>3}{marker}")

    print()
    print("    RAM  = system memory for CPU inference (slower)")
    print("    VRAM = GPU memory for full GPU offload (fastest)")
    print()
    print("    To add models:    ollama pull <model_name>")
    print("    To remove models: ollama rm <model_name>")
    print()

    while True:
        try:
            raw = input(f"  Select model [1-{len(models)}]: ").strip()
            if not raw:
                print("  Cancelled -- mode unchanged."); return None
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]["name"]
            print(f"  Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("  Please enter a number")
        except KeyboardInterrupt:
            print("\n  Cancelled -- mode unchanged."); return None
        except EOFError:
            return None


def main():
    try:
        with open("config/default_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("  [FAIL] config/default_config.yaml not found"); sys.exit(1)

    current_model = cfg.get("ollama", {}).get("model", "llama3")
    models = get_offline_models_with_specs()

    if not models:
        print("\n  [WARN] No Ollama models detected.")
        print("         Is Ollama installed? ollama pull llama3")
        cfg["mode"] = "offline"
        with open("config/default_config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        return

    if len(models) == 1:
        chosen = models[0]["name"]
        print(f"\n  Only one model installed: {chosen}")
        print(f"  ENG: {models[0]['tier_eng']}  GEN: {models[0]['tier_gen']}"
              f"  RAM: {models[0]['ram_gb']:.0f}GB  VRAM: {models[0]['vram_gb']:.0f}GB")
        print(f"  Auto-selected.")
    else:
        chosen = prompt_model_choice(models, current_model)
        if chosen is None:
            return

    cfg["mode"] = "offline"
    if "ollama" not in cfg:
        cfg["ollama"] = {}
    cfg["ollama"]["model"] = chosen

    with open("config/default_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    specs = next((m for m in models if m["name"] == chosen), None)
    print()
    print(f"  [OK] Mode set to: offline")
    print(f"       Model: {chosen}")
    if specs:
        print(f"       Params: {specs['params']}B ({specs['quant']})")
        print(f"       ENG: {specs['tier_eng']}  GEN: {specs['tier_gen']}")
        print(f"       RAM: {specs['ram_gb']:.0f} GB   VRAM: {specs['vram_gb']:.0f} GB")


if __name__ == "__main__":
    main()
