# ============================================================================
# HybridRAG v3 -- Admin Menu (src/gui/panels/engineering_menu.py)
# ============================================================================
# Separate child window for tuning retrieval, LLM, profile, and model settings.
# Changes take effect immediately and are written to config.
#
# INTERNET ACCESS: Test query may use API if in online mode.
#   Shows "Network: ACTIVE" indicator when online test queries run.
# ============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import sys
import os
import threading
import logging

from src.gui.theme import current_theme, FONT, FONT_BOLD, FONT_SMALL, FONT_MONO, bind_hover

logger = logging.getLogger(__name__)


class EngineeringMenu(tk.Toplevel):
    """
    Admin settings window with sliders for retrieval and LLM tuning,
    profile/model ranking, and a test query runner.
    """

    def __init__(self, parent, config, query_engine=None):
        super().__init__(parent)
        t = current_theme()
        self.title("Admin Settings")
        self.geometry("620x780")
        self.resizable(True, True)
        self.config = config
        self.query_engine = query_engine
        self._app = parent  # HybridRAGApp -- used for reload_config / reset_backends
        self.configure(bg=t["bg"])

        # Store original values for reset
        self._original_values = self._capture_values()

        # Build sections
        self._build_retrieval_section()
        self._build_llm_section()
        self._build_profile_section()
        self._build_test_section()
        self._build_buttons()

    def _capture_values(self):
        """Capture current config values for reset."""
        retrieval = getattr(self.config, "retrieval", None)
        api = getattr(self.config, "api", None)
        return {
            "top_k": getattr(retrieval, "top_k", 8) if retrieval else 8,
            "min_score": getattr(retrieval, "min_score", 0.20) if retrieval else 0.20,
            "hybrid_search": getattr(retrieval, "hybrid_search", True) if retrieval else True,
            "reranker_enabled": getattr(retrieval, "reranker_enabled", False) if retrieval else False,
            "max_tokens": getattr(api, "max_tokens", 2048) if api else 2048,
            "temperature": getattr(api, "temperature", 0.1) if api else 0.1,
            "timeout_seconds": getattr(api, "timeout_seconds", 30) if api else 30,
        }

    # ----------------------------------------------------------------
    # RETRIEVAL SETTINGS
    # ----------------------------------------------------------------

    def _build_retrieval_section(self):
        """Build retrieval settings section."""
        t = current_theme()
        frame = tk.LabelFrame(self, text="Retrieval Settings", padx=16, pady=8,
                               bg=t["panel_bg"], fg=t["accent"],
                               font=FONT_BOLD)
        frame.pack(fill=tk.X, padx=16, pady=(8, 4))

        retrieval = getattr(self.config, "retrieval", None)

        # top_k slider
        row_tk = tk.Frame(frame, bg=t["panel_bg"])
        row_tk.pack(fill=tk.X, pady=4)
        tk.Label(row_tk, text="top_k:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.topk_var = tk.IntVar(
            value=getattr(retrieval, "top_k", 8) if retrieval else 8
        )
        self.topk_scale = tk.Scale(
            row_tk, from_=1, to=50, orient=tk.HORIZONTAL,
            variable=self.topk_var, command=lambda v: self._on_retrieval_change(),
            bg=t["panel_bg"], fg=t["fg"], troughcolor=t["input_bg"],
            highlightthickness=0, font=FONT,
        )
        self.topk_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # min_score slider
        row_ms = tk.Frame(frame, bg=t["panel_bg"])
        row_ms.pack(fill=tk.X, pady=4)
        tk.Label(row_ms, text="min_score:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.minscore_var = tk.DoubleVar(
            value=getattr(retrieval, "min_score", 0.20) if retrieval else 0.20
        )
        self.minscore_scale = tk.Scale(
            row_ms, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            variable=self.minscore_var, command=lambda v: self._on_retrieval_change(),
            bg=t["panel_bg"], fg=t["fg"], troughcolor=t["input_bg"],
            highlightthickness=0, font=FONT,
        )
        self.minscore_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Hybrid search toggle
        row_hs = tk.Frame(frame, bg=t["panel_bg"])
        row_hs.pack(fill=tk.X, pady=4)
        tk.Label(row_hs, text="Hybrid search:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.hybrid_var = tk.BooleanVar(
            value=getattr(retrieval, "hybrid_search", True) if retrieval else True
        )
        tk.Checkbutton(
            row_hs, variable=self.hybrid_var,
            command=self._on_retrieval_change,
            bg=t["panel_bg"], fg=t["fg"],
            selectcolor=t["input_bg"], activebackground=t["panel_bg"],
            activeforeground=t["fg"], font=FONT,
        ).pack(side=tk.LEFT)

        # Reranker toggle
        row_rr = tk.Frame(frame, bg=t["panel_bg"])
        row_rr.pack(fill=tk.X, pady=4)
        tk.Label(row_rr, text="Reranker:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.reranker_var = tk.BooleanVar(
            value=getattr(retrieval, "reranker_enabled", False) if retrieval else False
        )
        tk.Checkbutton(
            row_rr, variable=self.reranker_var,
            command=self._on_retrieval_change,
            bg=t["panel_bg"], fg=t["fg"],
            selectcolor=t["input_bg"], activebackground=t["panel_bg"],
            activeforeground=t["fg"], font=FONT,
        ).pack(side=tk.LEFT)

    def _on_retrieval_change(self):
        """Write retrieval settings to config immediately."""
        retrieval = getattr(self.config, "retrieval", None)
        if retrieval:
            retrieval.top_k = self.topk_var.get()
            retrieval.min_score = self.minscore_var.get()
            retrieval.hybrid_search = self.hybrid_var.get()
            retrieval.reranker_enabled = self.reranker_var.get()

    # ----------------------------------------------------------------
    # LLM SETTINGS
    # ----------------------------------------------------------------

    def _build_llm_section(self):
        """Build LLM settings section."""
        t = current_theme()
        frame = tk.LabelFrame(self, text="LLM Settings", padx=16, pady=8,
                               bg=t["panel_bg"], fg=t["accent"],
                               font=FONT_BOLD)
        frame.pack(fill=tk.X, padx=16, pady=8)

        api = getattr(self.config, "api", None)

        # Max tokens slider
        row_mt = tk.Frame(frame, bg=t["panel_bg"])
        row_mt.pack(fill=tk.X, pady=4)
        tk.Label(row_mt, text="Max tokens:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.maxtokens_var = tk.IntVar(
            value=getattr(api, "max_tokens", 2048) if api else 2048
        )
        self.maxtokens_scale = tk.Scale(
            row_mt, from_=256, to=4096, orient=tk.HORIZONTAL,
            variable=self.maxtokens_var, command=lambda v: self._on_llm_change(),
            bg=t["panel_bg"], fg=t["fg"], troughcolor=t["input_bg"],
            highlightthickness=0, font=FONT,
        )
        self.maxtokens_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Temperature slider
        row_temp = tk.Frame(frame, bg=t["panel_bg"])
        row_temp.pack(fill=tk.X, pady=4)
        tk.Label(row_temp, text="Temperature:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.temp_var = tk.DoubleVar(
            value=getattr(api, "temperature", 0.1) if api else 0.1
        )
        self.temp_scale = tk.Scale(
            row_temp, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
            variable=self.temp_var, command=lambda v: self._on_llm_change(),
            bg=t["panel_bg"], fg=t["fg"], troughcolor=t["input_bg"],
            highlightthickness=0, font=FONT,
        )
        self.temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Timeout slider
        row_to = tk.Frame(frame, bg=t["panel_bg"])
        row_to.pack(fill=tk.X, pady=4)
        tk.Label(row_to, text="Timeout (s):", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)
        self.timeout_var = tk.IntVar(
            value=getattr(api, "timeout_seconds", 30) if api else 30
        )
        self.timeout_scale = tk.Scale(
            row_to, from_=10, to=120, orient=tk.HORIZONTAL,
            variable=self.timeout_var, command=lambda v: self._on_llm_change(),
            bg=t["panel_bg"], fg=t["fg"], troughcolor=t["input_bg"],
            highlightthickness=0, font=FONT,
        )
        self.timeout_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _on_llm_change(self):
        """Write LLM settings to config immediately."""
        api = getattr(self.config, "api", None)
        if api:
            api.max_tokens = self.maxtokens_var.get()
            api.temperature = self.temp_var.get()
            api.timeout_seconds = self.timeout_var.get()

    # ----------------------------------------------------------------
    # PERFORMANCE PROFILE
    # ----------------------------------------------------------------

    def _build_profile_section(self):
        """Build performance profile section with ranked model table."""
        t = current_theme()
        frame = tk.LabelFrame(self, text="Profile & Model Ranking", padx=16, pady=8,
                               bg=t["panel_bg"], fg=t["accent"],
                               font=FONT_BOLD)
        frame.pack(fill=tk.X, padx=16, pady=8)

        # Profile selector row
        row = tk.Frame(frame, bg=t["panel_bg"])
        row.pack(fill=tk.X, pady=4)

        tk.Label(row, text="Profile:", width=14, anchor=tk.W,
                 bg=t["panel_bg"], fg=t["fg"], font=FONT).pack(side=tk.LEFT)

        # Read profile names from profiles.yaml (fall back to hardcoded)
        profile_names = self._load_profile_names()
        self.profile_var = tk.StringVar(value="laptop_safe")
        self.profile_dropdown = ttk.Combobox(
            row, textvariable=self.profile_var, values=profile_names,
            state="readonly", width=20, font=FONT,
        )
        self.profile_dropdown.pack(side=tk.LEFT, padx=(8, 0))

        self.profile_apply_btn = tk.Button(
            row, text="Apply", command=self._on_profile_change, width=8,
            bg=t["accent"], fg=t["accent_fg"], font=FONT,
            relief=tk.FLAT, bd=0, padx=6, pady=2,
            activebackground=t["accent_hover"],
            activeforeground=t["accent_fg"],
        )
        self.profile_apply_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Profile info line (shows embedder + LLM for current selection)
        self.profile_info_label = tk.Label(
            frame, text="", anchor=tk.W, fg=t["fg"],
            bg=t["panel_bg"], font=FONT_SMALL,
        )
        self.profile_info_label.pack(fill=tk.X, pady=(2, 0))

        self.profile_status_label = tk.Label(
            frame, text="", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT,
        )
        self.profile_status_label.pack(fill=tk.X, pady=2)

        # Ranked model table (read-only text widget)
        self.model_table = tk.Text(
            frame, height=12, wrap=tk.NONE, state=tk.DISABLED,
            font=("Consolas", 9), bg=t["input_bg"], fg=t["input_fg"],
            relief=tk.FLAT, bd=2,
        )
        self.model_table.pack(fill=tk.X, pady=(4, 2))

        # Load current profile and show ranking
        self._detect_current_profile()
        self._refresh_profile_info()
        self._refresh_model_table()

    @staticmethod
    def _load_profile_names():
        """Read profile names from profiles.yaml."""
        try:
            import yaml
            root = os.environ.get("HYBRIDRAG_PROJECT_ROOT", ".")
            path = os.path.join(root, "config", "profiles.yaml")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return list(data.keys())
        except Exception:
            pass
        return ["laptop_safe", "desktop_power", "server_max"]

    def _detect_current_profile(self):
        """Infer current profile from config by matching embedding model."""
        try:
            import yaml
            root = os.environ.get("HYBRIDRAG_PROJECT_ROOT", ".")
            path = os.path.join(root, "config", "profiles.yaml")
            with open(path, "r", encoding="utf-8") as f:
                profiles_data = yaml.safe_load(f) or {}

            current_model = getattr(
                getattr(self.config, "embedding", None), "model_name", ""
            )
            current_device = getattr(
                getattr(self.config, "embedding", None), "device", "cpu"
            )

            # Match by embedding model_name + device (most reliable)
            for name, pdata in profiles_data.items():
                p_model = pdata.get("embedding", {}).get("model_name", "")
                p_device = pdata.get("embedding", {}).get("device", "cpu")
                if p_model == current_model and p_device == current_device:
                    self.profile_var.set(name)
                    return

            # Fallback: match by LLM model + device
            current_llm = getattr(
                getattr(self.config, "ollama", None), "model", ""
            )
            for name, pdata in profiles_data.items():
                p_llm = pdata.get("ollama", {}).get("model", "")
                p_device = pdata.get("embedding", {}).get("device", "cpu")
                if p_llm == current_llm and p_device == current_device:
                    self.profile_var.set(name)
                    return

            self.profile_var.set("laptop_safe")
        except Exception:
            self.profile_var.set("laptop_safe")

    def _refresh_profile_info(self):
        """Show embedder + LLM info for the currently detected profile."""
        embed = getattr(self.config, "embedding", None)
        ollama = getattr(self.config, "ollama", None)
        model_name = getattr(embed, "model_name", "?") if embed else "?"
        dim = getattr(embed, "dimension", "?") if embed else "?"
        device = getattr(embed, "device", "?") if embed else "?"
        llm = getattr(ollama, "model", "?") if ollama else "?"
        self.profile_info_label.config(
            text="Embedder: {} ({}d, {})  |  LLM: {}".format(
                model_name, dim, device, llm
            ),
        )

    def _refresh_model_table(self):
        """Populate the ranked model table for the current profile."""
        profile = self.profile_var.get()
        try:
            from scripts._model_meta import (
                get_profile_ranking_table, USE_CASES, use_case_score,
                WORK_ONLY_MODELS,
            )

            table = get_profile_ranking_table(profile)
            lines = []
            lines.append(
                "  {:<22s} {:<22s} {}".format(
                    "Use Case", "#1 (default)", "#2 (fallback)")
            )
            lines.append(
                "  {:<22s} {:<22s} {}".format("-" * 22, "-" * 22, "-" * 22)
            )

            display_order = [
                "sw", "eng", "sys", "draft", "log", "pm", "fe", "cyber", "gen",
            ]
            for uc_key in display_order:
                if uc_key not in table:
                    continue
                ranked = table[uc_key]
                label = USE_CASES[uc_key]["label"]
                col1 = ranked[0]["model"] if len(ranked) > 0 else "---"
                col2 = ranked[1]["model"] if len(ranked) > 1 else "---"
                lines.append(
                    "  {:<22s} {:<22s} {}".format(label, col1, col2)
                )

            text = "\n".join(lines)
        except Exception as e:
            text = "  [WARN] Could not load rankings: {}".format(e)

        self.model_table.config(state=tk.NORMAL)
        self.model_table.delete("1.0", tk.END)
        self.model_table.insert("1.0", text)
        self.model_table.config(state=tk.DISABLED)

    def _on_profile_change(self, event=None):
        """Apply profile switch: update YAML, reload config, rebuild backends.

        Full flow:
          1. Capture old embedding model name
          2. Run _profile_switch.py to update YAML on disk
          3. Reload Config from disk
          4. Detect embedding model change -> warn about re-index
          5. Clear embedder cache if model changed
          6. Propagate new config to app + all panels
          7. Reset backends (rebuild embedder/router/query engine)
          8. Refresh model table and info display
        """
        from tkinter import messagebox
        t = current_theme()
        profile = self.profile_var.get()
        root = os.environ.get("HYBRIDRAG_PROJECT_ROOT", ".")

        # 1. Capture old embedding model before the switch
        old_embed_model = getattr(
            getattr(self.config, "embedding", None), "model_name", ""
        )

        # 2. Run profile switch subprocess (updates YAML on disk)
        self.profile_status_label.config(
            text="Switching to {}...".format(profile), fg=t["gray"],
        )
        self.update_idletasks()

        try:
            proc = subprocess.run(
                [sys.executable,
                 os.path.join(root, "scripts", "_profile_switch.py"),
                 profile],
                capture_output=True, text=True, timeout=10,
                cwd=root,
            )
            if proc.returncode != 0:
                self.profile_status_label.config(
                    text="[FAIL] {}".format(proc.stderr.strip()[:80]),
                    fg=t["red"],
                )
                return
        except Exception as e:
            self.profile_status_label.config(
                text="[FAIL] {}".format(str(e)[:80]), fg=t["red"],
            )
            return

        # 3. Reload Config from disk
        try:
            from src.core.config import load_config
            new_config = load_config(root)
        except Exception as e:
            self.profile_status_label.config(
                text="[FAIL] Config reload: {}".format(str(e)[:60]),
                fg=t["red"],
            )
            return

        # Preserve the runtime mode (online/offline) -- the YAML always
        # says "offline" but the user may have toggled to online.
        new_config.mode = self.config.mode

        # 4. Detect embedding model change
        new_embed_model = getattr(
            getattr(new_config, "embedding", None), "model_name", ""
        )
        embedding_changed = (
            old_embed_model
            and new_embed_model
            and old_embed_model != new_embed_model
        )

        if embedding_changed:
            messagebox.showwarning(
                "Re-Index Required",
                "Embedding model changed:\n\n"
                "  Old: {}\n"
                "  New: {}\n\n"
                "Existing vectors are INCOMPATIBLE with the\n"
                "new model. You MUST re-index all documents\n"
                "before querying.\n\n"
                "Use the Index panel to start a new index.".format(
                    old_embed_model, new_embed_model,
                ),
            )

        # 5. Clear embedder cache if model changed
        if embedding_changed:
            try:
                from src.gui.launch_gui import clear_embedder_cache
                clear_embedder_cache()
                logger.info("Embedder cache cleared (model changed)")
            except Exception as e:
                logger.warning("Could not clear embedder cache: %s", e)

        # 6. Propagate new config to app + all panels
        self.config = new_config
        if hasattr(self._app, "reload_config"):
            self._app.reload_config(new_config)

        # 7. Reset backends (rebuilds embedder/router/query engine)
        if hasattr(self._app, "reset_backends"):
            self._app.reset_backends()

        # NOTE: query_engine will be None here because reset_backends()
        # launches a background thread. Our reference is refreshed lazily
        # in _on_test_query() which re-reads app.query_engine at call time.

        # 8. Refresh displays
        self._refresh_profile_info()
        self._refresh_model_table()

        # Update slider values to match new profile config
        self._sync_sliders_to_config()

        status_parts = ["[OK] Switched to {}".format(profile)]
        if embedding_changed:
            status_parts.append("-- RE-INDEX REQUIRED")
        self.profile_status_label.config(
            text=" ".join(status_parts), fg=t["green"],
        )

    def _sync_sliders_to_config(self):
        """Sync slider values to match the newly loaded config."""
        retrieval = getattr(self.config, "retrieval", None)
        api = getattr(self.config, "api", None)
        if retrieval:
            self.topk_var.set(getattr(retrieval, "top_k", 8))
            self.minscore_var.set(getattr(retrieval, "min_score", 0.20))
            self.hybrid_var.set(getattr(retrieval, "hybrid_search", True))
            self.reranker_var.set(getattr(retrieval, "reranker_enabled", False))
        if api:
            self.maxtokens_var.set(getattr(api, "max_tokens", 2048))
            self.temp_var.set(getattr(api, "temperature", 0.1))
            self.timeout_var.set(getattr(api, "timeout_seconds", 30))

    # ----------------------------------------------------------------
    # TEST QUERY
    # ----------------------------------------------------------------

    def _build_test_section(self):
        """Build test query section."""
        t = current_theme()
        frame = tk.LabelFrame(self, text="Test Query", padx=16, pady=8,
                               bg=t["panel_bg"], fg=t["accent"],
                               font=FONT_BOLD)
        frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        # Persistent label above input
        self.test_query_label = tk.Label(
            frame, text="Test query:", bg=t["panel_bg"],
            fg=t["fg"], font=FONT, anchor=tk.W,
        )
        self.test_query_label.pack(fill=tk.X, pady=(0, 4))

        # Input row
        row = tk.Frame(frame, bg=t["panel_bg"])
        row.pack(fill=tk.X, pady=(0, 8))

        self.test_entry = tk.Entry(
            row, font=FONT_SMALL, bg=t["input_bg"], fg=t["input_fg"],
            insertbackground=t["fg"], relief=tk.FLAT, bd=2,
        )
        self.test_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.test_entry.bind("<Return>", self._on_test_query)

        self.test_btn = tk.Button(
            row, text="Run Test", command=self._on_test_query, width=10,
            bg=t["accent"], fg=t["accent_fg"], font=FONT,
            relief=tk.FLAT, bd=0, padx=16, pady=8,
            activebackground=t["accent_hover"],
            activeforeground=t["accent_fg"],
        )
        self.test_btn.pack(side=tk.LEFT, padx=(8, 0))
        bind_hover(self.test_btn, normal_bg=t["accent"])

        # Network indicator for test query
        self.test_network_label = tk.Label(
            frame, text="", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT,
        )
        self.test_network_label.pack(fill=tk.X)

        # Result area
        self.test_result = scrolledtext.ScrolledText(
            frame, height=4, wrap=tk.WORD, state=tk.DISABLED,
            font=FONT_SMALL, bg=t["input_bg"], fg=t["input_fg"],
            insertbackground=t["fg"], relief=tk.FLAT, bd=1,
            selectbackground=t["accent"],
            selectforeground=t["accent_fg"],
        )
        self.test_result.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # Latency (monospace for aligned numbers)
        self.test_latency_label = tk.Label(
            frame, text="", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT_MONO,
        )
        self.test_latency_label.pack(fill=tk.X, pady=(4, 0))

    def _on_test_query(self, event=None):
        """Run a test query with current settings."""
        t = current_theme()
        question = self.test_entry.get().strip()
        if not question:
            return

        # Refresh from app (reset_backends rebuilds it asynchronously)
        self.query_engine = getattr(self._app, "query_engine", None)

        if self.query_engine is None:
            self.test_result.config(state=tk.NORMAL)
            self.test_result.delete("1.0", tk.END)
            self.test_result.insert("1.0", "[FAIL] Query engine not initialized")
            self.test_result.config(state=tk.DISABLED)
            return

        self.test_btn.config(state=tk.DISABLED)

        # Show network indicator
        mode = getattr(self.config, "mode", "offline")
        if mode == "online":
            self.test_network_label.config(text="Network: ACTIVE",
                                           fg=t["orange"])
        else:
            self.test_network_label.config(text="Network: offline",
                                           fg=t["gray"])

        thread = threading.Thread(
            target=self._run_test_query, args=(question,), daemon=True,
        )
        thread.start()

    def _run_test_query(self, question):
        """Execute test query in background."""
        try:
            result = self.query_engine.query(question)
            self.after(0, self._display_test_result, result)
        except Exception as e:
            self.after(0, self._display_test_error, str(e))

    def _display_test_result(self, result):
        """Show test query result."""
        try:
            self._display_test_result_inner(result)
        except Exception as e:
            logger.error("Test result display failed: %s", e)
            self.test_btn.config(state=tk.NORMAL)
            self.test_network_label.config(text="")

    def _display_test_result_inner(self, result):
        """Inner handler (separated so outer can catch and re-enable)."""
        self.test_btn.config(state=tk.NORMAL)
        self.test_network_label.config(text="")

        self.test_result.config(state=tk.NORMAL)
        self.test_result.delete("1.0", tk.END)
        if result.error:
            self.test_result.insert("1.0", "[FAIL] {}".format(result.error))
        else:
            self.test_result.insert("1.0", result.answer[:500])
        self.test_result.config(state=tk.DISABLED)

        self.test_latency_label.config(
            text="Latency: {:,.0f} ms".format(result.latency_ms),
        )

    def _display_test_error(self, error_msg):
        """Show test query error."""
        self.test_btn.config(state=tk.NORMAL)
        self.test_network_label.config(text="")

        self.test_result.config(state=tk.NORMAL)
        self.test_result.delete("1.0", tk.END)
        self.test_result.insert("1.0", "[FAIL] {}".format(error_msg))
        self.test_result.config(state=tk.DISABLED)

    # ----------------------------------------------------------------
    # BOTTOM BUTTONS
    # ----------------------------------------------------------------

    def _build_buttons(self):
        """Build Reset and Close buttons."""
        t = current_theme()
        btn_frame = tk.Frame(self, bg=t["bg"])
        btn_frame.pack(fill=tk.X, padx=16, pady=16)

        reset_btn = tk.Button(
            btn_frame, text="Reset to Defaults", command=self._on_reset,
            width=16, bg=t["inactive_btn_bg"], fg=t["inactive_btn_fg"],
            font=FONT, relief=tk.FLAT, bd=0, padx=12, pady=8,
        )
        reset_btn.pack(side=tk.LEFT)
        bind_hover(reset_btn)

        close_btn = tk.Button(
            btn_frame, text="Close", command=self.destroy, width=10,
            bg=t["accent"], fg=t["accent_fg"], font=FONT,
            relief=tk.FLAT, bd=0, padx=16, pady=8,
            activebackground=t["accent_hover"],
            activeforeground=t["accent_fg"],
        )
        close_btn.pack(side=tk.RIGHT)
        bind_hover(close_btn, normal_bg=t["accent"])

    def _on_reset(self):
        """Reset all sliders to original values."""
        orig = self._original_values
        self.topk_var.set(orig["top_k"])
        self.minscore_var.set(orig["min_score"])
        self.hybrid_var.set(orig["hybrid_search"])
        self.reranker_var.set(orig["reranker_enabled"])
        self.maxtokens_var.set(orig["max_tokens"])
        self.temp_var.set(orig["temperature"])
        self.timeout_var.set(orig["timeout_seconds"])

        # Write reset values to config
        self._on_retrieval_change()
        self._on_llm_change()
