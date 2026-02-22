# ============================================================================
# HybridRAG v3 -- Query Panel (src/gui/panels/query_panel.py)
# ============================================================================
# The main query interface: use case selector, question input, answer display.
#
# INTERNET ACCESS: Online mode sends query to API via QueryEngine.
#   Shows "Sending to API..." indicator during online queries.
# ============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import logging

from scripts._model_meta import USE_CASES, select_best_model
from src.core.llm_router import get_available_deployments
from src.gui.theme import current_theme, FONT, FONT_BOLD, FONT_MONO, bind_hover

logger = logging.getLogger(__name__)


class QueryPanel(tk.LabelFrame):
    """
    Query input and answer display panel.

    Shows use case dropdown, auto-selected model, question entry,
    answer area with sources and latency metrics.
    """

    def __init__(self, parent, config, query_engine=None):
        t = current_theme()
        super().__init__(parent, text="Query Panel", padx=16, pady=16,
                         bg=t["panel_bg"], fg=t["accent"],
                         font=FONT_BOLD)
        self.config = config
        self.query_engine = query_engine
        self._query_thread = None

        self._build_widgets(t)

        # Defer initial model selection so GUI renders immediately.
        # get_available_deployments() may do a network call on first use.
        self.after(100, self._on_use_case_change)

    def _build_widgets(self, t):
        """Build all child widgets with theme colors."""
        # -- Row 0: Use case selector --
        row0 = tk.Frame(self, bg=t["panel_bg"])
        row0.pack(fill=tk.X, pady=(0, 8))

        self.uc_label = tk.Label(row0, text="Use case:", bg=t["panel_bg"],
                                 fg=t["fg"], font=FONT)
        self.uc_label.pack(side=tk.LEFT)

        self._uc_keys = list(USE_CASES.keys())
        self._uc_labels = [USE_CASES[k]["label"] for k in self._uc_keys]
        self.uc_var = tk.StringVar(value=self._uc_labels[1])

        self.uc_dropdown = ttk.Combobox(
            row0, textvariable=self.uc_var, values=self._uc_labels,
            state="readonly", width=30, font=FONT,
        )
        self.uc_dropdown.pack(side=tk.LEFT, padx=(8, 0))
        self.uc_dropdown.bind("<<ComboboxSelected>>", self._on_use_case_change)

        # -- Row 1: Model display (read-only) --
        row1 = tk.Frame(self, bg=t["panel_bg"])
        row1.pack(fill=tk.X, pady=(0, 8))

        self.model_text_label = tk.Label(row1, text="Model:", bg=t["panel_bg"],
                                         fg=t["fg"], font=FONT)
        self.model_text_label.pack(side=tk.LEFT)

        self.model_var = tk.StringVar(value="(auto-selected)")
        self.model_label = tk.Label(
            row1, textvariable=self.model_var, anchor=tk.W,
            fg=t["accent"], bg=t["panel_bg"], padx=8, font=FONT,
        )
        self.model_label.pack(side=tk.LEFT, fill=tk.X)

        # -- Row 2: Question label + entry + Ask button --
        self.question_label = tk.Label(
            self, text="Question:", bg=t["panel_bg"],
            fg=t["fg"], font=FONT, anchor=tk.W,
        )
        self.question_label.pack(fill=tk.X, pady=(0, 4))

        row2 = tk.Frame(self, bg=t["panel_bg"])
        row2.pack(fill=tk.X, pady=(0, 8))

        self.question_entry = tk.Entry(
            row2, font=FONT, bg=t["input_bg"], fg=t["input_fg"],
            insertbackground=t["fg"], relief=tk.FLAT, bd=2,
        )
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.question_entry.insert(0, "Type your question here...")
        self.question_entry.bind("<FocusIn>", self._on_entry_focus)
        self.question_entry.bind("<Return>", self._on_ask)

        self.ask_btn = tk.Button(
            row2, text="Ask", command=self._on_ask, width=10,
            bg=t["inactive_btn_bg"], fg=t["inactive_btn_fg"],
            font=FONT_BOLD, relief=tk.FLAT, bd=0,
            padx=24, pady=8, state=tk.DISABLED,
            activebackground=t["accent_hover"],
            activeforeground=t["accent_fg"],
        )
        self.ask_btn.pack(side=tk.LEFT, padx=(8, 0))

        # -- Network activity indicator --
        self.network_label = tk.Label(
            self, text="", fg=t["gray"], anchor=tk.W,
            bg=t["panel_bg"], font=FONT,
        )
        self.network_label.pack(fill=tk.X)

        # -- Answer area (scrollable, selectable) --
        self.answer_text = scrolledtext.ScrolledText(
            self, height=10, wrap=tk.WORD, state=tk.DISABLED,
            font=FONT, bg=t["input_bg"], fg=t["input_fg"],
            insertbackground=t["fg"], relief=tk.FLAT, bd=1,
            selectbackground=t["accent"],
            selectforeground=t["accent_fg"],
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # -- Sources line --
        self.sources_label = tk.Label(
            self, text="Sources: (none)", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT,
        )
        self.sources_label.pack(fill=tk.X, pady=(8, 0))

        # -- Metrics line (monospace for aligned numbers) --
        self.metrics_label = tk.Label(
            self, text="", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT_MONO,
        )
        self.metrics_label.pack(fill=tk.X)

    def apply_theme(self, t):
        """Re-apply theme colors to all widgets."""
        self.configure(bg=t["panel_bg"], fg=t["accent"])

        for row in self.winfo_children():
            if isinstance(row, tk.Frame):
                row.configure(bg=t["panel_bg"])
                for child in row.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(bg=t["panel_bg"], fg=t["fg"])
                    elif isinstance(child, tk.Entry):
                        child.configure(bg=t["input_bg"], fg=t["input_fg"],
                                        insertbackground=t["fg"])
                    elif isinstance(child, tk.Button):
                        if str(child.cget("state")) == "disabled":
                            child.configure(bg=t["inactive_btn_bg"],
                                            fg=t["inactive_btn_fg"])
                        else:
                            child.configure(bg=t["accent"], fg=t["accent_fg"],
                                            activebackground=t["accent_hover"])

        self.model_label.configure(fg=t["accent"], bg=t["panel_bg"])
        self.question_label.configure(bg=t["panel_bg"], fg=t["fg"])
        self.network_label.configure(bg=t["panel_bg"], fg=t["gray"])
        self.answer_text.configure(bg=t["input_bg"], fg=t["input_fg"],
                                   insertbackground=t["fg"],
                                   selectbackground=t["accent"])
        self.sources_label.configure(bg=t["panel_bg"])
        self.metrics_label.configure(bg=t["panel_bg"])
        # Preserve gray/colored state of sources/metrics
        if "none" in self.sources_label.cget("text"):
            self.sources_label.configure(fg=t["gray"])
        else:
            self.sources_label.configure(fg=t["fg"])

    def _on_entry_focus(self, event=None):
        """Clear placeholder text on first focus."""
        if self.question_entry.get() == "Type your question here...":
            self.question_entry.delete(0, tk.END)

    def _on_use_case_change(self, event=None):
        """Update model display when use case changes."""
        idx = self._uc_labels.index(self.uc_var.get()) if self.uc_var.get() in self._uc_labels else 0
        uc_key = self._uc_keys[idx]

        mode = getattr(self.config, "mode", "offline")
        if mode == "offline":
            # Offline: show the configured Ollama model directly
            ollama_model = getattr(
                getattr(self.config, "ollama", None), "model", ""
            ) or "phi4-mini"
            self.model_var.set("{} (offline)".format(ollama_model))
        else:
            # Online: auto-select best from available API deployments
            deployments = get_available_deployments()
            best = select_best_model(uc_key, deployments)
            if best:
                self.model_var.set("{} (auto-selected)".format(best))
            else:
                self.model_var.set("(no model available)")

    def _on_ask(self, event=None):
        """Handle Ask button click or Enter key."""
        question = self.question_entry.get().strip()
        if not question or question == "Type your question here...":
            return

        if self.query_engine is None:
            self._show_error("[FAIL] Query engine not initialized. Run boot first.")
            return

        # Disable button during query
        self.ask_btn.config(state=tk.DISABLED)

        # Show network indicator for online mode
        t = current_theme()
        mode = getattr(self.config, "mode", "offline")
        if mode == "online":
            self.network_label.config(text="Sending to API...", fg=t["gray"])
        else:
            self.network_label.config(text="Querying local model...", fg=t["gray"])

        # Run query in background thread to avoid freezing UI
        self._query_thread = threading.Thread(
            target=self._run_query, args=(question,), daemon=True,
        )
        self._query_thread.start()

    def _run_query(self, question):
        """Execute query in background thread."""
        try:
            result = self.query_engine.query(question)
            self.after(0, self._display_result, result)
        except Exception as e:
            error_msg = "[FAIL] {}: {}".format(type(e).__name__, e)
            self.after(0, self._show_error, error_msg)

    def _display_result(self, result):
        """Display query result in the UI (called on main thread)."""
        try:
            self._display_result_inner(result)
        except Exception as e:
            logger.error("Display result failed: %s", e)
            self.ask_btn.config(state=tk.NORMAL)
            self.network_label.config(text="")

    def _display_result_inner(self, result):
        """Inner display logic (separated so outer can catch and re-enable)."""
        t = current_theme()
        self.ask_btn.config(state=tk.NORMAL)
        self.network_label.config(text="")

        # Check for error
        if result.error:
            self._show_error("[FAIL] {}".format(result.error))
            return

        # Display answer
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", result.answer)
        self.answer_text.config(state=tk.DISABLED)

        # Display sources
        if result.sources:
            source_strs = []
            for s in result.sources:
                path = s.get("path", "unknown")
                chunks = s.get("chunks", 0)
                # Show just the filename, not full path
                fname = path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                source_strs.append("{} ({} chunks)".format(fname, chunks))
            self.sources_label.config(
                text="Sources: {}".format(", ".join(source_strs)),
                fg=t["fg"],
            )
        else:
            self.sources_label.config(text="Sources: (none)", fg=t["gray"])

        # Display metrics
        latency = result.latency_ms
        tokens_in = result.tokens_in
        tokens_out = result.tokens_out
        self.metrics_label.config(
            text="Latency: {:,.0f} ms | Tokens in: {} | Tokens out: {}".format(
                latency, tokens_in, tokens_out
            ),
        )

    def _show_error(self, error_msg):
        """Display an error message in the answer area."""
        t = current_theme()
        self.ask_btn.config(state=tk.NORMAL)
        self.network_label.config(text="")

        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", error_msg)
        self.answer_text.tag_add("error", "1.0", tk.END)
        self.answer_text.tag_config("error", foreground=t["red"])
        self.answer_text.config(state=tk.DISABLED)

        self.sources_label.config(text="Sources: (none)", fg=t["gray"])
        self.metrics_label.config(text="")

    def set_ready(self, enabled):
        """Enable or disable the Ask button based on backend readiness."""
        t = current_theme()
        if enabled:
            self.ask_btn.config(state=tk.NORMAL, bg=t["accent"],
                                fg=t["accent_fg"])
        else:
            self.ask_btn.config(state=tk.DISABLED, bg=t["inactive_btn_bg"],
                                fg=t["inactive_btn_fg"])

    def get_current_use_case_key(self):
        """Return the currently selected use case key."""
        idx = self._uc_labels.index(self.uc_var.get()) if self.uc_var.get() in self._uc_labels else 0
        return self._uc_keys[idx]
