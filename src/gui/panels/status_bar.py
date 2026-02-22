# ============================================================================
# HybridRAG v3 -- Status Bar Panel (src/gui/panels/status_bar.py)
# ============================================================================
# Displays live system status: LLM backend, Ollama, and network gate state.
# Updates every 5 seconds via a background timer.
#
# INTERNET ACCESS: NONE (reads local state only)
# ============================================================================

import tkinter as tk
import threading
import logging

from src.gui.theme import current_theme, FONT

logger = logging.getLogger(__name__)


class StatusBar(tk.Frame):
    """
    Bottom status bar showing LLM, Ollama, and Gate status.

    Updates every 5 seconds by calling router.get_status().
    """

    REFRESH_MS = 5000  # 5 seconds

    def __init__(self, parent, config, router=None):
        t = current_theme()
        super().__init__(parent, relief=tk.FLAT, bd=1,
                         bg=t["panel_bg"])
        self.config = config
        self.router = router
        self._stop_event = threading.Event()
        self._loading = True
        self._loading_dots = 0
        self._dot_timer_id = None

        self._build_widgets(t)

        # -- Start periodic refresh --
        self._schedule_refresh()

    def _build_widgets(self, t):
        """Build all child widgets with theme colors."""
        # -- Loading indicator (left-most) --
        self.loading_label = tk.Label(
            self, text="Loading...", anchor=tk.W,
            padx=8, pady=4, bg=t["panel_bg"], fg=t["orange"], font=FONT,
        )
        self.loading_label.pack(side=tk.LEFT)

        self.sep_loading = tk.Frame(self, width=1, bg=t["separator"])
        self.sep_loading.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # -- LLM indicator --
        self.llm_label = tk.Label(
            self, text="LLM: Not configured", anchor=tk.W,
            padx=8, pady=4, bg=t["panel_bg"], fg=t["fg"], font=FONT,
        )
        self.llm_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -- Separator --
        self.sep1 = tk.Frame(self, width=1, bg=t["separator"])
        self.sep1.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # -- Ollama indicator --
        self.ollama_label = tk.Label(
            self, text="Ollama: Unknown", anchor=tk.W,
            padx=8, pady=4, bg=t["panel_bg"], fg=t["fg"], font=FONT,
        )
        self.ollama_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -- Separator --
        self.sep2 = tk.Frame(self, width=1, bg=t["separator"])
        self.sep2.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=4)

        # -- Gate indicator (clickable) --
        self.gate_dot = tk.Label(self, text=" ", width=2, padx=4,
                                 bg=t["panel_bg"])
        self.gate_dot.pack(side=tk.LEFT, padx=(8, 4))

        self.gate_label = tk.Label(
            self, text="Gate: OFFLINE", anchor=tk.W,
            padx=4, pady=4, cursor="hand2",
            bg=t["panel_bg"], fg=t["gray"], font=FONT,
        )
        self.gate_label.pack(side=tk.LEFT, padx=(0, 8))
        self.gate_label.bind("<Button-1>", self._on_gate_click)

    def apply_theme(self, t):
        """Re-apply theme colors to all widgets."""
        self.configure(bg=t["panel_bg"])
        self.loading_label.configure(bg=t["panel_bg"])
        if self._loading:
            self.loading_label.configure(fg=t["orange"])
        else:
            self.loading_label.configure(fg=t["green"])
        self.sep_loading.configure(bg=t["separator"])
        self.llm_label.configure(bg=t["panel_bg"])
        self.ollama_label.configure(bg=t["panel_bg"])
        self.gate_label.configure(bg=t["panel_bg"])
        self.gate_dot.configure(bg=t["panel_bg"])
        self.sep1.configure(bg=t["separator"])
        self.sep2.configure(bg=t["separator"])
        # Refresh status to set correct colors
        self._refresh_status()

    def _schedule_refresh(self):
        """Schedule next status refresh."""
        if not self._stop_event.is_set():
            self._refresh_status()
            self.after(self.REFRESH_MS, self._schedule_refresh)

    def _refresh_status(self):
        """Update all status indicators from current state."""
        try:
            self._update_gate_display()
            if self.router:
                self._update_from_router()
            else:
                self._update_no_router()
        except Exception as e:
            logger.debug("Status bar refresh error: %s", e)

    def _update_from_router(self):
        """Update LLM and Ollama indicators from router status."""
        t = current_theme()
        try:
            status = self.router.get_status()
        except Exception as e:
            logger.debug("Router status error: %s", e)
            self.llm_label.config(text="LLM: Error reading status", fg=t["fg"])
            self.ollama_label.config(text="Ollama: Unknown", fg=t["fg"])
            return

        # LLM
        mode = status.get("mode", "offline")
        if mode == "online" and status.get("api_configured"):
            provider = status.get("api_provider", "API")
            deployment = status.get("api_deployment", "")
            if deployment:
                self.llm_label.config(
                    text="LLM: {} ({})".format(deployment, provider),
                    fg=t["fg"],
                )
            else:
                self.llm_label.config(
                    text="LLM: {} ({})".format(
                        status.get("api_endpoint", "configured")[:30],
                        provider,
                    ),
                    fg=t["fg"],
                )
        elif mode == "offline":
            model = getattr(self.config, "ollama", None)
            model_name = getattr(model, "model", "phi4-mini") if model else "phi4-mini"
            self.llm_label.config(
                text="LLM: {} (Ollama)".format(model_name),
                fg=t["fg"],
            )
        else:
            self.llm_label.config(text="LLM: Not configured", fg=t["fg"])

        # Ollama
        ollama_up = status.get("ollama_available", False)
        if ollama_up:
            self.ollama_label.config(text="Ollama: Ready", fg=t["green"])
        else:
            self.ollama_label.config(text="Ollama: Offline", fg=t["gray"])

    def _update_no_router(self):
        """Display when no router is available."""
        t = current_theme()
        if self._loading:
            self.llm_label.config(text="LLM: Loading...", fg=t["gray"])
            self.ollama_label.config(text="Ollama: Loading...", fg=t["gray"])
        else:
            self.llm_label.config(text="LLM: Not initialized", fg=t["fg"])
            self.ollama_label.config(text="Ollama: Unknown", fg=t["gray"])

    def _update_gate_display(self):
        """Update gate indicator from config mode."""
        t = current_theme()
        mode = getattr(self.config, "mode", "offline")
        if mode == "online":
            self.gate_label.config(text="Gate: ONLINE", fg=t["green"])
            self.gate_dot.config(bg=t["green"])
        else:
            self.gate_label.config(text="Gate: OFFLINE", fg=t["gray"])
            self.gate_dot.config(bg=t["gray"])

    def _on_gate_click(self, event=None):
        """Toggle gate mode when clicked. Delegates to parent app."""
        parent_app = self._find_app()
        if parent_app and hasattr(parent_app, "toggle_mode"):
            current = getattr(self.config, "mode", "offline")
            new_mode = "offline" if current == "online" else "online"
            parent_app.toggle_mode(new_mode)

    def _find_app(self):
        """Walk up widget tree to find the HybridRAGApp instance."""
        widget = self.master
        while widget is not None:
            if hasattr(widget, "toggle_mode"):
                return widget
            widget = getattr(widget, "master", None)
        return None

    def set_loading_stage(self, stage_text):
        """Update the loading indicator with the current stage."""
        t = current_theme()
        self._loading = True
        self._loading_dots = 0
        self.loading_label.config(text="Loading: {}".format(stage_text),
                                  fg=t["orange"])
        # Start dot animation if not already running
        if self._dot_timer_id is None:
            self._animate_dots()

    def set_ready(self):
        """Mark loading as complete -- show green Ready text."""
        t = current_theme()
        self._loading = False
        # Cancel dot animation
        if self._dot_timer_id is not None:
            self.after_cancel(self._dot_timer_id)
            self._dot_timer_id = None
        self.loading_label.config(text="Ready", fg=t["green"])

    def _animate_dots(self):
        """Cycle dots (. -> .. -> ...) on the loading label."""
        if not self._loading or self._stop_event.is_set():
            self._dot_timer_id = None
            return
        self._loading_dots = (self._loading_dots % 3) + 1
        current_text = self.loading_label.cget("text")
        # Strip trailing dots and re-add
        base = current_text.rstrip(".")
        self.loading_label.config(text=base + "." * self._loading_dots)
        self._dot_timer_id = self.after(500, self._animate_dots)

    def force_refresh(self):
        """Immediately refresh all indicators."""
        self._refresh_status()

    def stop(self):
        """Stop the periodic refresh timer."""
        self._stop_event.set()
