# ============================================================================
# HybridRAG v3 -- Main GUI Application (src/gui/app.py)
# ============================================================================
# Technology Decision: tkinter (Python standard library)
#
# WHY TKINTER:
#   - Zero additional dependencies (already in Python stdlib)
#   - Works on every machine including work laptops with restricted installs
#   - No entry needed in requirements.txt
#   - Suitable for prototype / human review before production UI
#   - PyQt5/PySide6/wx/Dear PyGui are NOT in requirements.txt
#
# LAYOUT: Single window, four regions top to bottom:
#   1. Title bar with mode toggle (OFFLINE / ONLINE) + theme toggle
#   2. Query panel (use case, model, question, answer, sources, metrics)
#   3. Index panel (folder picker, progress bar, start/stop)
#   4. Status bar (LLM, Ollama, Gate indicators)
#
# Menu bar: File | Engineering | Help
#
# INTERNET ACCESS: Depends on mode.
#   Offline: NONE (all local)
#   Online: API calls through QueryEngine only
# ============================================================================

import tkinter as tk
from tkinter import messagebox
import logging
import threading

from src.gui.panels.query_panel import QueryPanel
from src.gui.panels.index_panel import IndexPanel
from src.gui.panels.status_bar import StatusBar
from src.gui.panels.engineering_menu import EngineeringMenu
from src.gui.theme import (
    DARK, LIGHT, FONT, FONT_BOLD, FONT_TITLE,
    current_theme, set_theme, apply_ttk_styles,
)

logger = logging.getLogger(__name__)


class HybridRAGApp(tk.Tk):
    """
    Main application window for HybridRAG v3.

    Owns all panels and coordinates mode switching, boot state,
    and backend references.
    """

    def __init__(self, boot_result=None, config=None, query_engine=None,
                 indexer=None, router=None):
        super().__init__()

        self.title("HybridRAG v3")
        self.geometry("780x720")
        self.minsize(640, 500)

        # Store backend references
        self.boot_result = boot_result
        self.config = config
        self.query_engine = query_engine
        self.indexer = indexer
        self.router = router

        # Apply initial theme
        self._theme = current_theme()
        apply_ttk_styles(self._theme)
        self.configure(bg=self._theme["bg"])

        # Build UI
        self._build_menu_bar()
        self._build_title_bar()
        self._build_query_panel()
        self._build_index_panel()
        self._build_status_bar()

        # Show boot warnings if any
        if boot_result and boot_result.warnings:
            for w in boot_result.warnings:
                logger.warning("Boot warning: %s", w)

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------------
    # MENU BAR
    # ----------------------------------------------------------------

    def _build_menu_bar(self):
        """Build File | Engineering | Help menu bar."""
        t = self._theme
        menubar = tk.Menu(self, bg=t["menu_bg"], fg=t["menu_fg"],
                          activebackground=t["accent"],
                          activeforeground=t["accent_fg"],
                          relief=tk.FLAT, font=FONT)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0,
                            bg=t["menu_bg"], fg=t["menu_fg"],
                            activebackground=t["accent"],
                            activeforeground=t["accent_fg"], font=FONT)
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        # Engineering menu
        eng_menu = tk.Menu(menubar, tearoff=0,
                           bg=t["menu_bg"], fg=t["menu_fg"],
                           activebackground=t["accent"],
                           activeforeground=t["accent_fg"], font=FONT)
        eng_menu.add_command(
            label="Engineering Settings...",
            command=self._open_engineering_menu,
        )
        menubar.add_cascade(label="Engineering", menu=eng_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0,
                            bg=t["menu_bg"], fg=t["menu_fg"],
                            activebackground=t["accent"],
                            activeforeground=t["accent_fg"], font=FONT)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config_menu = menubar
        self.configure(menu=menubar)

    # ----------------------------------------------------------------
    # TITLE BAR with mode toggle + theme toggle
    # ----------------------------------------------------------------

    def _build_title_bar(self):
        """Build title bar with OFFLINE/ONLINE toggle and theme toggle."""
        t = self._theme
        self.title_frame = tk.Frame(self, bg=t["panel_bg"], padx=8, pady=6)
        self.title_frame.pack(fill=tk.X)

        self.title_label = tk.Label(
            self.title_frame, text="HybridRAG v3", font=FONT_TITLE,
            bg=t["panel_bg"], fg=t["fg"],
        )
        self.title_label.pack(side=tk.LEFT)

        # Mode label
        self.mode_label = tk.Label(
            self.title_frame, text="Mode:", bg=t["panel_bg"], fg=t["label_fg"],
            font=FONT,
        )
        self.mode_label.pack(side=tk.LEFT, padx=(20, 4))

        # OFFLINE button
        self.offline_btn = tk.Button(
            self.title_frame, text="OFFLINE", width=10, font=FONT,
            command=lambda: self.toggle_mode("offline"),
            relief=tk.FLAT, bd=0, padx=6, pady=2,
        )
        self.offline_btn.pack(side=tk.LEFT, padx=2)

        # ONLINE button
        self.online_btn = tk.Button(
            self.title_frame, text="ONLINE", width=10, font=FONT,
            command=lambda: self.toggle_mode("online"),
            relief=tk.FLAT, bd=0, padx=6, pady=2,
        )
        self.online_btn.pack(side=tk.LEFT, padx=2)

        # -- Theme toggle (right side) --
        self.theme_btn = tk.Button(
            self.title_frame, text="Light", width=6, font=FONT,
            command=self._toggle_theme,
            relief=tk.FLAT, bd=0, padx=6, pady=2,
            bg=t["input_bg"], fg=t["fg"],
        )
        self.theme_btn.pack(side=tk.RIGHT, padx=4)

        self.theme_icon_label = tk.Label(
            self.title_frame, text="Theme:", bg=t["panel_bg"],
            fg=t["label_fg"], font=FONT,
        )
        self.theme_icon_label.pack(side=tk.RIGHT)

        # -- Reset button (right side, before Theme) --
        self.reset_btn = tk.Button(
            self.title_frame, text="Reset", width=6, font=FONT,
            command=self.reset_backends,
            relief=tk.FLAT, bd=0, padx=6, pady=2,
            bg=t["input_bg"], fg=t["fg"],
        )
        self.reset_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # Set initial button colors
        self._update_mode_buttons()

    def _update_mode_buttons(self):
        """Update mode button colors to reflect current state."""
        t = self._theme
        mode = getattr(self.config, "mode", "offline") if self.config else "offline"
        if mode == "online":
            self.online_btn.config(bg=t["active_btn_bg"], fg=t["active_btn_fg"],
                                   relief=tk.FLAT)
            self.offline_btn.config(bg=t["inactive_btn_bg"], fg=t["inactive_btn_fg"],
                                    relief=tk.FLAT)
        else:
            self.offline_btn.config(bg=t["active_btn_bg"], fg=t["active_btn_fg"],
                                    relief=tk.FLAT)
            self.online_btn.config(bg=t["inactive_btn_bg"], fg=t["inactive_btn_fg"],
                                   relief=tk.FLAT)

    # ----------------------------------------------------------------
    # THEME TOGGLE
    # ----------------------------------------------------------------

    def _toggle_theme(self):
        """Switch between dark and light themes and rebuild the UI."""
        if self._theme["name"] == "dark":
            new_theme = LIGHT
        else:
            new_theme = DARK

        set_theme(new_theme)
        self._theme = new_theme
        apply_ttk_styles(new_theme)
        self._apply_theme_to_all()

    def _apply_theme_to_all(self):
        """Re-apply theme colors to all widgets without rebuilding."""
        t = self._theme
        self.configure(bg=t["bg"])

        # Title bar
        self.title_frame.configure(bg=t["panel_bg"])
        self.title_label.configure(bg=t["panel_bg"], fg=t["fg"])
        self.mode_label.configure(bg=t["panel_bg"], fg=t["label_fg"])
        self.theme_icon_label.configure(bg=t["panel_bg"], fg=t["label_fg"])

        # Theme button label
        if t["name"] == "dark":
            self.theme_btn.configure(text="Light", bg=t["input_bg"], fg=t["fg"])
        else:
            self.theme_btn.configure(text="Dark", bg=t["input_bg"], fg=t["fg"])

        # Reset button
        self.reset_btn.configure(bg=t["input_bg"], fg=t["fg"])

        self._update_mode_buttons()

        # Rebuild menus
        self._build_menu_bar()

        # Propagate to panels
        if hasattr(self, "query_panel"):
            self.query_panel.apply_theme(t)
        if hasattr(self, "index_panel"):
            self.index_panel.apply_theme(t)
        if hasattr(self, "status_bar"):
            self.status_bar.apply_theme(t)

    # ----------------------------------------------------------------
    # PANELS
    # ----------------------------------------------------------------

    def _build_query_panel(self):
        """Build and pack the query panel."""
        self.query_panel = QueryPanel(
            self, config=self.config, query_engine=self.query_engine,
        )
        self.query_panel.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 2))

    def _build_index_panel(self):
        """Build and pack the index panel."""
        self.index_panel = IndexPanel(
            self, config=self.config, indexer=self.indexer,
        )
        self.index_panel.pack(fill=tk.X, padx=8, pady=2)

    def _build_status_bar(self):
        """Build and pack the status bar."""
        self.status_bar = StatusBar(
            self, config=self.config, router=self.router,
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ----------------------------------------------------------------
    # MODE TOGGLING
    # ----------------------------------------------------------------

    def toggle_mode(self, new_mode):
        """
        Switch between online and offline mode.

        Online: checks credentials first, shows error if missing.
        Offline: always succeeds (safe operation).
        """
        if new_mode == "online":
            self._switch_to_online()
        else:
            self._switch_to_offline()

    def _switch_to_online(self):
        """Attempt to switch to online mode."""
        # Check credentials
        try:
            from src.security.credentials import credential_status
            status = credential_status()

            if not status.get("api_key_set") or not status.get("api_endpoint_set"):
                missing = []
                if not status.get("api_key_set"):
                    missing.append("API key")
                if not status.get("api_endpoint_set"):
                    missing.append("API endpoint")
                messagebox.showwarning(
                    "Credentials Missing",
                    "Cannot switch to online mode.\n\n"
                    "Missing: {}\n\n"
                    "Run rag-store-key and rag-store-endpoint from "
                    "PowerShell first, then try again.".format(", ".join(missing)),
                )
                return
        except Exception as e:
            messagebox.showwarning(
                "Credential Check Failed",
                "Could not verify credentials: {}\n\n"
                "Run rag-store-key and rag-store-endpoint from "
                "PowerShell first, then try again.".format(e),
            )
            return

        # Switch mode
        if self.config:
            self.config.mode = "online"

        # Reconfigure network gate
        try:
            from src.core.network_gate import configure_gate
            from src.security.credentials import resolve_credentials
            creds = resolve_credentials()
            configure_gate(
                mode="online",
                api_endpoint=creds.endpoint or "",
                allowed_prefixes=getattr(
                    getattr(self.config, "api", None),
                    "allowed_endpoint_prefixes", [],
                ) if self.config else [],
            )
        except Exception as e:
            logger.warning("Gate reconfiguration failed: %s", e)

        self._update_mode_buttons()
        self.status_bar.force_refresh()
        logger.info("Switched to ONLINE mode")

    def _switch_to_offline(self):
        """Switch to offline mode (always safe)."""
        if self.config:
            self.config.mode = "offline"

        # Reconfigure network gate
        try:
            from src.core.network_gate import configure_gate
            configure_gate(mode="offline")
        except Exception as e:
            logger.warning("Gate reconfiguration failed: %s", e)

        self._update_mode_buttons()
        self.status_bar.force_refresh()
        logger.info("Switched to OFFLINE mode")

    # ----------------------------------------------------------------
    # BACKEND RESET + READY STATE
    # ----------------------------------------------------------------

    def reset_backends(self):
        """Tear down backends, show loading state, and reload in background.

        The Embedder is cached at module level in launch_gui.py so the
        expensive model-load (~8s) is paid only once per process.  Reset
        rebuilds VectorStore, Router, QueryEngine, and Indexer but reuses
        the cached Embedder -- making Reset near-instant.
        """
        # Clear backend references (embedder stays in module cache)
        self.query_engine = None
        self.indexer = None
        self.router = None

        # Propagate to panels
        if hasattr(self, "query_panel"):
            self.query_panel.query_engine = None
            self.query_panel.set_ready(False)
        if hasattr(self, "index_panel"):
            self.index_panel.indexer = None
            self.index_panel.set_ready(False)
        if hasattr(self, "status_bar"):
            self.status_bar.router = None
            self.status_bar.set_loading_stage("Restarting...")
            self.status_bar.force_refresh()

        # Launch reload in a new daemon thread
        from src.gui.launch_gui import _load_backends
        reload_thread = threading.Thread(
            target=_load_backends,
            args=(self, logging.getLogger("gui_launcher")),
            daemon=True,
        )
        reload_thread.start()
        logger.info("Backend reset -- reloading in background")

    def set_ready(self, enabled):
        """Propagate ready state to all panels."""
        if hasattr(self, "query_panel"):
            self.query_panel.set_ready(enabled)
        if hasattr(self, "index_panel"):
            self.index_panel.set_ready(enabled)
        if hasattr(self, "status_bar"):
            if enabled:
                self.status_bar.set_ready()
            else:
                self.status_bar.set_loading_stage("Loading...")

    # ----------------------------------------------------------------
    # ENGINEERING MENU
    # ----------------------------------------------------------------

    def _open_engineering_menu(self):
        """Open the engineering settings child window."""
        EngineeringMenu(self, config=self.config, query_engine=self.query_engine)

    # ----------------------------------------------------------------
    # HELP
    # ----------------------------------------------------------------

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About HybridRAG v3",
            "HybridRAG v3 -- GUI Prototype\n\n"
            "Local-first RAG system for technical document search.\n"
            "Zero-trust offline-default architecture.\n\n"
            "Technology: tkinter (Python standard library)\n"
            "Backend: SQLite + memmap + sentence-transformers",
        )

    # ----------------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------------

    def _on_close(self):
        """Clean up and close the application."""
        if hasattr(self, "status_bar"):
            self.status_bar.stop()
        self.destroy()
