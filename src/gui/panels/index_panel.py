# ============================================================================
# HybridRAG v3 -- Index Panel (src/gui/panels/index_panel.py)
# ============================================================================
# Folder picker, indexing progress bar, start/stop controls.
#
# INTERNET ACCESS: NONE (indexing is purely local)
# ============================================================================

import os
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import time
import logging
from datetime import datetime

from src.gui.theme import current_theme, FONT, FONT_BOLD

logger = logging.getLogger(__name__)


class IndexPanel(tk.LabelFrame):
    """
    Index management panel with folder picker, progress bar, and controls.
    """

    def __init__(self, parent, config, indexer=None):
        t = current_theme()
        super().__init__(parent, text="Index Panel", padx=8, pady=8,
                         bg=t["panel_bg"], fg=t["accent"],
                         font=FONT_BOLD)
        self.config = config
        self.indexer = indexer
        self._stop_flag = threading.Event()
        self._index_thread = None

        self._build_widgets(t)

    def _build_widgets(self, t):
        """Build all child widgets with theme colors."""
        # -- Row 0: Source folder --
        row0 = tk.Frame(self, bg=t["panel_bg"])
        row0.pack(fill=tk.X, pady=(0, 4))

        self.folder_label = tk.Label(row0, text="Source folder:",
                                     bg=t["panel_bg"], fg=t["fg"], font=FONT)
        self.folder_label.pack(side=tk.LEFT)

        default_source = getattr(
            getattr(self.config, "paths", None), "source_folder", ""
        ) or ""

        self.folder_var = tk.StringVar(value=default_source)
        self.folder_entry = tk.Entry(
            row0, textvariable=self.folder_var, width=50,
            bg=t["input_bg"], fg=t["input_fg"], font=FONT,
            insertbackground=t["fg"], relief=tk.FLAT, bd=2,
        )
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        self.browse_btn = tk.Button(
            row0, text="Browse", command=self._on_browse, width=8,
            bg=t["accent"], fg=t["accent_fg"], font=FONT,
            relief=tk.FLAT, bd=0, padx=6, pady=2,
            activebackground=t["accent_hover"],
            activeforeground=t["accent_fg"],
        )
        self.browse_btn.pack(side=tk.LEFT, padx=(8, 0))

        # -- Row 1: Controls --
        row1 = tk.Frame(self, bg=t["panel_bg"])
        row1.pack(fill=tk.X, pady=(0, 4))

        self.start_btn = tk.Button(
            row1, text="Start Indexing", command=self._on_start, width=14,
            bg=t["accent"], fg=t["accent_fg"], font=FONT,
            relief=tk.FLAT, bd=0, padx=6, pady=2,
            activebackground=t["accent_hover"],
            activeforeground=t["accent_fg"],
        )
        self.start_btn.pack(side=tk.LEFT)

        self.stop_btn = tk.Button(
            row1, text="Stop", command=self._on_stop, width=8,
            state=tk.DISABLED,
            bg=t["inactive_btn_bg"], fg=t["inactive_btn_fg"], font=FONT,
            relief=tk.FLAT, bd=0, padx=6, pady=2,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.progress_file_label = tk.Label(
            row1, text="", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT,
        )
        self.progress_file_label.pack(side=tk.LEFT, padx=(16, 0), fill=tk.X, expand=True)

        # -- Row 2: Progress bar --
        row2 = tk.Frame(self, bg=t["panel_bg"])
        row2.pack(fill=tk.X, pady=(0, 4))

        self.progress_bar = ttk.Progressbar(
            row2, mode="determinate", length=400,
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_count_label = tk.Label(
            row2, text="0 / 0 files", anchor=tk.W, padx=8,
            bg=t["panel_bg"], fg=t["fg"], font=FONT,
        )
        self.progress_count_label.pack(side=tk.LEFT)

        # -- Row 3: Last run info --
        self.last_run_label = tk.Label(
            self, text="Last run: (none)", anchor=tk.W, fg=t["gray"],
            bg=t["panel_bg"], font=FONT,
        )
        self.last_run_label.pack(fill=tk.X)

    def apply_theme(self, t):
        """Re-apply theme colors to all widgets."""
        self.configure(bg=t["panel_bg"], fg=t["accent"])

        for row in self.winfo_children():
            if isinstance(row, tk.Frame):
                row.configure(bg=t["panel_bg"])
                for child in row.winfo_children():
                    if isinstance(child, tk.Label):
                        child.configure(bg=t["panel_bg"])
                        # Keep colored status labels
                        cur_fg = str(child.cget("fg"))
                        if cur_fg in ("#888888", "gray"):
                            child.configure(fg=t["gray"])
                        elif cur_fg not in ("green", "red", "orange",
                                            t["green"], t["red"], t["orange"]):
                            child.configure(fg=t["fg"])
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

        self.last_run_label.configure(bg=t["panel_bg"])
        if "none" in self.last_run_label.cget("text"):
            self.last_run_label.configure(fg=t["gray"])
        else:
            self.last_run_label.configure(fg=t["fg"])
        self.progress_count_label.configure(bg=t["panel_bg"], fg=t["fg"])

    def _on_browse(self):
        """Open folder picker dialog."""
        folder = filedialog.askdirectory(
            title="Select Source Document Folder",
            initialdir=self.folder_var.get() or None,
        )
        if folder:
            self.folder_var.set(folder)

    def _on_start(self):
        """Start indexing in a background thread."""
        t = current_theme()
        folder = self.folder_var.get().strip()
        if not folder:
            self.progress_file_label.config(
                text="[FAIL] No folder selected", fg=t["red"],
            )
            return

        if not os.path.isdir(folder):
            self.progress_file_label.config(
                text="[FAIL] Folder does not exist: {}".format(folder),
                fg=t["red"],
            )
            return

        if self.indexer is None:
            self.progress_file_label.config(
                text="[FAIL] Indexer not initialized. Run boot first.",
                fg=t["red"],
            )
            return

        # Reset UI
        self._stop_flag.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar["value"] = 0
        self.progress_count_label.config(text="0 / 0 files")
        self.progress_file_label.config(text="Starting...", fg=t["gray"])

        # Run in background
        self._index_thread = threading.Thread(
            target=self._run_indexing, args=(folder,), daemon=True,
        )
        self._index_thread.start()

    def _on_stop(self):
        """Signal the indexing thread to stop after current file."""
        t = current_theme()
        self._stop_flag.set()
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_file_label.config(text="Stopping after current file...",
                                        fg=t["orange"])

    def _run_indexing(self, folder):
        """Execute indexing in background thread with progress callback."""
        try:
            callback = _GUIProgressCallback(self)
            result = self.indexer.index_folder(
                folder, progress_callback=callback, recursive=True,
            )
            self.after(0, self._on_indexing_done, result)
        except Exception as e:
            error_msg = "[FAIL] {}: {}".format(type(e).__name__, e)
            self.after(0, self._on_indexing_error, error_msg)

    def _on_indexing_done(self, result):
        """Handle indexing completion (called on main thread)."""
        try:
            self._on_indexing_done_inner(result)
        except Exception as e:
            logger.error("Indexing done handler failed: %s", e)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _on_indexing_done_inner(self, result):
        """Inner handler (separated so outer can catch and re-enable)."""
        t = current_theme()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        total_chunks = result.get("total_chunks_added", 0)
        elapsed = result.get("elapsed_seconds", 0)
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        self.last_run_label.config(
            text="Last run: {} | {:,} chunks indexed | {:.0f}s".format(
                now, total_chunks, elapsed
            ),
            fg=t["fg"],
        )
        self.progress_file_label.config(
            text="[OK] Indexing complete", fg=t["green"],
        )

    def _on_indexing_error(self, error_msg):
        """Handle indexing error (called on main thread)."""
        t = current_theme()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_file_label.config(text=error_msg, fg=t["red"])


class _GUIProgressCallback:
    """
    Indexing progress callback that updates the GUI panel.

    Methods are called from the indexing background thread,
    so all GUI updates use panel.after() to run on the main thread.
    """

    def __init__(self, panel):
        self.panel = panel
        self._file_count = 0
        self._total_files = 0

    def on_file_start(self, file_path, file_num, total_files):
        """Called when a file starts processing."""
        self._total_files = total_files
        fname = os.path.basename(file_path)

        # Check stop flag
        if self.panel._stop_flag.is_set():
            raise InterruptedError("Indexing stopped by user")

        self.panel.after(0, self._update_file_start, fname, file_num, total_files)

    def _update_file_start(self, fname, file_num, total_files):
        t = current_theme()
        self.panel.progress_file_label.config(
            text="Processing: {}".format(fname), fg=t["gray"],
        )
        self.panel.progress_count_label.config(
            text="{} / {} files".format(file_num, total_files),
        )
        if total_files > 0:
            self.panel.progress_bar["maximum"] = total_files
            self.panel.progress_bar["value"] = file_num - 1

    def on_file_complete(self, file_path, chunks_created):
        """Called when a file finishes processing."""
        self._file_count += 1
        self.panel.after(0, self._update_file_complete)

    def _update_file_complete(self):
        self.panel.progress_bar["value"] = self._file_count

    def on_file_skipped(self, file_path, reason):
        """Called when a file is skipped."""
        self._file_count += 1
        self.panel.after(0, self._update_file_complete)

    def on_indexing_complete(self, total_chunks, elapsed_seconds):
        """Called when indexing finishes."""
        pass  # Handled by _on_indexing_done in the panel

    def on_error(self, file_path, error):
        """Called when a file has an error (continues to next file)."""
        t = current_theme()
        fname = os.path.basename(file_path)
        self.panel.after(
            0,
            lambda: self.panel.progress_file_label.config(
                text="[WARN] Error on {}: {}".format(fname, error[:60]),
                fg=t["orange"],
            ),
        )
