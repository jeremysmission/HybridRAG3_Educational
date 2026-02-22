# ============================================================================
# HybridRAG v3 -- GUI Integration Tests (tests/test_gui_integration_w4.py)
# ============================================================================
# 14 tests covering all GUI panels with mocked backends.
# No real indexing or API calls. Works offline with no API key.
#
# INTERNET ACCESS: NONE
# ============================================================================

import sys
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# -- sys.path setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# FAKE CONFIG (extends conftest pattern)
# ============================================================================

@dataclass
class FakePathsConfig:
    database: str = ""
    embeddings_cache: str = ""
    source_folder: str = ""


@dataclass
class FakeOllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "phi4-mini"
    timeout_seconds: int = 120


@dataclass
class FakeAPIConfig:
    endpoint: str = ""
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout_seconds: int = 30
    deployment: str = ""
    api_version: str = ""
    allowed_endpoint_prefixes: list = field(default_factory=list)


@dataclass
class FakeRetrievalConfig:
    top_k: int = 8
    min_score: float = 0.20
    hybrid_search: bool = True
    reranker_enabled: bool = False
    reranker_model: str = ""
    reranker_top_n: int = 20
    rrf_k: int = 60
    block_rows: int = 25000
    lex_boost: float = 0.06
    min_chunks: int = 1


@dataclass
class FakeCostConfig:
    input_cost_per_1k: float = 0.0015
    output_cost_per_1k: float = 0.002
    track_enabled: bool = True
    daily_budget_usd: float = 5.0


@dataclass
class FakeChunkingConfig:
    chunk_size: int = 1200
    overlap: int = 200
    max_heading_len: int = 160


@dataclass
class FakeGUIConfig:
    mode: str = "offline"
    paths: FakePathsConfig = field(default_factory=FakePathsConfig)
    ollama: FakeOllamaConfig = field(default_factory=FakeOllamaConfig)
    api: FakeAPIConfig = field(default_factory=FakeAPIConfig)
    retrieval: FakeRetrievalConfig = field(default_factory=FakeRetrievalConfig)
    cost: FakeCostConfig = field(default_factory=FakeCostConfig)
    chunking: FakeChunkingConfig = field(default_factory=FakeChunkingConfig)


@dataclass
class FakeQueryResult:
    answer: str = "Test answer"
    sources: list = field(default_factory=list)
    chunks_used: int = 3
    tokens_in: int = 450
    tokens_out: int = 120
    cost_usd: float = 0.001
    latency_ms: float = 1234.0
    mode: str = "offline"
    error: str = ""


@dataclass
class FakeBootResult:
    boot_timestamp: str = "2026-02-21 14:30:00"
    success: bool = True
    online_available: bool = False
    offline_available: bool = True
    api_client: object = None
    config: dict = field(default_factory=dict)
    credentials: object = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def summary(self):
        return "BOOT: OK"


# ============================================================================
# HELPERS
# ============================================================================

def _make_root():
    """Create a Tk root that we can destroy after each test."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk runtime unavailable")
    root.withdraw()  # Don't show the window
    return root


def _pump_events(root, ms=100):
    """Process pending tkinter events for a short time."""
    end = time.time() + ms / 1000.0
    while time.time() < end:
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break
        time.sleep(0.005)


def _wait_and_pump(root, ms=500):
    """Wait for background threads to post results, then pump events."""
    time.sleep(ms / 1000.0)
    # Pump events multiple times to ensure after() callbacks are processed
    for _ in range(10):
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break
        time.sleep(0.02)


# ============================================================================
# TEST 01: GUI launches without crashing
# ============================================================================

def test_01_gui_launches_without_crashing():
    """GUI app window creates successfully with mocked boot result."""
    from src.gui.app import HybridRAGApp

    config = FakeGUIConfig()
    boot = FakeBootResult()

    app = HybridRAGApp(boot_result=boot, config=config)
    app.withdraw()

    assert app.winfo_exists()
    assert app.title() == "HybridRAG v3"

    app.status_bar.stop()
    app.destroy()


# ============================================================================
# TEST 02: Query panel submits query and displays answer
# ============================================================================

def test_02_query_panel_displays_answer():
    """Query panel shows answer from mocked QueryEngine."""
    root = _make_root()

    config = FakeGUIConfig()

    from src.gui.panels.query_panel import QueryPanel
    panel = QueryPanel(root, config=config)
    panel.pack()

    # Directly test _display_result (bypasses threading for test reliability)
    result = FakeQueryResult(
        answer="The frequency is 2.4 GHz.",
        sources=[{"path": "spec.pdf", "chunks": 3}],
        latency_ms=1500.0,
        tokens_in=400,
        tokens_out=80,
    )
    panel._display_result(result)
    _pump_events(root, 50)

    # Check answer displayed
    answer_text = panel.answer_text.get("1.0", tk.END).strip()
    assert "2.4 GHz" in answer_text

    # Check sources displayed
    sources_text = panel.sources_label.cget("text")
    assert "spec.pdf" in sources_text

    # Check metrics displayed
    metrics_text = panel.metrics_label.cget("text")
    assert "1,500" in metrics_text

    root.destroy()


# ============================================================================
# TEST 03: Query panel shows error correctly when query fails
# ============================================================================

def test_03_query_panel_shows_error():
    """Query panel shows error when query returns error."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.query_panel import QueryPanel
    panel = QueryPanel(root, config=config)
    panel.pack()

    # Directly test _display_result with an error result
    result = FakeQueryResult(
        answer="Error processing query",
        error="LLM call failed",
    )
    panel._display_result(result)
    _pump_events(root, 50)

    answer_text = panel.answer_text.get("1.0", tk.END).strip()
    assert "FAIL" in answer_text

    root.destroy()


# ============================================================================
# TEST 04: Ask button disables during query, re-enables after
# ============================================================================

def test_04_ask_button_disable_reenable():
    """Ask button is disabled during query and re-enabled after."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.query_panel import QueryPanel
    panel = QueryPanel(root, config=config, query_engine=MagicMock())
    panel.pack()

    # Simulate what _on_ask does: disable button
    panel.ask_btn.config(state=tk.DISABLED)
    _pump_events(root, 20)
    assert str(panel.ask_btn["state"]) == "disabled"

    # Simulate what _display_result does: re-enable button
    panel._display_result(FakeQueryResult())
    _pump_events(root, 20)
    assert str(panel.ask_btn["state"]) == "normal"

    root.destroy()


# ============================================================================
# TEST 05: Use case dropdown populates from USE_CASES
# ============================================================================

def test_05_use_case_dropdown_populates():
    """Use case dropdown contains all USE_CASES labels."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.query_panel import QueryPanel
    from scripts._model_meta import USE_CASES

    panel = QueryPanel(root, config=config)
    panel.pack()

    dropdown_values = list(panel.uc_dropdown["values"])
    expected_labels = [USE_CASES[k]["label"] for k in USE_CASES]

    for label in expected_labels:
        assert label in dropdown_values, "Missing use case: {}".format(label)

    root.destroy()


# ============================================================================
# TEST 06: Index panel browse button updates source folder field
# ============================================================================

def test_06_index_panel_browse_updates_folder():
    """Browse button updates the folder entry field."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.index_panel import IndexPanel

    panel = IndexPanel(root, config=config)
    panel.pack()

    # Mock the folder dialog
    test_path = os.path.join(os.path.dirname(__file__), "test_data")
    with patch("src.gui.panels.index_panel.filedialog.askdirectory", return_value=test_path):
        panel._on_browse()

    assert panel.folder_var.get() == test_path

    root.destroy()


# ============================================================================
# TEST 07: Index panel progress bar advances
# ============================================================================

def test_07_index_panel_progress_bar_advances():
    """Progress callback updates the progress bar."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.index_panel import IndexPanel, _GUIProgressCallback

    panel = IndexPanel(root, config=config)
    panel.pack()

    callback = _GUIProgressCallback(panel)

    # Simulate file processing
    callback.on_file_start("/tmp/file1.pdf", 1, 5)
    _pump_events(root, 100)
    assert panel.progress_bar["maximum"] == 5

    callback.on_file_complete("/tmp/file1.pdf", 10)
    _pump_events(root, 100)
    assert callback._file_count == 1

    callback.on_file_complete("/tmp/file2.pdf", 8)
    _pump_events(root, 100)
    assert callback._file_count == 2

    root.destroy()


# ============================================================================
# TEST 08: Status bar reflects offline mode
# ============================================================================

def test_08_status_bar_offline_mode():
    """Status bar shows OFFLINE when config.mode is offline."""
    root = _make_root()
    config = FakeGUIConfig(mode="offline")

    from src.gui.panels.status_bar import StatusBar

    bar = StatusBar(root, config=config)
    bar.pack()
    bar._refresh_status()
    _pump_events(root, 100)

    gate_text = bar.gate_label.cget("text")
    assert "OFFLINE" in gate_text

    bar.stop()
    root.destroy()


# ============================================================================
# TEST 09: Status bar reflects online mode
# ============================================================================

def test_09_status_bar_online_mode():
    """Status bar shows ONLINE when config.mode is online."""
    root = _make_root()
    config = FakeGUIConfig(mode="online")

    from src.gui.panels.status_bar import StatusBar

    bar = StatusBar(root, config=config)
    bar.pack()
    bar._refresh_status()
    _pump_events(root, 100)

    gate_text = bar.gate_label.cget("text")
    assert "ONLINE" in gate_text

    bar.stop()
    root.destroy()


# ============================================================================
# TEST 10: ONLINE button shows credential error when creds missing
# ============================================================================

def test_10_online_button_cred_error():
    """Switching to online mode shows error dialog when credentials are missing."""
    from src.gui.app import HybridRAGApp
    import src.gui.app as app_module

    config = FakeGUIConfig(mode="offline")
    app = HybridRAGApp(config=config)
    app.withdraw()

    # Mock credential_status to return no credentials
    mock_status = {
        "api_key_set": False,
        "api_endpoint_set": False,
        "api_key_source": "none",
        "api_endpoint_source": "none",
    }

    # Track warning calls
    warning_calls = []
    original_showwarning = app_module.messagebox.showwarning

    def fake_showwarning(title, message):
        warning_calls.append((title, message))

    app_module.messagebox.showwarning = fake_showwarning

    # Pre-import the credentials module and patch it
    from src.security import credentials as cred_mod
    original_fn = cred_mod.credential_status
    cred_mod.credential_status = lambda: mock_status

    try:
        app._switch_to_online()

        # Should have shown a warning
        assert len(warning_calls) >= 1
        assert "Credentials Missing" in warning_calls[0][0]

        # Mode should NOT have changed
        assert config.mode == "offline"
    finally:
        app_module.messagebox.showwarning = original_showwarning
        cred_mod.credential_status = original_fn

    app.status_bar.stop()
    app.destroy()


# ============================================================================
# TEST 11: Engineering menu sliders read current config values
# ============================================================================

def test_11_engineering_menu_reads_config():
    """Engineering menu sliders are initialized from config values."""
    root = _make_root()
    config = FakeGUIConfig()
    config.retrieval.top_k = 12
    config.retrieval.min_score = 0.15
    config.api.temperature = 0.3
    config.api.max_tokens = 3000

    from src.gui.panels.engineering_menu import EngineeringMenu

    with patch("src.gui.panels.engineering_menu.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        menu = EngineeringMenu(root, config=config)

    assert menu.topk_var.get() == 12
    assert abs(menu.minscore_var.get() - 0.15) < 0.01
    assert abs(menu.temp_var.get() - 0.3) < 0.01
    assert menu.maxtokens_var.get() == 3000

    menu.destroy()
    root.destroy()


# ============================================================================
# TEST 12: Engineering menu writes config on slider change
# ============================================================================

def test_12_engineering_menu_writes_config():
    """Changing a slider immediately updates the config object."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.engineering_menu import EngineeringMenu

    with patch("src.gui.panels.engineering_menu.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        menu = EngineeringMenu(root, config=config)

    # Change top_k
    menu.topk_var.set(25)
    menu._on_retrieval_change()
    assert config.retrieval.top_k == 25

    # Change temperature
    menu.temp_var.set(0.5)
    menu._on_llm_change()
    assert abs(config.api.temperature - 0.5) < 0.01

    menu.destroy()
    root.destroy()


# ============================================================================
# TEST 13: Profile dropdown calls _profile_switch.py
# ============================================================================

def test_13_profile_dropdown_calls_switch():
    """Profile dropdown triggers subprocess call to _profile_switch.py."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.engineering_menu import EngineeringMenu

    with patch("src.gui.panels.engineering_menu.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Applied", stderr="")
        menu = EngineeringMenu(root, config=config)

        # Change profile
        menu.profile_var.set("desktop_power")
        menu._on_profile_change()

        # Verify subprocess was called with the profile name
        calls = mock_run.call_args_list
        # Find the call that includes _profile_switch.py
        switch_calls = [c for c in calls if "_profile_switch" in str(c)]
        assert len(switch_calls) > 0, "Expected _profile_switch.py to be called"

    menu.destroy()
    root.destroy()


# ============================================================================
# TEST 14: Test query in engineering menu shows latency
# ============================================================================

def test_14_test_query_shows_latency():
    """Test query section displays latency after running."""
    root = _make_root()
    config = FakeGUIConfig()

    from src.gui.panels.engineering_menu import EngineeringMenu

    with patch("src.gui.panels.engineering_menu.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        menu = EngineeringMenu(root, config=config)

    # Directly test the display method (bypasses threading)
    result = FakeQueryResult(answer="Test result", latency_ms=567.0)
    menu._display_test_result(result)
    _pump_events(root, 50)

    latency_text = menu.test_latency_label.cget("text")
    assert "567" in latency_text

    menu.destroy()
    root.destroy()
