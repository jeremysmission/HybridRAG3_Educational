# ============================================================================
# HybridRAG v3 -- GUI Launcher (src/gui/launch_gui.py)
# ============================================================================
# Entry point: python src/gui/launch_gui.py
#
# Opens the GUI window FIRST, then loads backends in a background thread.
# This ensures the user sees the window immediately instead of waiting
# for heavy imports (torch, sentence-transformers) to finish.
#
# PERFORMANCE: The Embedder is the cold-start bottleneck (~16s on 8GB
# laptop). Three tricks to minimize perceived wait:
#   1. Eager preload -- start building the Embedder at t=0, BEFORE
#      boot/config/GUI, so the 16s overlaps with the 2s of setup.
#   2. Embedder cache -- keep the Embedder across Reset clicks so the
#      7.6s model-load is paid only once per process lifetime.
#   3. Warm encode -- fire a dummy encode() after load so the first
#      real query pays zero lazy-init cost.
#
# INTERNET ACCESS: Depends on boot result and user mode selection.
# ============================================================================

import os
import sys
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Set HYBRIDRAG_PROJECT_ROOT if not already set
if not os.environ.get("HYBRIDRAG_PROJECT_ROOT"):
    os.environ["HYBRIDRAG_PROJECT_ROOT"] = _project_root

# ============================================================================
# EAGER PRELOAD: start the heaviest work (import torch + sentence-transformers
# + load model weights) immediately, before boot/config/GUI.  The result is
# stashed in _preload_result and picked up by _load_backends() later.
# On this laptop the overlap saves ~2s; on faster hardware the ratio is even
# better because boot+config+GUI take longer relative to model-load time.
# ============================================================================

_preload_result = {}   # {"embedder": Embedder | None, "error": str | None}
_preload_done = threading.Event()


def _preload_embedder():
    """Build the Embedder (torch + model weights) as early as possible."""
    try:
        from src.core.embedder import Embedder
        e = Embedder()   # default model_name = "all-MiniLM-L6-v2"
        # Warm encode: force any lazy init (tokenizer buffers, etc.)
        e.embed_query("warmup")
        _preload_result["embedder"] = e
        _preload_result["error"] = None
    except Exception as exc:
        _preload_result["embedder"] = None
        _preload_result["error"] = str(exc)
    finally:
        _preload_done.set()


_preload_thread = threading.Thread(target=_preload_embedder, daemon=True)
_preload_thread.start()

# ============================================================================
# Module-level embedder cache -- survives Reset clicks so the expensive
# model-load is paid once per process.  _load_backends() stores the
# Embedder here after first use; reset_backends() in app.py re-reads it.
# ============================================================================

_cached_embedder = None
_cached_embedder_lock = threading.Lock()


def _get_or_build_embedder(model_name, logger):
    """Return a cached Embedder if model_name matches, else build a new one."""
    global _cached_embedder

    # Try the preload first (only blocks if preload is still running)
    if not _preload_done.is_set():
        logger.info("Waiting for eager preload to finish...")
    _preload_done.wait()

    with _cached_embedder_lock:
        # Use cached if model matches
        if (_cached_embedder is not None
                and _cached_embedder.model is not None
                and _cached_embedder.model_name == model_name):
            logger.info("[OK] Embedder reused from cache")
            return _cached_embedder

        # Use preload result if model matches and cache is empty
        preloaded = _preload_result.get("embedder")
        if (preloaded is not None
                and preloaded.model is not None
                and preloaded.model_name == model_name):
            _cached_embedder = preloaded
            logger.info("[OK] Embedder loaded (from eager preload)")
            return _cached_embedder

        # Fallback: build fresh (different model_name or preload failed)
        from src.core.embedder import Embedder
        e = Embedder(model_name=model_name)
        e.embed_query("warmup")
        _cached_embedder = e
        logger.info("[OK] Embedder loaded (fresh build)")
        return _cached_embedder


def _set_stage(app, stage_text):
    """Schedule a loading-stage update on the GUI main thread."""
    try:
        app.after(0, lambda: (
            app.status_bar.set_loading_stage(stage_text)
            if hasattr(app, "status_bar") else None
        ))
    except Exception:
        pass


def _load_backends(app, logger):
    """Load heavy backends in a background thread, then attach to the GUI."""
    config = app.config
    query_engine = None
    indexer = None
    router = None
    store = None
    embedder = None

    try:
        logger.info("Loading backends (this may take a moment)...")
        from src.core.vector_store import VectorStore
        from src.core.llm_router import LLMRouter
        from src.core.query_engine import QueryEngine
        from src.core.chunker import Chunker
        from src.core.indexer import Indexer

        model_name = getattr(
            getattr(config, "embedding", None), "model_name",
            "all-MiniLM-L6-v2"
        )

        # -- Parallel phase: VectorStore, Embedder, LLMRouter --
        def _init_store():
            _set_stage(app, "VectorStore...")
            db_path = getattr(getattr(config, "paths", None), "database", "")
            if db_path and os.path.exists(os.path.dirname(db_path) or "."):
                s = VectorStore(
                    db_path=db_path,
                    embedding_dim=getattr(
                        getattr(config, "embedding", None), "dimension", 384
                    ),
                )
                s.connect()
                logger.info("[OK] Vector store connected")
                return s
            logger.warning("[WARN] No database path configured")
            return None

        def _init_embedder():
            _set_stage(app, "Embedder...")
            return _get_or_build_embedder(model_name, logger)

        def _init_router():
            _set_stage(app, "LLM Router...")
            r = LLMRouter(config)
            logger.info("[OK] LLM router ready")
            return r

        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_store = pool.submit(_init_store)
            fut_embedder = pool.submit(_init_embedder)
            fut_router = pool.submit(_init_router)

            store = fut_store.result()
            embedder = fut_embedder.result()
            router = fut_router.result()

        # -- Sequential phase: assemble QueryEngine + Indexer --
        _set_stage(app, "QueryEngine...")
        if store and embedder:
            query_engine = QueryEngine(config, store, embedder, router)
            logger.info("[OK] Query engine ready")

            chunker = Chunker(config)
            indexer = Indexer(config, store, embedder, chunker)
            logger.info("[OK] Indexer ready")

    except Exception as e:
        logger.warning("[WARN] Backend loading partial: %s", e)

    # Attach backends to the GUI (schedule on main thread)
    def _attach():
        app.query_engine = query_engine
        app.indexer = indexer
        app.router = router
        if hasattr(app, "query_panel"):
            app.query_panel.query_engine = query_engine
            app.query_panel.set_ready(query_engine is not None)
        if hasattr(app, "index_panel"):
            app.index_panel.indexer = indexer
            app.index_panel.set_ready(indexer is not None)
        if hasattr(app, "status_bar"):
            app.status_bar.router = router
            app.status_bar.set_ready()
            app.status_bar.force_refresh()
        logger.info("[OK] Backends attached to GUI")

    try:
        app.after(0, _attach)
    except Exception as e:
        logger.debug("after() failed during backend attach: %s", e)


def main():
    """Boot config, open GUI immediately, load backends in background."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("gui_launcher")

    # NOTE: _preload_thread is already running (started at module load).
    # While we boot + load config + build the GUI (~2s), torch and the
    # embedding model are loading in parallel.

    # -- Step 1: Boot the system (lightweight -- config + creds + gate) --
    logger.info("Booting HybridRAG...")
    boot_result = None
    config = None

    try:
        from src.core.boot import boot_hybridrag
        boot_result = boot_hybridrag()
        if boot_result.success:
            logger.info("[OK] Boot succeeded")
        else:
            logger.warning("[WARN] Boot completed with errors")
            for err in boot_result.errors:
                logger.warning("  %s", err)
    except Exception as e:
        logger.error("[FAIL] Boot failed: %s", e)

    # -- Step 2: Load config --
    try:
        from src.core.config import load_config
        config = load_config(_project_root)
        logger.info("[OK] Config loaded (mode=%s)", config.mode)
    except Exception as e:
        logger.warning("[WARN] Config load failed, using defaults: %s", e)
        from src.core.config import Config
        config = Config()

    # -- Step 3: Open GUI immediately --
    logger.info("Opening GUI window...")
    from src.gui.app import HybridRAGApp

    app = HybridRAGApp(
        boot_result=boot_result,
        config=config,
    )

    # -- Step 4: Load backends in background thread --
    backend_thread = threading.Thread(
        target=_load_backends, args=(app, logger), daemon=True,
    )
    backend_thread.start()

    # -- Step 5: Run the GUI event loop --
    app.mainloop()


if __name__ == "__main__":
    main()
