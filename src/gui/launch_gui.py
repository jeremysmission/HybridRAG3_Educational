# ============================================================================
# HybridRAG v3 -- GUI Launcher (src/gui/launch_gui.py)
# ============================================================================
# Entry point: python src/gui/launch_gui.py
#
# Opens the GUI window FIRST, then loads backends in a background thread.
# This ensures the user sees the window immediately instead of waiting
# for heavy imports (torch, sentence-transformers) to finish.
#
# INTERNET ACCESS: Depends on boot result and user mode selection.
# ============================================================================

import os
import sys
import logging
import threading

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Set HYBRIDRAG_PROJECT_ROOT if not already set
if not os.environ.get("HYBRIDRAG_PROJECT_ROOT"):
    os.environ["HYBRIDRAG_PROJECT_ROOT"] = _project_root


def _load_backends(app, logger):
    """Load heavy backends in a background thread, then attach to the GUI."""
    config = app.config
    query_engine = None
    indexer = None
    router = None

    try:
        logger.info("Loading backends (this may take a moment)...")
        from src.core.vector_store import VectorStore
        from src.core.embedder import Embedder
        from src.core.llm_router import LLMRouter
        from src.core.query_engine import QueryEngine
        from src.core.chunker import Chunker
        from src.core.indexer import Indexer

        # Vector store
        db_path = getattr(getattr(config, "paths", None), "database", "")
        store = None
        if db_path and os.path.exists(os.path.dirname(db_path) or "."):
            store = VectorStore(
                db_path=db_path,
                embedding_dim=getattr(
                    getattr(config, "embedding", None), "dimension", 384
                ),
            )
            store.connect()
            logger.info("[OK] Vector store connected")
        else:
            logger.warning("[WARN] No database path configured")

        # Embedder
        model_name = getattr(
            getattr(config, "embedding", None), "model_name",
            "all-MiniLM-L6-v2"
        )
        embedder = Embedder(model_name=model_name)
        logger.info("[OK] Embedder loaded")

        # LLM Router
        router = LLMRouter(config)
        logger.info("[OK] LLM router ready")

        # Query Engine
        if store:
            query_engine = QueryEngine(config, store, embedder, router)
            logger.info("[OK] Query engine ready")

        # Indexer
        if store:
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
        if hasattr(app, "index_panel"):
            app.index_panel.indexer = indexer
        if hasattr(app, "status_bar"):
            app.status_bar.router = router
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

    # -- Step 3: Open GUI immediately (no heavy imports yet) --
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
