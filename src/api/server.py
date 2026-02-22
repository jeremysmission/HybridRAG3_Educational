# ============================================================================
# HybridRAG -- FastAPI Server (src/api/server.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Creates and configures the FastAPI application with lifecycle
#   management for RAG pipeline components (embedder, vector store,
#   LLM router, query engine, indexer).
#
# USAGE:
#   python -m src.api.server                    # Start on port 8000
#   python -m src.api.server --port 9000        # Custom port
#   python -m src.api.server --host 0.0.0.0     # Expose to network
#
# INTERNET ACCESS:
#   Offline mode: NONE (localhost Ollama only)
#   Online mode: API endpoint only (HuggingFace still blocked)
# ============================================================================

from __future__ import annotations

import os
import sys
import time
import threading
import argparse
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Ensure project root is on the path so imports work
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config import load_config, Config
from src.core.vector_store import VectorStore
from src.core.embedder import Embedder
from src.core.llm_router import LLMRouter
from src.core.query_engine import QueryEngine, QueryResult
from src.core.indexer import Indexer, IndexingProgressCallback


# -------------------------------------------------------------------
# Application version
# -------------------------------------------------------------------
APP_VERSION = "3.1.0"


# -------------------------------------------------------------------
# Shared state (populated during lifespan)
# -------------------------------------------------------------------
class AppState:
    """Mutable container for pipeline components."""
    config: Optional[Config] = None
    vector_store: Optional[VectorStore] = None
    embedder: Optional[Embedder] = None
    llm_router: Optional[LLMRouter] = None
    query_engine: Optional[QueryEngine] = None

    # Indexing state (background thread)
    indexing_active: bool = False
    indexing_lock: threading.Lock = threading.Lock()
    index_progress: dict = {
        "files_processed": 0,
        "files_total": 0,
        "files_skipped": 0,
        "files_errored": 0,
        "current_file": "",
        "start_time": 0.0,
    }


state = AppState()


# -------------------------------------------------------------------
# Indexing progress callback
# -------------------------------------------------------------------
class APIProgressCallback(IndexingProgressCallback):
    """Captures indexing progress for the /index/status endpoint."""

    def on_file_start(
        self, file_path: str, file_num: int, total_files: int
    ) -> None:
        state.index_progress["current_file"] = os.path.basename(file_path)
        state.index_progress["files_total"] = total_files

    def on_file_complete(
        self, file_path: str, chunks_created: int, file_num: int, total_files: int
    ) -> None:
        state.index_progress["files_processed"] += 1

    def on_file_skipped(
        self, file_path: str, reason: str, file_num: int, total_files: int
    ) -> None:
        state.index_progress["files_skipped"] += 1

    def on_error(
        self, file_path: str, error: str, file_num: int, total_files: int
    ) -> None:
        state.index_progress["files_errored"] += 1


# -------------------------------------------------------------------
# Lifespan: startup and shutdown
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup, clean up on shutdown."""
    # -- Startup --
    logger.info("[OK] Loading configuration...")
    state.config = load_config(".")

    logger.info("[OK] Connecting to vector store...")
    state.vector_store = VectorStore(
        state.config.paths.database,
        state.config.embedding.dimension,
    )
    state.vector_store.connect()

    logger.info("[OK] Loading embedding model...")
    state.embedder = Embedder(state.config.embedding.model_name)

    logger.info("[OK] Initializing LLM router...")
    state.llm_router = LLMRouter(state.config)

    logger.info("[OK] Building query engine...")
    state.query_engine = QueryEngine(
        state.config,
        state.vector_store,
        state.embedder,
        state.llm_router,
    )

    logger.info("[OK] FastAPI server ready.")
    yield

    # -- Shutdown --
    logger.info("[OK] Shutting down...")
    if state.vector_store:
        state.vector_store.close()
    if state.embedder:
        state.embedder.close()
    logger.info("[OK] Cleanup complete.")


# -------------------------------------------------------------------
# Create the FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="HybridRAG API",
    description=(
        "REST API for HybridRAG document search and question answering. "
        "Supports offline (Ollama) and online (API) modes with hybrid "
        "vector + keyword search."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# -------------------------------------------------------------------
# Register routes
# -------------------------------------------------------------------
from src.api.routes import router  # noqa: E402

app.include_router(router)


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
def main():
    """Run the server with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(description="HybridRAG FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    logger.info("[OK] Starting HybridRAG API on http://%s:%s", args.host, args.port)
    logger.info("[OK] API docs at http://%s:%s/docs", args.host, args.port)

    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
