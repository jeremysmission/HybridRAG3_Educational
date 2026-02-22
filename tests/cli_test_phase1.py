#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 - CLI Query Test (tests/cli_test_phase1.py)
# ============================================================================
# PURPOSE:
#   This is the script that runs when you type: rag-query "your question"
#   It wires together all the pieces of the RAG pipeline:
#     1. Load config (paths, model settings)
#     2. Connect to the SQLite database (where your indexed chunks live)
#     3. Load the embedding model (converts your question to a vector)
#     4. Create the LLM router (decides Ollama vs API)
#     5. Run the query through the QueryEngine pipeline
#     6. Display the answer with source citations
#
# USAGE:
#   python tests\cli_test_phase1.py --query "What is a digisonde?"
#   python tests\cli_test_phase1.py --query "frequency range" --mode online
#
# INTERNET ACCESS:
#   - offline mode: Connects to localhost only (Ollama) — NO internet
#   - online mode:  Connects to configured API endpoint — REQUIRES internet
# ============================================================================

import argparse
import sys
from pathlib import Path

# ============================================================================
# PATH FIX: Make sure Python can find the src/ package
# ============================================================================
# This file lives in HybridRAG3/tests/, but the src/ package is in
# HybridRAG3/src/. We need to add HybridRAG3/ (the parent of tests/)
# to Python's import search path so "from src.core.config import ..."
# works correctly.
#
# Path(__file__)         = .../HybridRAG3/tests/cli_test_phase1.py
# Path(__file__).parent  = .../HybridRAG3/tests/
# .parent.parent         = .../HybridRAG3/          <-- this is what we need
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import Config, load_config, validate_config, ensure_directories
from src.core.vector_store import VectorStore
from src.core.embedder import Embedder
from src.core.llm_router import LLMRouter
from src.core.query_engine import QueryEngine


def main():
    # --- Parse command-line arguments ---
    # argparse handles the --query and --mode flags you type after the command
    parser = argparse.ArgumentParser(description="HybridRAG v3 - Query CLI")
    parser.add_argument("--query", type=str, required=True, help="Query to run")
    parser.add_argument("--project-dir", type=str, default=str(PROJECT_ROOT),
                        help="Project directory (auto-detected)")
    parser.add_argument("--mode", type=str, choices=["offline", "online"],
                        default=None, help="Query mode (default: use config)")
    args = parser.parse_args()

    # --- Load configuration ---
    config = load_config(args.project_dir)
    errors = validate_config(config)
    if errors:
        for e in errors:
            print(f"Config error: {e}")
        return 1
    ensure_directories(config)

    # Only override mode if explicitly passed on command line
    # Otherwise use whatever is in default_config.yaml
    if args.mode:
        config.mode = args.mode

    # --- Initialize components ---
    print(f"Mode: {config.mode}")
    print(f"Database: {config.paths.database}")

    vs = VectorStore(db_path=config.paths.database,
                     embedding_dim=config.embedding.dimension)
    vs.connect()

    stats = vs.get_stats()
    print(f"Indexed: {stats.get('chunk_count', 0)} chunks from "
          f"{stats.get('source_count', 0)} files")
    print()

    if stats.get("chunk_count", 0) == 0:
        print("No indexed data. Run rag-index first.")
        return 1

    # Load embedding model (for converting your question into a vector)
    embedder = Embedder(config.embedding.model_name)

    # Create the LLM router (decides whether to use Ollama or API)
    llm_router = LLMRouter(config)

    # Wire everything together into the query engine
    query_engine = QueryEngine(config, vs, embedder, llm_router)

    # --- Check LLM availability before wasting time ---
    if config.mode == "offline":
        if not llm_router.ollama.is_available():
            print(f"\nOllama not running at {config.ollama.base_url}")
            print("Fix: Open a separate terminal and run:")
            print('  & "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe" serve')
            print("\nOr add Ollama to PATH first:")
            print('  $env:PATH += ";$env:LOCALAPPDATA\\Programs\\Ollama"')
            return 1

    # --- Run the query ---
    print(f"Query: {args.query}")
    print()

    result = query_engine.query(args.query)

    # --- Display the answer ---
    print("Answer:")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    print()

    # --- Display sources with relevance scores ---
    if result.sources:
        print("Retrieved Sources:")
        for source in result.sources:
            path = Path(source['path']).name
            chunks = source['chunks']
            relevance = source['avg_relevance']
            print(f"  - {path} ({chunks} chunks, relevance: {relevance:.0%})")
    print()

    # --- Display performance metadata ---
    print(f"Chunks used: {result.chunks_used}")
    if result.tokens_in:
        print(f"Tokens: {result.tokens_in} in / {result.tokens_out} out")
    if result.cost_usd > 0:
        print(f"Cost: ${result.cost_usd:.4f}")
    print(f"Latency: {result.latency_ms:.0f}ms")

    if result.error:
        print(f"Error: {result.error}")

    # --- Clean up resources to prevent memory leaks ---
    # BUG-003 pattern: always close what you open
    try:
        embedder.close()
    except Exception:
        pass
    try:
        vs.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
