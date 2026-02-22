# HybridRAG3 -- Educational Reference Implementation

An educational reference implementation of Retrieval-Augmented Generation (RAG)
patterns for studying AI engineering concepts.

## What This Is

A complete, working RAG system built from scratch with:
- Zero "magic" dependencies (no LangChain)
- Extensive code comments explaining every design decision
- Production-grade architecture patterns
- Full diagnostic and testing toolkit

## Purpose

This repository exists for **educational purposes** -- to study and learn:
- How RAG systems work at every layer
- Python engineering patterns and best practices
- Vector database design with SQLite + numpy
- LLM API integration (Azure OpenAI, Ollama)
- Diagnostic and fault analysis systems
- Security-conscious software design

## Architecture

```
src/
  core/           # Core RAG engine
    indexer.py        # Document ingestion pipeline
    chunker.py        # Text splitting with overlap
    chunk_ids.py      # Deterministic ID generation
    embedder.py       # Sentence-transformer embeddings
    vector_store.py   # SQLite + numpy vector search
    retriever.py      # Hybrid retrieval (vector + keyword)
    llm_router.py     # Multi-provider LLM routing
    config.py         # Dataclass-based configuration
  diagnostic/     # Health monitoring
  tools/          # System utilities
tools/
  py/             # Extracted Python toolkit scripts
  master_toolkit.ps1  # PowerShell command interface
tests/            # Test suites
config/           # YAML configuration templates
```

## Key Design Principles

1. **Zero Magic** -- Every operation is explicit and traceable
2. **Offline-First** -- Works without internet by default
3. **Auditable** -- Full logging, deterministic behavior
4. **Minimal Dependencies** -- Only what's needed, pinned versions
5. **Readable** -- Extensive comments for learning

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

See `config/default_config.yaml` for configuration options.

## Requirements

- Python 3.11+
- ~200MB disk for dependencies
- ~87MB for embedding model (downloads on first run)
- Optional: Ollama for local LLM inference
- Optional: Azure OpenAI API for cloud inference

## License

Educational and research use.
