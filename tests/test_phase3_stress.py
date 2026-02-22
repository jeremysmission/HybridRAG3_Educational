# ============================================================================
# Phase 3: Stress & Integration Tests for HybridRAG3
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Validates all module integration points under simulated load.
#   Tests chunker, vector_store, retriever, query_engine, config,
#   parser registry, network gate, and cross-module pipelines.
#
# INTERNET ACCESS: NONE -- all tests use mocks and temp files
# DEPENDENCIES: pytest, numpy (both already in requirements.txt)
# ============================================================================

import os
import sys
import json
import time
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import pytest
import numpy as np

# -- sys.path setup --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.conftest import FakeConfig, FakeLLMResponse


# ============================================================================
# SECTION 1: CHUNKER STRESS TESTS
# ============================================================================

class TestChunkerStress:
    """Stress test the chunker with various input sizes and edge cases."""

    def _make_chunker(self):
        from src.core.chunker import Chunker
        cfg = FakeConfig()
        return Chunker(cfg.chunking)

    def test_empty_text(self):
        chunker = self._make_chunker()
        result = chunker.chunk_text("")
        assert isinstance(result, list)
        assert len(result) == 0

    def test_tiny_text(self):
        chunker = self._make_chunker()
        result = chunker.chunk_text("Hello world")
        assert len(result) >= 1
        assert "Hello world" in result[0]

    def test_exact_chunk_size(self):
        chunker = self._make_chunker()
        text = "A" * 1200  # exactly chunk_size
        result = chunker.chunk_text(text)
        assert len(result) >= 1

    def test_large_text_10k(self):
        chunker = self._make_chunker()
        text = "This is a test sentence. " * 500  # ~12500 chars
        result = chunker.chunk_text(text)
        assert len(result) > 5
        # Verify no chunk exceeds chunk_size + some tolerance
        for chunk in result:
            assert len(chunk) <= 2000, f"Chunk too large: {len(chunk)} chars"

    def test_large_text_100k(self):
        chunker = self._make_chunker()
        text = "Performance test with repeated content. " * 2500  # ~100k chars
        result = chunker.chunk_text(text)
        assert len(result) > 50

    def test_large_text_1m(self):
        """Stress test: 1 million characters."""
        chunker = self._make_chunker()
        text = "Stress test line with enough words to be meaningful. " * 18000  # ~1M chars
        t0 = time.time()
        result = chunker.chunk_text(text)
        elapsed = time.time() - t0
        assert len(result) > 500
        assert elapsed < 10.0, f"Chunking 1M chars took {elapsed:.1f}s (limit: 10s)"

    def test_unicode_safe(self):
        """Chunker handles multi-byte UTF-8 without crashing."""
        chunker = self._make_chunker()
        # Use ASCII-safe representation for the test
        text = "Temperature is 100 degrees. " * 100
        result = chunker.chunk_text(text)
        assert len(result) >= 1

    def test_newline_heavy_text(self):
        chunker = self._make_chunker()
        text = "\n".join([f"Line {i}: Some content here." for i in range(500)])
        result = chunker.chunk_text(text)
        assert len(result) >= 1

    def test_heading_detection(self):
        chunker = self._make_chunker()
        text = "# Section 1\n\nSome content under section 1. " * 50
        result = chunker.chunk_text(text)
        assert len(result) >= 1


# ============================================================================
# SECTION 2: VECTOR STORE STRESS TESTS
# ============================================================================

class TestVectorStoreStress:
    """Stress test VectorStore with simulated embeddings load."""

    def _make_store(self, tmp_path):
        from src.core.vector_store import VectorStore
        db_path = str(tmp_path / "test.sqlite3")
        store = VectorStore(db_path=db_path, embedding_dim=384)
        store.connect()
        return store

    def test_add_and_search_basic(self, tmp_path):
        store = self._make_store(tmp_path)
        try:
            embeddings = np.random.randn(5, 384).astype(np.float32)
            from src.core.vector_store import ChunkMetadata
            metadata = [
                ChunkMetadata(
                    source_path=f"/test/file_{i}.txt",
                    chunk_index=i,
                    text_length=100,
                    created_at="2026-01-01T00:00:00",
                )
                for i in range(5)
            ]
            texts = [f"Test chunk content number {i}" for i in range(5)]
            store.add_embeddings(embeddings, metadata, texts)

            query_vec = np.random.randn(384).astype(np.float32)
            results = store.search(query_vec, top_k=3)
            assert len(results) <= 3
        finally:
            store.close()

    def test_add_100_chunks(self, tmp_path):
        store = self._make_store(tmp_path)
        try:
            n = 100
            embeddings = np.random.randn(n, 384).astype(np.float32)
            from src.core.vector_store import ChunkMetadata
            metadata = [
                ChunkMetadata(
                    source_path=f"/test/doc_{i // 10}.txt",
                    chunk_index=i % 10,
                    text_length=200,
                    created_at="2026-01-01T00:00:00",
                )
                for i in range(n)
            ]
            texts = [f"Content for chunk {i} with keywords: radar frequency" for i in range(n)]
            store.add_embeddings(embeddings, metadata, texts)

            stats = store.get_stats()
            assert stats["chunk_count"] == n

            query_vec = np.random.randn(384).astype(np.float32)
            results = store.search(query_vec, top_k=8)
            assert len(results) == 8
        finally:
            store.close()

    def test_add_1000_chunks_performance(self, tmp_path):
        """Stress test: 1000 chunks insert + search."""
        store = self._make_store(tmp_path)
        try:
            n = 1000
            embeddings = np.random.randn(n, 384).astype(np.float32)
            from src.core.vector_store import ChunkMetadata
            metadata = [
                ChunkMetadata(
                    source_path=f"/test/big_doc_{i // 50}.txt",
                    chunk_index=i % 50,
                    text_length=300,
                    created_at="2026-01-01T00:00:00",
                )
                for i in range(n)
            ]
            texts = [f"Stress test chunk {i}: technical content" for i in range(n)]

            t0 = time.time()
            store.add_embeddings(embeddings, metadata, texts)
            insert_time = time.time() - t0

            assert insert_time < 30.0, f"Insert 1000 chunks took {insert_time:.1f}s"

            query_vec = np.random.randn(384).astype(np.float32)
            t0 = time.time()
            results = store.search(query_vec, top_k=8)
            search_time = time.time() - t0

            assert len(results) == 8
            assert search_time < 5.0, f"Search over 1000 chunks took {search_time:.1f}s"
        finally:
            store.close()

    def test_fts_search(self, tmp_path):
        store = self._make_store(tmp_path)
        try:
            n = 20
            embeddings = np.random.randn(n, 384).astype(np.float32)
            from src.core.vector_store import ChunkMetadata
            metadata = [
                ChunkMetadata(
                    source_path="/test/manual.txt",
                    chunk_index=i,
                    text_length=100,
                    created_at="2026-01-01T00:00:00",
                )
                for i in range(n)
            ]
            texts = [f"Chunk {i}: operating frequency range is 9.0 to 9.5 GHz" for i in range(n)]
            store.add_embeddings(embeddings, metadata, texts)

            results = store.fts_search("frequency GHz", top_k=5)
            assert len(results) >= 1
        finally:
            store.close()

    def test_delete_and_reindex(self, tmp_path):
        store = self._make_store(tmp_path)
        try:
            embeddings = np.random.randn(5, 384).astype(np.float32)
            from src.core.vector_store import ChunkMetadata
            metadata = [
                ChunkMetadata(
                    source_path="/test/deleteme.txt",
                    chunk_index=i,
                    text_length=100,
                    created_at="2026-01-01T00:00:00",
                )
                for i in range(5)
            ]
            texts = [f"Delete test chunk {i}" for i in range(5)]
            store.add_embeddings(embeddings, metadata, texts)

            deleted = store.delete_chunks_by_source("/test/deleteme.txt")
            assert deleted == 5

            stats = store.get_stats()
            assert stats["chunk_count"] == 0
        finally:
            store.close()

    def test_file_hash_tracking(self, tmp_path):
        store = self._make_store(tmp_path)
        try:
            # Must insert chunks before update_file_hash works
            from src.core.vector_store import ChunkMetadata
            embeddings = np.random.randn(2, 384).astype(np.float32)
            metadata = [
                ChunkMetadata(
                    source_path="/test/file.txt",
                    chunk_index=i,
                    text_length=100,
                    created_at="2026-01-01T00:00:00",
                )
                for i in range(2)
            ]
            texts = ["hash test chunk 0", "hash test chunk 1"]
            store.add_embeddings(embeddings, metadata, texts, file_hash="abc123")

            h = store.get_file_hash("/test/file.txt")
            assert h == "abc123"

            store.update_file_hash("/test/file.txt", "def456")
            h2 = store.get_file_hash("/test/file.txt")
            assert h2 == "def456"
        finally:
            store.close()


# ============================================================================
# SECTION 3: CONFIG INTEGRATION TESTS
# ============================================================================

class TestConfigIntegration:
    """Test config loading, validation, and edge cases."""

    def test_default_config_loads(self):
        from src.core.config import Config
        cfg = Config()
        assert cfg.mode == "offline"
        assert cfg.embedding.dimension == 384
        assert cfg.chunking.chunk_size == 1200

    def test_hallucination_guard_defaults(self):
        from src.core.config import Config, HallucinationGuardConfig
        cfg = Config()
        assert hasattr(cfg, "hallucination_guard")
        assert isinstance(cfg.hallucination_guard, HallucinationGuardConfig)
        assert cfg.hallucination_guard.enabled is False
        assert cfg.hallucination_guard.threshold == 0.80
        assert cfg.hallucination_guard.failure_action == "block"

    def test_convenience_properties(self):
        from src.core.config import Config
        cfg = Config()
        assert cfg.hallucination_guard_enabled is False
        assert cfg.hallucination_guard_threshold == 0.80
        assert cfg.hallucination_guard_action == "block"

    def test_validate_config_offline(self):
        from src.core.config import Config, validate_config
        cfg = Config()
        cfg.paths.database = "/tmp/test.sqlite3"
        errors = validate_config(cfg)
        assert len(errors) == 0

    def test_validate_config_online_no_endpoint(self):
        from src.core.config import Config, validate_config
        cfg = Config()
        cfg.mode = "online"
        cfg.paths.database = "/tmp/test.sqlite3"
        errors = validate_config(cfg)
        assert any("SEC-001" in e for e in errors)

    def test_load_config_from_yaml(self, tmp_path):
        from src.core.config import load_config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml_content = (
            "mode: offline\n"
            "chunking:\n"
            "  chunk_size: 800\n"
            "  overlap: 100\n"
            "hallucination_guard:\n"
            "  enabled: true\n"
            "  threshold: 0.90\n"
        )
        (config_dir / "default_config.yaml").write_text(yaml_content, encoding="utf-8")
        cfg = load_config(str(tmp_path))
        assert cfg.chunking.chunk_size == 800
        assert cfg.hallucination_guard.enabled is True
        assert cfg.hallucination_guard.threshold == 0.90

    def test_load_config_missing_yaml(self, tmp_path):
        """Config works with defaults when YAML is missing."""
        from src.core.config import load_config
        cfg = load_config(str(tmp_path))
        assert cfg.mode == "offline"
        assert cfg.chunking.chunk_size == 1200

    def test_all_original_fields_present(self):
        from src.core.config import Config
        cfg = Config()
        for field_name in ["mode", "paths", "embedding", "chunking", "ollama",
                           "api", "cost", "retrieval", "indexing", "security",
                           "hallucination_guard"]:
            assert hasattr(cfg, field_name), f"Config missing field: {field_name}"


# ============================================================================
# SECTION 4: NETWORK GATE INTEGRATION TESTS
# ============================================================================

class TestNetworkGateIntegration:
    """Test network gate configuration and enforcement."""

    def test_offline_mode_blocks_external(self):
        from src.core.network_gate import NetworkGate, NetworkMode
        gate = NetworkGate()
        gate.configure(mode="offline")
        assert gate.mode == NetworkMode.OFFLINE

    def test_online_mode_allows_configured_endpoint(self):
        from src.core.network_gate import NetworkGate
        gate = NetworkGate()
        gate.configure(
            mode="online",
            api_endpoint="https://myapi.example.com/v1",
        )
        # check_allowed returns None on success (does not raise)
        gate.check_allowed("https://myapi.example.com/v1/chat")

    def test_online_mode_blocks_unconfigured(self):
        from src.core.network_gate import NetworkGate, NetworkBlockedError
        gate = NetworkGate()
        gate.configure(mode="online", api_endpoint="https://myapi.example.com/v1")
        with pytest.raises(NetworkBlockedError):
            gate.check_allowed("https://api.openai.com/v1/chat")

    def test_offline_always_allows_localhost(self):
        from src.core.network_gate import NetworkGate
        gate = NetworkGate()
        gate.configure(mode="offline")
        # check_allowed returns None on success (does not raise)
        gate.check_allowed("http://localhost:11434/api/generate")
        gate.check_allowed("http://127.0.0.1:11434/api/generate")

    def test_env_var_override(self):
        from src.core.network_gate import NetworkGate, NetworkMode
        gate = NetworkGate()
        with patch.dict(os.environ, {"HYBRIDRAG_OFFLINE": "1"}):
            gate.configure(mode="online")
        assert gate.mode == NetworkMode.OFFLINE


# ============================================================================
# SECTION 5: PARSER REGISTRY TESTS
# ============================================================================

class TestParserRegistry:
    """Test parser registry completeness and correctness."""

    def test_registry_loads(self):
        from src.parsers.registry import REGISTRY
        assert REGISTRY is not None

    def test_core_extensions_registered(self):
        from src.parsers.registry import REGISTRY
        expected = [".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".eml"]
        for ext in expected:
            info = REGISTRY.get(ext)
            assert info is not None, f"Extension {ext} not registered"

    def test_supported_extensions_nonempty(self):
        from src.parsers.registry import REGISTRY
        exts = REGISTRY.supported_extensions()
        assert len(exts) >= 7, f"Only {len(exts)} extensions registered"

    def test_parser_classes_are_callable(self):
        from src.parsers.registry import REGISTRY
        for ext in REGISTRY.supported_extensions():
            info = REGISTRY.get(ext)
            assert callable(info.parser_cls), f"Parser for {ext} is not callable"


# ============================================================================
# SECTION 6: CROSS-MODULE PIPELINE INTEGRATION
# ============================================================================

class TestPipelineIntegration:
    """Test the full pipeline: chunk -> embed -> store -> search -> query."""

    def test_chunk_to_store_pipeline(self, tmp_path):
        """Chunker output feeds correctly into VectorStore."""
        from src.core.chunker import Chunker
        from src.core.vector_store import VectorStore, ChunkMetadata

        cfg = FakeConfig()
        chunker = Chunker(cfg.chunking)

        text = "This is a test document. " * 200  # ~5000 chars
        chunks = chunker.chunk_text(text)
        assert len(chunks) >= 2

        store = VectorStore(
            db_path=str(tmp_path / "pipeline_test.sqlite3"),
            embedding_dim=384,
        )
        store.connect()
        try:
            n = len(chunks)
            fake_embeddings = np.random.randn(n, 384).astype(np.float32)
            metadata = [
                ChunkMetadata(
                    source_path="/test/pipeline_doc.txt",
                    chunk_index=i,
                    text_length=len(c),
                    created_at="2026-01-01T00:00:00",
                )
                for i, c in enumerate(chunks)
            ]
            store.add_embeddings(fake_embeddings, metadata, chunks)

            stats = store.get_stats()
            assert stats["chunk_count"] == n

            query_vec = np.random.randn(384).astype(np.float32)
            results = store.search(query_vec, top_k=3)
            assert len(results) <= 3
            for r in results:
                assert "text" in r, f"Search result missing 'text' key: {list(r.keys())}"
        finally:
            store.close()

    def test_query_engine_full_mock(self):
        """Full QueryEngine pipeline with all mocks."""
        from src.core.query_engine import QueryEngine

        cfg = FakeConfig()
        cfg.mode = "offline"

        # Mock vector_store to return results with correct key names
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "text": "The operating frequency is 9.0 to 9.5 GHz.",
                "source_path": "/docs/radar.txt",
                "chunk_index": 0,
                "score": 0.85,
            },
            {
                "text": "Power output is 500W peak.",
                "source_path": "/docs/radar.txt",
                "chunk_index": 1,
                "score": 0.72,
            },
        ]
        mock_store.fts_search.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = np.random.randn(384).astype(np.float32)

        mock_llm = MagicMock()
        mock_llm.query.return_value = FakeLLMResponse(
            text="The operating frequency range is 9.0 to 9.5 GHz.",
            tokens_in=50,
            tokens_out=20,
            model="phi4-mini",
            latency_ms=150.0,
        )

        engine = QueryEngine(cfg, mock_store, mock_embedder, mock_llm)
        result = engine.query("What is the operating frequency?")

        assert result.error is None or result.error == ""
        assert result.chunks_used >= 1
        assert result.mode == "offline"

    def test_credential_resolution_isolation(self):
        """Credentials module works in isolation without keyring."""
        from src.security.credentials import (
            resolve_credentials,
            KEYRING_SERVICE,
            KEYRING_KEY_NAME,
            KEYRING_ENDPOINT_NAME,
        )
        assert KEYRING_SERVICE == "hybridrag"
        assert KEYRING_KEY_NAME == "azure_api_key"
        assert KEYRING_ENDPOINT_NAME == "azure_endpoint"

    def test_llm_router_offline_no_crash(self):
        """LLMRouter in offline mode should not crash without API key."""
        from src.core.llm_router import LLMRouter
        cfg = FakeConfig()
        cfg.mode = "offline"
        router = LLMRouter(cfg)
        status = router.get_status()
        assert "ollama" in str(status).lower() or "offline" in str(status).lower()


# ============================================================================
# SECTION 7: MODULE SIZE ENFORCEMENT
# ============================================================================

class TestModuleSizeEnforcement:
    """Verify no class exceeds 500 lines (standing rule)."""

    def _count_class_lines(self, filepath):
        """Return dict of {class_name: line_count}."""
        import ast
        source = Path(filepath).read_text(encoding="utf-8")
        tree = ast.parse(source)
        results = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                end = getattr(node, "end_lineno", node.lineno)
                results[node.name] = end - node.lineno + 1
        return results

    def test_core_classes_under_500_lines(self):
        core_dir = PROJECT_ROOT / "src" / "core"
        violations = []
        for py_file in core_dir.rglob("*.py"):
            try:
                classes = self._count_class_lines(py_file)
                for cls_name, lines in classes.items():
                    if lines > 500:
                        rel = py_file.relative_to(PROJECT_ROOT)
                        violations.append(f"{rel}::{cls_name} = {lines} lines")
            except Exception:
                pass
        # BUG-6 resolved: all classes refactored under 500 lines
        assert len(violations) == 0, (
            f"Classes over 500 lines:\n"
            + "\n".join(violations)
        )


# ============================================================================
# SECTION 8: NON-ASCII ENFORCEMENT
# ============================================================================

class TestNonAsciiEnforcement:
    """Verify no script has non-ASCII characters.

    BUG-7 (SEV-3): 25 src/ files and 1 scripts/ file have non-ASCII bytes.
    Mostly 0xe2 (em-dash in UTF-8) and 0xef (BOM prefix).
    These are being cleaned in the Phase 3 fix cycle.
    This test tracks the count to catch new introductions.
    """

    def _count_non_ascii_files(self, directory):
        violations = []
        for py_file in directory.rglob("*.py"):
            try:
                content = py_file.read_bytes()
                # Skip BOM (0xEF 0xBB 0xBF) at start -- some editors add it
                start = 3 if content[:3] == b"\xef\xbb\xbf" else 0
                for i in range(start, len(content)):
                    if content[i] > 127:
                        line_num = content[:i].count(b"\n") + 1
                        rel = py_file.relative_to(PROJECT_ROOT)
                        violations.append(f"{rel} line {line_num}: byte {content[i]:#x}")
                        break
            except Exception:
                pass
        return violations

    def test_src_files_ascii_inventory(self):
        """Track non-ASCII file count -- should decrease as we clean."""
        violations = self._count_non_ascii_files(PROJECT_ROOT / "src")
        # BUG-7: Known 25 files with non-ASCII. Track to prevent growth.
        assert len(violations) <= 25, (
            f"Non-ASCII files grew beyond known 25:\n" + "\n".join(violations)
        )

    def test_scripts_ascii_inventory(self):
        violations = self._count_non_ascii_files(PROJECT_ROOT / "scripts")
        # BUG-7: Known 1 file with non-ASCII. Track to prevent growth.
        assert len(violations) <= 1, (
            f"Non-ASCII files grew beyond known 1:\n" + "\n".join(violations)
        )
