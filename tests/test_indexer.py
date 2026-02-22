# ============================================================================
# test_indexer.py -- Tests for Indexer and Integration
# ============================================================================
#
# COVERS:
#   TestIndexer     -- file scanning, chunking, hashing, storage pipeline
#   TestIntegration -- end-to-end integration tests
#
# RUN:
#   python -m pytest tests/test_indexer.py -v
#
# INTERNET ACCESS: NONE -- all external calls are mocked
# ============================================================================

import os
import time
import shutil
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pytest
# Import shared fixtures from conftest.py in the same directory.
# WHY this style: avoids requiring tests/__init__.py to exist.
# Works on both home PC and work laptop regardless of package structure.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from conftest import FakeConfig, FakeLLMResponse

class TestIndexer:
    """
    Tests for the Indexer pipeline.

    Uses a real temporary directory with real files, but mocks
    the database, embedder, and chunker to avoid needing actual
    AI models or SQLite during tests.
    """

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self, tmp_path):
        """
        WHAT: Create a temporary directory for each test with sample files.
        WHY:  Tests need real files on disk to scan, but we don't want to
              touch your actual documents. tmp_path is automatically cleaned
              up after each test.

        WHAT IS @pytest.fixture?
          A "fixture" is code that runs before each test to set up the
          environment. autouse=True means it runs for EVERY test in this
          class without being explicitly requested.
        """
        self.test_dir = tmp_path / "source"
        self.test_dir.mkdir()

        # Create sample files of different types
        (self.test_dir / "document.txt").write_text(
            "This is a test document about radio frequency engineering. "
            "The system operates at 5.2 GHz with a power output of 100 watts. "
            "The antenna gain is measured in dBi." * 20,  # Make it chunky
            encoding="utf-8",
        )

        (self.test_dir / "notes.md").write_text(
            "# Engineering Notes\n\n"
            "## Section 1: RF Design\n\n"
            "The receiver sensitivity is -95 dBm at the input. "
            "Signal-to-noise ratio must exceed 10 dB." * 15,
            encoding="utf-8",
        )

        (self.test_dir / "data.csv").write_text(
            "frequency,power,gain\n"
            "5200,100,15\n"
            "5300,95,14\n"
            "5400,90,13\n",
            encoding="utf-8",
        )

        # File type we don't support -- should be skipped
        (self.test_dir / "image.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        )

        # Excluded directory -- should be skipped entirely
        excluded = self.test_dir / ".git"
        excluded.mkdir()
        (excluded / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

        # Subdirectory with a valid file
        sub = self.test_dir / "subdir"
        sub.mkdir()
        (sub / "deep_doc.txt").write_text(
            "This is a document in a subdirectory." * 30,
            encoding="utf-8",
        )

    def _make_indexer(self, config=None):
        """
        Helper: create an Indexer with all dependencies mocked.

        Returns:
            (indexer, mocks) -- the indexer and a dict of mock objects
        """
        if config is None:
            config = FakeConfig()

        mock_vector_store = MagicMock()
        mock_embedder = MagicMock()
        mock_chunker = MagicMock()

        # VectorStore: no existing data (everything is new)
        mock_vector_store.get_file_hash.return_value = None
        mock_vector_store.delete_chunks_by_source.return_value = 0

        # Embedder: return dummy vectors (384-dim for MiniLM)
        # Each chunk gets a list of 384 zeros as its "embedding"
        def fake_embed_batch(texts):
            return [[0.0] * 384 for _ in texts]

        mock_embedder.embed_batch.side_effect = fake_embed_batch

        # Chunker: split text into fixed-size chunks for testing
        def fake_chunk_text(text):
            # Simple chunker: split into 200-char pieces
            chunks = []
            for i in range(0, len(text), 200):
                chunk = text[i:i + 200]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks

        mock_chunker.chunk_text.side_effect = fake_chunk_text

        # We need to patch several imports that the Indexer uses
        with patch("src.core.indexer.make_chunk_id") as mock_make_id:
            # Return deterministic IDs
            call_counter = [0]

            def make_fake_id(**kwargs):
                call_counter[0] += 1
                return f"chunk_{call_counter[0]:06d}"

            mock_make_id.side_effect = make_fake_id

            from src.core.indexer import Indexer
            indexer = Indexer(config, mock_vector_store, mock_embedder, mock_chunker)

        mocks = {
            "vector_store": mock_vector_store,
            "embedder": mock_embedder,
            "chunker": mock_chunker,
        }

        return indexer, mocks

    # ------------------------------------------------------------------
    # Test 5.1: Discovers supported files, ignores unsupported
    # ------------------------------------------------------------------
    def test_discovers_supported_files_only(self):
        """
        WHAT: The indexer should find .txt, .md, .csv but skip .png
        WHY:  Indexing a binary PNG as text would create garbage chunks
              that pollute search results.
        """
        indexer, mocks = self._make_indexer()

        result = indexer.index_folder(str(self.test_dir))

        # We created: document.txt, notes.md, data.csv, deep_doc.txt = 4 files
        # Skipped: image.png (unsupported), .git/HEAD (excluded dir)
        assert result["total_files_scanned"] == 4, (
            f"Expected 4 supported files, found {result['total_files_scanned']}"
        )

    # ------------------------------------------------------------------
    # Test 5.2: Skips excluded directories
    # ------------------------------------------------------------------
    def test_skips_excluded_directories(self):
        """
        WHAT: Files inside .git, __pycache__, .venv etc. are skipped.
        WHY:  These contain internal tool files, not user documents.
              Indexing .git objects would create nonsensical search results.
        """
        indexer, mocks = self._make_indexer()

        result = indexer.index_folder(str(self.test_dir))

        # The .git/HEAD file should NOT appear in the results
        # If it did, total_files_scanned would be 5 instead of 4
        assert result["total_files_scanned"] == 4

    # ------------------------------------------------------------------
    # Test 5.3: Skips unchanged files (hash match)
    # ------------------------------------------------------------------
    def test_skips_unchanged_files(self):
        """
        WHAT: If a file was already indexed and hasn't changed, skip it.
        WHY:  Your 100GB corporate drive has thousands of files. Most don't
              change. Without skip logic, every re-index takes days.
              With it, a re-index takes minutes.
        """
        indexer, mocks = self._make_indexer()

        # First run: index everything
        result1 = indexer.index_folder(str(self.test_dir))
        first_run_indexed = result1["total_files_indexed"]

        # Simulate: all files now have stored hashes (already indexed)
        def return_matching_hash(file_path):
            # Return the same hash that _compute_file_hash would produce
            p = Path(file_path)
            if p.exists():
                stat = p.stat()
                return f"{stat.st_size}:{stat.st_mtime_ns}"
            return None

        mocks["vector_store"].get_file_hash.side_effect = return_matching_hash

        # Reset the call counters
        mocks["embedder"].embed_batch.reset_mock()

        # Second run: everything should be skipped
        result2 = indexer.index_folder(str(self.test_dir))

        assert result2["total_files_skipped"] == result2["total_files_scanned"], (
            f"All {result2['total_files_scanned']} files should be skipped "
            f"on re-index, but only {result2['total_files_skipped']} were"
        )
        assert result2["total_files_indexed"] == 0

    # ------------------------------------------------------------------
    # Test 5.4: Re-indexes modified files
    # ------------------------------------------------------------------
    def test_reindexes_modified_files(self):
        """
        WHAT: If a file was indexed but has since been modified, re-index it.
        WHY:  Someone updates a document on the shared drive. HybridRAG
              needs to pick up those changes without re-indexing everything.
        """
        indexer, mocks = self._make_indexer()

        # Simulate: file was previously indexed with a DIFFERENT hash
        mocks["vector_store"].get_file_hash.return_value = "old_size:old_mtime"
        mocks["vector_store"].delete_chunks_by_source.return_value = 5

        result = indexer.index_folder(str(self.test_dir))

        # All files should be re-indexed because hash doesn't match
        assert result["total_files_reindexed"] > 0, (
            "Modified files should be detected and re-indexed"
        )

        # Old chunks should have been deleted before re-indexing
        assert mocks["vector_store"].delete_chunks_by_source.call_count > 0

    # ------------------------------------------------------------------
    # Test 5.5: Text validation catches binary garbage
    # ------------------------------------------------------------------
    def test_validate_text_catches_binary_garbage(self):
        """
        WHAT: _validate_text() returns False for binary-looking strings.
        WHY:  Corrupted PDFs sometimes return binary garbage as "text."
              Without validation, that garbage gets embedded and pollutes
              search results with nonsensical matches.
        """
        indexer, _ = self._make_indexer()

        # Good text -- should pass validation
        good_text = (
            "This is a perfectly normal document about RF engineering. "
            "The system operates at 5.2 GHz with an antenna gain of 15 dBi."
        )
        assert indexer._validate_text(good_text) is True

        # Binary garbage -- should FAIL validation
        binary_garbage = "\x00\x89PNG\r\n\x1a\x00" * 100
        assert indexer._validate_text(binary_garbage) is False

        # Too short -- should FAIL validation
        assert indexer._validate_text("Hi") is False
        assert indexer._validate_text("") is False
        assert indexer._validate_text(None) is False

    # ------------------------------------------------------------------
    # Test 5.6: File hash computation is deterministic
    # ------------------------------------------------------------------
    def test_file_hash_is_deterministic(self):
        """
        WHAT: Same file produces the same hash every time.
        WHY:  Hash comparison is how we detect changes. If the hash
              changes randomly, we'd re-index everything every time,
              defeating the skip optimization.
        """
        indexer, _ = self._make_indexer()

        test_file = self.test_dir / "document.txt"

        hash1 = indexer._compute_file_hash(test_file)
        hash2 = indexer._compute_file_hash(test_file)

        assert hash1 == hash2, "Same file should produce same hash"
        assert ":" in hash1, "Hash should be in format 'size:mtime_ns'"

    # ------------------------------------------------------------------
    # Test 5.7: File hash changes when file is modified
    # ------------------------------------------------------------------
    def test_file_hash_changes_on_modification(self):
        """
        WHAT: Modifying a file changes its hash.
        WHY:  If the hash doesn't change on modification, the skip logic
              would never re-index changed files, giving stale results.
        """
        indexer, _ = self._make_indexer()

        test_file = self.test_dir / "document.txt"

        hash_before = indexer._compute_file_hash(test_file)

        # Modify the file (changes size and mtime)
        import time as _time
        _time.sleep(0.1)  # Ensure mtime changes
        test_file.write_text(
            "Modified content that is different from the original.",
            encoding="utf-8",
        )

        hash_after = indexer._compute_file_hash(test_file)

        assert hash_before != hash_after, (
            "Modified file should produce a different hash"
        )

    # ------------------------------------------------------------------
    # Test 5.8: Single file error doesn't crash the entire run
    # ------------------------------------------------------------------
    def test_single_file_error_continues(self):
        """
        WHAT: If one file fails to parse, the indexer continues to the next.
        WHY:  Your 100GB drive might have corrupted PDFs buried in
              subdirectories. If the indexer crashes on file #3,000, you
              lose 3 days of progress. Instead, it logs the error and moves on.
        """
        indexer, mocks = self._make_indexer()

        # Make the chunker fail on the FIRST call, then work normally
        call_count = [0]
        original_side_effect = mocks["chunker"].chunk_text.side_effect

        def sometimes_failing_chunker(text):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated corrupted file")
            return original_side_effect(text)

        mocks["chunker"].chunk_text.side_effect = sometimes_failing_chunker

        # Should NOT raise an exception
        result = indexer.index_folder(str(self.test_dir))

        # At least some files should have been indexed despite the error
        assert result["total_files_indexed"] >= 1, (
            "Indexer should continue after a single file error"
        )

    # ------------------------------------------------------------------
    # Test 5.9: Excluded directory detection
    # ------------------------------------------------------------------
    def test_is_excluded_detects_git_dir(self):
        """
        WHAT: _is_excluded() returns True for paths inside .git/
        WHY:  Git objects look like valid text files but contain
              internal version control data, not user documents.
        """
        indexer, _ = self._make_indexer()

        git_file = Path("/repo/.git/objects/ab/cdef123")
        assert indexer._is_excluded(git_file) is True

        normal_file = Path("/repo/docs/readme.md")
        assert indexer._is_excluded(normal_file) is False

        venv_file = Path("/repo/.venv/lib/python3.11/site.py")
        assert indexer._is_excluded(venv_file) is True

    # ------------------------------------------------------------------
    # Test 5.10: Text block iterator
    # ------------------------------------------------------------------
    def test_iter_text_blocks_splits_large_text(self):
        """
        WHAT: _iter_text_blocks() splits text into manageable chunks.
        WHY:  A 500-page PDF produces ~2 million characters. Loading all
              that into RAM at once would spike memory. The block iterator
              processes 200K chars at a time, keeping RAM stable.
        """
        indexer, _ = self._make_indexer()

        # Create a large text that's bigger than one block
        # block_chars defaults to 200,000
        large_text = "A" * 500_000  # 500K chars = at least 3 blocks

        blocks = list(indexer._iter_text_blocks(large_text))

        assert len(blocks) >= 2, "500K chars should produce multiple blocks"
        total_chars = sum(len(b) for b in blocks)
        assert total_chars == 500_000, "All text should be accounted for"

    # ------------------------------------------------------------------
    # Test 5.11: Resource cleanup
    # ------------------------------------------------------------------
    def test_close_releases_resources(self):
        """
        WHAT: close() calls close() on the embedder and vector store.
        WHY:  The embedding model stays in RAM (~100MB) until explicitly
              released. Over repeated indexing runs, this leaks memory.
              close() prevents this.
        """
        indexer, mocks = self._make_indexer()

        indexer.close()

        mocks["embedder"].close.assert_called_once()
        mocks["vector_store"].close.assert_called_once()

    # ------------------------------------------------------------------
    # Test 5.12: Large file clamping
    # ------------------------------------------------------------------
    def test_large_file_is_clamped(self):
        """
        WHAT: Files larger than max_chars_per_file are truncated.
        WHY:  A single 10,000-page document could overwhelm the system.
              Clamping at 2 million chars (~1,000 pages) keeps processing
              time and memory reasonable.
        """
        indexer, mocks = self._make_indexer()

        # Create a file larger than the limit
        huge_file = self.test_dir / "huge_doc.txt"
        # Use a small limit for testing
        indexer.max_chars_per_file = 1000
        huge_file.write_text("X" * 5000, encoding="utf-8")

        result = indexer.index_folder(str(self.test_dir))

        # The file should still be indexed, just truncated
        assert result["total_files_indexed"] >= 1

    # ------------------------------------------------------------------
    # Test 5.13: Empty folder handling
    # ------------------------------------------------------------------
    def test_empty_folder(self):
        """
        WHAT: Indexing an empty folder returns zero counts without crashing.
        WHY:  A user might point the indexer at an empty directory by mistake.
        """
        indexer, _ = self._make_indexer()

        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()

        result = indexer.index_folder(str(empty_dir))

        assert result["total_files_scanned"] == 0
        assert result["total_files_indexed"] == 0
        assert result["total_chunks_added"] == 0

    # ------------------------------------------------------------------
    # Test 5.14: Nonexistent folder raises error
    # ------------------------------------------------------------------
    def test_nonexistent_folder_raises(self):
        """
        WHAT: Pointing the indexer at a folder that doesn't exist raises
              FileNotFoundError with a clear message.
        WHY:  Better to fail fast with a clear error than silently index
              zero files and let the user wonder why nothing happened.
        """
        indexer, _ = self._make_indexer()

        with pytest.raises(FileNotFoundError):
            indexer.index_folder("/this/path/does/not/exist")

    # ------------------------------------------------------------------
    # Test 5.15: Progress callback is called
    # ------------------------------------------------------------------
    def test_progress_callback_called(self):
        """
        WHAT: Verify the progress callback methods are invoked during indexing.
        WHY:  The GUI progress bar depends on these callbacks. If they stop
              firing, the GUI appears frozen during long indexing runs.
        """
        indexer, _ = self._make_indexer()

        from src.core.indexer import IndexingProgressCallback

        class TrackingCallback(IndexingProgressCallback):
            def __init__(self):
                self.file_starts = []
                self.file_completes = []
                self.file_skips = []
                self.errors = []
                self.completed = False

            def on_file_start(self, path, num, total):
                self.file_starts.append((path, num, total))

            def on_file_complete(self, path, chunks):
                self.file_completes.append((path, chunks))

            def on_file_skipped(self, path, reason):
                self.file_skips.append((path, reason))

            def on_error(self, path, error):
                self.errors.append((path, error))

            def on_indexing_complete(self, total, elapsed):
                self.completed = True

        tracker = TrackingCallback()
        result = indexer.index_folder(str(self.test_dir), progress_callback=tracker)

        assert len(tracker.file_starts) > 0, "on_file_start should be called"
        assert tracker.completed is True, "on_indexing_complete should be called"


# ============================================================================
# SECTION 4: INTEGRATION-STYLE TESTS
# ============================================================================
#
# These tests check that different modules work together correctly.
# Still uses mocks for external services, but tests the wiring between
# internal components.
# ============================================================================

class TestIntegration:
    """
    Tests that verify components work together correctly.
    """

    # ------------------------------------------------------------------
    # Test 6.1: Config validation
    # ------------------------------------------------------------------
    def test_fake_config_has_all_required_fields(self):
        """
        WHAT: Verify our FakeConfig has all the fields the real code expects.
        WHY:  If someone adds a new config field to the real Config class
              but forgets to add it to FakeConfig, all tests would break
              with confusing AttributeError messages.
        """
        config = FakeConfig()

        # Fields used by LLMRouter
        assert hasattr(config, "mode")
        assert hasattr(config.ollama, "base_url")
        assert hasattr(config.ollama, "model")
        assert hasattr(config.ollama, "timeout_seconds")
        assert hasattr(config.api, "endpoint")
        assert hasattr(config.api, "model")

        # Fields used by QueryEngine
        assert hasattr(config, "cost")
        assert hasattr(config.cost, "input_cost_per_1k")
        assert hasattr(config.cost, "output_cost_per_1k")

        # Fields used by Indexer
        assert hasattr(config, "chunking")
        assert hasattr(config.chunking, "chunk_size")
        assert hasattr(config.chunking, "overlap")

    # ------------------------------------------------------------------
    # Test 6.2: LLMResponse dataclass completeness
    # ------------------------------------------------------------------
    def test_llm_response_fields(self):
        """
        WHAT: Verify LLMResponse has all the fields that QueryEngine expects.
        WHY:  QueryEngine reads tokens_in, tokens_out, text, and model
              from LLMResponse. If any are missing, cost calculation or
              logging breaks.
        """
        response = FakeLLMResponse(
            text="Answer",
            tokens_in=100,
            tokens_out=20,
            model="phi4-mini",
            latency_ms=5000.0,
        )

        assert response.text == "Answer"
        assert response.tokens_in == 100
        assert response.tokens_out == 20
        assert response.model == "phi4-mini"
        assert response.latency_ms == 5000.0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# If you run this file directly (python test_hybridrag3.py), it runs all tests.
# But the recommended way is: python -m pytest tests/test_hybridrag3.py -v
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
