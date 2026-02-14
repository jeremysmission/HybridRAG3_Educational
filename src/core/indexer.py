# ============================================================================
# HybridRAG — Indexer (src/core/indexer.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   This is the "brain" of the indexing pipeline. It orchestrates:
#     scan folder -> parse each file -> chunk text -> embed chunks -> store
#
#   This is the module that runs for your week-long indexing job.
#
# KEY DESIGN DECISIONS:
#
#   1. Process files in blocks (not all text at once)
#      WHY: A single 500-page PDF can produce 2 million characters of text.
#      Loading that all into RAM, chunking it, and embedding it would spike
#      memory. Instead we break the text into blocks of ~200K chars, process
#      each block, write to disk, then move on. RAM stays stable.
#
#   2. Determisecurity standardic chunk IDs (from chunk_ids.py)
#      WHY: If indexing crashes at file #4,000 out of 10,000 and you restart,
#      determisecurity standardic IDs mean "INSERT OR IGNORE" in SQLite skips the first
#      4,000 files' chunks automatically. No duplicates, no manual cleanup.
#
#   3. Skip unchanged files (hash-based change detection)
#      WHY: Your corporate drive has 100GB of documents. Most don't change
#      week to week. We store a hash (size + mtime) with each file's chunks.
#      On restart, we compare the stored hash to the current file. If they
#      match, skip it. If they differ, the file was modified — delete old
#      chunks and re-index. This turns a 7-day re-index into minutes.
#
#   4. Never crash on a single file failure
#      WHY: File #3,000 might be a corrupted PDF. If we crash, you lose
#      3 days of indexing progress. Instead, we log the error and continue
#      to file #3,001. You can review failures in the log after.
#
# BUGS FIXED (2026-02-08):
#   BUG-001: Hash detection uses vector_store.get_file_hash() in index_folder()
#            of raw SQL against a column that didn't exist.
#   BUG-002: Change detection logic inlined in index_folder() using
#            _compute_file_hash() + get_file_hash(). Dead _file_changed()
#            method removed 2026-02-14.
#            Previously only _file_already_indexed() was called, which just
#            checked "do chunks exist?" without checking if the file changed.
#   BUG-003: Added close() method to release the embedder model from RAM.
#   BUG-004: Added _validate_text() to catch binary garbage before chunking.
#
# ALTERNATIVES CONSIDERED:
#   - LangChain DirectoryLoader: hides logic in "magic" class, impossible
#     to debug. We control every step.
#   - Async/parallel indexing: faster but much harder to debug and resume.
#   - Hash-based detection using xxhash on file content: more accurate but
#     reads every file on every run. Size+mtime is instant and good enough.
# ============================================================================

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import Config
from .vector_store import VectorStore, ChunkMetadata
from .chunker import Chunker, ChunkerConfig
from .embedder import Embedder
from .chunk_ids import make_chunk_id
import gc
import hashlib


# -------------------------------------------------------------------
# Progress callback interface
# -------------------------------------------------------------------

class IndexingProgressCallback:
    """
    Override these methods to receive progress updates during indexing.

    Default implementations do nothing (safe no-op). Subclass this and
    override methods for progress bars, GUI updates, or logging.
    """

    def on_file_start(self, file_path: str, file_num: int, total_files: int) -> None:
        pass

    def on_file_complete(self, file_path: str, chunks_created: int) -> None:
        pass

    def on_file_skipped(self, file_path: str, reason: str) -> None:
        pass

    def on_indexing_complete(self, total_chunks: int, elapsed_seconds: float) -> None:
        pass

    def on_error(self, file_path: str, error: str) -> None:
        pass


# -------------------------------------------------------------------
# Indexer
# -------------------------------------------------------------------

class Indexer:
    """
    Scans a folder, parses files, chunks text, embeds, and stores.

    Designed for multi-day indexing runs on laptops (24/7 for a week),
    resumable-safe operation, low memory usage, and auditable environments.
    """

    def __init__(
        self,
        config: Config,
        vector_store: VectorStore,
        embedder: Embedder,
        chunker: Chunker,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker

        idx_cfg = config.indexing if config else None

        # max_chars_per_file: Safety limit — truncate files larger than this.
        # 2 million chars ~ a 1,000-page document.
        self.max_chars_per_file = getattr(idx_cfg, "max_chars_per_file", 2_000_000)

        # block_chars: How much text to process at a time before writing to
        # disk. 200K chars ~ 100 pages. Keeps RAM usage predictable.
        self.block_chars = getattr(idx_cfg, "block_chars", 200_000)

        # File extensions we know how to parse
        self._supported_extensions = set(
            getattr(idx_cfg, "supported_extensions", [
                ".txt", ".md", ".csv", ".json", ".xml", ".log", ".pdf",
                ".docx", ".pptx", ".xlsx", ".eml",
            ])
        )

        # Directories to skip (virtual environments, git history, etc.)
        self._excluded_dirs = set(
            getattr(idx_cfg, "excluded_dirs", [
                ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode",
                "node_modules",
            ])
        )

    # ------------------------------------------------------------------
    # Public API — this is what you call to index a folder
    # ------------------------------------------------------------------

    def index_folder(
        self,
        folder_path: str,
        progress_callback: Optional[IndexingProgressCallback] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Index all supported files in a folder.

        Returns dict with: total_files_scanned, total_files_indexed,
        total_files_skipped, total_chunks_added, elapsed_seconds.
        """
        if progress_callback is None:
            progress_callback = IndexingProgressCallback()

        start_time = time.time()
        total_chunks = 0
        total_files_indexed = 0
        total_files_skipped = 0
        total_files_reindexed = 0    # NEW: tracks modified file re-indexes

        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Source folder not found: {folder_path}")

        # --- Step 1: Discover all supported files ---
        raw_files = list(folder.rglob("*")) if recursive else list(folder.glob("*"))

        supported_files: List[Path] = []
        for f in raw_files:
            if not f.is_file():
                continue
            if self._is_excluded(f):
                continue
            if f.suffix.lower() not in self._supported_extensions:
                continue
            supported_files.append(f)

        print(f"Found {len(supported_files)} supported files in {folder}")

        # --- Step 2: Process each file ---
        for idx, file_path in enumerate(supported_files, start=1):
            file_chunks_created = 0
            try:
                progress_callback.on_file_start(
                    str(file_path), idx, len(supported_files)
                )

                # ==========================================================
                # BUG-002 FIX: Change detection now uses file hashes
                # ==========================================================
                # OLD behavior: just checked "do chunks exist?" — if yes,
                #   skip. This meant modified files were never re-indexed.
                # NEW behavior: check hash. Three possible outcomes:
                #   1. No chunks exist -> new file, index it
                #   2. Chunks exist, hash matches -> unchanged, skip
                #   3. Chunks exist, hash differs -> modified, re-index
                # ==========================================================
                current_hash = self._compute_file_hash(file_path)
                stored_hash = self.vector_store.get_file_hash(str(file_path))

                if stored_hash:
                    # Chunks exist for this file — check if it changed
                    if stored_hash == current_hash:
                        # File unchanged since last index — skip it
                        total_files_skipped += 1
                        progress_callback.on_file_skipped(
                            str(file_path), "unchanged (hash match)"
                        )
                        continue
                    else:
                        # File was modified — delete old chunks, re-index
                        deleted = self.vector_store.delete_chunks_by_source(
                            str(file_path)
                        )
                        print(
                            f"  RE-INDEX: {file_path.name} changed "
                            f"(deleted {deleted} old chunks)"
                        )
                        total_files_reindexed += 1

                # --- Parse the file into text ---
                text = self._process_file_with_retry(file_path)
                if not text or not text.strip():
                    total_files_skipped += 1
                    progress_callback.on_file_skipped(
                        str(file_path), "no text extracted"
                    )
                    continue

                # ==========================================================
                # BUG-004 FIX: Validate text before chunking
                # ==========================================================
                # Check if the extracted text is actually readable words,
                # not binary garbage from a corrupted file. If it fails
                # validation, skip the file and log a warning.
                # ==========================================================
                if not self._validate_text(text):
                    total_files_skipped += 1
                    progress_callback.on_file_skipped(
                        str(file_path), "binary garbage detected"
                    )
                    print(
                        f"  WARNING: {file_path.name} — text looks like "
                        f"binary garbage, skipping"
                    )
                    continue

                # Safety: clamp oversized files
                if len(text) > self.max_chars_per_file:
                    print(
                        f"  WARNING: Clamping {file_path.name} from "
                        f"{len(text):,} to {self.max_chars_per_file:,} chars"
                    )
                    text = text[: self.max_chars_per_file]

                # Get file modification time for determisecurity standardic chunk IDs
                try:
                    file_mtime_ns = file_path.stat().st_mtime_ns
                except Exception:
                    file_mtime_ns = 0

                # --- Chunk + Embed + Store (in blocks for RAM safety) ---
                char_offset = 0
                for block in self._iter_text_blocks(text):
                    if not block.strip():
                        char_offset += len(block)
                        continue

                    chunks = self.chunker.chunk_text(block)
                    if not chunks:
                        char_offset += len(block)
                        continue

                    embeddings = self.embedder.embed_batch(chunks)

                    metadata_list = []
                    chunk_ids = []
                    for i, chunk_text in enumerate(chunks):
                        chunk_size = self.config.chunking.chunk_size
                        chunk_overlap = self.config.chunking.overlap
                        chunk_start = char_offset + (
                            i * (chunk_size - chunk_overlap)
                        )
                        chunk_end = chunk_start + len(chunk_text)

                        cid = make_chunk_id(
                            file_path=str(file_path),
                            file_mtime_ns=file_mtime_ns,
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            chunk_text=chunk_text,
                        )
                        chunk_ids.append(cid)

                        metadata_list.append(
                            ChunkMetadata(
                                source_path=str(file_path),
                                chunk_index=file_chunks_created + i,
                                text_length=len(chunk_text),
                                created_at=datetime.utcnow().isoformat(),
                            )
                        )

                    # BUG-001 FIX: Pass file_hash so it's stored with chunks
                    self.vector_store.add_embeddings(
                        embeddings, metadata_list,
                        texts=chunks,
                        chunk_ids=chunk_ids,
                        file_hash=current_hash,
                    )

                    file_chunks_created += len(chunks)
                    total_chunks += len(chunks)
                    char_offset += len(block)

                if file_chunks_created == 0:
                    total_files_skipped += 1
                    progress_callback.on_file_skipped(
                        str(file_path), "no chunks produced"
                    )
                    continue

                total_files_indexed += 1
                gc.collect()  # Free RAM between files
                progress_callback.on_file_complete(
                    str(file_path), file_chunks_created
                )

            except Exception as e:
                # Never crash on a single file — log and continue
                error_msg = f"{type(e).__name__}: {e}"
                print(f"  ERROR on {file_path.name}: {error_msg}")
                progress_callback.on_error(str(file_path), error_msg)

        # --- Done ---
        elapsed = time.time() - start_time
        progress_callback.on_indexing_complete(total_chunks, elapsed)

        result = {
            "total_files_scanned": len(supported_files),
            "total_files_indexed": total_files_indexed,
            "total_files_skipped": total_files_skipped,
            "total_files_reindexed": total_files_reindexed,
            "total_chunks_added": total_chunks,
            "elapsed_seconds": elapsed,
        }

        print(f"\nIndexing complete:")
        print(f"  Files scanned:    {result['total_files_scanned']}")
        print(f"  Files indexed:    {result['total_files_indexed']}")
        print(f"  Files re-indexed: {result['total_files_reindexed']}")
        print(f"  Files skipped:    {result['total_files_skipped']}")
        print(f"  Chunks added:     {result['total_chunks_added']}")
        print(f"  Time: {elapsed:.1f}s")

        return result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _process_file_with_retry(self, file_path, max_retries=3):
        """Retry file processing up to max_retries times with backoff."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                text = self._parse_file(file_path)
                return text
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(
                        f"  Retry {attempt}/{max_retries} for "
                        f"{file_path.name} in {wait}s: {e}"
                    )
                    import time as _time
                    _time.sleep(wait)
        raise last_error

    # =================================================================
    # BUG-001 FIX: _compute_file_hash unchanged but now USED properly
    # =================================================================
    def _compute_file_hash(self, file_path):
        """
        Compute a fast fingerprint of a file: "filesize:mtime_nanoseconds".

        Example: "284519:132720938471230000"

        WHY size + mtime instead of reading file content?
          Reading file content (e.g., SHA-256) would require reading every
          byte of every file on every indexing run — that's 100GB+ of I/O
          on a corporate network drive. Size + mtime is instant (just a
          stat() call) and catches the vast majority of real modifications.

        WHEN THIS FAILS:
          If someone modifies a file but the OS doesn't update mtime (rare),
          or if a file is replaced with a same-size different file at the
          exact same nanosecond (essentially impossible). For defence use,
          we could add SHA-256 as a future config option for higher assurance.
        """
        stat = file_path.stat()
        fast_key = f"{stat.st_size}:{stat.st_mtime_ns}"
        return fast_key

    def _parse_file(self, file_path: Path) -> str:
        """
        Extract text from a file using the parser registry.

        Falls back to reading as plain text if no specialized parser
        is available. Returns empty string on failure.
        """
        try:
            from ..parsers.registry import REGISTRY
            parser = REGISTRY.get(file_path.suffix.lower())
            if parser:
                result = parser.parser_cls().parse(str(file_path))
                if hasattr(result, "text"):
                    return result.text or ""
                if isinstance(result, str):
                    return result
        except ImportError:
            pass
        except Exception as e:
            print(f"  Parser error on {file_path.name}: {e}")

        # Fallback: try reading as plain text
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                return file_path.read_text(encoding="latin-1", errors="replace")
            except Exception:
                return ""

    # =================================================================
    # BUG-004 FIX: Validate text before chunking to catch garbage
    # =================================================================
    def _validate_text(self, text: str) -> bool:
        """
        Check if extracted text is actually readable, not binary garbage.

        WHY THIS EXISTS (BUG-004):
          Some corrupted PDFs don't crash the parser — they just return
          garbage like "\\x00\\x89PNG\\r\\n\\x1a\\x00..." as text. Without
          this check, that garbage gets chunked, embedded, and stored,
          polluting search results with nonsensical matches.

        HOW IT WORKS:
          1. Take a sample of the text (first 2000 chars is enough)
          2. Count how many characters are "normal" (letters, digits,
             spaces, common punctuation)
          3. If less than 30% of characters are normal, it's garbage

        WHY 30%?
          - Pure English text: ~95% printable
          - Technical docs with equations/symbols: ~70% printable
          - Mixed language docs (CJK, Arabic, etc.): ~60% printable
          - Binary garbage: typically < 10% printable
          - 30% gives plenty of margin for all real document types

        Returns True if text looks valid, False if it looks like garbage.
        """
        if not text or len(text) < 20:
            return False  # Too short to be useful

        # Take a sample — no need to scan millions of chars
        sample = text[:2000]

        # Count "normal" characters: letters, digits, spaces, newlines,
        # and common punctuation that appears in real documents
        normal_count = 0
        for ch in sample:
            if ch.isalnum() or ch in ' \t\n\r.,;:!?()-/\'\"@#$%&*+=[]{}|<>~':
                normal_count += 1

        ratio = normal_count / len(sample)
        return ratio >= 0.30

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if any parent directory is in the exclusion list."""
        parts_lower = {p.lower() for p in file_path.parts}
        return bool(parts_lower & {d.lower() for d in self._excluded_dirs})

    def _iter_text_blocks(self, text: str):
        """
        Yield text in blocks of self.block_chars.

        Breaks on newlines when possible to avoid splitting mid-sentence.
        200K chars ~ 100 pages. Keeps RAM usage predictable.
        """
        n = len(text)
        start = 0
        while start < n:
            end = min(start + self.block_chars, n)
            if end < n:
                nl = text.rfind("\n", start, end)
                if nl != -1 and nl > start + 10_000:
                    end = nl
            yield text[start:end]
            start = end

    # =================================================================
    # BUG-003 FIX: close() to release the embedding model from RAM
    # =================================================================
    def close(self) -> None:
        """
        Release resources held by the indexer.

        BUG-003 FIX: The embedding model (SentenceTransformer) stays in
        RAM (~100MB) until explicitly deleted. Over repeated indexing
        runs without restarting Python, this leaks memory. The embedder
        and vector_store now have close() methods that this calls.

        Safe to call multiple times. Call in a "finally" block.
        """
        if hasattr(self, 'embedder') and self.embedder is not None:
            self.embedder.close()
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            self.vector_store.close()
