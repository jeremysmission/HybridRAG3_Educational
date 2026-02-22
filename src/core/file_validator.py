# ============================================================================
# HybridRAG -- File Validator (src/core/file_validator.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Pre-flight integrity checks and text validation for the indexing pipeline.
#   Extracted from indexer.py to keep class sizes under 500 lines.
#
# WHY SEPARATE:
#   The Indexer class was 599 lines. Extracting validation logic (165 lines)
#   into this module keeps both under 500 while maintaining a clean single-
#   responsibility boundary: this module decides IF a file should be indexed,
#   while indexer.py handles HOW to index it.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional, Set


# -------------------------------------------------------------------
# Pre-flight check constants
# -------------------------------------------------------------------

_ZIP_BASED_FORMATS = {".docx", ".pptx", ".xlsx"}

_MIN_FILE_SIZES = {
    ".pdf":  200,
    ".docx": 2000,
    ".pptx": 5000,
    ".xlsx": 2000,
    ".eml":  100,
}

_PREFLIGHT_SAMPLE_SIZE = 16384  # 16KB for null-byte sampling
_NULL_BYTE_THRESHOLD = 0.20     # 20% null bytes = suspect


class FileValidator:
    """
    Pre-flight and post-parse validation for indexing candidates.

    Usage:
        validator = FileValidator(excluded_dirs={"__pycache__", ".git"})
        reason = validator.preflight_check(file_path)
        if reason:
            skip(file_path, reason)
        else:
            text = parse(file_path)
            if not validator.validate_text(text):
                skip(file_path, "binary garbage")
    """

    def __init__(self, excluded_dirs: Optional[Set[str]] = None):
        self._excluded_dirs = excluded_dirs or set()

    def preflight_check(self, file_path: Path) -> Optional[str]:
        """
        Fast structural integrity check BEFORE parsing.

        Runs <1ms per file (just stat() + small reads). Returns None
        if the file looks OK, or a reason string if it should be skipped.

        Checks (in order):
          1. Word/Office temp lock file (~$ prefix)
          2. Zero-byte file
          3. ZIP integrity for Office formats (.docx, .pptx, .xlsx)
          4. PDF structure (header + %%EOF)
          5. High null-byte ratio (incomplete download signature)
          6. Too small for file type
        """
        name = file_path.name
        ext = file_path.suffix.lower()

        # --- Check 1: Word/Office temp lock file ---
        if name.startswith("~$"):
            return "Office temp lock file (not a document)"

        # --- Check 2: Zero-byte file ---
        try:
            size = file_path.stat().st_size
        except OSError as e:
            return f"Cannot stat file: {e}"

        if size == 0:
            return "Zero-byte file (empty)"

        # --- Check 3: ZIP integrity for Office formats ---
        if ext in _ZIP_BASED_FORMATS:
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    bad = zf.testzip()
                    if bad:
                        return f"Corrupt ZIP (bad entry: {bad})"
            except zipfile.BadZipFile:
                return "Invalid ZIP structure (not a valid Office doc)"
            except Exception as e:
                return f"Cannot verify ZIP: {type(e).__name__}"

        # --- Check 4: PDF structure ---
        if ext == ".pdf":
            try:
                with open(file_path, "rb") as f:
                    header = f.read(10)
                    if not header.startswith(b"%PDF"):
                        return "Not a valid PDF (missing %PDF header)"
                    read_size = min(1024, size)
                    f.seek(-read_size, 2)
                    tail = f.read(read_size)
                    if b"%%EOF" not in tail:
                        return "Truncated PDF (missing %%EOF marker)"
            except (PermissionError, OSError) as e:
                return f"Cannot read PDF: {e}"

        # --- Check 5: Null-byte ratio ---
        if size > 1000:
            try:
                with open(file_path, "rb") as f:
                    if size > _PREFLIGHT_SAMPLE_SIZE * 2:
                        f.seek(size // 2 - _PREFLIGHT_SAMPLE_SIZE // 2)
                        sample = f.read(_PREFLIGHT_SAMPLE_SIZE)
                    else:
                        sample = f.read(_PREFLIGHT_SAMPLE_SIZE)

                null_count = sample.count(b'\x00')
                ratio = null_count / len(sample) if sample else 0

                if ratio > _NULL_BYTE_THRESHOLD:
                    pct = f"{ratio * 100:.0f}%"
                    return (
                        f"High null-byte ratio ({pct}) -- "
                        f"likely incomplete download"
                    )
            except (PermissionError, OSError):
                pass  # Non-fatal, continue to parsing

        # --- Check 6: Too small for file type ---
        min_size = _MIN_FILE_SIZES.get(ext)
        if min_size and size < min_size:
            return (
                f"Too small for {ext} ({size}B, "
                f"min expected: {min_size}B)"
            )

        # All checks passed
        return None

    def validate_text(self, text: str) -> bool:
        """
        Check if extracted text is readable, not binary garbage.

        Takes a sample of the text (first 2000 chars) and checks
        what proportion of characters are "normal" (letters, digits,
        spaces, common punctuation). If less than 30%, it is garbage.

        Returns True if text looks valid, False otherwise.
        """
        if not text or len(text) < 20:
            return False

        sample = text[:2000]

        # Fast-reject: null bytes never appear in real text documents
        null_count = sample.count('\x00')
        if null_count / len(sample) > 0.01:
            return False

        normal_count = 0
        for ch in sample:
            if ch.isalnum() or ch in ' \t\n\r.,;:!?()-/\'"@#$%&*+=[]{}|<>~':
                normal_count += 1

        ratio = normal_count / len(sample)
        return ratio >= 0.30

    def is_excluded(self, file_path: Path) -> bool:
        """Check if any parent directory is in the exclusion list."""
        parts_lower = {p.lower() for p in file_path.parts}
        return bool(parts_lower & {d.lower() for d in self._excluded_dirs})
