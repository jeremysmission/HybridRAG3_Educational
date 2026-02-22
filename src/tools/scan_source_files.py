#!/usr/bin/env python3
# ============================================================================
# HybridRAG v3 -- Source File Integrity Scanner
# ============================================================================
# FILE: src/tools/scan_source_files.py
#
# WHAT THIS DOES (plain English):
#   Scans your source data folder BEFORE you index, looking for files that
#   would cause problems:
#     - Word temp/lock files (~$ prefix -- not real documents)
#     - Zero-byte files (empty, nothing to index)
#     - Truncated downloads (incomplete torrents, partial copies)
#     - Corrupt Office files (broken ZIP archives inside .docx/.pptx/.xlsx)
#     - Corrupt PDFs (missing structure markers)
#     - Binary garbage (files that parse but produce nonsense text)
#
#   When it finds problems, it shows you a report with:
#     - WHAT is wrong (reason)
#     - HOW it was detected (detection method -- for learning/debugging)
#     - WHAT TO DO about it (suggestion)
#
#   Then asks what you want to do:
#     [Q] Quarantine all   -- move to _quarantine/ folder (recoverable)
#     [D] Delete all       -- permanently remove
#     [I] Individual       -- decide file-by-file (quarantine/delete/skip)
#     [S] Skip all         -- leave everything in place
#
# WHY THIS EXISTS:
#   BUG-004 showed that the indexer's _validate_text() only checks the
#   first 2000 characters. Incomplete torrent downloads often have valid
#   headers but garbage data deeper in the file. Those garbage chunks
#   get embedded into the vector store and pollute ALL search results.
#
#   This tool catches those files BEFORE they ever reach the indexer.
#
# USAGE:
#   python src\tools\scan_source_files.py                    # Interactive
#   python src\tools\scan_source_files.py --auto-quarantine  # Non-interactive
#   python src\tools\scan_source_files.py --report-only      # Just show report
#   python src\tools\scan_source_files.py --deep             # Full parse test
#
# PowerShell (after sourcing start_hybridrag.ps1):
#   rag-scan                    # Interactive scan
#   rag-scan --deep             # Deep parse validation
#   rag-scan --report-only      # Report only, no prompts
#
# INTERNET ACCESS: NONE -- all checks are local file operations
#
# CHANGE LOG:
#   2026-02-15  v1: Initial version
#   2026-02-15  v2: Added Word temp file (~$) detection with diagnostic
#                   messaging. Added detection_method and suggestion fields
#                   to all findings for learning/debugging. Reordered checks
#                   so definitive structural checks run before heuristics.
# ============================================================================

import argparse
import os
import sys
import time
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================
# These match the indexer's file discovery logic exactly.
# If you add extensions to the indexer, add them here too.

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".log", ".pdf",
    ".docx", ".pptx", ".xlsx", ".eml",
    ".yaml", ".yml", ".ini",
}

EXCLUDED_DIRS = {
    ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode",
    "node_modules", "_quarantine",  # Don't scan our own quarantine folder
}

# Office formats that are actually ZIP archives internally
ZIP_BASED_FORMATS = {".docx", ".pptx", ".xlsx"}

# Minimum reasonable file sizes (bytes) -- files smaller than this
# for their type are almost certainly corrupt or placeholder stubs
MIN_FILE_SIZES = {
    ".pdf":  200,    # Smallest valid PDF is ~67 bytes, but real docs > 200
    ".docx": 2000,   # Smallest valid DOCX ZIP is ~1.5KB
    ".pptx": 5000,   # PPTX is always larger than DOCX
    ".xlsx": 2000,   # Similar to DOCX
    ".eml":  100,    # Headers alone are usually > 100 bytes
}

# How many bytes to sample for null-byte detection
SAMPLE_SIZE = 16384  # 16KB

# Null-byte ratio threshold -- above this = likely binary/incomplete
NULL_BYTE_THRESHOLD = 0.20  # 20% null bytes = suspicious

# Text garbage ratio -- below this = binary garbage (matches indexer's 30%)
TEXT_QUALITY_THRESHOLD = 0.30


# ============================================================================
# SEVERITY LEVELS
# ============================================================================

class Severity:
    CRITICAL = "CRITICAL"  # Will definitely cause indexing problems
    WARNING  = "WARNING"   # Probably bad, worth investigating
    INFO     = "INFO"      # Might be fine, but flagged for review


# ============================================================================
# SCAN RESULT
# ============================================================================

class ScanFinding:
    """
    One problematic file with details about what's wrong.

    Every finding includes:
      - reason:           What is wrong (short summary)
      - details:          Why it matters (plain English explanation)
      - detection_method: HOW we detected it (for learning/debugging)
      - suggestion:       What to do about it (actionable next step)
    """

    def __init__(self, filepath: Path, severity: str, reason: str,
                 details: str = "", detection_method: str = "",
                 suggestion: str = "", file_size: int = 0):
        self.filepath = filepath
        self.severity = severity
        self.reason = reason
        self.details = details
        self.detection_method = detection_method
        self.suggestion = suggestion
        self.file_size = file_size

    def __str__(self):
        size_str = self._human_size(self.file_size)
        lines = [
            f"[{self.severity:8s}] {self.filepath.name}",
            f"           Size: {size_str} | {self.reason}",
            f"           Path: {self.filepath}",
            f"           Why:  {self.details}",
        ]
        if self.detection_method:
            lines.append(f"           How detected: {self.detection_method}")
        if self.suggestion:
            lines.append(f"           Suggestion:   {self.suggestion}")
        return "\n".join(lines)

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size."""
        if size_bytes == 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


# ============================================================================
# LEVEL 1: FILE STRUCTURE CHECKS (fast, no parsing)
# ============================================================================
# These checks examine the raw file without trying to parse its contents.
# They catch the obvious problems: Word temp files, zero-byte, truncated
# ZIPs, missing PDF markers, high null-byte ratio, suspiciously small.
#
# CHECK ORDER MATTERS:
#   1. Word temp files (~$ prefix) -- instant, definitive
#   2. Zero-byte -- instant, definitive
#   3. ZIP integrity -- definitive for Office formats
#   4. PDF structure -- definitive for PDFs
#   5. Null-byte ratio -- heuristic (can have false positives on image PDFs)
#   6. Too-small for type -- heuristic (weakest signal)
#
# Definitive checks run FIRST so you get the most accurate reason.
# Only one finding per file (first match wins).


def check_word_temp_file(filepath: Path) -> Optional[ScanFinding]:
    """
    Check if the file is a Microsoft Word/Office temporary lock file.

    WHAT ARE ~$ FILES?
      Word, Excel, and PowerPoint create hidden temporary files whenever
      a document is open for editing. These files always start with "~$"
      followed by part of the original filename. For example:
        - Original: "Discussion_Personnel_Management.docx"
        - Lock file: "~$scussion_Personnel_Management.docx"

      Notice the first two characters of the real name are replaced
      by "~$". These files are typically 162 bytes and contain only
      the username of whoever has the document open.

    WHY THEY CAUSE PROBLEMS:
      - They are NOT valid .docx/.pptx/.xlsx files
      - They don't contain ZIP/XML structure, just a raw byte stream
      - The parser will crash (BadZipFile) or extract garbage
      - They should have been auto-deleted when Office closed, but
        persist when Office crashes or files are on a network drive
    """
    name = filepath.name
    if not name.startswith("~$"):
        return None

    size = filepath.stat().st_size
    # Try to figure out the real document name
    real_name = name[2:]  # Remove the ~$ prefix

    return ScanFinding(
        filepath, Severity.CRITICAL,
        "Office temporary lock file (not a real document)",
        "Files starting with '~$' are temporary lock files that "
        "Microsoft Office creates while a document is open for "
        "editing. They contain the editor's username, not document "
        "content. Office should delete these when the document "
        "closes, but they persist after crashes or on network drives.",
        detection_method=(
            "Filename prefix '~$' is the universal marker for Office "
            "lock files. These are created alongside the real document "
            "and hidden by default in Windows Explorer. The typical "
            "size of 162 bytes is the standard lock file owner-info "
            "block -- far too small to be a real document."
        ),
        suggestion=(
            f"Safe to quarantine or delete -- no document content "
            f"is lost. The real document should be named '{real_name}' "
            f"in the same folder. Verify that file exists before "
            f"removing this lock file."
        ),
        file_size=size,
    )


def check_zero_byte(filepath: Path) -> Optional[ScanFinding]:
    """
    Check if file is zero bytes (completely empty).

    WHY: Empty files waste indexer time and produce no useful content.
    An empty file means the download never started, was interrupted
    at the very beginning, or was a placeholder that was never filled.
    """
    size = filepath.stat().st_size
    if size == 0:
        return ScanFinding(
            filepath, Severity.CRITICAL,
            "Zero-byte file (completely empty)",
            "This file contains no data at all. Likely a failed "
            "download, an empty placeholder, or a file that was "
            "created but never written to.",
            detection_method=(
                "os.stat() reports file size as exactly 0 bytes. "
                "No content exists to parse or index."
            ),
            suggestion=(
                "Quarantine or delete. If this was supposed to be "
                "a real document, re-download from the original source."
            ),
            file_size=size,
        )
    return None


def check_zip_integrity(filepath: Path) -> Optional[ScanFinding]:
    """
    Check if Office documents (.docx, .pptx, .xlsx) have valid ZIP structure.

    WHY: These file formats are actually ZIP archives containing XML files.
    If the download was incomplete, the ZIP structure is broken -- the
    central directory (the "table of contents" at the end of the ZIP) is
    missing or truncated. Python's zipfile module can detect this.
    """
    ext = filepath.suffix.lower()
    if ext not in ZIP_BASED_FORMATS:
        return None

    size = filepath.stat().st_size
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            # testzip() reads every file in the archive and checks CRC32
            # Returns the name of the first bad file, or None if all OK
            bad_file = zf.testzip()
            if bad_file:
                return ScanFinding(
                    filepath, Severity.CRITICAL,
                    f"Corrupt ZIP archive (bad entry: {bad_file})",
                    f"This {ext} file is a ZIP archive internally, "
                    f"and the entry '{bad_file}' has corrupt data "
                    "(CRC32 checksum mismatch). The file was likely "
                    "truncated during download.",
                    detection_method=(
                        f"Python zipfile.ZipFile.testzip() reads every "
                        f"entry in the archive and verifies CRC32 "
                        f"checksums. Entry '{bad_file}' failed the "
                        f"checksum -- its data doesn't match what was "
                        f"originally written."
                    ),
                    suggestion=(
                        "Quarantine and re-download from the original "
                        "source. The file's internal structure is damaged."
                    ),
                    file_size=size,
                )
    except zipfile.BadZipFile:
        return ScanFinding(
            filepath, Severity.CRITICAL,
            "Invalid ZIP structure (not a valid Office document)",
            f"This {ext} file should be a ZIP archive internally, "
            "but the ZIP structure is completely broken. The file "
            "is corrupt -- either the download was incomplete or "
            "the file was damaged in transit.",
            detection_method=(
                "Python zipfile.ZipFile() raised BadZipFile when "
                "trying to open the file. The ZIP magic bytes or "
                "central directory are missing/invalid. This is "
                "definitive -- the file cannot be a valid Office "
                "document."
            ),
            suggestion=(
                "Quarantine or delete. This file is unrecoverable "
                "without the original source. Re-download if needed."
            ),
            file_size=size,
        )
    except Exception as e:
        return ScanFinding(
            filepath, Severity.WARNING,
            f"Cannot verify ZIP integrity: {type(e).__name__}: {e}",
            "Could not open the file to check its structure.",
            detection_method=(
                f"zipfile.ZipFile() raised {type(e).__name__}. "
                "The file may be locked by another process or have "
                "permission issues."
            ),
            suggestion="Try closing any programs that might have this file open.",
            file_size=size,
        )
    return None


def check_pdf_structure(filepath: Path) -> Optional[ScanFinding]:
    """
    Check if a PDF has valid structural markers.

    A valid PDF must:
      - Start with %PDF (magic bytes identifying the format)
      - End with %%EOF (end-of-file marker, possibly with whitespace)
      - Contain an xref table or /XRef stream (page directory)

    Missing any of these means the file is truncated, corrupt, or not
    actually a PDF despite its extension.
    """
    if filepath.suffix.lower() != ".pdf":
        return None

    size = filepath.stat().st_size
    try:
        with open(filepath, "rb") as f:
            # Check header (first 10 bytes)
            header = f.read(10)
            if not header.startswith(b"%PDF"):
                return ScanFinding(
                    filepath, Severity.CRITICAL,
                    "Not a valid PDF (missing %PDF header)",
                    "This file has a .pdf extension but doesn't start "
                    "with the PDF magic bytes (%PDF). It might be an "
                    "HTML error page saved with .pdf extension, a "
                    "renamed file, or a completely corrupt download.",
                    detection_method=(
                        "Read first 10 bytes and checked for the PDF "
                        "magic byte sequence '%PDF'. Every valid PDF "
                        "file must start with these 4 bytes -- this is "
                        "defined in the PDF specification (ISO 32000)."
                    ),
                    suggestion=(
                        "Open the file in a text editor to see what it "
                        "actually contains. If it's HTML, it's probably "
                        "an error page. Quarantine or delete."
                    ),
                    file_size=size,
                )

            # Check trailer (last 1KB for %%EOF)
            read_size = min(1024, size)
            f.seek(-read_size, 2)  # Seek from end
            tail = f.read(read_size)

            if b"%%EOF" not in tail:
                return ScanFinding(
                    filepath, Severity.CRITICAL,
                    "Truncated PDF (missing %%EOF marker)",
                    "This PDF is missing its end-of-file marker, "
                    "which means it was truncated during download. "
                    "Some pages may be missing or contain garbled "
                    "data. The parser may extract partial text from "
                    "early pages but it will be unreliable.",
                    detection_method=(
                        "Read the last 1KB of the file and searched "
                        "for the '%%EOF' marker. Per the PDF spec, "
                        "every valid PDF must end with this marker "
                        "(possibly followed by whitespace). Its absence "
                        "means the file was cut short."
                    ),
                    suggestion=(
                        "Quarantine and re-download. If partial content "
                        "is needed, try opening in a PDF viewer -- some "
                        "viewers can recover pages from truncated files."
                    ),
                    file_size=size,
                )

            # Check for xref table or cross-reference stream
            # Some modern PDFs use /XRef streams instead of xref tables
            check_size = min(4096, size)
            f.seek(-check_size, 2)
            bigger_tail = f.read(check_size)
            if b"xref" not in bigger_tail and b"/XRef" not in bigger_tail:
                return ScanFinding(
                    filepath, Severity.WARNING,
                    "PDF may be damaged (no xref table found)",
                    "The cross-reference table was not found in the "
                    "expected location. The PDF might still be readable "
                    "but some pages could be inaccessible.",
                    detection_method=(
                        "Searched the last 4KB for 'xref' (traditional "
                        "cross-reference table) or '/XRef' (modern "
                        "cross-reference stream). The xref is the PDF's "
                        "page directory -- without it, the reader has "
                        "to guess where each page starts."
                    ),
                    suggestion=(
                        "Try opening in a PDF viewer. If it displays "
                        "correctly, this may be a false positive (some "
                        "PDF generators put the xref far from the end). "
                        "If pages are missing or garbled, quarantine."
                    ),
                    file_size=size,
                )
    except (PermissionError, OSError) as e:
        return ScanFinding(
            filepath, Severity.WARNING,
            f"Cannot read PDF: {e}",
            "File may be locked or have permission issues.",
            detection_method=f"open() raised {type(e).__name__}.",
            suggestion="Close any programs using this file and retry.",
            file_size=size,
        )
    return None


def check_null_bytes(filepath: Path) -> Optional[ScanFinding]:
    """
    Check if file has a high ratio of null bytes (0x00).

    WHY: Incomplete downloads often have valid data at the start but
    long stretches of null bytes where the download never filled in.
    This is the #1 sign of incomplete torrent data -- the torrent
    client pre-allocates the file at full size but only fills in the
    pieces it actually downloaded. The unfilled pieces are all zeros.

    IMPORTANT: Scanned PDFs with embedded images can have some null
    bytes naturally (image binary streams). The 20% threshold is set
    high enough to avoid most false positives, but image-heavy PDFs
    might occasionally trigger this. The --deep flag can confirm by
    actually parsing the file and checking the text output.
    """
    size = filepath.stat().st_size
    if size == 0:
        return None  # Caught by check_zero_byte

    try:
        with open(filepath, "rb") as f:
            # Read a sample from the MIDDLE of the file, not just start.
            # Torrent files often have valid headers but null middles.
            if size > SAMPLE_SIZE * 2:
                f.seek(size // 2 - SAMPLE_SIZE // 2)
                sample = f.read(SAMPLE_SIZE)
            else:
                sample = f.read(SAMPLE_SIZE)

        null_count = sample.count(b'\x00')
        ratio = null_count / len(sample) if sample else 0

        if ratio > NULL_BYTE_THRESHOLD:
            pct = f"{ratio * 100:.1f}%"
            return ScanFinding(
                filepath, Severity.CRITICAL,
                f"High null-byte ratio: {pct} (threshold: "
                f"{NULL_BYTE_THRESHOLD*100:.0f}%)",
                "This file contains large stretches of zero bytes, "
                "which is the classic signature of an incomplete "
                "torrent download or truncated network copy. The "
                "torrent client pre-allocates the file at full size "
                "but only fills in the pieces it downloaded -- the "
                "rest stays as zeros.",
                detection_method=(
                    f"Read {len(sample):,} bytes from the MIDDLE of "
                    f"the file (not the start, since headers are often "
                    f"valid). Counted null bytes (0x00): "
                    f"{null_count:,} out of {len(sample):,} = {pct}. "
                    f"Threshold is {NULL_BYTE_THRESHOLD*100:.0f}%. "
                    f"NOTE: Image-heavy scanned PDFs can have ~10-15% "
                    f"null bytes naturally from embedded image streams. "
                    f"If this is a scanned PDF at 20-30%, run "
                    f"rag-scan --deep to verify via actual parsing."
                ),
                suggestion=(
                    "If this is a torrent download, the file is "
                    "incomplete -- quarantine and re-download. "
                    "If this is a scanned/image-heavy PDF, run "
                    "rag-scan --deep to confirm. If the deep scan "
                    "shows the text is readable, the file is OK."
                ),
                file_size=size,
            )
    except (PermissionError, OSError) as e:
        return ScanFinding(
            filepath, Severity.WARNING,
            f"Cannot read file: {e}",
            "File may be locked by another process.",
            detection_method=f"open() raised {type(e).__name__}.",
            suggestion="Close any programs using this file and retry.",
            file_size=size,
        )
    return None


def check_too_small(filepath: Path) -> Optional[ScanFinding]:
    """
    Check if file is suspiciously small for its type.

    WHY: A 50-byte .pdf or a 500-byte .docx is almost certainly not
    a real document. It might be an error page saved with the wrong
    extension, or a download that started but barely got any data.
    """
    ext = filepath.suffix.lower()
    size = filepath.stat().st_size
    min_size = MIN_FILE_SIZES.get(ext)

    if min_size and size < min_size and size > 0:
        return ScanFinding(
            filepath, Severity.WARNING,
            f"Too small for {ext} ({size} bytes, minimum "
            f"expected: {min_size})",
            "Real documents of this type are always larger than "
            "this. The file is likely a stub, an error page saved "
            "with the wrong extension, or a heavily truncated "
            "download.",
            detection_method=(
                f"Compared file size ({size} bytes) against the "
                f"minimum expected size for {ext} files "
                f"({min_size} bytes). This threshold was determined "
                f"by examining the smallest valid files of each type."
            ),
            suggestion=(
                "Open the file manually to check if it has real "
                "content. If it's an error page or stub, quarantine."
            ),
            file_size=size,
        )
    return None


# ============================================================================
# LEVEL 2: DEEP PARSE VALIDATION (slower, uses actual parsers)
# ============================================================================
# These checks actually try to parse the file and validate the output.
# Slower than Level 1 but catches files that have valid structure but
# produce garbage text.

def check_parse_output(filepath: Path) -> Optional[ScanFinding]:
    """
    Actually parse the file and check if the output text is valid.

    Unlike the indexer's _validate_text() which only checks the first
    2000 characters, this checks MULTIPLE windows throughout the text
    to catch files that start clean but go bad partway through.

    THE CLASSIC INCOMPLETE-TORRENT PATTERN:
      - First few pages: valid text (passes _validate_text)
      - Middle pages: binary garbage from undownloaded pieces
      - Last pages: might be valid again (torrent downloaded them)

    By sampling windows at 0%, 25%, 50%, 75% through the text, we
    catch the garbage in the middle.
    """
    ext = filepath.suffix.lower()
    size = filepath.stat().st_size

    try:
        from src.parsers.registry import REGISTRY
        parser_info = REGISTRY.get(ext)
        if not parser_info:
            return None  # No parser for this type

        parser = parser_info.parser_cls()
        text = parser.parse(str(filepath))

        if not text or len(text) < 20:
            return ScanFinding(
                filepath, Severity.WARNING,
                "Parser returned empty or near-empty text",
                f"The {ext} parser ran without errors but extracted "
                f"only {len(text) if text else 0} characters. The "
                "file may be a scanned image without OCR, a blank "
                "template, or genuinely empty.",
                detection_method=(
                    f"Parsed the file with {parser_info.name} and "
                    f"measured the output length. Documents under 20 "
                    f"characters contain too little text to be useful "
                    f"for RAG retrieval."
                ),
                suggestion=(
                    "Open the file manually. If it's a scanned image, "
                    "OCR would be needed to extract text. If it's a "
                    "blank template, quarantine it."
                ),
                file_size=size,
            )

        # Multi-window garbage detection
        text_len = len(text)
        window_size = 2000
        bad_windows = []

        for pct in [0, 25, 50, 75]:
            start = int(text_len * pct / 100)
            end = min(start + window_size, text_len)
            window = text[start:end]

            if len(window) < 50:
                continue  # Not enough text to judge

            normal = sum(
                1 for ch in window
                if ch.isalnum()
                or ch in ' \t\n\r.,;:!?()-/\'"@#$%&*+=[]{}|<>~'
            )
            ratio = normal / len(window)

            if ratio < TEXT_QUALITY_THRESHOLD:
                bad_windows.append(
                    f"{pct}% ({ratio*100:.0f}% readable)"
                )

        if bad_windows:
            return ScanFinding(
                filepath, Severity.CRITICAL,
                f"Binary garbage in parsed output at "
                f"{len(bad_windows)} position(s)",
                f"Garbage windows: {', '.join(bad_windows)}. "
                "The file parses but produces unreadable text in "
                "sections. This is the classic pattern of an "
                "incomplete torrent download where some pieces "
                "were received and others were not. If indexed, "
                "these garbage chunks pollute search results.",
                detection_method=(
                    f"Parsed the file with {parser_info.name}, then "
                    f"sampled 2000-char windows at 0%, 25%, 50%, and "
                    f"75% through the {text_len:,}-character output. "
                    f"Each window was checked for the ratio of 'normal' "
                    f"characters (letters, digits, punctuation) vs "
                    f"binary garbage. Threshold: {TEXT_QUALITY_THRESHOLD*100:.0f}% "
                    f"(same as the indexer's _validate_text). "
                    f"Unlike the indexer, we check multiple windows -- "
                    f"not just the first 2000 chars."
                ),
                suggestion=(
                    "Quarantine and re-download. If partial content "
                    "is acceptable, you could manually extract the "
                    "readable sections, but the garbage chunks will "
                    "degrade RAG search quality if indexed."
                ),
                file_size=size,
            )

    except Exception as e:
        return ScanFinding(
            filepath, Severity.CRITICAL,
            f"Parser crashed: {type(e).__name__}: {e}",
            "The parser threw an exception trying to read this "
            "file. It is definitely corrupt or in an unsupported "
            "sub-format.",
            detection_method=(
                f"Called {ext} parser.parse() which raised "
                f"{type(e).__name__}. This means the file's internal "
                f"structure is too damaged for the parser to handle."
            ),
            suggestion=(
                "Quarantine. If the file opens in its native app "
                "(Word, Acrobat, etc.), the parser may need updating. "
                "If it won't open anywhere, the file is corrupt."
            ),
            file_size=size,
        )
    return None


# ============================================================================
# SCANNER ENGINE
# ============================================================================

def discover_files(source_dir: Path) -> List[Path]:
    """
    Walk the source directory and find all files the indexer would try
    to process. Uses the SAME logic as src/core/indexer.py so we scan
    exactly what would be indexed.
    """
    files = []
    for f in sorted(source_dir.rglob("*")):
        if not f.is_file():
            continue

        # Skip excluded directories
        parts_lower = {p.lower() for p in f.parts}
        if parts_lower & {d.lower() for d in EXCLUDED_DIRS}:
            continue

        # Skip unsupported extensions
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(f)
    return files


def scan_files(
    files: List[Path],
    deep: bool = False,
    progress: bool = True,
) -> List[ScanFinding]:
    """
    Run all integrity checks on a list of files.

    Level 1 (always): Structure checks (fast, no parsing)
    Level 2 (--deep): Parse validation (slower, catches garbage text)

    Check order (definitive first, heuristic last):
      1. Word temp file (~$ prefix)
      2. Zero-byte
      3. ZIP integrity (Office formats)
      4. PDF structure
      5. Null-byte ratio
      6. Too-small for type
    """
    findings = []
    total = len(files)

    for i, filepath in enumerate(files, 1):
        if progress and i % 50 == 0:
            print(f"  Scanning... {i}/{total} files", end="\r")

        # --- Level 1: Structure checks (always run) ---
        # Order: definitive checks first, heuristics last.
        # Only one finding per file (first match wins).
        for check_fn in [
            check_word_temp_file,
            check_zero_byte,
            check_zip_integrity,
            check_pdf_structure,
            check_null_bytes,
            check_too_small,
        ]:
            finding = check_fn(filepath)
            if finding:
                findings.append(finding)
                break  # One finding per file is enough
        else:
            # No Level 1 findings -- run Level 2 if requested
            if deep:
                finding = check_parse_output(filepath)
                if finding:
                    findings.append(finding)

    if progress:
        print(f"  Scanning... {total}/{total} files -- done")

    return findings


# ============================================================================
# USER INTERACTION
# ============================================================================

def print_report(findings: List[ScanFinding], total_files: int):
    """Print a formatted report of all findings."""
    print()
    print("=" * 72)
    print("  SOURCE FILE INTEGRITY REPORT")
    print("=" * 72)

    if not findings:
        print()
        print("  No problems found!")
        print(f"  Scanned {total_files} files -- all look healthy.")
        print()
        print("=" * 72)
        return

    # Count by severity
    critical = [f for f in findings if f.severity == Severity.CRITICAL]
    warnings = [f for f in findings if f.severity == Severity.WARNING]
    info     = [f for f in findings if f.severity == Severity.INFO]

    print()
    print(f"  Scanned: {total_files} files")
    print(f"  Problems found: {len(findings)}")
    print(f"    CRITICAL: {len(critical)} (will cause indexing problems)")
    print(f"    WARNING:  {len(warnings)} (probably bad, investigate)")
    print(f"    INFO:     {len(info)} (might be fine)")
    print()

    for i, finding in enumerate(findings, 1):
        print(f"  --- #{i} ---")
        print(f"  {finding}")
        print()

    print("=" * 72)


def prompt_action(findings: List[ScanFinding]) -> str:
    """
    Ask the user what to do with the suspect files.

    Options:
      Q = Quarantine all (move to _quarantine/ -- safe, recoverable)
      D = Delete all permanently
      I = Individual (decide file-by-file: quarantine, delete, or skip)
      S = Skip all (do nothing)
    """
    print()
    print("  What would you like to do with these files?")
    print()
    print("    [Q] Quarantine all -- Move to _quarantine/ folder (RECOMMENDED)")
    print("                         Files preserved, won't be indexed.")
    print("                         You can recover them later if needed.")
    print()
    print("    [D] Delete all     -- Permanently remove all flagged files")
    print("                         Cannot be undone!")
    print()
    print("    [I] Individual     -- Decide for each file one by one")
    print("                         You'll see each finding and choose")
    print("                         quarantine, delete, or skip per file.")
    print()
    print("    [S] Skip all       -- Do nothing, leave all files in place")
    print()

    while True:
        choice = input("  Your choice [Q/D/I/S]: ").strip().upper()
        if choice in ("Q", "D", "I", "S"):
            return choice
        print("  Please enter Q, D, I, or S.")


def prompt_individual(finding: ScanFinding, index: int,
                      total: int) -> str:
    """
    Ask the user what to do with ONE specific file.

    Shows the full finding details so the user can make an informed
    decision, then prompts for quarantine, delete, or skip.
    """
    print()
    print(f"  === File {index}/{total} ===")
    print(f"  {finding}")
    print()
    while True:
        ans = input("  [Q]uarantine / [D]elete / [S]kip this file? ").strip().upper()
        if ans in ("Q", "D", "S"):
            return ans
        print("  Please enter Q, D, or S.")


def quarantine_file(filepath: Path, source_dir: Path) -> bool:
    """
    Move a file to the _quarantine/ subfolder, preserving relative path.

    Example:
      Source: D:\\RAG Source Data\\reports\\annual.pdf
      Quarantine: D:\\RAG Source Data\\_quarantine\\reports\\annual.pdf
    """
    try:
        quarantine_dir = source_dir / "_quarantine"
        relative = filepath.relative_to(source_dir)
        dest = quarantine_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(filepath), str(dest))
        return True
    except Exception as e:
        print(f"  [ERROR] Could not quarantine {filepath.name}: {e}")
        return False


def delete_file(filepath: Path) -> bool:
    """Permanently delete a file."""
    try:
        filepath.unlink()
        return True
    except Exception as e:
        print(f"  [ERROR] Could not delete {filepath.name}: {e}")
        return False


def execute_action(choice: str, findings: List[ScanFinding],
                   source_dir: Path) -> Dict[str, int]:
    """Execute the user's chosen action on all findings."""
    stats = {"quarantined": 0, "deleted": 0, "skipped": 0, "errors": 0}

    for i, finding in enumerate(findings, 1):
        file_action = choice

        # In individual mode, prompt for each file
        if choice == "I":
            file_action = prompt_individual(finding, i, len(findings))

        if file_action == "Q":
            if quarantine_file(finding.filepath, source_dir):
                stats["quarantined"] += 1
                print(f"  [QUARANTINED] {finding.filepath.name}")
            else:
                stats["errors"] += 1
        elif file_action == "D":
            if delete_file(finding.filepath):
                stats["deleted"] += 1
                print(f"  [DELETED] {finding.filepath.name}")
            else:
                stats["errors"] += 1
        else:
            stats["skipped"] += 1
            if choice != "S":
                # Only print skip message in individual mode
                print(f"  [SKIPPED] {finding.filepath.name}")

    return stats


def print_action_summary(stats: Dict[str, int], source_dir: Path):
    """Print what was done."""
    print()
    print("  ACTION SUMMARY")
    print("  " + "-" * 40)

    if stats["quarantined"] > 0:
        q_dir = source_dir / "_quarantine"
        print(f"  Quarantined: {stats['quarantined']} files")
        print(f"  Location:    {q_dir}")
        print(f"  To recover:  Move files back from _quarantine/")
        print(f"               to their original location.")

    if stats["deleted"] > 0:
        print(f"  Deleted:     {stats['deleted']} files (permanent)")

    if stats["skipped"] > 0:
        print(f"  Skipped:     {stats['skipped']} files (left in place)")

    if stats["errors"] > 0:
        print(f"  Errors:      {stats['errors']} files (could not process)")

    total_removed = stats["quarantined"] + stats["deleted"]
    if total_removed > 0:
        print()
        print("  NEXT STEP: Re-index to rebuild the vector store without")
        print("  the bad files. Run:  rag-index")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Scan source files for corruption before indexing"
    )
    parser.add_argument(
        "--source", "-s",
        help="Source directory to scan "
             "(default: HYBRIDRAG_INDEX_FOLDER env var)"
    )
    parser.add_argument(
        "--deep", "-d",
        action="store_true",
        help="Run deep parse validation (slower, catches garbage text)"
    )
    parser.add_argument(
        "--report-only", "-r",
        action="store_true",
        help="Show report only, don't prompt for action"
    )
    parser.add_argument(
        "--auto-quarantine",
        action="store_true",
        help="Automatically quarantine all findings (non-interactive)"
    )
    args = parser.parse_args()

    # Determine source directory
    source_dir = None
    if args.source:
        source_dir = Path(args.source)
    else:
        env_dir = os.environ.get("HYBRIDRAG_INDEX_FOLDER")
        if env_dir:
            source_dir = Path(env_dir)
        else:
            try:
                from src.core.config import load_config
                cfg = load_config(str(PROJECT_ROOT))
                if cfg.indexing.source_folder:
                    source_dir = Path(cfg.indexing.source_folder)
            except Exception:
                pass

    if not source_dir or not source_dir.exists():
        print()
        print("  [ERROR] Source directory not found.")
        print()
        if source_dir:
            print(f"  Checked: {source_dir}")
        print("  Set HYBRIDRAG_INDEX_FOLDER or use --source PATH")
        print()
        sys.exit(1)

    # Run scan
    print()
    print("=" * 72)
    print("  SOURCE FILE INTEGRITY SCANNER")
    print("=" * 72)
    print(f"  Source:     {source_dir}")
    scan_label = "Deep (structure + parse)" if args.deep else "Fast (structure only)"
    print(f"  Scan mode: {scan_label}")
    print()

    start_time = time.time()

    # Step 1: Discover files
    print("  Discovering files...")
    files = discover_files(source_dir)
    print(f"  Found {len(files)} indexable files")
    print()

    if not files:
        print("  No files found to scan. Check your source directory.")
        sys.exit(0)

    # Step 2: Run checks
    findings = scan_files(files, deep=args.deep)
    elapsed = time.time() - start_time

    # Step 3: Report
    print_report(findings, len(files))

    if not findings:
        print(f"  Scan completed in {elapsed:.1f} seconds")
        sys.exit(0)

    print(f"  Scan completed in {elapsed:.1f} seconds")

    # Step 4: Action
    if args.report_only:
        print()
        print("  (Report-only mode -- no files were modified)")
        print()
        sys.exit(0)

    if args.auto_quarantine:
        choice = "Q"
        print()
        print("  Auto-quarantine mode: moving all findings to _quarantine/")
    else:
        choice = prompt_action(findings)

    if choice == "S":
        print()
        print("  Skipped -- no files were modified.")
        print()
        sys.exit(0)

    # Execute
    stats = execute_action(choice, findings, source_dir)
    print_action_summary(stats, source_dir)


if __name__ == "__main__":
    main()
