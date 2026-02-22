# ============================================================================
# HybridRAG -- Legacy Word .doc Parser (src/parsers/doc_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads legacy Microsoft Word (.doc) files from the Word 97-2003 era.
#   These are BINARY files (not the modern .docx ZIP format).
#
#   The .doc format uses OLE2 (Object Linking and Embedding) -- the same
#   container format used by old Excel, PowerPoint, and Outlook files.
#   We use the olefile library to crack open the OLE container and extract
#   the text stream inside.
#
# WHY THIS MATTERS:
#   Many organizations still have thousands of .doc files from before
#   the switch to .docx (2007). These contain procedures, specifications,
#   and reports that are still referenced. python-docx CANNOT read .doc
#   files -- it only handles .docx. This parser fills that gap.
#
# HOW IT WORKS:
#   1. Open the .doc as an OLE2 container using olefile
#   2. Read the "WordDocument" stream (raw binary)
#   3. Read the text from the binary stream (simplified extraction)
#   4. Also extract metadata (author, title, subject, dates)
#
# LIMITATIONS:
#   - The binary .doc format is complex. This parser extracts the main
#     text body but may miss some formatting, tables, or embedded objects.
#   - For perfect extraction, the external tool "antiword" is better.
#     If antiword is available, we prefer it.
#
# DEPENDENCIES:
#   pip install olefile  (BSD-2 license)
#   Optional: antiword installed on system PATH (GPL-2.0, external tool)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple


class DocParser:
    """
    Extract text from legacy Word .doc files.

    NON-PROGRAMMER NOTE:
      This parser tries two approaches:
        1. antiword (external tool) -- best quality, but must be installed
        2. olefile (Python library) -- fallback, extracts text + metadata
      If both fail, it returns empty text with an error message.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "DocParser"}

        # Strategy 1: Try antiword (best quality)
        text = _try_antiword(str(path), details)
        if text:
            details["method"] = "antiword"
            details["total_len"] = len(text)
            return text, details

        # Strategy 2: OLE-based extraction (fallback)
        text = _try_olefile(str(path), details)
        if text:
            details["method"] = "olefile"
            details["total_len"] = len(text)
            return text, details

        # Strategy 3: Raw binary text extraction (last resort)
        text = _try_raw_extract(str(path), details)
        details["method"] = "raw_binary"
        details["total_len"] = len(text)
        return text, details


def _try_antiword(file_path: str, details: Dict) -> str:
    """Try extracting text using the antiword command-line tool."""
    try:
        result = subprocess.run(
            ["antiword", file_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            details["antiword_available"] = True
            return result.stdout.strip()
        details["antiword_available"] = True
        details["antiword_error"] = result.stderr.strip()
    except FileNotFoundError:
        details["antiword_available"] = False
    except Exception as e:
        details["antiword_error"] = str(e)
    return ""


def _try_olefile(file_path: str, details: Dict) -> str:
    """Extract text and metadata from .doc via OLE2 structure."""
    try:
        import olefile
    except ImportError:
        details["olefile_available"] = False
        return ""

    details["olefile_available"] = True
    parts = []

    try:
        ole = olefile.OleFileIO(file_path)

        # Extract document metadata (title, author, subject, etc.)
        meta = ole.get_metadata()
        for field in ["title", "subject", "author", "comments", "keywords"]:
            val = getattr(meta, field, None)
            if val:
                if isinstance(val, bytes):
                    val = val.decode("utf-8", errors="ignore")
                val = str(val).strip()
                if val:
                    parts.append(f"{field.title()}: {val}")

        # Try to read the WordDocument stream
        if ole.exists("WordDocument"):
            data = ole.openstream("WordDocument").read()
            # Extract ASCII/Unicode text runs from the binary data
            text = _extract_text_from_binary(data)
            if text.strip():
                parts.append(text.strip())

        ole.close()
    except Exception as e:
        details["olefile_error"] = str(e)

    return "\n\n".join(parts).strip()


def _try_raw_extract(file_path: str, details: Dict) -> str:
    """Last resort: scan binary for printable text runs."""
    try:
        with open(file_path, "rb") as f:
            data = f.read(2_000_000)  # Cap at 2 MB
        return _extract_text_from_binary(data)
    except Exception as e:
        details["raw_error"] = str(e)
        return ""


def _extract_text_from_binary(data: bytes) -> str:
    """
    Extract readable text runs from binary data.

    NON-PROGRAMMER NOTE:
      Binary .doc files have text mixed in with formatting codes and
      binary structures. We scan through the bytes looking for sequences
      of printable characters (letters, numbers, spaces, punctuation).
      Any run of 8+ printable characters in a row is likely real text.
      Shorter runs are probably formatting noise and are discarded.
    """
    MIN_RUN = 8  # Minimum length of a text run to keep
    parts = []
    current = []

    for byte in data:
        # Printable ASCII range: space (32) through tilde (126), plus common
        # characters like newline (10) and tab (9)
        if 32 <= byte <= 126 or byte in (9, 10, 13):
            current.append(chr(byte))
        else:
            if len(current) >= MIN_RUN:
                parts.append("".join(current).strip())
            current = []

    if len(current) >= MIN_RUN:
        parts.append("".join(current).strip())

    # Filter out obvious non-text artifacts
    filtered = []
    for p in parts:
        # Skip runs that are mostly special characters
        alpha = sum(1 for c in p if c.isalpha() or c.isspace())
        if alpha > len(p) * 0.5 and len(p) > MIN_RUN:
            filtered.append(p)

    return "\n".join(filtered).strip()
