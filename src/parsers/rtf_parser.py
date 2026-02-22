# ============================================================================
# HybridRAG -- RTF Parser (src/parsers/rtf_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Rich Text Format (.rtf) files and extracts the plain text from them.
#   RTF is an older Microsoft format that predates .docx. Many legacy documents,
#   automated reports, and exported emails use RTF.
#
# WHY THIS MATTERS:
#   RTF files look like gibberish if you open them in a text editor:
#     {\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}
#     \pard Hello World\par}
#   The striprtf library understands this markup and extracts just the
#   readable text: "Hello World"
#
# DEPENDENCIES:
#   pip install striprtf  (BSD-3 license, ~10 KB, pure Python)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


class RtfParser:
    """
    Extract plain text from RTF documents using striprtf.

    NON-PROGRAMMER NOTE:
      RTF is a text-based format (unlike .doc which is binary), but it
      contains formatting codes mixed with the actual content. The
      striprtf library strips out all the formatting codes, leaving
      just the readable text.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "RtfParser"}

        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install striprtf"
            )
            return "", details

        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = rtf_to_text(raw)
            text = (text or "").strip()
            details["total_len"] = len(text)
            return text, details
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {e}"
            return "", details
