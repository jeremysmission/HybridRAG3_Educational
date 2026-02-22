# ============================================================================
# HybridRAG -- Plain Text Parser (src/parsers/plain_text_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   The simplest parser in HybridRAG. It reads files that are already
#   plain text -- no conversion needed. This covers:
#     .txt, .md, .csv, .json, .xml, .log, .yaml, .ini
#
#   Unlike PDF or Word files, these are already human-readable text.
#   We just read the bytes, decode them as UTF-8, and pass along.
#
# WHY errors="ignore"?
#   Some text files may contain a few non-UTF-8 bytes (e.g., a log
#   file from a system that uses Latin-1 encoding). Rather than crash
#   on one bad byte, we skip it and keep the rest. Losing one character
#   out of 100,000 is better than failing to index the entire file.
#
# WHY THIS IS A SEPARATE FILE:
#   Prevents circular imports. The REGISTRY imports parsers, and if
#   this lived in text_parser.py, the import chain would loop.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any


class PlainTextParser:
    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "PlainTextParser"}

        try:
            data = path.read_text(encoding="utf-8", errors="ignore")
            details["total_len"] = len(data)
            return data, details
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            return "", details
