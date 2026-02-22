# ============================================================================
# HybridRAG -- Local HTML File Parser (src/parsers/html_file_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Reads .html / .htm files from disk and extracts readable text.
#   Uses the same html_parser.py text extraction as the HTTP parser.
#
# WHY SEPARATE FROM HTTP PARSER:
#   HttpParser fetches from URLs (requires network access + gate check).
#   This parser reads local files (no network, registered in extension
#   registry alongside PDF/DOCX/etc).
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

from .html_parser import extract_text_from_html


class HtmlFileParser:
    """Parse local .html / .htm files to extract readable text."""

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(
        self, file_path: str
    ) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {
            "file": str(path),
            "parser": "HtmlFileParser",
        }

        try:
            # Try UTF-8 first, fall back to latin-1
            try:
                html = path.read_text(encoding="utf-8")
                details["encoding"] = "utf-8"
            except UnicodeDecodeError:
                html = path.read_text(encoding="latin-1")
                details["encoding"] = "latin-1"

            details["file_size_bytes"] = path.stat().st_size

            text, parse_details = extract_text_from_html(html)
            details["html_parse"] = parse_details
            return text, details

        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            return "", details
