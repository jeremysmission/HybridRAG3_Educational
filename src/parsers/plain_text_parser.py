# ============================================================================
# HybridRAG - Plain Text Parser (src/parsers/plain_text_parser.py)
# ============================================================================
# Extracts text from "text-like" files:
#   .txt, .md, .csv, .json, .xml, .log, .yaml, etc.
#
# Why this is separate:
# - Prevents circular imports between registry.py and text_parser.py
# - Keeps things modular as we add many parsers (emails, images, CAD, etc.)
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
