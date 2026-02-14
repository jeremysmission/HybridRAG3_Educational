# ============================================================================
# HybridRAG - Routing Parser (src/parsers/text_parser.py)
# ============================================================================
# This module is a ROUTER that chooses the correct parser by file extension.
#
# Why:
# - PDFs need PDFParser (and OCR)
# - Office files need docx/pptx/xlsx parsers
# - Unknown types should not crash indexing
#
# IMPORTANT:
# - PlainTextParser was moved to plain_text_parser.py to avoid circular imports.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

from .registry import REGISTRY


class TextParser:
    """
    Routing parser.
    """
    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        ext = path.suffix.lower()

        info = REGISTRY.get(ext)
        if info is None:
            return "", {"file": str(path), "parser": "NONE", "error": f"UNSUPPORTED_EXTENSION: {ext}"}

        parser = info.parser_cls()
        if hasattr(parser, "parse_with_details"):
            return parser.parse_with_details(str(path))  # type: ignore

        return parser.parse(str(path)), {"file": str(path), "parser": info.name}
