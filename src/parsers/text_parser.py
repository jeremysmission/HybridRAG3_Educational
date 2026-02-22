# ============================================================================
# HybridRAG -- Routing Parser (src/parsers/text_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   This is the "traffic cop" for file parsing. When the indexer needs
#   to read a file, it hands the file to TextParser. TextParser looks
#   at the file extension (.pdf, .docx, .xlsx, etc.) and routes it to
#   the correct specialized parser. Think of it like a hospital triage
#   desk: the patient (file) arrives, triage (TextParser) decides which
#   department (parser) handles them.
#
# HOW IT WORKS:
#   1. Look at the file extension (e.g., ".pdf", ".docx", ".txt")
#   2. Look up that extension in the REGISTRY (a mapping table)
#   3. If found, create the matching parser and call its parse method
#   4. If not found, return empty text with an UNSUPPORTED_EXTENSION error
#      (the indexer logs this and moves on -- no crash)
#
# WHY THIS IS SEPARATE FROM plain_text_parser.py:
#   PlainTextParser handles .txt/.md/.csv files specifically. It lives
#   in its own file to avoid circular imports: the REGISTRY imports
#   parsers, and if TextParser and PlainTextParser were in the same
#   file, the import chain would loop.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

from .registry import REGISTRY


class TextParser:
    """
    Routing parser -- looks up the file extension in the REGISTRY
    and delegates to the correct specialized parser.
    """
    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        ext = path.suffix.lower()

        # Look up this file extension in the registry.
        # The registry maps extensions like ".pdf" -> PDFParser, ".docx" -> DocxParser.
        info = REGISTRY.get(ext)
        if info is None:
            # No parser registered for this extension -- return empty.
            # The indexer will log this and skip the file.
            return "", {"file": str(path), "parser": "NONE", "error": f"UNSUPPORTED_EXTENSION: {ext}"}

        # Create the matching parser and call it.
        # Most parsers have parse_with_details() which returns (text, info_dict).
        # Older parsers only have parse() which returns just text.
        parser = info.parser_cls()
        if hasattr(parser, "parse_with_details"):
            return parser.parse_with_details(str(path))  # type: ignore

        return parser.parse(str(path)), {"file": str(path), "parser": info.name}
