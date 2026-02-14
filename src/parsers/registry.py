# ============================================================================
# HybridRAG - Parser Registry (src/parsers/registry.py)
# ============================================================================
# Purpose:
# - Map file extensions -> parser classes
# - One place to add support for new formats (AutoCAD, emails, images, etc.)
#
# Design rules:
# - registry.py must NOT import text_parser.py (prevents circular imports)
# - "router" logic lives in text_parser.py
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

from .plain_text_parser import PlainTextParser
from .pdf_parser import PDFParser

# These may already exist in your project; if not, we’ll add them later.
# If any import fails, you’ll see it immediately and we’ll correct quickly.
from .office_docx_parser import DocxParser
from .office_pptx_parser import PptxParser
from .office_xlsx_parser import XlsxParser

# NEW in Step 5B
from .eml_parser import EmlParser
from .image_parser import ImageOCRParser


@dataclass(frozen=True)
class ParserInfo:
    name: str
    parser_cls: Type


class ParserRegistry:
    """
    Registry of extension -> parser.
    Extensions must be lowercase and include the dot (".pdf").
    """

    def __init__(self) -> None:
        self._map: Dict[str, ParserInfo] = {}

        # --- Text-like formats ---
        for ext in [".txt", ".md", ".csv", ".json", ".xml", ".log", ".yaml", ".yml", ".ini"]:
            self.register(ext, "PlainTextParser", PlainTextParser)

        # --- PDFs ---
        self.register(".pdf", "PDFParser", PDFParser)

        # --- Office docs ---
        self.register(".docx", "DocxParser", DocxParser)
        self.register(".pptx", "PptxParser", PptxParser)
        self.register(".xlsx", "XlsxParser", XlsxParser)

        # --- Emails ---
        self.register(".eml", "EmlParser", EmlParser)

        # --- Images (OCR) ---
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"]:
            self.register(ext, "ImageOCRParser", ImageOCRParser)

        # --- Scaffolding placeholders for engineering/CAD formats ---
        # Add your 20+ extensions here soon.
        # Example:
        #   self.register(".dwg", "DwgParser", DwgParser)
        #
        # For formats that require proprietary tools, we can:
        # - "recognize and skip" with an audit log reason
        # - or extract metadata only
        # - or call an external converter if you have one approved

    def register(self, ext: str, name: str, parser_cls) -> None:
        self._map[ext.lower()] = ParserInfo(name=name, parser_cls=parser_cls)

    def get(self, ext: str) -> Optional[ParserInfo]:
        return self._map.get(ext.lower())

    def supported_extensions(self) -> list[str]:
        return sorted(self._map.keys())


REGISTRY = ParserRegistry()
