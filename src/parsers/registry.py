# ============================================================================
# HybridRAG -- Parser Registry (src/parsers/registry.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   The central mapping of file extensions to parser classes. When the
#   indexer encounters a file, it looks up the extension here to find
#   which parser to use. If the extension is not registered, the file
#   is skipped.
#
# HOW TO ADD A NEW FORMAT:
#   1. Create a parser class in src/parsers/ (see existing ones for template)
#   2. Import it at the top of this file
#   3. Register it in the __init__ method below with self.register()
#   4. Add it to docs/FORMAT_SUPPORT.md
#
# DESIGN RULES:
#   - registry.py must NOT import text_parser.py (prevents circular imports)
#   - All extensions are lowercase with leading dot (".pdf", not "pdf")
#   - Each parser must have parse(file_path) and parse_with_details(file_path)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Type

# --- Original parsers ---
from .plain_text_parser import PlainTextParser
from .pdf_parser import PDFParser
from .office_docx_parser import DocxParser
from .office_pptx_parser import PptxParser
from .office_xlsx_parser import XlsxParser
from .eml_parser import EmlParser
from .image_parser import ImageOCRParser
from .html_file_parser import HtmlFileParser

# --- New parsers (expanded parsing) ---
from .dxf_parser import DxfParser
from .stl_parser import StlParser
from .rtf_parser import RtfParser
from .doc_parser import DocParser
from .msg_parser import MsgParser
from .psd_parser import PsdParser
from .step_iges_parser import StepParser, IgesParser
from .visio_parser import VsdxParser
from .certificate_parser import CertificateParser
from .evtx_parser import EvtxParser
from .access_db_parser import AccessDbParser
from .mbox_parser import MboxParser
from .pcap_parser import PcapParser
from .placeholder_parser import PlaceholderParser


@dataclass(frozen=True)
class ParserInfo:
    name: str
    parser_cls: Type


class ParserRegistry:
    """
    Registry of extension -> parser.

    NON-PROGRAMMER NOTE:
      This is the "phone book" that tells HybridRAG which parser to
      use for each file type. When you add a new file format, you
      register it here. Extensions must be lowercase with a leading
      dot (e.g., ".pdf", not "pdf" or ".PDF").
    """

    def __init__(self) -> None:
        self._map: Dict[str, ParserInfo] = {}

        # ==============================================================
        # PLAIN TEXT FORMATS
        # These files are already human-readable text. We just read them.
        # ==============================================================
        for ext in [
            ".txt", ".md", ".csv", ".json", ".xml", ".log",
            ".yaml", ".yml", ".ini", ".cfg", ".conf", ".properties",
            ".reg",  # Windows registry export (text-based)
        ]:
            self.register(ext, "PlainTextParser", PlainTextParser)

        # ==============================================================
        # DOCUMENT FORMATS (office, legacy, rich text)
        # ==============================================================
        self.register(".pdf",  "PDFParser",  PDFParser)
        self.register(".docx", "DocxParser",  DocxParser)
        self.register(".pptx", "PptxParser",  PptxParser)
        self.register(".xlsx", "XlsxParser",  XlsxParser)
        self.register(".doc",  "DocParser",   DocParser)   # Legacy Word 97-2003
        self.register(".rtf",  "RtfParser",   RtfParser)   # Rich Text Format

        # .ai files (Adobe Illustrator) are internally PDF since CS era.
        # We parse them with the PDF parser to extract any embedded text.
        self.register(".ai",   "PDFParser",   PDFParser)

        # ==============================================================
        # EMAIL FORMATS
        # ==============================================================
        self.register(".eml",  "EmlParser",   EmlParser)    # RFC 822 email
        self.register(".msg",  "MsgParser",   MsgParser)    # Outlook .msg
        self.register(".mbox", "MboxParser",  MboxParser)   # Unix mbox archive

        # ==============================================================
        # WEB FORMATS
        # ==============================================================
        self.register(".html", "HtmlFileParser", HtmlFileParser)
        self.register(".htm",  "HtmlFileParser", HtmlFileParser)

        # ==============================================================
        # IMAGE FORMATS (parsed via OCR using Tesseract)
        # ==============================================================
        for ext in [
            ".png", ".jpg", ".jpeg", ".tif", ".tiff",
            ".bmp", ".gif", ".webp",
        ]:
            self.register(ext, "ImageOCRParser", ImageOCRParser)

        # WMF/EMF: Windows Metafile formats. Pillow can rasterize them
        # on Windows (GDI+), then OCR extracts text from the raster.
        self.register(".wmf", "ImageOCRParser", ImageOCRParser)
        self.register(".emf", "ImageOCRParser", ImageOCRParser)

        # PSD: Photoshop files. Extract text layers + layer names.
        self.register(".psd", "PsdParser", PsdParser)

        # ==============================================================
        # CAD / ENGINEERING FORMATS
        # ==============================================================

        # DXF: AutoCAD exchange format (OPEN, fully supported)
        self.register(".dxf", "DxfParser", DxfParser)

        # STEP: ISO 10303 CAD exchange (OPEN, text-based)
        self.register(".stp",  "StepParser", StepParser)
        self.register(".step", "StepParser", StepParser)
        self.register(".ste",  "StepParser", StepParser)

        # IGES: older CAD exchange format (OPEN, text-based)
        self.register(".igs",  "IgesParser", IgesParser)
        self.register(".iges", "IgesParser", IgesParser)

        # STL: 3D printing mesh format (OPEN, geometry metadata)
        self.register(".stl", "StlParser", StlParser)

        # ==============================================================
        # VISIO DIAGRAMS
        # ==============================================================
        self.register(".vsdx", "VsdxParser", VsdxParser)  # Visio 2013+

        # ==============================================================
        # CYBERSECURITY / SYSTEM ADMIN FORMATS
        # ==============================================================
        self.register(".evtx", "EvtxParser",        EvtxParser)        # Windows event logs
        self.register(".pcap", "PcapParser",         PcapParser)        # Network captures
        self.register(".pcapng", "PcapParser",       PcapParser)        # PCAPNG format
        self.register(".cer",  "CertificateParser",  CertificateParser) # X.509 certs
        self.register(".crt",  "CertificateParser",  CertificateParser)
        self.register(".pem",  "CertificateParser",  CertificateParser)

        # ==============================================================
        # DATABASE FORMATS
        # ==============================================================
        self.register(".accdb", "AccessDbParser", AccessDbParser)  # Access 2007+
        self.register(".mdb",   "AccessDbParser", AccessDbParser)  # Access 97-2003

        # ==============================================================
        # PLACEHOLDER FORMATS (recognized but not fully parseable)
        # These create "identity cards" with file metadata.
        # See placeholder_parser.py for requirements to upgrade each.
        # ==============================================================
        for ext in [".prt", ".sldprt", ".asm", ".sldasm"]:
            self.register(ext, "PlaceholderParser", _make_placeholder(ext))

        for ext in [".dwg", ".dwt"]:
            self.register(ext, "PlaceholderParser", _make_placeholder(ext))

        for ext in [".mpp", ".vsd", ".one", ".ost", ".eps"]:
            self.register(ext, "PlaceholderParser", _make_placeholder(ext))

    def register(self, ext: str, name: str, parser_cls) -> None:
        """Register a parser class for a file extension."""
        self._map[ext.lower()] = ParserInfo(name=name, parser_cls=parser_cls)

    def get(self, ext: str) -> Optional[ParserInfo]:
        """Look up the parser for a given extension. Returns None if unknown."""
        return self._map.get(ext.lower())

    def supported_extensions(self) -> list[str]:
        """Return sorted list of all registered extensions."""
        return sorted(self._map.keys())

    def fully_supported_extensions(self) -> list[str]:
        """Return extensions with real parsers (not placeholders)."""
        return sorted(
            ext for ext, info in self._map.items()
            if info.name != "PlaceholderParser"
        )

    def placeholder_extensions(self) -> list[str]:
        """Return extensions that only have placeholder parsers."""
        return sorted(
            ext for ext, info in self._map.items()
            if info.name == "PlaceholderParser"
        )


def _make_placeholder(ext: str):
    """
    Create a PlaceholderParser class pre-configured for a specific extension.

    NON-PROGRAMMER NOTE:
      Each placeholder extension needs its own parser instance so it
      knows which format description to include. This factory function
      creates a small wrapper class for each one.
    """
    class _Placeholder(PlaceholderParser):
        def __init__(self):
            super().__init__(extension=ext)
    _Placeholder.__name__ = f"PlaceholderParser_{ext.lstrip('.')}"
    return _Placeholder


REGISTRY = ParserRegistry()
