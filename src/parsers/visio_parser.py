# ============================================================================
# HybridRAG -- Visio Parser (src/parsers/visio_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Microsoft Visio diagram files (.vsdx) and extracts all text
#   content from shapes, connectors, and page titles.
#
#   Visio is used extensively in engineering and IT for:
#     - Network diagrams
#     - Process flowcharts
#     - Organizational charts
#     - System architecture diagrams
#
#   Each shape in a Visio diagram can contain text labels. This parser
#   extracts all of those labels plus page names.
#
# SUPPORTED FORMATS:
#   .vsdx -- Visio 2013+ (XML-based, fully supported)
#   .vsd  -- Legacy Visio (OLE binary, metadata only via placeholder)
#
# DEPENDENCIES:
#   pip install vsdx  (BSD license, for .vsdx files)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class VsdxParser:
    """
    Extract text from Visio .vsdx files (Visio 2013+).

    NON-PROGRAMMER NOTE:
      .vsdx files are ZIP archives containing XML, similar to .docx.
      The vsdx library understands the Visio XML schema and can walk
      through pages, shapes, and connectors to extract text.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "VsdxParser"}

        try:
            import vsdx
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install vsdx"
            )
            return "", details

        try:
            doc = vsdx.VisioFile(str(path))
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read VSDX: {e}"
            return "", details

        parts: List[str] = [f"Visio Diagram: {path.name}"]
        shape_count = 0

        try:
            for page in doc.pages:
                page_name = getattr(page, "name", "Unnamed Page")
                parts.append(f"\n--- Page: {page_name} ---")

                for shape in page.child_shapes:
                    shape_count += 1
                    text = shape.text or ""
                    text = text.strip()
                    if text:
                        parts.append(text)

                    # Check sub-shapes (grouped shapes)
                    if hasattr(shape, "sub_shapes"):
                        for sub in (shape.sub_shapes() if callable(shape.sub_shapes) else []):
                            sub_text = getattr(sub, "text", "") or ""
                            if sub_text.strip():
                                parts.append(sub_text.strip())
                                shape_count += 1

            doc.close()
        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["shapes"] = shape_count
        return full, details
