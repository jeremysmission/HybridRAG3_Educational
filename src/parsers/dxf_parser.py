# ============================================================================
# HybridRAG -- DXF Parser (src/parsers/dxf_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads AutoCAD DXF (.dxf) files and extracts all text content from them.
#   DXF is AutoCAD's open exchange format -- unlike .dwg (which is proprietary),
#   DXF files can be read by any program.
#
#   A DXF file contains drawing entities like lines, circles, arcs, and TEXT.
#   For RAG purposes, we extract:
#     - All TEXT and MTEXT entities (labels, annotations, dimensions)
#     - Block names and descriptions
#     - Layer names
#     - Drawing metadata (title, author, comments)
#
# WHY THIS MATTERS:
#   Engineering drawings contain critical information as text annotations:
#   part numbers, dimensions, tolerances, notes, revision history. Extracting
#   this text lets HybridRAG answer questions like "What is the tolerance
#   on the main bearing?" or "What revision is drawing 12345?"
#
# DEPENDENCIES:
#   pip install ezdxf  (MIT license, pure Python + optional Cython)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class DxfParser:
    """
    Extract text content from AutoCAD DXF files using ezdxf.

    NON-PROGRAMMER NOTE:
      ezdxf is the industry-standard Python library for reading DXF files.
      It understands every entity type AutoCAD can produce (R12 through R2018).
      We focus on text entities because those are what matter for search.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "DxfParser"}

        try:
            import ezdxf
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install ezdxf"
            )
            return "", details

        try:
            doc = ezdxf.readfile(str(path))
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read DXF: {e}"
            return "", details

        parts: List[str] = []

        try:
            # --- Drawing metadata ---
            # The header section contains document properties
            header = doc.header
            title = header.get("$TITLE", "") if hasattr(header, "get") else ""
            if title:
                parts.append(f"Title: {title}")

            # --- Layer names ---
            # Layers organize drawing elements. Names often encode meaning
            # (e.g., "DIMENSIONS", "HIDDEN_LINES", "STRUCTURAL")
            layer_names = [layer.dxf.name for layer in doc.layers]
            if layer_names:
                parts.append("Layers: " + ", ".join(layer_names))

            # --- Text entities from modelspace ---
            # Modelspace is where the actual drawing lives.
            # TEXT = single-line text (labels, part numbers)
            # MTEXT = multi-line text (notes, descriptions)
            msp = doc.modelspace()

            for entity in msp:
                etype = entity.dxftype()

                if etype == "TEXT":
                    t = entity.dxf.text or ""
                    if t.strip():
                        parts.append(t.strip())

                elif etype == "MTEXT":
                    # MTEXT can contain formatting codes like {\fArial;...}
                    # The .text property gives us the raw string including codes.
                    # We strip common formatting codes for cleaner output.
                    raw = entity.text or ""
                    clean = _strip_mtext_formatting(raw)
                    if clean.strip():
                        parts.append(clean.strip())

                elif etype == "ATTRIB":
                    # Block attributes: key-value pairs on inserted blocks
                    tag = getattr(entity.dxf, "tag", "")
                    val = getattr(entity.dxf, "text", "")
                    if val.strip():
                        prefix = f"{tag}: " if tag else ""
                        parts.append(f"{prefix}{val.strip()}")

                elif etype == "DIMENSION":
                    # Dimension text (the measured value shown on drawing)
                    t = getattr(entity.dxf, "text", "")
                    if t and t.strip() and t.strip() != "<>":
                        parts.append(f"DIM: {t.strip()}")

            # --- Block definitions ---
            # Blocks are reusable drawing components (like symbols).
            # Block names and their text content can be informative.
            for block in doc.blocks:
                if block.name.startswith("*"):
                    continue  # Skip anonymous blocks
                for entity in block:
                    etype = entity.dxftype()
                    if etype in ("TEXT", "MTEXT"):
                        t = entity.dxf.text if etype == "TEXT" else entity.text
                        t = t or ""
                        if etype == "MTEXT":
                            t = _strip_mtext_formatting(t)
                        if t.strip():
                            parts.append(t.strip())

        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"

        full = "\n\n".join(parts).strip()
        details["total_len"] = len(full)
        details["text_entities"] = len(parts)
        details["layers"] = len(layer_names) if "layer_names" in dir() else 0
        return full, details


def _strip_mtext_formatting(raw: str) -> str:
    """
    Remove common MTEXT formatting codes for cleaner text output.

    NON-PROGRAMMER NOTE:
      MTEXT in DXF uses inline formatting codes like:
        {\\fArial|b1;Bold text}  -- font specification
        \\P                      -- paragraph break
        \\S...^...;              -- stacked fractions
      We strip these to get plain readable text.
    """
    import re
    text = raw
    # Remove font/style specifications: {\fArial|b1;...}
    text = re.sub(r"\{\\f[^;]*;", "", text)
    text = re.sub(r"\{\\[A-Za-z][^;]*;", "", text)
    text = text.replace("}", "")
    # Convert paragraph breaks to newlines
    text = text.replace("\\P", "\n")
    text = text.replace("\\p", "\n")
    # Remove other common codes
    text = re.sub(r"\\[A-Za-z]", " ", text)
    # Clean up multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()
