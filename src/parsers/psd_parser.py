# ============================================================================
# HybridRAG -- Photoshop PSD Parser (src/parsers/psd_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Adobe Photoshop (.psd) files and extracts text content from
#   text layers, plus image metadata (dimensions, color mode, layer names).
#
#   A PSD file is a layered image. Some layers contain text (labels,
#   titles, annotations). We extract that text plus structural info
#   about the document.
#
# WHY THIS MATTERS:
#   Engineers and designers create annotated diagrams, flowcharts, and
#   templates in Photoshop. The text in those layers contains searchable
#   information (labels, part numbers, descriptions).
#
# DEPENDENCIES:
#   pip install psd-tools  (MIT license)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class PsdParser:
    """
    Extract text layers and metadata from Photoshop PSD files.

    NON-PROGRAMMER NOTE:
      PSD files are layered images. Each layer has a name and may contain
      text. We extract the text from all "type layers" (Photoshop's term
      for text layers) plus the names of all layers. This lets HybridRAG
      search for content inside Photoshop files.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "PsdParser"}

        try:
            from psd_tools import PSDImage
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install psd-tools"
            )
            return "", details

        try:
            psd = PSDImage.open(str(path))
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read PSD: {e}"
            return "", details

        parts: List[str] = []

        try:
            # Document metadata
            parts.append(f"Photoshop Document: {path.name}")
            parts.append(f"Dimensions: {psd.width} x {psd.height}")
            parts.append(f"Color mode: {psd.color_mode}")
            parts.append(f"Layers: {len(list(psd.descendants()))}")

            # Walk all layers and extract text
            layer_names = []
            text_contents = []
            for layer in psd.descendants():
                layer_names.append(layer.name)

                # Type layers contain editable text
                if layer.kind == "type":
                    # The text content is in the type layer's data
                    try:
                        if hasattr(layer, "text"):
                            t = layer.text or ""
                            if t.strip():
                                text_contents.append(t.strip())
                    except Exception:
                        pass

            if layer_names:
                parts.append("Layer names: " + ", ".join(layer_names))

            if text_contents:
                parts.append("")
                parts.append("Text content:")
                for t in text_contents:
                    parts.append(f"  {t}")

        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["text_layers"] = len(text_contents) if "text_contents" in dir() else 0
        return full, details
