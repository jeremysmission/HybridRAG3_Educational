# ============================================================================
# HybridRAG - PPTX Parser (src/parsers/office_pptx_parser.py)
# ============================================================================
# Extracts text from PowerPoint (.pptx) slides.
#
# Why this matters:
# - Many briefings and engineering slides contain critical info
# - We want slides searchable just like manuals
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any


class PptxParser:
    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "PptxParser"}

        try:
            from pptx import Presentation  # python-pptx
        except Exception as e:
            details["error"] = f"IMPORT_ERROR: {type(e).__name__}: {e}"
            return "", details

        try:
            pres = Presentation(str(path))

            parts = []
            slide_text_count = 0

            for si, slide in enumerate(pres.slides):
                for shape in slide.shapes:
                    if not hasattr(shape, "text"):
                        continue
                    t = (shape.text or "").strip()
                    if t:
                        slide_text_count += 1
                        parts.append(f"[SLIDE {si+1}] {t}")

            full = "\n\n".join(parts).strip()
            details["total_len"] = len(full)
            details["slides"] = len(pres.slides)
            details["text_blocks"] = slide_text_count

            return full, details
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            return "", details
