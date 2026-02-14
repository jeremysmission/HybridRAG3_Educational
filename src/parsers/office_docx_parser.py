# ============================================================================
# HybridRAG - DOCX Parser (src/parsers/office_docx_parser.py)
# ============================================================================
# Extracts text from Microsoft Word (.docx) files.
#
# Why this matters:
# - Many engineering groups store reports and procedures in docx
# - We want the same index/search behavior as PDFs
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any


class DocxParser:
    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "DocxParser"}

        try:
            from docx import Document  # python-docx
        except Exception as e:
            details["error"] = f"IMPORT_ERROR: {type(e).__name__}: {e}"
            return "", details

        try:
            doc = Document(str(path))

            parts = []
            for p in doc.paragraphs:
                t = (p.text or "").strip()
                if t:
                    parts.append(t)

            full = "\n\n".join(parts).strip()
            details["total_len"] = len(full)
            details["paragraphs"] = len(doc.paragraphs)

            return full, details
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            return "", details
