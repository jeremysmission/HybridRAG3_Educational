# ============================================================================
# HybridRAG - XLSX Parser (src/parsers/office_xlsx_parser.py)
# ============================================================================
# Extracts text from Excel (.xlsx) spreadsheets.
#
# Strategy:
# - Read each sheet
# - Convert rows into pipe-delimited text lines
# - Skip completely empty rows
#
# This is not "perfect Excel understanding", but it makes key content searchable.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any


class XlsxParser:
    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "XlsxParser"}

        try:
            import openpyxl
        except Exception as e:
            details["error"] = f"IMPORT_ERROR: {type(e).__name__}: {e}"
            return "", details

        try:
            wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)

            parts = []
            sheets = 0
            rows_emitted = 0

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                sheets += 1
                parts.append(f"[SHEET] {sheet_name}")

                for row in ws.iter_rows(values_only=True):
                    # Convert row values to strings, skip empty rows
                    vals = []
                    for v in row:
                        if v is None:
                            vals.append("")
                        else:
                            vals.append(str(v).strip())

                    if all(x == "" for x in vals):
                        continue

                    rows_emitted += 1
                    parts.append(" | ".join(vals))

            full = "\n".join(parts).strip()
            details["total_len"] = len(full)
            details["sheets"] = sheets
            details["rows_emitted"] = rows_emitted

            return full, details
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            return "", details
