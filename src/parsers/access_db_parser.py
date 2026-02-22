# ============================================================================
# HybridRAG -- Access Database Parser (src/parsers/access_db_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Microsoft Access database files (.accdb, .mdb) and extracts
#   the table names, column names, and a sample of row data as text.
#
# WHY THIS MATTERS:
#   Access databases are widespread in logistics, program management,
#   and field engineering. They contain structured data (equipment lists,
#   inventory tracking, personnel records) that becomes searchable when
#   extracted as text.
#
# DEPENDENCIES:
#   pip install access-parser  (Apache 2.0 license, pure Python)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class AccessDbParser:
    """
    Extract table structure and sample data from Access .accdb/.mdb files.

    NON-PROGRAMMER NOTE:
      An Access database contains multiple tables, each with columns
      and rows (like a collection of spreadsheets). We extract the
      table names, column names, and a few sample rows to give HybridRAG
      enough context to answer questions about the data.
    """

    MAX_ROWS_PER_TABLE = 50  # Cap rows to prevent enormous output

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "AccessDbParser"}

        try:
            from access_parser import AccessParser
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install access-parser"
            )
            return "", details

        try:
            db = AccessParser(str(path))
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read database: {e}"
            return "", details

        parts: List[str] = [f"Access Database: {path.name}"]
        table_count = 0

        try:
            tables = db.catalog
            for table_name in tables:
                # Skip system tables (start with MSys or ~)
                if table_name.startswith("MSys") or table_name.startswith("~"):
                    continue

                table_count += 1
                parts.append(f"\n--- Table: {table_name} ---")

                try:
                    table = db.parse_table(table_name)
                    columns = list(table.keys()) if isinstance(table, dict) else []
                    if columns:
                        parts.append(f"Columns: {', '.join(columns)}")

                        # Sample rows
                        row_count = len(table[columns[0]]) if columns else 0
                        cap = min(row_count, self.MAX_ROWS_PER_TABLE)

                        for i in range(cap):
                            row_parts = []
                            for col in columns:
                                val = table[col][i] if i < len(table[col]) else ""
                                if isinstance(val, bytes):
                                    val = val.decode("utf-8", errors="ignore")
                                row_parts.append(f"{col}={val}")
                            parts.append("  " + " | ".join(row_parts))

                        if row_count > cap:
                            parts.append(f"  ... {row_count - cap} more rows")
                except Exception as e:
                    parts.append(f"  [ERROR reading table: {e}]")

        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["tables"] = table_count
        return full, details
