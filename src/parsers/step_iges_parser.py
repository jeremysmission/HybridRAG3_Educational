# ============================================================================
# HybridRAG -- STEP/IGES CAD Parser (src/parsers/step_iges_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads STEP (.stp, .step, .ste) and IGES (.igs, .iges) CAD exchange
#   files and extracts text metadata from them.
#
#   STEP and IGES are open CAD exchange formats -- the "universal translator"
#   of the CAD world. When an engineer sends a model to a supplier, they
#   export it as STEP. Unlike proprietary formats (SolidWorks, CATIA),
#   these can be read by any CAD program.
#
#   Both formats are TEXT-BASED (not binary), so we can extract:
#     - File description and header comments
#     - Product/part names
#     - Author, organization, originating system
#     - Schema version and timestamp
#     - Entity counts (how complex the model is)
#
# WHY THIS MATTERS:
#   STEP/IGES files contain product names, descriptions, and provenance
#   data that helps answer "Who created this model?", "What CAD system
#   was it exported from?", "When was it last modified?"
#
# DEPENDENCIES:
#   Optional: pip install steputils  (MIT license, for structured STEP parsing)
#   Falls back to plain text regex extraction if steputils is not installed.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class StepParser:
    """
    Extract metadata from STEP (.stp, .step, .ste) CAD files.

    NON-PROGRAMMER NOTE:
      STEP files start with a HEADER section containing metadata:
        FILE_DESCRIPTION, FILE_NAME, FILE_SCHEMA
      followed by a DATA section containing geometry entities.
      We extract the header metadata (human-readable) and count
      the data entities (tells you model complexity).
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "StepParser"}

        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read file: {e}"
            return "", details

        parts: List[str] = [f"STEP CAD File: {path.name}"]

        # Extract FILE_DESCRIPTION
        m = re.search(r"FILE_DESCRIPTION\s*\(\s*\('([^']*)'\)", raw)
        if m:
            parts.append(f"Description: {m.group(1)}")

        # Extract FILE_NAME fields
        m = re.search(r"FILE_NAME\s*\(\s*'([^']*)'", raw)
        if m:
            parts.append(f"Name: {m.group(1)}")

        # Extract author/organization from FILE_NAME
        fn = re.search(r"FILE_NAME\s*\([^)]*\)", raw, re.DOTALL)
        if fn:
            fn_text = fn.group(0)
            strings = re.findall(r"'([^']+)'", fn_text)
            if len(strings) >= 5:
                parts.append(f"Author: {strings[2]}")
                parts.append(f"Organization: {strings[3]}")
                parts.append(f"CAD System: {strings[4]}")

        # Extract FILE_SCHEMA
        m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", raw)
        if m:
            parts.append(f"Schema: {m.group(1)}")

        # Count entities in the DATA section
        entity_count = len(re.findall(r"^#\d+\s*=", raw, re.MULTILINE))
        parts.append(f"Entities: {entity_count:,}")

        # Extract PRODUCT entity names (the part/assembly names)
        products = re.findall(
            r"PRODUCT\s*\(\s*'([^']*)'[^)]*'([^']*)'", raw
        )
        for pid, pname in products[:10]:
            parts.append(f"Product: {pid} -- {pname}")

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["entities"] = entity_count
        return full, details


class IgesParser:
    """
    Extract metadata from IGES (.igs, .iges) CAD files.

    NON-PROGRAMMER NOTE:
      IGES is an older CAD exchange format (predates STEP). It uses a
      fixed-column text format (80 characters per line, like old punch
      cards). The first section ("Start") contains human-readable comments.
      The "Global" section contains metadata. We extract both.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "IgesParser"}

        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read file: {e}"
            return "", details

        parts: List[str] = [f"IGES CAD File: {path.name}"]
        lines = raw.split("\n")

        # IGES files have a column 73 section flag:
        # S = Start (comments), G = Global (metadata),
        # D = Directory, P = Parameter, T = Terminate
        start_lines = []
        global_text = []
        d_count = 0

        for line in lines:
            if len(line) >= 73:
                flag = line[72]
                if flag == "S":
                    start_lines.append(line[:72].strip())
                elif flag == "G":
                    global_text.append(line[:72].strip())
                elif flag == "D":
                    d_count += 1

        # Start section: human-readable comments
        if start_lines:
            comment = " ".join(start_lines)
            parts.append(f"Comment: {comment}")

        # Global section: parse delimited fields
        # Fields are separated by parameter delimiter (usually comma)
        global_str = "".join(global_text)
        if global_str:
            # Extract strings between delimiters
            strings = re.findall(r"\d+H([^,;]+)", global_str)
            if len(strings) >= 4:
                parts.append(f"Filename: {strings[1]}")
                parts.append(f"System ID: {strings[2]}")
                parts.append(f"Preprocessor: {strings[3]}")
            if len(strings) >= 6:
                parts.append(f"Author: {strings[5]}")
                parts.append(f"Organization: {strings[6]}" if len(strings) > 6 else "")

        parts.append(f"Entities: {d_count // 2:,}")

        full = "\n".join(p for p in parts if p).strip()
        details["total_len"] = len(full)
        details["entities"] = d_count // 2
        return full, details
