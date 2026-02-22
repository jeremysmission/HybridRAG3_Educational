# ============================================================================
# HybridRAG -- Placeholder Parser (src/parsers/placeholder_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   A "recognize and log" parser for file formats that HybridRAG knows
#   about but cannot yet fully parse. Instead of silently ignoring these
#   files, the placeholder parser:
#     1. Recognizes the file by extension
#     2. Returns a text summary saying what the file IS
#     3. Logs what would be needed to fully parse it
#     4. Extracts whatever metadata it can (file size, dates, etc.)
#
# WHY THIS MATTERS:
#   If someone searches "SolidWorks assembly", HybridRAG can still find
#   .sldasm files even though we can't extract the 3D geometry. The
#   filename, file type, and location metadata are still searchable.
#
# FORMATS COVERED (with notes on what's needed):
#
#   PROPRIETARY CAD (requires commercial software):
#     .prt / .sldprt  -- SolidWorks part (needs SolidWorks installed + COM API)
#     .asm / .sldasm  -- SolidWorks assembly (same as above)
#     .dwg            -- AutoCAD drawing (needs ODA File Converter or GPL LibreDWG)
#     .dwt            -- AutoCAD template (same as .dwg)
#
#   COMPLEX DEPENDENCIES:
#     .mpp            -- MS Project (needs Java + MPXJ library)
#     .vsd            -- Legacy Visio (binary OLE, no good Python parser)
#     .one            -- OneNote (semi-proprietary, limited extraction)
#     .ost            -- Outlook offline storage (needs C toolchain for libpff)
#
#   LICENSE CONCERNS:
#     .eps            -- Encapsulated PostScript (Ghostscript is AGPL)
#     .ai (pre-CS)    -- Older Illustrator files (pure PostScript, needs GS)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

# Maps extensions to (format_name, what_is_needed) for placeholder reporting.
# This is the "wish list" database for future parser development.
_PLACEHOLDER_INFO = {
    # --- SolidWorks (proprietary, requires SolidWorks installed) ---
    ".prt": (
        "SolidWorks Part",
        "Requires SolidWorks installed + pywin32 COM API (pySldWrap). "
        "Alternative: export to STEP/IGES from SolidWorks, then parse those.",
    ),
    ".sldprt": (
        "SolidWorks Part",
        "Same as .prt -- requires SolidWorks installed.",
    ),
    ".asm": (
        "SolidWorks Assembly",
        "Requires SolidWorks installed + pywin32 COM API. "
        "Alternative: export to STEP from SolidWorks.",
    ),
    ".sldasm": (
        "SolidWorks Assembly",
        "Same as .asm -- requires SolidWorks installed.",
    ),

    # --- AutoCAD DWG (proprietary binary format) ---
    ".dwg": (
        "AutoCAD Drawing",
        "DWG is proprietary. Options: (1) Install ODA File Converter (free) "
        "to convert DWG->DXF, then parse DXF with ezdxf. (2) LibreDWG is "
        "open-source but GPL-3.0 (viral license). No MIT/BSD Python library exists.",
    ),
    ".dwt": (
        "AutoCAD Drawing Template",
        "Same binary format as DWG. Same conversion approach needed.",
    ),

    # --- MS Project (requires Java runtime for MPXJ) ---
    ".mpp": (
        "Microsoft Project",
        "The MPXJ library (pip install mpxj) can read .mpp files, but it "
        "requires a Java Runtime Environment (JRE) installed because it "
        "wraps a Java library via JPype. License: LGPL.",
    ),

    # --- Legacy Visio (binary OLE format) ---
    ".vsd": (
        "Visio Diagram (Legacy)",
        "Legacy .vsd is a binary OLE format with no good open-source Python "
        "parser. The 'vsdx' library only handles .vsdx (Visio 2013+). "
        "Workaround: convert .vsd to .vsdx using Visio or LibreOffice.",
    ),

    # --- OneNote (semi-proprietary) ---
    ".one": (
        "OneNote Section",
        "pyOneNote (pip install pyOneNote) can extract some content, but "
        "extraction quality varies. The .one format is semi-proprietary. "
        "Best approach: export OneNote sections to PDF or HTML first.",
    ),

    # --- Outlook Offline Storage (needs C toolchain) ---
    ".ost": (
        "Outlook Offline Storage",
        "libpff-python can parse .ost files, but it requires a C compiler "
        "toolchain to install (compiles C code). Works better on Linux. "
        "Windows installation is difficult. Alternative: use Outlook to "
        "export individual messages as .msg or .eml files.",
    ),

    # --- PostScript (Ghostscript is AGPL) ---
    ".eps": (
        "Encapsulated PostScript",
        "Full EPS rendering requires Ghostscript (AGPL-3.0 license). "
        "Pillow can rasterize EPS if Ghostscript is installed, then OCR "
        "can extract text from the rasterized image. Without Ghostscript, "
        "we can only extract DSC (Document Structuring Convention) comments.",
    ),
}


class PlaceholderParser:
    """
    Recognize-and-log parser for formats that can't be fully parsed yet.

    NON-PROGRAMMER NOTE:
      This parser doesn't extract the full content of a file. Instead,
      it creates a brief "identity card" for the file:
        - What type of file it is
        - Its size, name, and location
        - What would be needed to fully parse it

      This means the file still shows up in search results (by name
      and type) even though we can't read its internal content.
    """

    def __init__(self, extension: str = "") -> None:
        self._ext = extension.lower()

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        ext = self._ext or path.suffix.lower()
        details: Dict[str, Any] = {
            "file": str(path),
            "parser": "PlaceholderParser",
            "extension": ext,
            "placeholder": True,
        }

        info = _PLACEHOLDER_INFO.get(ext, ("Unknown Format", "No parser available."))
        format_name, requirement = info

        # Even without a real parser, we can report file metadata
        parts = [
            f"File: {path.name}",
            f"Type: {format_name} ({ext})",
        ]

        try:
            st = os.stat(str(path))
            size_mb = st.st_size / (1024 * 1024)
            parts.append(f"Size: {size_mb:.1f} MB ({st.st_size:,} bytes)")

            from datetime import datetime, timezone
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            parts.append(f"Modified: {mtime.isoformat()}")
        except Exception:
            pass

        parts.append(f"Parser status: PLACEHOLDER (content not yet extractable)")
        parts.append(f"Requirement: {requirement}")

        full = "\n".join(parts)
        details["total_len"] = len(full)
        details["format_name"] = format_name
        details["requirement"] = requirement
        return full, details
