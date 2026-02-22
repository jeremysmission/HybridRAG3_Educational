# ============================================================================
# HybridRAG -- Windows Event Log Parser (src/parsers/evtx_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Windows Event Log files (.evtx) and extracts event records
#   as searchable text. Each event has: timestamp, source, event ID,
#   level (info/warning/error), and a message.
#
# WHY THIS MATTERS:
#   System administrators and cybersecurity analysts work with event logs
#   daily. Being able to search "show me all authentication failures in
#   the last month" or "what errors did the application generate?" across
#   collected .evtx files is extremely valuable.
#
# DEPENDENCIES:
#   pip install python-evtx  (Apache 2.0 license, pure Python)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class EvtxParser:
    """
    Extract event records from Windows .evtx log files.

    NON-PROGRAMMER NOTE:
      .evtx files are binary files created by Windows Event Viewer.
      Each file contains thousands of event records. We extract the
      first ~500 events to keep the output manageable, with key fields
      formatted as readable text.
    """

    MAX_EVENTS = 500  # Cap to prevent enormous output

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "EvtxParser"}

        try:
            import Evtx.Evtx as evtx
            import Evtx.Views as evtx_views
        except ImportError as e:
            details["error"] = (
                f"IMPORT_ERROR: {e}. Install with: pip install python-evtx"
            )
            return "", details

        parts: List[str] = [f"Windows Event Log: {path.name}"]
        event_count = 0

        try:
            with evtx.Evtx(str(path)) as log:
                for record in log.records():
                    if event_count >= self.MAX_EVENTS:
                        parts.append(
                            f"\n... truncated at {self.MAX_EVENTS} events"
                        )
                        break
                    event_count += 1
                    try:
                        xml = record.xml()
                        # Extract key fields from XML
                        text = _extract_event_text(xml)
                        if text:
                            parts.append(text)
                    except Exception:
                        pass  # Skip malformed records
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: {e}"

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["events"] = event_count
        return full, details


def _extract_event_text(xml_str: str) -> str:
    """Extract key fields from an event XML record as plain text."""
    import re
    fields = {}

    for tag in ["TimeCreated", "EventID", "Level", "Provider",
                 "Channel", "Computer"]:
        m = re.search(
            rf"<{tag}[^>]*?(?:SystemTime=['\"]([^'\"]+)['\"]|>([^<]+)<)",
            xml_str,
        )
        if m:
            fields[tag] = m.group(1) or m.group(2)

    # Provider has Name attribute
    m = re.search(r"<Provider\s+Name=['\"]([^'\"]+)['\"]", xml_str)
    if m:
        fields["Provider"] = m.group(1)

    # Event data fields
    data_parts = re.findall(r"<Data[^>]*>([^<]+)</Data>", xml_str)

    line = ""
    ts = fields.get("TimeCreated", "")
    eid = fields.get("EventID", "")
    prov = fields.get("Provider", "")
    if ts or eid:
        line = f"[{ts}] EventID={eid} Provider={prov}"
        if data_parts:
            line += " | " + " ".join(data_parts[:5])

    return line.strip()
