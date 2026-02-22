# ============================================================================
# HybridRAG -- Outlook MSG Parser (src/parsers/msg_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Microsoft Outlook message (.msg) files and extracts the email
#   content: sender, recipients, subject, date, and body text.
#
#   .msg files are what you get when you drag an email out of Outlook
#   and save it to disk. They use the OLE2 binary format (same container
#   as legacy .doc and .xls files).
#
# WHY THIS MATTERS:
#   Many organizations save important correspondence as .msg files on
#   shared drives. Extracting the text lets HybridRAG search through
#   saved emails alongside other documents.
#
# DEPENDENCIES:
#   pip install python-oxmsg  (MIT license, by the python-docx author)
#   Fallback: pip install olefile  (BSD-2 license)
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


class MsgParser:
    """
    Extract email content from Outlook .msg files.

    NON-PROGRAMMER NOTE:
      We try python-oxmsg first (MIT license, best quality), then fall
      back to olefile for basic metadata extraction. The extracted text
      includes sender, recipients, subject, date, and body -- formatted
      like you'd see in an email client.
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "MsgParser"}

        # Strategy 1: python-oxmsg (MIT, best quality)
        text = _try_oxmsg(str(path), details)
        if text:
            details["method"] = "python-oxmsg"
            details["total_len"] = len(text)
            return text, details

        # Strategy 2: olefile metadata extraction (fallback)
        text = _try_olefile_msg(str(path), details)
        details["method"] = "olefile"
        details["total_len"] = len(text)
        return text, details


def _try_oxmsg(file_path: str, details: Dict) -> str:
    """Extract email using python-oxmsg (MIT license)."""
    try:
        from oxmsg import Message
    except ImportError:
        details["oxmsg_available"] = False
        return ""

    details["oxmsg_available"] = True
    try:
        msg = Message.load(file_path)
        parts = []

        if hasattr(msg, "sender") and msg.sender:
            parts.append(f"From: {msg.sender}")
        if hasattr(msg, "to") and msg.to:
            parts.append(f"To: {msg.to}")
        if hasattr(msg, "cc") and msg.cc:
            parts.append(f"CC: {msg.cc}")
        if hasattr(msg, "subject") and msg.subject:
            parts.append(f"Subject: {msg.subject}")
        if hasattr(msg, "date") and msg.date:
            parts.append(f"Date: {msg.date}")

        if parts:
            parts.append("")  # Blank line between headers and body

        body = ""
        if hasattr(msg, "body") and msg.body:
            body = msg.body
        elif hasattr(msg, "html_body") and msg.html_body:
            # Strip HTML tags for plain text
            import re
            body = re.sub(r"<[^>]+>", " ", msg.html_body)
            body = re.sub(r"\s+", " ", body)

        if body:
            parts.append(body.strip())

        return "\n".join(parts).strip()
    except Exception as e:
        details["oxmsg_error"] = str(e)
        return ""


def _try_olefile_msg(file_path: str, details: Dict) -> str:
    """Fallback: extract basic fields from MSG via OLE2 structure."""
    try:
        import olefile
    except ImportError:
        details["olefile_available"] = False
        details["error"] = (
            "Neither python-oxmsg nor olefile installed. "
            "Install with: pip install python-oxmsg"
        )
        return ""

    details["olefile_available"] = True
    parts = []
    try:
        ole = olefile.OleFileIO(file_path)

        # Common MSG OLE stream names for email fields
        field_map = {
            "__substg1.0_0037001F": "Subject",
            "__substg1.0_0042001F": "From",
            "__substg1.0_0E04001F": "To",
            "__substg1.0_1000001F": "Body",
        }

        for stream, label in field_map.items():
            if ole.exists(stream):
                try:
                    data = ole.openstream(stream).read()
                    text = data.decode("utf-16-le", errors="ignore").strip()
                    if text:
                        if label == "Body":
                            parts.append("")
                            parts.append(text)
                        else:
                            parts.append(f"{label}: {text}")
                except Exception:
                    pass

        ole.close()
    except Exception as e:
        details["olefile_error"] = str(e)

    return "\n".join(parts).strip()
