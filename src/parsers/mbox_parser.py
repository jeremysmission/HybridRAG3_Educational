# ============================================================================
# HybridRAG -- Mbox Email Archive Parser (src/parsers/mbox_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES (plain English):
#   Reads Unix mbox email archive files (.mbox) and extracts email headers
#   and body text from all messages in the archive.
#
#   An mbox file is a single file containing multiple email messages
#   concatenated together. It is a standard format used by Thunderbird,
#   Gmail exports, and many email archiving tools.
#
# DEPENDENCIES:
#   None -- uses Python's built-in mailbox and email modules.
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import email
import mailbox
from pathlib import Path
from typing import Any, Dict, List, Tuple


class MboxParser:
    """
    Extract emails from mbox archive files.

    NON-PROGRAMMER NOTE:
      An mbox file is like a filing cabinet of emails stored in one file.
      We open the cabinet, pull out each email, and extract the headers
      (From, To, Subject, Date) and body text. We cap at 200 messages
      to prevent enormous output from massive archives.
    """

    MAX_MESSAGES = 200

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {"file": str(path), "parser": "MboxParser"}

        try:
            mbox = mailbox.mbox(str(path))
        except Exception as e:
            details["error"] = f"RUNTIME_ERROR: Cannot read mbox: {e}"
            return "", details

        parts: List[str] = [f"Email Archive: {path.name}"]
        msg_count = 0

        try:
            for message in mbox:
                if msg_count >= self.MAX_MESSAGES:
                    parts.append(
                        f"\n... truncated at {self.MAX_MESSAGES} messages"
                    )
                    break
                msg_count += 1
                parts.append(f"\n--- Message {msg_count} ---")

                for header in ["From", "To", "Subject", "Date"]:
                    val = message.get(header, "")
                    if val:
                        parts.append(f"{header}: {val}")

                # Extract body text
                body = _get_email_body(message)
                if body:
                    parts.append(body[:5000])  # Cap body length
        except Exception as e:
            details["error"] = f"PARSE_ERROR: {e}"

        full = "\n".join(parts).strip()
        details["total_len"] = len(full)
        details["messages"] = msg_count
        return full, details


def _get_email_body(msg) -> str:
    """Extract plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="ignore")
    return ""
