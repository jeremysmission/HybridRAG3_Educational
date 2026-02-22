# ============================================================================
# HybridRAG -- Email Parser (src/parsers/eml_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Extracts readable text from .eml email files so they can be indexed
#   and searched by the RAG system. Emails contain valuable information
#   like decisions, approvals, specifications, and meeting notes.
#
# HOW IT WORKS:
#   1. Opens the .eml file and parses it using Python's built-in email library
#   2. Extracts headers: Subject, From, To, Cc, Date, Message-ID
#   3. Finds the email body -- prefers plain text, falls back to HTML
#   4. If the body is HTML, strips out all tags to get clean text
#   5. Combines headers + body into one text block for indexing
#   6. Returns diagnostics showing exactly what happened
#
# EMAIL ANATOMY (for nonprogrammers):
#   An .eml file is a text file that contains:
#   - Headers (metadata): who sent it, who received it, date, subject
#   - Body: the actual message content
#   - Attachments: files attached to the email (we skip these for now)
#
#   Emails can be "multipart" (have multiple sections) or "singlepart".
#   Multipart emails might have both a plain text version AND an HTML
#   version of the same message. We prefer the plain text version because
#   it's cleaner for indexing.
#
# DEPENDENCIES:
#   - Python's built-in `email` module (no extra installs needed)
#   - No external libraries required -- everything is stdlib
#
# INTERNET ACCESS: None -- purely local file processing
# ============================================================================

import os
import re        # Regular expressions for stripping HTML tags
import hashlib   # For generating a content fingerprint (audit trail)
from typing import Tuple, Dict, Any, Optional
from email import policy                # Email parsing settings
from email.parser import BytesParser    # Parses raw email bytes into objects


def _strip_html(html: str) -> str:
    """
    Convert HTML to plain text without any external dependencies.

    Why we need this:
        Some emails only have an HTML body (no plain text version).
        We need to strip out all the HTML tags to get readable text.

    How it works (step by step):
        1. Remove <script> and <style> blocks entirely (code/styling, not content)
        2. Convert <br> tags to newlines (line breaks)
        3. Convert </p> tags to double newlines (paragraph breaks)
        4. Remove all remaining HTML tags
        5. Clean up extra whitespace and blank lines

    Args:
        html: Raw HTML string from the email body

    Returns:
        Clean plain text with no HTML tags
    """
    if not html:
        return ""
    # Remove script and style blocks (they contain code, not readable text)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    # Convert <br> tags to actual line breaks
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    # Convert closing </p> tags to paragraph breaks
    text = re.sub(r"(?is)</p\s*>", "\n\n", text)
    # Strip all remaining HTML tags (anything between < and >)
    text = re.sub(r"(?is)<.*?>", " ", text)
    # Clean up trailing whitespace on each line
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse 3+ consecutive newlines into just 2 (paragraph spacing)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_decode(payload: Optional[bytes], charset: Optional[str]) -> str:
    """
    Safely decode email body bytes into a text string.

    Why this exists:
        Emails can use many different character encodings (UTF-8, ISO-8859-1,
        Windows-1252, etc.). If we guess wrong, we get garbled text or crashes.
        This function tries the declared encoding first, then falls back to
        UTF-8 with error replacement (replaces unreadable chars with ?).

    Args:
        payload: Raw bytes from the email body
        charset: The character encoding declared in the email headers

    Returns:
        Decoded text string (may contain replacement characters for bad bytes)
    """
    if not payload:
        return ""
    # Use the declared charset, defaulting to UTF-8 if none specified
    enc = (charset or "utf-8").strip() if charset else "utf-8"
    try:
        # errors="replace" means replace undecodable bytes with ? instead of crashing
        return payload.decode(enc, errors="replace")
    except Exception:
        # If even that fails, force UTF-8 with replacement
        return payload.decode("utf-8", errors="replace")


class EmlParser:
    """
    Production-safe .eml email parser.

    Design principles:
        - Uses only Python's built-in email library (no pip installs)
        - Never crashes the indexing pipeline -- always returns gracefully
        - Prefers plain text body, falls back to stripped HTML
        - Returns full diagnostics for debugging and audit trail

    Usage:
        parser = EmlParser()

        # Simple: just get the text
        text = parser.parse("message.eml")

        # Detailed: get text + diagnostics
        text, details = parser.parse_with_details("message.eml")
    """

    def parse(self, file_path: str) -> str:
        """
        Simple interface: extract text from an .eml file.
        Returns the combined headers + body as a single text string.
        """
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Full interface: extract text AND return detailed diagnostics.

        The diagnostics dictionary records:
            - Which headers were found
            - Whether the body was plain text or stripped HTML
            - How many attachments were present (but not extracted)
            - A SHA1 hash of the extracted content (for audit/deduplication)
            - Any errors that occurred

        Returns:
            Tuple of (extracted_text, details_dictionary)
        """
        # Initialize the diagnostics dictionary
        details: Dict[str, Any] = {
            "parser": "EmlParser",
            "file_path": file_path,
            "file_size_bytes": None,
            "status": None,            # "OK", "NO_BODY_TEXT", or "ERROR:..."
            "headers": {},             # Extracted email headers
            "body_choice": None,       # "text/plain", "text/html_stripped", or "none"
            "attachment_count": 0,     # Number of attachments found (not extracted)
            "errors": [],              # Any errors during parsing
        }

        # Record file size for diagnostics
        try:
            details["file_size_bytes"] = os.path.getsize(file_path)
        except Exception:
            details["file_size_bytes"] = None

        try:
            # ============================================================
            # STEP 1: Parse the .eml file using Python's email library
            # ============================================================
            # We open in binary mode ("rb") because email files can contain
            # multiple character encodings. BytesParser handles this for us.
            with open(file_path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)

            # ============================================================
            # STEP 2: Extract email headers
            # ============================================================
            # These headers are the "envelope" of the email -- who sent it,
            # who received it, when, and what the subject was.
            def _h(name: str) -> str:
                """Safely get a header value, returning empty string if missing."""
                v = msg.get(name)
                return str(v) if v is not None else ""

            headers = {
                "subject": _h("Subject"),
                "from": _h("From"),
                "to": _h("To"),
                "cc": _h("Cc"),
                "date": _h("Date"),
                "message_id": _h("Message-ID"),
            }
            details["headers"] = headers

            # ============================================================
            # STEP 3: Find the email body
            # ============================================================
            # Emails can be "multipart" (multiple sections) or simple.
            # Multipart emails may have:
            #   - text/plain (the readable text we want)
            #   - text/html (HTML-formatted version)
            #   - attachments (files -- we skip these)
            #
            # We walk through all parts looking for the body text.
            # Preference: text/plain > text/html (stripped)
            text_plain = ""
            text_html = ""
            attachment_count = 0

            if msg.is_multipart():
                # Walk through all parts of the email
                for part in msg.walk():
                    ctype = part.get_content_type()     # e.g., "text/plain"
                    disp = part.get_content_disposition()  # "attachment" or None

                    # Skip attachments (we count them but don't extract them)
                    if disp == "attachment":
                        attachment_count += 1
                        continue

                    # Look for text bodies
                    if ctype in ("text/plain", "text/html"):
                        # decode=True converts from base64/quoted-printable to raw bytes
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset()
                        decoded = _safe_decode(payload, charset).strip()
                        if not decoded:
                            continue
                        # Keep the first plain text body and first HTML body we find
                        if ctype == "text/plain" and not text_plain:
                            text_plain = decoded
                        elif ctype == "text/html" and not text_html:
                            text_html = decoded
            else:
                # Simple (non-multipart) email -- just one body
                ctype = msg.get_content_type()
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset()
                decoded = _safe_decode(payload, charset).strip()
                if ctype == "text/plain":
                    text_plain = decoded
                elif ctype == "text/html":
                    text_html = decoded

            details["attachment_count"] = attachment_count

            # ============================================================
            # STEP 4: Choose the best body text
            # ============================================================
            # Prefer plain text (cleaner for indexing), fall back to HTML stripped
            body = ""
            if text_plain:
                body = text_plain
                details["body_choice"] = "text/plain"
            elif text_html:
                # Strip HTML tags to get readable text
                body = _strip_html(text_html)
                details["body_choice"] = "text/html_stripped"
            else:
                details["body_choice"] = "none"
                details["status"] = "NO_BODY_TEXT"
                return "", details

            # ============================================================
            # STEP 5: Combine headers + body into final text
            # ============================================================
            # We include the headers because they contain searchable info:
            # - Subject line is often the best summary of the email
            # - From/To helps identify who said what
            # - Date helps with time-based queries
            header_block = "\n".join([
                f"Subject: {headers.get('subject','')}",
                f"From: {headers.get('from','')}",
                f"To: {headers.get('to','')}",
                f"Cc: {headers.get('cc','')}",
                f"Date: {headers.get('date','')}",
                f"Message-ID: {headers.get('message_id','')}",
            ]).strip()

            extracted = (header_block + "\n\n" + body).strip()

            # Generate a content fingerprint for audit/deduplication
            # SHA1 is a hash function that produces a unique "fingerprint" of the text
            try:
                details["content_sha1"] = hashlib.sha1(
                    extracted.encode("utf-8", errors="replace")
                ).hexdigest()
            except Exception:
                details["content_sha1"] = None

            details["status"] = "OK"
            return extracted, details

        except Exception as e:
            # If anything goes wrong, return empty text with the error recorded
            details["status"] = f"ERROR:{type(e).__name__}"
            details["errors"].append(repr(e))
            return "", details
