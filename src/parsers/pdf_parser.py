# ============================================================================
# HybridRAG -- PDF Parser (src/parsers/pdf_parser.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Extracts readable text from PDF files so the indexer can chunk and
#   embed it for search. PDFs are the most complex file format HybridRAG
#   handles because they come in many varieties:
#
#   1. DIGITAL PDFs -- Created from Word/PowerPoint. Text is stored as
#      characters. Easy to extract. Most PDFs are this type.
#
#   2. SCANNED PDFs -- Someone scanned a paper document. The PDF contains
#      images of pages, not actual text. Requires OCR (Optical Character
#      Recognition) to "read" the images and convert them to text.
#
#   3. MIXED PDFs -- Some pages are digital, some are scanned images.
#
#   4. ENCRYPTED PDFs -- Password-protected. May or may not allow text
#      extraction depending on the encryption settings.
#
# HOW IT WORKS (the extraction pipeline):
#   Step 1: Try pypdf (fast, handles most digital PDFs)
#   Step 2: If pypdf fails or gets no text -> try pdfplumber (more robust)
#   Step 3: If both fail or get very little text -> trigger OCR fallback
#   Step 4: OCR converts page images to text using Tesseract
#   Step 5: Return whatever text we got + detailed diagnostic info
#
# WHY TWO NORMAL EXTRACTORS?
#   pypdf is faster but sometimes fails on complex layouts.
#   pdfplumber is slower but handles tables and unusual formatting better.
#   Using both gives us the best chance of extracting text without OCR.
#
# DIAGNOSTICS:
#   Every parse call returns a "details" dictionary that records exactly
#   what happened at each step. This is critical for debugging when a PDF
#   fails to index -- you can see whether pypdf failed, whether OCR was
#   triggered, what errors occurred, etc.
#
# DEPENDENCIES:
#   - pypdf: Fast PDF text extraction (required)
#   - pdfplumber: Backup PDF text extraction (optional but recommended)
#   - pdf_ocr_fallback.py: OCR pipeline using Tesseract + pdf2image
# ============================================================================

import os
from typing import Tuple, Dict, Any

# Import OCR fallback utilities from our companion module
from .pdf_ocr_fallback import (
    _get_int_env,       # Read integer from environment variable (with default)
    _get_str_env,       # Read string from environment variable (with default)
    ocr_deps_available, # Check if Tesseract + pdf2image are installed
    ocr_pdf_pages,      # Actually perform OCR on PDF pages
)


class PDFParser:
    """
    Production-safe PDF parsing with forensic diagnostics + OCR fallback.

    This parser is designed to NEVER crash the indexing pipeline.
    Even if a PDF is corrupt, encrypted, or completely unreadable,
    the parser returns gracefully with diagnostic information about
    what went wrong.

    Usage:
        parser = PDFParser()

        # Simple: just get the text
        text = parser.parse("manual.pdf")

        # Detailed: get text + full diagnostics
        text, details = parser.parse_with_details("manual.pdf")
        # details tells you exactly what happened during extraction
    """

    def __init__(self) -> None:
        # No initialization needed -- the parser is stateless.
        # Each call to parse() is independent.
        pass

    def parse(self, file_path: str) -> str:
        """
        Simple interface: extract text from a PDF file.

        This is what the indexer calls. If you just need the text
        and don't care about diagnostics, use this method.

        Args:
            file_path: Full path to the PDF file

        Returns:
            Extracted text as a string (empty string if extraction failed)
        """
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Full interface: extract text AND return detailed diagnostics.

        The diagnostics dictionary records every step of the extraction
        process. This is invaluable for debugging PDFs that fail to index.

        Args:
            file_path: Full path to the PDF file

        Returns:
            Tuple of (extracted_text, details_dictionary)
            - extracted_text: The text content (may be empty)
            - details: Dictionary with extraction diagnostics including:
                - file_size_bytes: Size of the PDF
                - pdf_page_count: Number of pages
                - pdf_encrypted: Whether the PDF is password-protected
                - normal_extract: What happened with pypdf/pdfplumber
                - ocr_fallback: What happened with OCR (if triggered)
                - likely_reason: Best guess at why extraction failed (if it did)
        """
        # Initialize the diagnostics dictionary
        # Every field starts with a default "nothing happened yet" value
        details: Dict[str, Any] = {
            "parser": "PDFParser",
            "file_path": file_path,
            "file_size_bytes": None,
            "pdf_page_count": None,
            "pdf_encrypted": None,
            "normal_extract": {
                "attempted": False,
                "method": None,        # "pypdf" or "pdfplumber"
                "chars": 0,            # How many characters were extracted
                "status": None,        # "OK", "NO_TEXT", or "ERROR:..."
                "errors": [],          # List of any errors that occurred
            },
            "ocr_fallback": {
                "triggered": False,    # Did we decide OCR was needed?
                "used": False,         # Did we actually run OCR?
                "status": None,        # Result status
                "dependency_check": {},  # Are Tesseract + pdf2image installed?
                "settings": {},        # OCR settings used (DPI, max pages, etc.)
                "result": None,        # "OCR_TEXT_PRODUCED" or "OCR_PRODUCED_NO_TEXT"
                "stats": {},           # Per-page OCR timing and character counts
            },
            "likely_reason": None,     # Best guess at why extraction failed
        }

        # Get the file size (useful for diagnostics -- large files may be slow)
        try:
            details["file_size_bytes"] = os.path.getsize(file_path)
        except Exception:
            details["file_size_bytes"] = None

        text = ""
        normal_errors = []

        # ================================================================
        # STEP 1: Try pypdf (fast, handles most digital PDFs)
        # ================================================================
        # pypdf reads the PDF's internal text objects directly.
        # This is fast because it doesn't render the pages -- it just
        # reads the text data that's already stored in the PDF file.
        # ================================================================
        details["normal_extract"]["attempted"] = True
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)

            # Check if the PDF is encrypted (password-protected)
            details["pdf_encrypted"] = bool(getattr(reader, "is_encrypted", False))

            if details["pdf_encrypted"]:
                # Try to decrypt with empty password (many PDFs use this)
                try:
                    reader.decrypt("")
                except Exception as e:
                    normal_errors.append(f"pypdf_decrypt_error:{type(e).__name__}")

            # Record the page count
            details["pdf_page_count"] = len(getattr(reader, "pages", []))

            # Extract text from each page individually
            # We do this page-by-page so one bad page doesn't kill the whole PDF
            extracted_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text:
                        extracted_parts.append(page_text)
                except Exception as e:
                    # Log which page failed but keep going with the rest
                    normal_errors.append(f"pypdf_page_{i+1}_error:{type(e).__name__}")

            # Join all pages with double-newline separators
            text = "\n\n".join(extracted_parts).strip()
            details["normal_extract"]["method"] = "pypdf"
            details["normal_extract"]["chars"] = len(text)
            details["normal_extract"]["status"] = "OK" if text else "NO_TEXT"

        except Exception as e:
            details["normal_extract"]["method"] = "pypdf"
            details["normal_extract"]["status"] = f"ERROR:{type(e).__name__}"
            normal_errors.append(f"pypdf_error:{type(e).__name__}")

        # ================================================================
        # STEP 2: If pypdf got nothing, try pdfplumber (more robust)
        # ================================================================
        # pdfplumber uses a different approach -- it analyzes the visual
        # layout of each page to find text. This is slower but works
        # better on PDFs with complex tables, columns, or unusual fonts.
        # ================================================================
        if not text:
            try:
                import pdfplumber

                extracted_parts = []
                with pdfplumber.open(file_path) as pdf:
                    # Update page count if pypdf didn't get it
                    if details["pdf_page_count"] is None:
                        details["pdf_page_count"] = len(pdf.pages)

                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text = page.extract_text() or ""
                            if page_text:
                                extracted_parts.append(page_text)
                        except Exception as e:
                            normal_errors.append(f"pdfplumber_page_{i+1}_error:{type(e).__name__}")

                text = "\n\n".join(extracted_parts).strip()
                details["normal_extract"]["method"] = "pdfplumber"
                details["normal_extract"]["chars"] = len(text)
                details["normal_extract"]["status"] = "OK" if text else "NO_TEXT"

            except Exception as e:
                normal_errors.append(f"pdfplumber_error:{type(e).__name__}")

        # Save any errors that occurred during normal extraction
        if normal_errors:
            details["normal_extract"]["errors"] = normal_errors

        # ================================================================
        # STEP 3: Decide whether to trigger OCR
        # ================================================================
        # If normal extraction produced fewer characters than the minimum
        # threshold (default: 20 chars), the PDF is probably scanned/image-based.
        # In that case, we need OCR to "read" the page images.
        #
        # The threshold is configurable via environment variable so you can
        # adjust it without changing code.
        # ================================================================
        trigger_min_chars = _get_int_env("HYBRIDRAG_OCR_TRIGGER_MIN_CHARS", 20)
        should_ocr = (len((text or "").strip()) < trigger_min_chars)

        if should_ocr:
            # ============================================================
            # STEP 4: Run OCR fallback
            # ============================================================
            # OCR (Optical Character Recognition) converts page images to text.
            # It requires two external tools:
            #   - Tesseract: The OCR engine (reads images -> text)
            #   - pdf2image: Converts PDF pages to images for Tesseract
            #
            # OCR is SLOW (several seconds per page) and imperfect, so we
            # only use it when normal extraction fails.
            # ============================================================
            details["ocr_fallback"]["triggered"] = True
            details["normal_extract"]["status"] = details["normal_extract"]["status"] or "NO_TEXT"

            # Check if OCR tools are installed
            ok, dep_details = ocr_deps_available()
            details["ocr_fallback"]["dependency_check"] = dep_details

            # Read OCR settings from environment variables (with sensible defaults)
            ocr_max_pages = _get_int_env("HYBRIDRAG_OCR_MAX_PAGES", 200)   # Don't OCR more than this
            ocr_dpi = _get_int_env("HYBRIDRAG_OCR_DPI", 200)               # Image quality (higher = slower)
            ocr_timeout_s = _get_int_env("HYBRIDRAG_OCR_TIMEOUT_S", 20)    # Timeout per page
            ocr_lang = _get_str_env("HYBRIDRAG_OCR_LANG", "eng")           # Language for OCR

            # Don't try to OCR more pages than the PDF actually has
            page_count = details.get("pdf_page_count")
            if isinstance(page_count, int) and page_count > 0:
                effective_max_pages = min(ocr_max_pages, page_count)
            else:
                effective_max_pages = ocr_max_pages

            # Record the settings used (for diagnostics)
            details["ocr_fallback"]["settings"] = {
                "trigger_min_chars": trigger_min_chars,
                "max_pages": effective_max_pages,
                "dpi": ocr_dpi,
                "timeout_s": ocr_timeout_s,
                "lang": ocr_lang,
            }

            # If OCR tools aren't installed, we can't proceed
            if not ok:
                details["ocr_fallback"]["used"] = False
                details["ocr_fallback"]["status"] = "OCR_DEPS_MISSING"
                details["likely_reason"] = "LIKELY_SCANNED_OR_IMAGE_ONLY_OR_UNUSUAL_ENCODING"
                return "", details

            # Actually run OCR
            try:
                ocr_text, ocr_stats = ocr_pdf_pages(
                    file_path,
                    max_pages=effective_max_pages,
                    dpi=ocr_dpi,
                    timeout_s=ocr_timeout_s,
                    lang=ocr_lang,
                )

                details["ocr_fallback"]["used"] = True
                details["ocr_fallback"]["status"] = "OCR_ATTEMPTED"
                details["ocr_fallback"]["stats"] = ocr_stats

                # Check if OCR actually produced usable text
                if (ocr_text or "").strip():
                    details["ocr_fallback"]["result"] = "OCR_TEXT_PRODUCED"
                    details["likely_reason"] = "SCANNED_OR_IMAGE_ONLY_PDF_OCR_RECOVERED_TEXT"
                    return ocr_text, details

                # OCR ran but produced no text (blank pages, bad scan quality, etc.)
                details["ocr_fallback"]["result"] = "OCR_PRODUCED_NO_TEXT"
                details["likely_reason"] = "LIKELY_SCANNED_OR_IMAGE_ONLY_OR_UNUSUAL_ENCODING"
                return "", details

            except Exception as e:
                # OCR crashed -- return empty text with diagnostic info
                details["ocr_fallback"]["used"] = False
                details["ocr_fallback"]["status"] = f"OCR_ERROR:{type(e).__name__}"
                details["likely_reason"] = "LIKELY_SCANNED_OR_IMAGE_ONLY_OR_UNUSUAL_ENCODING"
                return "", details

        # ================================================================
        # Normal extraction produced enough text -- no OCR needed
        # ================================================================
        details["ocr_fallback"]["triggered"] = False
        details["ocr_fallback"]["used"] = False
        details["ocr_fallback"]["status"] = "NOT_NEEDED"
        return (text or ""), details
