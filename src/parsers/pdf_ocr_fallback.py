# ============================================================================
# HybridRAG -- PDF OCR Fallback (src/parsers/pdf_ocr_fallback.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   When a PDF is scanned (contains images instead of text), normal text
#   extraction fails. This module provides OCR (Optical Character Recognition)
#   to "read" the page images and convert them to searchable text.
#
# HOW OCR WORKS (plain English):
#   1. pdf2image converts each PDF page into a picture (PNG image)
#   2. Tesseract (an OCR engine) "reads" each picture and outputs text
#   3. We collect the text from all pages and return it
#
# WHY IT'S A SEPARATE FILE:
#   OCR requires two external tools that may not be installed:
#   - Tesseract OCR engine (the "reader")
#   - Poppler (converts PDF pages to images)
#   By keeping OCR in a separate file, the rest of HybridRAG works fine
#   even if these tools are missing. PDFs that don't need OCR still index.
#
# SAFETY FEATURES:
#   - Per-page timeout: If one page takes too long, we skip it and move on
#   - Max pages limit: Don't try to OCR a 10,000-page document
#   - Graceful failure: If OCR tools are missing, returns empty text + reason
#   - Thread isolation: OCR runs in a separate thread so timeouts actually work
#
# CONFIGURATION (via environment variables):
#   HYBRIDRAG_OCR_TRIGGER_MIN_CHARS = 20    (OCR if normal extraction < this)
#   HYBRIDRAG_OCR_MAX_PAGES = 200           (don't OCR more than this)
#   HYBRIDRAG_OCR_DPI = 200                 (image quality -- higher = slower)
#   HYBRIDRAG_OCR_TIMEOUT_S = 20            (seconds per page before giving up)
#   HYBRIDRAG_OCR_LANG = "eng"              (language for OCR recognition)
#   HYBRIDRAG_TESSERACT_CMD = ""            (path to tesseract.exe if not in PATH)
#   HYBRIDRAG_POPPLER_BIN = ""              (path to poppler bin/ folder)
#
# INTERNET ACCESS: None -- purely local processing
# ============================================================================

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Tuple, Dict, Any


# ============================================================================
# Helper: Read environment variables safely
# ============================================================================
# These let you configure OCR behavior without changing code.
# If the environment variable isn't set, the default value is used.

def _get_int_env(name: str, default: int) -> int:
    """
    Read an integer from an environment variable, with a fallback default.

    Example:
        _get_int_env("HYBRIDRAG_OCR_DPI", 200)
        -> Returns 200 if the env var isn't set
        -> Returns the env var's value as an integer if it is set
    """
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _get_str_env(name: str, default: str) -> str:
    """
    Read a string from an environment variable, with a fallback default.
    """
    val = os.getenv(name)
    return default if not val else str(val).strip()


# ============================================================================
# Dependency Check: Are OCR tools installed?
# ============================================================================

def ocr_deps_available() -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the required OCR tools (Tesseract + pdf2image) are installed.

    This is called BEFORE attempting OCR so we can fail gracefully with
    a clear error message instead of crashing mid-process.

    Returns:
        Tuple of (is_available, details_dict)
        - is_available: True if OCR can be used, False otherwise
        - details_dict: What was found/missing (for diagnostics)
    """
    details: Dict[str, Any] = {}
    try:
        # Try importing both required libraries
        import pytesseract  # noqa -- The Python wrapper for Tesseract OCR
        from pdf2image import convert_from_path  # noqa -- Converts PDF pages to images
        details["pytesseract_import"] = True
        details["pdf2image_import"] = True
    except Exception as e:
        # If either import fails, OCR is not available
        details["import_error"] = repr(e)
        return False, details

    # Check if custom paths are configured for the OCR tools
    # (needed when tools aren't in the system PATH)
    tess_cmd = _get_str_env("HYBRIDRAG_TESSERACT_CMD", "")
    poppler_bin = _get_str_env("HYBRIDRAG_POPPLER_BIN", "")

    details["tesseract_cmd_set"] = bool(tess_cmd)
    details["poppler_bin_set"] = bool(poppler_bin)
    details["tesseract_cmd"] = tess_cmd if tess_cmd else None
    details["poppler_bin"] = poppler_bin if poppler_bin else None

    return True, details


# ============================================================================
# Single Page OCR: Convert one page image to text
# ============================================================================

def _ocr_page_image_to_text(pil_image, lang: str) -> str:
    """
    Run Tesseract OCR on a single page image and return the extracted text.

    This function runs inside a thread with a timeout, so if Tesseract
    hangs on a complex page, the main process can kill it and move on.

    Args:
        pil_image: A PIL/Pillow Image object (one page of the PDF)
        lang: Language code for OCR (e.g., "eng" for English)

    Returns:
        Extracted text from the page (may be empty if OCR found nothing)
    """
    import pytesseract

    # If a custom Tesseract path was configured, use it
    tess_cmd = _get_str_env("HYBRIDRAG_TESSERACT_CMD", "")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    # Run OCR and return the text
    return pytesseract.image_to_string(pil_image, lang=lang) or ""


# ============================================================================
# Main OCR Pipeline: Process multiple PDF pages
# ============================================================================

def ocr_pdf_pages(
    pdf_path: str,
    *,
    max_pages: int,
    dpi: int,
    timeout_s: int,
    lang: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    OCR a PDF file page by page and return the combined text.

    How the pipeline works:
        For each page (up to max_pages):
            1. Convert the PDF page to a PNG image using pdf2image + Poppler
            2. Send the image to Tesseract OCR in a separate thread
            3. Wait up to timeout_s seconds for the result
            4. If the page times out, skip it and continue with the next page
            5. Collect all extracted text with page markers

    The output text includes page markers like [OCR_PAGE=1] so you can
    tell which text came from which page.

    Args:
        pdf_path:   Path to the PDF file
        max_pages:  Maximum number of pages to OCR
        dpi:        Image quality (200 = good balance of speed/accuracy)
        timeout_s:  Seconds to wait per page before giving up
        lang:       OCR language code (e.g., "eng" for English)

    Returns:
        Tuple of (combined_text, stats_dictionary)
        - combined_text: All extracted text with [OCR_PAGE=N] markers
        - stats: Page-by-page statistics (timing, success/failure counts)
    """
    from pdf2image import convert_from_path

    # Get the custom Poppler path if configured
    # Poppler is required by pdf2image to render PDF pages as images
    poppler_bin = _get_str_env("HYBRIDRAG_POPPLER_BIN", "") or None

    # Initialize statistics for diagnostics
    details: Dict[str, Any] = {
        "enabled": True,
        "lang": lang,
        "dpi": dpi,
        "max_pages": max_pages,
        "timeout_s": timeout_s,
        "pages_attempted": 0,      # How many pages we tried
        "pages_successful": 0,     # How many returned text
        "pages_timed_out": 0,      # How many exceeded the timeout
        "pages_failed": 0,         # How many crashed
        "total_chars": 0,          # Total characters extracted
        "runtime_s": None,         # Total wall-clock time
    }

    parts = []        # Collected text from each successful page
    t0 = time.time()  # Start the overall timer

    # Process pages one at a time (not all at once) to control memory
    for page_num in range(1, max_pages + 1):
        details["pages_attempted"] += 1
        try:
            # STEP 1: Convert this single PDF page to a PNG image
            # We do one page at a time to avoid loading a huge PDF into RAM
            images = convert_from_path(
                pdf_path,
                dpi=dpi,                    # Image resolution (pixels per inch)
                first_page=page_num,        # Only convert this page
                last_page=page_num,
                poppler_path=poppler_bin,   # Path to Poppler tools
                fmt="png",                  # Output format
                thread_count=1,             # Single thread for stability
            )
            # If no images returned, we've gone past the end of the PDF
            if not images:
                break

            img = images[0]  # The rendered page image

            # STEP 2: Run OCR in a separate thread with a timeout
            # ThreadPoolExecutor lets us set a time limit on the OCR
            # If Tesseract takes longer than timeout_s, we give up on this page
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_ocr_page_image_to_text, img, lang)
                try:
                    page_text = fut.result(timeout=timeout_s)
                except FuturesTimeoutError:
                    # This page took too long -- skip it and continue
                    details["pages_timed_out"] += 1
                    continue

            # STEP 3: Collect the extracted text
            page_text = (page_text or "").strip()
            if page_text:
                details["pages_successful"] += 1
                details["total_chars"] += len(page_text)
                # Add a page marker so we know where this text came from
                parts.append(f"\n\n[OCR_PAGE={page_num}]\n{page_text}")

        except Exception:
            # This page crashed OCR -- skip it and continue
            details["pages_failed"] += 1
            continue

    # Record total time
    details["runtime_s"] = round(time.time() - t0, 3)

    # Combine all page texts into one string
    return "".join(parts).strip(), details
