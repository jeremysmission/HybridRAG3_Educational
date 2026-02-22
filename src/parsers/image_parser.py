# ============================================================================
# HybridRAG - Image OCR Parser (src/parsers/image_parser.py)
# ============================================================================
# Purpose:
# - Parse image files by OCR (extract readable text)
#
# Supported formats we will register:
# - .png, .jpg, .jpeg, .tif, .tiff, .bmp, .gif, .webp
#
# Dependencies:
# - pillow (PIL)  -> pip install pillow
# - pytesseract   -> pip install pytesseract
# - Tesseract OCR application must be installed on Windows
#   (the EXE is not installed by pip).
#
# Behavior:
# - If OCR is unavailable, we return "" text but include a clear reason
#   in details so the indexer can log it.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def _try_import_pil():
    try:
        from PIL import Image  # type: ignore
        return True, Image, None
    except Exception as e:
        return False, None, f"IMPORT_ERROR: {type(e).__name__}: {e}"


def _try_import_pytesseract():
    try:
        import pytesseract  # type: ignore
        return True, pytesseract, None
    except Exception as e:
        return False, None, f"IMPORT_ERROR: {type(e).__name__}: {e}"


class ImageOCRParser:
    """
    OCR parser for images using Tesseract (via pytesseract).
    """

    def parse(self, file_path: str) -> str:
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        path = Path(file_path)
        details: Dict[str, Any] = {
            "file": str(path),
            "parser": "ImageOCRParser",
        }

        ok_pil, Image, pil_err = _try_import_pil()
        ok_ts, pytesseract, ts_err = _try_import_pytesseract()

        details["pillow_installed"] = bool(ok_pil)
        details["pytesseract_installed"] = bool(ok_ts)

        if not ok_pil:
            details["winner"] = "none"
            details["likely_reason"] = "PILLOW_NOT_INSTALLED"
            details["error"] = pil_err
            return "", details

        if not ok_ts:
            details["winner"] = "none"
            details["likely_reason"] = "PYTESSERACT_NOT_INSTALLED"
            details["error"] = ts_err
            return "", details

        # Optional: allow user to configure tesseract.exe path via env var
        # Example:
        #   $env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
        tcmd = (str(Path(pytesseract.pytesseract.tesseract_cmd)) if hasattr(pytesseract, "pytesseract") else "")
        env_cmd = (Path(str(Path.cwd())) / "tesseract.exe")  # not used, just a placeholder

        tesseract_cmd_env = (str(Path(Path.cwd())) if False else None)  # placeholder to keep comments clear
        tesseract_cmd_env = None  # actual env read below

        tesseract_cmd_env = __import__("os").getenv("TESSERACT_CMD")

        if tesseract_cmd_env:
            try:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_env
                details["tesseract_cmd_source"] = "env:TESSERACT_CMD"
                details["tesseract_cmd"] = tesseract_cmd_env
            except Exception:
                # If that fails, we keep default and let OCR attempt reveal error
                details["tesseract_cmd_source"] = "env:TESSERACT_CMD (failed_to_set)"
                details["tesseract_cmd"] = tesseract_cmd_env
        else:
            details["tesseract_cmd_source"] = "pytesseract_default"
            details["tesseract_cmd"] = tcmd

        try:
            img = Image.open(path)

            # Convert to RGB to avoid some palette/alpha issues
            img = img.convert("RGB")

            # OCR language can be set later (e.g. eng)
            # For technical/engineering docs, "eng" is fine initially.
            text = pytesseract.image_to_string(img)

            text = (text or "").strip()
            details["total_len"] = len(text)
            details["winner"] = "tesseract"

            if not text:
                details["likely_reason"] = "OCR_RETURNED_EMPTY_TEXT"
            return text, details

        except Exception as e:
            details["winner"] = "none"
            details["likely_reason"] = "OCR_UNAVAILABLE_OR_FAILED"
            details["error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
            return "", details
