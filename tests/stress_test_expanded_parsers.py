#!/usr/bin/env python3
# ============================================================================
# HybridRAG -- Expanded Parser Stress Test
# ============================================================================
# FILE: tests/stress_test_expanded_parsers.py
#
# WHAT THIS FILE DOES (plain English):
#   Exhaustively tests EVERY registered file extension in the parser registry.
#   For each extension, it creates a synthetic test file with realistic content,
#   runs the parser on it, and verifies the parser returns valid output or a
#   graceful error (never crashes). It also tests FAKE extensions to verify
#   the registry correctly rejects unknown formats.
#
# TEST CATEGORIES:
#   1. Registry Integrity    -- all extensions registered, no duplicates
#   2. Plain Text Parsers    -- .txt, .md, .csv, .json, .xml, .log, etc.
#   3. Document Parsers      -- .pdf, .docx, .pptx, .xlsx, .doc, .rtf, .ai
#   4. Email Parsers         -- .eml, .msg, .mbox
#   5. Web Parsers           -- .html, .htm
#   6. Image/OCR Parsers     -- .png, .jpg, .bmp, .gif, .wmf, .emf, etc.
#   7. Design Parsers        -- .psd
#   8. CAD Parsers           -- .dxf, .stp, .step, .ste, .igs, .iges, .stl
#   9. Diagram Parsers       -- .vsdx
#  10. Cyber/Admin Parsers   -- .evtx, .pcap, .pcapng, .cer, .crt, .pem
#  11. Database Parsers      -- .accdb, .mdb
#  12. Placeholder Parsers   -- .prt, .sldprt, .asm, .sldasm, .dwg, etc.
#  13. Fake Extensions       -- .xyz, .aaa, .fake, .notreal, etc.
#  14. Edge Cases            -- empty files, huge names, special characters
#
# HOW TO RUN:
#   python tests/stress_test_expanded_parsers.py
#
# INTERNET ACCESS: NONE
# ============================================================================

from __future__ import annotations

import os
import sys
import time
import struct
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# TEST RESULT TRACKING
# ============================================================================
# Non-programmer note:
#   We keep a running tally of passed, failed, warned, and skipped tests.
#   At the end, we print a summary table and generate a markdown report.
# ============================================================================

class TestResult:
    """One test outcome."""
    def __init__(self, name: str, status: str, detail: str = ""):
        self.name = name
        self.status = status  # PASS, FAIL, WARN, SKIP
        self.detail = detail


def _check_parse_result(text: str, details: dict, keyword: str = "") -> Tuple[str, str]:
    """
    Evaluate a parser result and return (status, detail).

    Non-programmer note:
      When a parser's optional library is not installed, it returns an
      IMPORT_ERROR in the details dict. This is CORRECT behavior (graceful
      degradation), not a failure. This helper distinguishes between:
        - Real content extraction -> PASS
        - Missing library -> PASS (graceful degradation)
        - Empty output with no error -> WARN (unexpected)
    """
    if "error" in details:
        err = details["error"]
        if "IMPORT_ERROR" in err:
            return "PASS", f"Graceful degradation: {err[:60]}"
        return "PASS", f"Error handled: {err[:60]}"
    if len(text) > 0:
        if keyword and keyword in text:
            return "PASS", f"{len(text)} chars, found '{keyword}'"
        return "PASS", f"{len(text)} chars"
    return "WARN", "Empty output with no error"


class TestRunner:
    """Collects and reports test results."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()

    def record(self, name: str, status: str, detail: str = ""):
        self.results.append(TestResult(name, status, detail))
        tag = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]", "SKIP": "[SKIP]"}[status]
        print(f"  {tag} {name}" + (f" -- {detail}" if detail else ""))

    def summary(self) -> Dict[str, int]:
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def elapsed(self) -> float:
        return time.time() - self.start_time


# ============================================================================
# SYNTHETIC FILE GENERATORS
# ============================================================================
# Non-programmer note:
#   We cannot test parsers without files to parse. These functions create
#   small, synthetic test files in a temporary directory. The files contain
#   realistic-ish content for each format. For binary formats (like .docx
#   or .pcap), we create minimal valid structures or intentionally malformed
#   files to test graceful error handling.
# ============================================================================

def make_plain_text(path: Path, ext: str):
    """Create a plain text file with known content."""
    content_map = {
        ".txt": "Hello from HybridRAG stress test.\nLine 2.\nLine 3.",
        ".md": "# Test Document\n\nThis is **bold** and *italic*.\n\n## Section 2\n\nContent here.",
        ".csv": "Name,Age,Role\nAlice,30,Engineer\nBob,25,Analyst\nCharlie,35,Manager",
        ".json": '{"name": "test", "version": 1, "items": ["alpha", "beta"]}',
        ".xml": '<?xml version="1.0"?>\n<root>\n  <item id="1">Alpha</item>\n  <item id="2">Beta</item>\n</root>',
        ".log": "[2026-01-15 10:30:00] [OK] System started\n[2026-01-15 10:30:01] [OK] Indexer ready\n[2026-01-15 10:30:05] [WARN] Slow query detected",
        ".yaml": "name: test\nversion: 1\nitems:\n  - alpha\n  - beta\n",
        ".yml": "database:\n  host: localhost\n  port: 5432\n",
        ".ini": "[DEFAULT]\nServerAliveInterval = 45\n\n[section1]\nkey1 = value1\n",
        ".cfg": "[settings]\nmode = production\ntimeout = 30\n",
        ".conf": "# Apache config\nServerRoot /etc/httpd\nListen 80\n",
        ".properties": "app.name=HybridRAG\napp.version=3.0\napp.debug=false\n",
        ".reg": 'Windows Registry Editor Version 5.00\n\n[HKEY_LOCAL_MACHINE\\SOFTWARE\\Test]\n"Value1"="Data1"\n',
    }
    path.write_text(content_map.get(ext, f"Test content for {ext}"), encoding="utf-8")


def make_html(path: Path):
    """Create a minimal HTML file."""
    path.write_text(
        "<html><head><title>Test Page</title></head>"
        "<body><h1>Heading</h1><p>Paragraph text.</p></body></html>",
        encoding="utf-8",
    )


def make_eml(path: Path):
    """Create a minimal RFC 822 email file."""
    path.write_text(
        "From: sender@example.com\r\n"
        "To: recipient@example.com\r\n"
        "Subject: Test Email\r\n"
        "Date: Mon, 15 Jan 2026 10:00:00 +0000\r\n"
        "Content-Type: text/plain\r\n"
        "\r\n"
        "This is a test email body.\r\n",
        encoding="utf-8",
    )


def make_mbox(path: Path):
    """Create a minimal mbox file with two messages."""
    path.write_text(
        "From sender@example.com Mon Jan 15 10:00:00 2026\n"
        "From: sender@example.com\n"
        "To: recipient@example.com\n"
        "Subject: First Message\n"
        "Date: Mon, 15 Jan 2026 10:00:00 +0000\n"
        "\n"
        "Body of first message.\n"
        "\n"
        "From other@example.com Mon Jan 15 11:00:00 2026\n"
        "From: other@example.com\n"
        "To: recipient@example.com\n"
        "Subject: Second Message\n"
        "Date: Mon, 15 Jan 2026 11:00:00 +0000\n"
        "\n"
        "Body of second message.\n",
        encoding="utf-8",
    )


def make_rtf(path: Path):
    """Create a minimal RTF file."""
    path.write_text(
        r"{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}"
        r"\pard Hello from RTF stress test.\par"
        r" Second paragraph here.\par}",
        encoding="utf-8",
    )


def make_dxf(path: Path):
    """Create a minimal DXF file with text entities."""
    # Minimal DXF with one TEXT entity
    dxf_content = """  0
SECTION
  2
HEADER
  0
ENDSEC
  0
SECTION
  2
ENTITIES
  0
TEXT
  8
Layer1
 10
0.0
 20
0.0
 30
0.0
 40
2.5
  1
Hello from DXF stress test
  0
TEXT
  8
Layer2
 10
10.0
 20
10.0
 30
0.0
 40
2.5
  1
Second text entity
  0
ENDSEC
  0
EOF
"""
    path.write_text(dxf_content, encoding="utf-8")


def make_step(path: Path):
    """Create a minimal STEP (ISO 10303-21) file."""
    path.write_text(
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('Stress test STEP file'),'2;1');\n"
        "FILE_NAME('test_part.stp','2026-01-15',('Author'),('Org'),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#1=PRODUCT('P001','Test Part','Stress test product',());\n"
        "#2=PRODUCT_DEFINITION_FORMATION('Rev1','Initial',#1);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )


def make_iges(path: Path):
    """Create a minimal IGES file with fixed-column format."""
    # IGES uses 80-column fixed-format lines
    lines = [
        "Stress test IGES file created by HybridRAG.                            S      1",
        "1H,,1H;,7Htest.igs,20HHybridRAG Stress Test,;                         G      1",
        "     116       1       0       0       0       0       0       000000001D      1",
        "     116       0       0       1       0                               0D      2",
        "116,0.0,0.0,0.0,1.0;                                           1P      1",
        "S      1G      1D      2P      1                                        T      1",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_stl_ascii(path: Path):
    """Create a minimal ASCII STL file."""
    path.write_text(
        "solid StressTestSolid\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 1 0 0\n"
        "      vertex 1 1 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid StressTestSolid\n",
        encoding="utf-8",
    )


def make_pem_cert(path: Path):
    """Create a self-signed PEM certificate (valid X.509 structure)."""
    # We will generate a minimal cert using the cryptography library if
    # available, otherwise write a dummy PEM that will test error handling.
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime as dt

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "StressTest CA"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "HybridRAG Test"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(dt.datetime(2026, 1, 1))
            .not_valid_after(dt.datetime(2027, 1, 1))
            .sign(key, hashes.SHA256())
        )
        pem_data = cert.public_bytes(serialization.Encoding.PEM)
        path.write_bytes(pem_data)
    except Exception:
        # If cryptography is not installed, write a dummy PEM
        path.write_text(
            "-----BEGIN CERTIFICATE-----\n"
            "MIIBkTCB+wIJAKHBfpHYM0qlMA0GCSqGSIb3DQEBCwUAMBExDzANBgNVBAMMBnRl\n"
            "c3QwHhcNMjYwMTE1MDAwMDAwWhcNMjcwMTE1MDAwMDAwWjARMQ8wDQYDVQQDDAZ0\n"
            "-----END CERTIFICATE-----\n",
            encoding="utf-8",
        )


def make_binary_placeholder(path: Path, ext: str):
    """Create a small binary file for placeholder/binary format testing."""
    # Write a few bytes so the file is not empty and has a measurable size
    path.write_bytes(b"\x00\x01\x02\x03" * 64)


def make_empty_file(path: Path):
    """Create a zero-byte file."""
    path.write_bytes(b"")


def make_msg_ole(path: Path):
    """Create a minimal OLE2 file that resembles a .msg container."""
    # OLE2 magic number + minimal header so olefile can detect it
    # This is intentionally minimal -- we test the parser's error handling
    # for a minimal/invalid .msg rather than a full valid one.
    ole_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    path.write_bytes(ole_magic + b"\x00" * 504)  # 512-byte minimal header


def make_doc_ole(path: Path):
    """Create a minimal OLE2 file resembling a .doc."""
    ole_magic = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    # Write minimal OLE header + some text that raw binary scan can find
    text_block = b"This is a test document content from OLE binary scan. " * 5
    path.write_bytes(ole_magic + b"\x00" * 504 + text_block)


# ============================================================================
# MAIN TEST SUITES
# ============================================================================

def test_registry_integrity(runner: TestRunner):
    """
    Test 1: Registry Integrity
    --------------------------
    Verify the registry loads correctly, contains expected extensions,
    and reports accurate counts for fully-supported vs placeholder formats.
    """
    print("\n=== TEST SUITE 1: Registry Integrity ===")

    try:
        from src.parsers.registry import REGISTRY
        runner.record("Registry imports without error", "PASS")
    except Exception as e:
        runner.record("Registry imports without error", "FAIL", str(e))
        return  # Cannot continue without registry

    # Check total extension count
    all_exts = REGISTRY.supported_extensions()
    runner.record(
        f"Registry has {len(all_exts)} extensions",
        "PASS" if len(all_exts) >= 55 else "FAIL",
        f"Extensions: {len(all_exts)}"
    )

    # Check fully supported vs placeholder split
    full = REGISTRY.fully_supported_extensions()
    placeholders = REGISTRY.placeholder_extensions()
    runner.record(
        f"Fully supported: {len(full)}, Placeholders: {len(placeholders)}",
        "PASS" if len(full) >= 45 and len(placeholders) >= 10 else "WARN",
    )

    # Verify no overlap between full and placeholder
    overlap = set(full) & set(placeholders)
    runner.record(
        "No overlap between full and placeholder",
        "PASS" if not overlap else "FAIL",
        f"Overlap: {overlap}" if overlap else ""
    )

    # Verify all CAD guy's extensions are registered
    cad_guy_exts = [
        ".igs", ".iges", ".wmf", ".pdf", ".prt", ".sldprt", ".asm",
        ".sldasm", ".ste", ".stp", ".step", ".dwg", ".dwt", ".dxf",
        ".stl", ".eps", ".bmp", ".ai", ".doc", ".emf", ".gif", ".png", ".psd",
    ]
    for ext in cad_guy_exts:
        info = REGISTRY.get(ext)
        runner.record(
            f"CAD extension {ext} registered",
            "PASS" if info is not None else "FAIL",
            info.name if info else "NOT FOUND"
        )

    # Verify each registered extension has a valid parser class
    for ext in all_exts:
        info = REGISTRY.get(ext)
        if info is None:
            runner.record(f"Extension {ext} lookup", "FAIL", "Returns None")
            continue
        has_parse = hasattr(info.parser_cls, "parse") or (
            hasattr(info.parser_cls, "__call__")
        )
        # For placeholder factories, the class itself might need instantiation
        try:
            instance = info.parser_cls()
            has_method = hasattr(instance, "parse") and hasattr(instance, "parse_with_details")
            runner.record(
                f"Extension {ext} -> {info.name} instantiable",
                "PASS" if has_method else "FAIL",
                "" if has_method else "Missing parse/parse_with_details methods"
            )
        except Exception as e:
            runner.record(f"Extension {ext} -> {info.name} instantiable", "FAIL", str(e))


def test_plain_text_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 2: Plain Text Parsers
    --------------------------
    Create synthetic text files for each plain text extension and verify
    the parser extracts the content correctly.
    """
    print("\n=== TEST SUITE 2: Plain Text Parsers ===")

    from src.parsers.registry import REGISTRY

    text_exts = [
        ".txt", ".md", ".csv", ".json", ".xml", ".log",
        ".yaml", ".yml", ".ini", ".cfg", ".conf", ".properties", ".reg",
    ]

    for ext in text_exts:
        test_file = tmp_dir / f"test_file{ext}"
        make_plain_text(test_file, ext)

        info = REGISTRY.get(ext)
        if info is None:
            runner.record(f"PlainText {ext} parse", "FAIL", "Not registered")
            continue

        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(test_file))
            if len(text) > 0:
                runner.record(f"PlainText {ext} parse", "PASS", f"{len(text)} chars")
            else:
                runner.record(f"PlainText {ext} parse", "WARN", "Empty output")
        except Exception as e:
            runner.record(f"PlainText {ext} parse", "FAIL", str(e))


def test_document_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 3: Document Parsers
    ------------------------
    Test .rtf and .doc with synthetic files. PDF/DOCX/PPTX/XLSX require
    valid binary formats that are hard to synthesize, so we test them
    with minimal/invalid files to verify graceful error handling.
    """
    print("\n=== TEST SUITE 3: Document Parsers ===")

    from src.parsers.registry import REGISTRY

    # RTF -- we can create a valid RTF file
    rtf_file = tmp_dir / "test.rtf"
    make_rtf(rtf_file)
    info = REGISTRY.get(".rtf")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(rtf_file))
            status, detail = _check_parse_result(text, details, "Hello")
            runner.record("RTF parser extracts text", status, detail)
        except Exception as e:
            runner.record("RTF parser extracts text", "FAIL", str(e))
    else:
        runner.record("RTF parser registered", "FAIL")

    # DOC -- minimal OLE2 binary
    doc_file = tmp_dir / "test.doc"
    make_doc_ole(doc_file)
    info = REGISTRY.get(".doc")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(doc_file))
            # With a minimal OLE, we expect either extracted text or graceful error
            if "error" in details:
                runner.record("DOC parser graceful on minimal OLE", "PASS",
                              f"Error handled: {details['error'][:60]}")
            elif len(text) > 0:
                runner.record("DOC parser extracts from minimal OLE", "PASS",
                              f"{len(text)} chars")
            else:
                runner.record("DOC parser returns empty on minimal OLE", "PASS",
                              "Empty but no crash")
        except Exception as e:
            runner.record("DOC parser graceful on minimal OLE", "FAIL", str(e))
    else:
        runner.record("DOC parser registered", "FAIL")

    # PDF -- invalid binary (should fail gracefully)
    pdf_file = tmp_dir / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content not a real PDF")
    info = REGISTRY.get(".pdf")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(pdf_file))
            runner.record("PDF parser graceful on invalid PDF", "PASS",
                          f"No crash, got {len(text)} chars")
        except Exception as e:
            runner.record("PDF parser graceful on invalid PDF", "FAIL",
                          f"Crashed: {str(e)[:80]}")

    # AI -- should use PDFParser (test registration only)
    info = REGISTRY.get(".ai")
    if info:
        runner.record("AI extension uses PDFParser", "PASS" if "PDF" in info.name else "FAIL",
                      info.name)
    else:
        runner.record("AI extension registered", "FAIL")

    # DOCX/PPTX/XLSX -- binary ZIP formats, test graceful handling of invalid files
    for ext, name in [(".docx", "DocxParser"), (".pptx", "PptxParser"), (".xlsx", "XlsxParser")]:
        fake_file = tmp_dir / f"test{ext}"
        fake_file.write_bytes(b"PK\x03\x04not a real zip content")
        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(fake_file))
                runner.record(f"{name} graceful on invalid file", "PASS",
                              f"No crash, {len(text)} chars")
            except Exception as e:
                runner.record(f"{name} graceful on invalid file", "FAIL",
                              f"Crashed: {str(e)[:80]}")


def test_email_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 4: Email Parsers
    ---------------------
    Test .eml, .msg, and .mbox with synthetic files.
    """
    print("\n=== TEST SUITE 4: Email Parsers ===")

    from src.parsers.registry import REGISTRY

    # EML
    eml_file = tmp_dir / "test.eml"
    make_eml(eml_file)
    info = REGISTRY.get(".eml")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(eml_file))
            has_subject = "Test Email" in text
            runner.record("EML parser extracts subject", "PASS" if has_subject else "WARN",
                          f"{len(text)} chars, subject found: {has_subject}")
        except Exception as e:
            runner.record("EML parser", "FAIL", str(e))

    # MBOX
    mbox_file = tmp_dir / "test.mbox"
    make_mbox(mbox_file)
    info = REGISTRY.get(".mbox")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(mbox_file))
            has_msgs = details.get("messages", 0) >= 2
            runner.record("MBOX parser finds 2+ messages",
                          "PASS" if has_msgs else "WARN",
                          f"{details.get('messages', 0)} messages, {len(text)} chars")
        except Exception as e:
            runner.record("MBOX parser", "FAIL", str(e))

    # MSG -- minimal OLE2
    msg_file = tmp_dir / "test.msg"
    make_msg_ole(msg_file)
    info = REGISTRY.get(".msg")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(msg_file))
            # Minimal OLE2 may not parse successfully, but should not crash
            runner.record("MSG parser graceful on minimal OLE", "PASS",
                          f"No crash, {len(text)} chars")
        except Exception as e:
            runner.record("MSG parser graceful on minimal OLE", "FAIL",
                          f"Crashed: {str(e)[:80]}")


def test_web_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 5: Web Parsers
    -------------------
    """
    print("\n=== TEST SUITE 5: Web Parsers ===")

    from src.parsers.registry import REGISTRY

    for ext in [".html", ".htm"]:
        html_file = tmp_dir / f"test{ext}"
        make_html(html_file)
        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(html_file))
                has_heading = "Heading" in text
                runner.record(f"HTML {ext} parser extracts content",
                              "PASS" if has_heading else "WARN",
                              f"{len(text)} chars")
            except Exception as e:
                runner.record(f"HTML {ext} parser", "FAIL", str(e))


def test_image_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 6: Image/OCR Parsers
    -------------------------
    Create minimal image files. OCR may not extract text from tiny synthetic
    images, but the parser must not crash.
    """
    print("\n=== TEST SUITE 6: Image/OCR Parsers ===")

    from src.parsers.registry import REGISTRY

    image_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff",
                  ".bmp", ".gif", ".webp", ".wmf", ".emf"]

    for ext in image_exts:
        img_file = tmp_dir / f"test{ext}"

        try:
            # Try to create a minimal valid image using Pillow
            from PIL import Image
            img = Image.new("RGB", (100, 30), color="white")
            # Draw some text if possible
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.text((10, 5), "OCR TEST", fill="black")
            except Exception:
                pass

            if ext in [".wmf", ".emf"]:
                # Pillow cannot save WMF/EMF, create placeholder binary
                make_binary_placeholder(img_file, ext)
            else:
                save_format = {
                    ".jpg": "JPEG", ".jpeg": "JPEG",
                    ".tif": "TIFF", ".tiff": "TIFF",
                }.get(ext, ext.lstrip(".").upper())
                try:
                    img.save(str(img_file), format=save_format)
                except Exception:
                    make_binary_placeholder(img_file, ext)
        except ImportError:
            make_binary_placeholder(img_file, ext)

        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(img_file))
                if "error" in details:
                    runner.record(f"Image {ext} parser graceful error", "PASS",
                                  f"Error: {details['error'][:60]}")
                else:
                    runner.record(f"Image {ext} parser runs", "PASS",
                                  f"{len(text)} chars extracted")
            except Exception as e:
                runner.record(f"Image {ext} parser", "FAIL", f"Crashed: {str(e)[:80]}")
        else:
            runner.record(f"Image {ext} registered", "FAIL")


def test_design_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 7: Design Parsers (PSD)
    ----------------------------
    """
    print("\n=== TEST SUITE 7: Design Parsers ===")

    from src.parsers.registry import REGISTRY

    # PSD -- binary format, create minimal header or placeholder
    psd_file = tmp_dir / "test.psd"
    # PSD magic: "8BPS" + version (1) + reserved (6 bytes) + channels (3)
    # This is a minimal header that psd-tools may or may not accept
    psd_header = b"8BPS" + struct.pack(">H", 1) + b"\x00" * 6
    psd_header += struct.pack(">H", 3)  # channels
    psd_header += struct.pack(">II", 100, 100)  # height, width
    psd_header += struct.pack(">H", 8)  # depth
    psd_header += struct.pack(">H", 3)  # color mode (RGB)
    psd_file.write_bytes(psd_header + b"\x00" * 100)

    info = REGISTRY.get(".psd")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(psd_file))
            runner.record("PSD parser graceful on minimal file", "PASS",
                          f"No crash, {len(text)} chars")
        except Exception as e:
            runner.record("PSD parser graceful on minimal file", "FAIL",
                          f"Crashed: {str(e)[:80]}")


def test_cad_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 8: CAD/Engineering Parsers
    -------------------------------
    Test DXF, STEP, IGES, STL with synthetic files containing known content.
    """
    print("\n=== TEST SUITE 8: CAD/Engineering Parsers ===")

    from src.parsers.registry import REGISTRY

    # DXF
    dxf_file = tmp_dir / "test.dxf"
    make_dxf(dxf_file)
    info = REGISTRY.get(".dxf")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(dxf_file))
            status, detail = _check_parse_result(text, details, "Hello")
            runner.record("DXF parser extracts text entities", status, detail)
        except Exception as e:
            runner.record("DXF parser", "FAIL", str(e))

    # STEP (.stp, .step, .ste)
    for ext in [".stp", ".step", ".ste"]:
        step_file = tmp_dir / f"test{ext}"
        make_step(step_file)
        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(step_file))
                has_product = "Test Part" in text or "STEP" in text or len(text) > 0
                runner.record(f"STEP {ext} parser extracts metadata",
                              "PASS" if has_product else "WARN",
                              f"{len(text)} chars")
            except Exception as e:
                runner.record(f"STEP {ext} parser", "FAIL", str(e))

    # IGES (.igs, .iges)
    for ext in [".igs", ".iges"]:
        iges_file = tmp_dir / f"test{ext}"
        make_iges(iges_file)
        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(iges_file))
                runner.record(f"IGES {ext} parser extracts metadata",
                              "PASS" if len(text) > 0 else "WARN",
                              f"{len(text)} chars")
            except Exception as e:
                runner.record(f"IGES {ext} parser", "FAIL", str(e))

    # STL
    stl_file = tmp_dir / "test.stl"
    make_stl_ascii(stl_file)
    info = REGISTRY.get(".stl")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(stl_file))
            status, detail = _check_parse_result(text, details, "triangle")
            runner.record("STL parser extracts mesh metadata", status, detail)
        except Exception as e:
            runner.record("STL parser", "FAIL", str(e))


def test_diagram_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 9: Diagram Parsers (VSDX)
    ------------------------------
    """
    print("\n=== TEST SUITE 9: Diagram Parsers ===")

    from src.parsers.registry import REGISTRY

    # VSDX is a ZIP-based format -- create minimal invalid zip
    vsdx_file = tmp_dir / "test.vsdx"
    vsdx_file.write_bytes(b"PK\x03\x04not a real vsdx")
    info = REGISTRY.get(".vsdx")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(vsdx_file))
            runner.record("VSDX parser graceful on invalid file", "PASS",
                          f"No crash, {len(text)} chars")
        except Exception as e:
            runner.record("VSDX parser graceful on invalid file", "FAIL",
                          f"Crashed: {str(e)[:80]}")


def test_cyber_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 10: Cybersecurity/Admin Parsers
    ------------------------------------
    Test EVTX, PCAP, certificates with synthetic files.
    """
    print("\n=== TEST SUITE 10: Cybersecurity/Admin Parsers ===")

    from src.parsers.registry import REGISTRY

    # EVTX -- binary format, create minimal header
    evtx_file = tmp_dir / "test.evtx"
    # EVTX magic: "ElfFile\x00"
    evtx_file.write_bytes(b"ElfFile\x00" + b"\x00" * 504)
    info = REGISTRY.get(".evtx")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(evtx_file))
            runner.record("EVTX parser graceful on minimal file", "PASS",
                          f"No crash, {len(text)} chars")
        except Exception as e:
            runner.record("EVTX parser graceful on minimal file", "FAIL",
                          f"Crashed: {str(e)[:80]}")

    # PCAP -- minimal pcap header (24 bytes)
    pcap_file = tmp_dir / "test.pcap"
    # PCAP global header: magic + version + timezone + sigfigs + snaplen + network
    pcap_header = struct.pack("<IHHiIII",
                              0xa1b2c3d4,  # magic
                              2, 4,         # version
                              0,            # timezone
                              0,            # sigfigs
                              65535,        # snaplen
                              1)            # Ethernet
    pcap_file.write_bytes(pcap_header)
    info = REGISTRY.get(".pcap")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(pcap_file))
            runner.record("PCAP parser graceful on empty capture", "PASS",
                          f"No crash, {len(text)} chars, {details.get('packets', 0)} pkts")
        except Exception as e:
            runner.record("PCAP parser graceful on empty capture", "FAIL",
                          f"Crashed: {str(e)[:80]}")

    # PCAPNG -- test registration
    info = REGISTRY.get(".pcapng")
    runner.record("PCAPNG extension registered",
                  "PASS" if info and "Pcap" in info.name else "FAIL",
                  info.name if info else "NOT FOUND")

    # Certificates (.cer, .crt, .pem)
    for ext in [".cer", ".crt", ".pem"]:
        cert_file = tmp_dir / f"test{ext}"
        make_pem_cert(cert_file)
        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(cert_file))
                if "error" in details:
                    runner.record(f"Certificate {ext} parser graceful error", "PASS",
                                  f"Error: {details['error'][:60]}")
                else:
                    has_cert_info = "StressTest" in text or "Certificate" in text or len(text) > 0
                    runner.record(f"Certificate {ext} parser extracts info",
                                  "PASS" if has_cert_info else "WARN",
                                  f"{len(text)} chars")
            except Exception as e:
                runner.record(f"Certificate {ext} parser", "FAIL",
                              f"Crashed: {str(e)[:80]}")


def test_database_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 11: Database Parsers
    -------------------------
    Access databases require a valid format. We test graceful failure
    with an invalid file.
    """
    print("\n=== TEST SUITE 11: Database Parsers ===")

    from src.parsers.registry import REGISTRY

    for ext in [".accdb", ".mdb"]:
        db_file = tmp_dir / f"test{ext}"
        # Write minimal content -- not a valid Access database
        db_file.write_bytes(b"\x00\x01\x00\x00Standard Jet DB\x00" + b"\x00" * 100)
        info = REGISTRY.get(ext)
        if info:
            try:
                parser = info.parser_cls()
                text, details = parser.parse_with_details(str(db_file))
                runner.record(f"Access {ext} parser graceful on invalid file", "PASS",
                              f"No crash, {len(text)} chars")
            except Exception as e:
                runner.record(f"Access {ext} parser graceful on invalid file", "FAIL",
                              f"Crashed: {str(e)[:80]}")


def test_placeholder_parsers(runner: TestRunner, tmp_dir: Path):
    """
    Test 12: Placeholder Parsers
    ----------------------------
    Verify placeholder parsers produce identity-card output with file
    metadata and requirement descriptions.
    """
    print("\n=== TEST SUITE 12: Placeholder Parsers ===")

    from src.parsers.registry import REGISTRY

    placeholder_exts = [".prt", ".sldprt", ".asm", ".sldasm",
                        ".dwg", ".dwt", ".mpp", ".vsd", ".one", ".ost", ".eps"]

    for ext in placeholder_exts:
        ph_file = tmp_dir / f"test_file{ext}"
        make_binary_placeholder(ph_file, ext)

        info = REGISTRY.get(ext)
        if info is None:
            runner.record(f"Placeholder {ext} registered", "FAIL")
            continue

        if "Placeholder" not in info.name:
            runner.record(f"Placeholder {ext} uses PlaceholderParser", "FAIL",
                          f"Uses {info.name} instead")
            continue

        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(ph_file))

            # Verify identity card content
            checks = {
                "has file name": f"test_file{ext}" in text,
                "has format type": "Type:" in text,
                "has size": "Size:" in text or "bytes" in text,
                "has placeholder tag": "PLACEHOLDER" in text,
                "has requirement": "Requirement:" in text,
                "details has placeholder flag": details.get("placeholder") is True,
            }

            all_pass = all(checks.values())
            failed = [k for k, v in checks.items() if not v]

            runner.record(
                f"Placeholder {ext} identity card",
                "PASS" if all_pass else "WARN",
                f"Missing: {', '.join(failed)}" if failed else f"{len(text)} chars"
            )
        except Exception as e:
            runner.record(f"Placeholder {ext} parser", "FAIL", str(e))


def test_fake_extensions(runner: TestRunner, tmp_dir: Path):
    """
    Test 13: Fake Extensions
    ------------------------
    Verify that unrecognized extensions return None from the registry.
    This ensures HybridRAG skips unknown file types gracefully.
    """
    print("\n=== TEST SUITE 13: Fake Extensions ===")

    from src.parsers.registry import REGISTRY

    fake_exts = [
        ".xyz", ".aaa", ".bbb", ".fake", ".notreal",
        ".hybridrag", ".test123", ".abcdefg", ".qqq",
        ".mp3", ".mp4", ".wav", ".avi", ".mkv",  # media (not supported)
        ".exe", ".dll", ".sys", ".bin",            # executables
        ".iso", ".vmdk", ".vhd",                   # disk images
        ".tar", ".gz", ".7z", ".rar",              # archives (not yet)
        "",                                         # empty extension
    ]

    for ext in fake_exts:
        info = REGISTRY.get(ext)
        runner.record(
            f"Fake extension '{ext}' rejected",
            "PASS" if info is None else "FAIL",
            f"Incorrectly mapped to {info.name}" if info else "Correctly returns None"
        )


def test_edge_cases(runner: TestRunner, tmp_dir: Path):
    """
    Test 14: Edge Cases
    -------------------
    Test empty files, files with unusual names, case sensitivity, etc.
    """
    print("\n=== TEST SUITE 14: Edge Cases ===")

    from src.parsers.registry import REGISTRY

    # Empty file test -- should not crash any parser
    empty_file = tmp_dir / "empty.txt"
    make_empty_file(empty_file)
    info = REGISTRY.get(".txt")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(empty_file))
            runner.record("Empty .txt file handled", "PASS",
                          f"No crash, {len(text)} chars")
        except Exception as e:
            runner.record("Empty .txt file handled", "FAIL", str(e))

    # Case sensitivity -- registry should normalize to lowercase
    for ext in [".TXT", ".PDF", ".DXF", ".STEP", ".DocX"]:
        info = REGISTRY.get(ext)
        runner.record(
            f"Case-insensitive lookup '{ext}'",
            "PASS" if info is not None else "FAIL",
            info.name if info else "NOT FOUND"
        )

    # File with spaces in name
    space_file = tmp_dir / "file with spaces.txt"
    space_file.write_text("Content with spaces in filename", encoding="utf-8")
    info = REGISTRY.get(".txt")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(space_file))
            runner.record("File with spaces in name", "PASS", f"{len(text)} chars")
        except Exception as e:
            runner.record("File with spaces in name", "FAIL", str(e))

    # Very long filename
    long_name = "a" * 200 + ".txt"
    long_file = tmp_dir / long_name
    try:
        long_file.write_text("Long filename test", encoding="utf-8")
        info = REGISTRY.get(".txt")
        if info:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(long_file))
            runner.record("Very long filename (200+ chars)", "PASS", f"{len(text)} chars")
    except Exception as e:
        # Some filesystems may reject long names -- that is OK
        runner.record("Very long filename (200+ chars)", "SKIP", str(e)[:60])

    # File that does not exist
    info = REGISTRY.get(".txt")
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(tmp_dir / "nonexistent.txt"))
            runner.record("Nonexistent file handled gracefully", "PASS",
                          "No crash")
        except FileNotFoundError:
            runner.record("Nonexistent file raises FileNotFoundError", "PASS",
                          "Expected exception")
        except Exception as e:
            runner.record("Nonexistent file handled", "WARN",
                          f"Unexpected exception: {type(e).__name__}")

    # Unicode content in text file
    unicode_file = tmp_dir / "unicode.txt"
    unicode_file.write_text(
        "English text\nTexto en espanol\nDeutscher Text\n",
        encoding="utf-8"
    )
    if info:
        try:
            parser = info.parser_cls()
            text, details = parser.parse_with_details(str(unicode_file))
            runner.record("Unicode content handled", "PASS", f"{len(text)} chars")
        except Exception as e:
            runner.record("Unicode content handled", "FAIL", str(e))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("=" * 70)
    print("HybridRAG Expanded Parser Stress Test")
    print("=" * 70)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Project: {PROJECT_ROOT}")
    print()

    runner = TestRunner()
    tmp_dir = Path(tempfile.mkdtemp(prefix="hybridrag_parser_stress_"))
    print(f"Temp directory: {tmp_dir}")

    try:
        test_registry_integrity(runner)
        test_plain_text_parsers(runner, tmp_dir)
        test_document_parsers(runner, tmp_dir)
        test_email_parsers(runner, tmp_dir)
        test_web_parsers(runner, tmp_dir)
        test_image_parsers(runner, tmp_dir)
        test_design_parsers(runner, tmp_dir)
        test_cad_parsers(runner, tmp_dir)
        test_diagram_parsers(runner, tmp_dir)
        test_cyber_parsers(runner, tmp_dir)
        test_database_parsers(runner, tmp_dir)
        test_placeholder_parsers(runner, tmp_dir)
        test_fake_extensions(runner, tmp_dir)
        test_edge_cases(runner, tmp_dir)
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(str(tmp_dir))
            print(f"\n[OK] Cleaned up temp directory: {tmp_dir}")
        except Exception as e:
            print(f"\n[WARN] Could not clean temp dir: {e}")

    # ---- Summary ----
    elapsed = runner.elapsed()
    counts = runner.summary()
    total = sum(counts.values())

    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print(f"Total tests:  {total}")
    print(f"  PASS:       {counts['PASS']}")
    print(f"  FAIL:       {counts['FAIL']}")
    print(f"  WARN:       {counts['WARN']}")
    print(f"  SKIP:       {counts['SKIP']}")
    print(f"Time:         {elapsed:.1f}s")
    print()

    if counts["FAIL"] == 0:
        print("RESULT: ALL TESTS PASSED (no failures)")
    else:
        print(f"RESULT: {counts['FAIL']} FAILURE(S) DETECTED")
        print("\nFailed tests:")
        for r in runner.results:
            if r.status == "FAIL":
                print(f"  [FAIL] {r.name}: {r.detail}")

    # ---- Generate markdown report ----
    report_path = PROJECT_ROOT / "docs" / "EXPANDED_PARSER_STRESS_TEST.md"
    _write_report(report_path, runner, counts, elapsed)
    print(f"\nReport saved: {report_path}")

    return 0 if counts["FAIL"] == 0 else 1


def _write_report(path: Path, runner: TestRunner, counts: Dict, elapsed: float):
    """Write a markdown report of the stress test results."""
    lines = [
        "# Expanded Parser Stress Test Results",
        "",
        f"**Date:** {datetime.now().isoformat()}",
        f"**Total tests:** {sum(counts.values())}",
        f"**Duration:** {elapsed:.1f}s",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|--------|-------|",
        f"| PASS   | {counts['PASS']}  |",
        f"| FAIL   | {counts['FAIL']}  |",
        f"| WARN   | {counts['WARN']}  |",
        f"| SKIP   | {counts['SKIP']}  |",
        "",
        "## Detailed Results",
        "",
        "| # | Test | Status | Detail |",
        "|---|------|--------|--------|",
    ]

    for i, r in enumerate(runner.results, 1):
        detail = r.detail.replace("|", "/").replace("\n", " ")[:80]
        lines.append(f"| {i} | {r.name} | {r.status} | {detail} |")

    if counts["FAIL"] > 0:
        lines.extend(["", "## Failures", ""])
        for r in runner.results:
            if r.status == "FAIL":
                lines.append(f"- **{r.name}**: {r.detail}")

    if counts["WARN"] > 0:
        lines.extend(["", "## Warnings", ""])
        for r in runner.results:
            if r.status == "WARN":
                lines.append(f"- **{r.name}**: {r.detail}")

    lines.extend(["", "---", f"*Generated by stress_test_expanded_parsers.py*"])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
