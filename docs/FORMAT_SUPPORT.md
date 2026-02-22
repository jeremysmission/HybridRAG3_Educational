# HybridRAG File Format Support

> **Non-programmer note:** This document lists every file type HybridRAG can
> process. "Fully supported" means we extract text and metadata right now.
> "Placeholder" means we recognize the file but need additional software to
> read its contents. "Wish list" means we intend to add support in the future.

---

## Quick Stats

| Category              | Count |
|-----------------------|-------|
| Fully supported       |   49  |
| Placeholder (recognized) | 11 |
| Total registered      |   60  |
| Wish list (future)    |   14  |

---

## 1. Fully Supported Formats

These formats are parsed NOW. HybridRAG extracts searchable text from them.

### 1.1 Plain Text Formats

| Extension    | Format Name                  | Parser          | Dependencies         |
|-------------|------------------------------|-----------------|----------------------|
| `.txt`      | Plain text                   | PlainTextParser | None (stdlib)        |
| `.md`       | Markdown                     | PlainTextParser | None (stdlib)        |
| `.csv`      | Comma-separated values       | PlainTextParser | None (stdlib)        |
| `.json`     | JSON data                    | PlainTextParser | None (stdlib)        |
| `.xml`      | XML markup                   | PlainTextParser | None (stdlib)        |
| `.log`      | Log files                    | PlainTextParser | None (stdlib)        |
| `.yaml`     | YAML configuration           | PlainTextParser | None (stdlib)        |
| `.yml`      | YAML configuration (alt)     | PlainTextParser | None (stdlib)        |
| `.ini`      | INI configuration            | PlainTextParser | None (stdlib)        |
| `.cfg`      | Config files                 | PlainTextParser | None (stdlib)        |
| `.conf`     | Config files (Unix style)    | PlainTextParser | None (stdlib)        |
| `.properties` | Java properties            | PlainTextParser | None (stdlib)        |
| `.reg`      | Windows Registry export      | PlainTextParser | None (stdlib)        |

> **Non-programmer note:** All these are already human-readable text files.
> We just read them in and index the content as-is. No special library needed.

### 1.2 Document Formats

| Extension | Format Name              | Parser     | Dependencies                     |
|-----------|--------------------------|------------|----------------------------------|
| `.pdf`    | PDF (Portable Document)  | PDFParser  | pdfplumber (MIT)                 |
| `.docx`   | Word 2007+ document      | DocxParser | python-docx (MIT)                |
| `.pptx`   | PowerPoint 2007+ slides  | PptxParser | python-pptx (MIT)                |
| `.xlsx`   | Excel 2007+ spreadsheet  | XlsxParser | openpyxl (MIT)                   |
| `.doc`    | Word 97-2003 (legacy)    | DocParser  | olefile (BSD), antiword optional  |
| `.rtf`    | Rich Text Format         | RtfParser  | striprtf (BSD)                   |
| `.ai`     | Adobe Illustrator        | PDFParser  | pdfplumber (MIT) -- see note (a) |

> **(a) Adobe Illustrator note:** Modern .ai files (CS era and later) embed
> a PDF representation inside the file. We parse that embedded PDF to extract
> any text. Pre-CS .ai files are pure PostScript and would need Ghostscript
> (AGPL license) -- those are rare in practice.

### 1.3 Email Formats

| Extension | Format Name             | Parser    | Dependencies                      |
|-----------|-------------------------|-----------|-----------------------------------|
| `.eml`    | RFC 822 email message   | EmlParser | None (stdlib email module)        |
| `.msg`    | Outlook email message   | MsgParser | python-oxmsg (MIT), olefile (BSD) |
| `.mbox`   | Unix mbox email archive | MboxParser| None (stdlib mailbox module)      |

> **Non-programmer note:** `.eml` and `.mbox` use Python's built-in email
> libraries, so no extra installation is needed. `.msg` files use a
> MIT-licensed library (python-oxmsg) chosen specifically because the more
> popular `extract-msg` library uses a GPL license (viral -- would restrict
> how we distribute HybridRAG).

### 1.4 Web Formats

| Extension | Format Name | Parser         | Dependencies         |
|-----------|-------------|----------------|----------------------|
| `.html`   | HTML page   | HtmlFileParser | beautifulsoup4 (MIT) |
| `.htm`    | HTML page   | HtmlFileParser | beautifulsoup4 (MIT) |

### 1.5 Image Formats (OCR-based)

| Extension | Format Name                | Parser         | Dependencies                  |
|-----------|----------------------------|----------------|-------------------------------|
| `.png`    | PNG image                  | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.jpg`    | JPEG image                 | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.jpeg`   | JPEG image                 | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.tif`    | TIFF image                 | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.tiff`   | TIFF image                 | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.bmp`    | Bitmap image               | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.gif`    | GIF image                  | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.webp`   | WebP image                 | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.wmf`    | Windows Metafile           | ImageOCRParser | Pillow (MIT), Tesseract OCR   |
| `.emf`    | Enhanced Metafile          | ImageOCRParser | Pillow (MIT), Tesseract OCR   |

> **Non-programmer note:** We cannot "read" images directly. Instead, we use
> OCR (Optical Character Recognition) via Tesseract to find text visible in
> the image. WMF/EMF are Windows vector graphics -- Pillow rasterizes them
> using Windows GDI+, then we OCR the result. This means WMF/EMF OCR only
> works on Windows.

### 1.6 Design Formats

| Extension | Format Name      | Parser    | Dependencies          |
|-----------|------------------|-----------|-----------------------|
| `.psd`    | Adobe Photoshop  | PsdParser | psd-tools (MIT)       |

> **Non-programmer note:** Photoshop files contain layers. We extract the
> names of all layers and any text layers (type tool text). This makes PSD
> files searchable by layer name and text content. We do NOT extract the
> actual image pixels for OCR -- just the structured text data.

### 1.7 CAD / Engineering Formats

| Extension | Format Name              | Parser      | Dependencies           |
|-----------|--------------------------|-------------|------------------------|
| `.dxf`    | AutoCAD DXF exchange     | DxfParser   | ezdxf (MIT)            |
| `.stp`    | STEP (ISO 10303)         | StepParser  | None (regex parsing)   |
| `.step`   | STEP (ISO 10303)         | StepParser  | None (regex parsing)   |
| `.ste`    | STEP (ISO 10303, short)  | StepParser  | None (regex parsing)   |
| `.igs`    | IGES CAD exchange        | IgesParser  | None (text parsing)    |
| `.iges`   | IGES CAD exchange        | IgesParser  | None (text parsing)    |
| `.stl`    | STL 3D mesh              | StlParser   | numpy-stl (BSD)        |

> **Non-programmer note:** CAD formats vary widely. DXF is AutoCAD's open
> exchange format and we extract all text annotations, dimensions, layer
> names, and block attributes. STEP and IGES are open text-based formats
> used for CAD data exchange -- we extract product names, descriptions, and
> file metadata. STL files contain only 3D geometry (triangles) with no text,
> so we extract structural metadata (triangle count, bounding box, volume).

### 1.8 Diagram Formats

| Extension | Format Name       | Parser     | Dependencies   |
|-----------|-------------------|------------|----------------|
| `.vsdx`   | Visio 2013+ diagram | VsdxParser | vsdx (BSD)   |

### 1.9 Cybersecurity / System Admin Formats

| Extension  | Format Name            | Parser            | Dependencies               |
|------------|------------------------|-------------------|-----------------------------|
| `.evtx`    | Windows Event Log      | EvtxParser        | python-evtx (Apache 2.0)   |
| `.pcap`    | Network capture        | PcapParser        | dpkt (BSD)                  |
| `.pcapng`  | Network capture (NG)   | PcapParser        | dpkt (BSD)                  |
| `.cer`     | X.509 certificate      | CertificateParser | cryptography (Apache/BSD)   |
| `.crt`     | X.509 certificate      | CertificateParser | cryptography (Apache/BSD)   |
| `.pem`     | PEM-encoded cert/key   | CertificateParser | cryptography (Apache/BSD)   |

> **Non-programmer note:** These formats are common in IT and cybersecurity.
> Event logs (.evtx) come from Windows Event Viewer. Packet captures (.pcap)
> come from Wireshark. Certificates (.cer/.crt/.pem) are used for SSL/TLS.
> We chose `dpkt` over the more popular `scapy` because dpkt is BSD-licensed
> while scapy is GPL (viral license).

### 1.10 Database Formats

| Extension | Format Name           | Parser         | Dependencies                |
|-----------|-----------------------|----------------|-----------------------------|
| `.accdb`  | Access 2007+ database | AccessDbParser | access-parser (Apache 2.0)  |
| `.mdb`    | Access 97-2003 DB     | AccessDbParser | access-parser (Apache 2.0)  |

> **Non-programmer note:** Access databases are surprisingly common in
> engineering and logistics. We extract table names, column names, and a
> sample of row data (capped at 50 rows per table to keep output manageable).

---

## 2. Placeholder Formats (Recognized but Not Fully Parsed)

These file types are **recognized** by HybridRAG. The file will appear in
search results by filename, type, and location, but the internal content
cannot be extracted yet. Each entry explains what would be needed.

| Extension   | Format Name             | What Is Needed                                        |
|-------------|-------------------------|-------------------------------------------------------|
| `.prt`      | SolidWorks Part         | SolidWorks installed + pywin32 COM API (pySldWrap)    |
| `.sldprt`   | SolidWorks Part         | Same as .prt                                          |
| `.asm`      | SolidWorks Assembly     | SolidWorks installed + pywin32 COM API                |
| `.sldasm`   | SolidWorks Assembly     | Same as .asm                                          |
| `.dwg`      | AutoCAD Drawing         | ODA File Converter (free) or LibreDWG (GPL-3.0)      |
| `.dwt`      | AutoCAD Template        | Same as .dwg                                          |
| `.mpp`      | Microsoft Project       | Java Runtime + MPXJ library (LGPL)                    |
| `.vsd`      | Visio Legacy (binary)   | No good Python parser; convert to .vsdx first         |
| `.one`      | OneNote Section         | pyOneNote (limited quality); export to PDF preferred   |
| `.ost`      | Outlook Offline Storage | libpff-python (needs C compiler); hard on Windows     |
| `.eps`      | Encapsulated PostScript | Ghostscript (AGPL-3.0) for rendering                  |

> **Non-programmer note:** "Placeholder" does NOT mean the file is ignored.
> HybridRAG still records the file's name, type, size, and location. If
> someone searches "SolidWorks assembly for pump housing", a .sldasm file
> named "pump_housing.sldasm" can still appear in results -- we just cannot
> extract the 3D geometry or internal annotations.

### Upgrade Paths for Placeholder Formats

**SolidWorks (.prt, .sldprt, .asm, .sldasm):**
The easiest path is to ask users to export their SolidWorks files as STEP
(.stp) or IGES (.igs) before ingestion. Those open formats are already
fully supported. Alternatively, if SolidWorks is installed on the indexing
machine, the pywin32 COM API can automate SolidWorks to extract metadata.

**AutoCAD DWG (.dwg, .dwt):**
Option 1: Install the free ODA File Converter to batch-convert DWG to DXF,
then parse the DXF with our existing DxfParser. Option 2: LibreDWG is
open-source but GPL-3.0 (viral license). No MIT/BSD Python library exists
for DWG reading.

**Microsoft Project (.mpp):**
The MPXJ library can read .mpp files, but it wraps a Java library via
JPype, so a Java Runtime Environment (JRE) must be installed. This adds
significant complexity to the deployment.

**Legacy Visio (.vsd):**
The .vsd format is a binary OLE container with no good open-source parser.
The recommended workflow is to convert .vsd to .vsdx using Visio or
LibreOffice, then parse the .vsdx with our existing VsdxParser.

**Outlook Offline Storage (.ost):**
The libpff-python package can parse .ost files but requires a C compiler
toolchain to install. Works better on Linux than Windows. The preferred
alternative is to export individual messages as .msg or .eml files.

**Encapsulated PostScript (.eps):**
Full EPS rendering requires Ghostscript, which is AGPL-3.0 licensed. We
could extract DSC (Document Structuring Convention) comments without
Ghostscript, but that provides minimal useful text.

---

## 3. Wish List (Future Formats)

These are formats we have identified as valuable but have not yet built
parsers for. They are organized by professional domain.

### 3.1 Program Management / Office

| Extension | Format Name            | Difficulty | Notes                                    |
|-----------|------------------------|------------|------------------------------------------|
| `.xls`    | Excel 97-2003          | Medium     | xlrd (BSD) can read; legacy but common    |
| `.ppt`    | PowerPoint 97-2003     | Medium     | python-pptx cannot read; need OLE parsing |
| `.odt`    | OpenDocument Text      | Easy       | odfpy (Apache 2.0) or python-docx fork   |
| `.ods`    | OpenDocument Spreadsheet| Easy      | odfpy (Apache 2.0)                       |
| `.odp`    | OpenDocument Presentation| Easy     | odfpy (Apache 2.0)                       |

### 3.2 Engineering / Technical

| Extension | Format Name            | Difficulty | Notes                                    |
|-----------|------------------------|------------|------------------------------------------|
| `.ipt`    | Inventor Part          | Hard       | Proprietary (Autodesk); no open parser   |
| `.iam`    | Inventor Assembly      | Hard       | Proprietary (Autodesk); no open parser   |
| `.catpart`| CATIA Part             | Hard       | Proprietary (Dassault); no open parser   |
| `.3mf`    | 3MF 3D Manufacturing   | Easy       | XML-based ZIP; stdlib can read           |

### 3.3 Cybersecurity / IT Admin

| Extension | Format Name            | Difficulty | Notes                                    |
|-----------|------------------------|------------|------------------------------------------|
| `.cap`    | Packet capture (alt)   | Easy       | Same format as .pcap; just register ext  |
| `.dmp`    | Windows crash dump     | Hard       | Needs specialized parsing (minidump fmt) |
| `.vmdk`   | VMware disk image      | Hard       | Very large; forensic use case only       |
| `.ova`    | VM appliance           | Medium     | TAR archive with OVF XML + VMDK         |

### 3.4 Logistics / Field Engineering

| Extension | Format Name            | Difficulty | Notes                                    |
|-----------|------------------------|------------|------------------------------------------|
| `.gpx`    | GPS Exchange Format    | Easy       | XML-based; stdlib xml.etree can parse    |
| `.kml`    | Keyhole Markup (Maps)  | Easy       | XML-based; used in Google Earth          |
| `.shp`    | ESRI Shapefile         | Medium     | pyshp (MIT) can read; geospatial data    |

---

## 4. Dependency Summary

All dependencies use permissive licenses (MIT, BSD, Apache 2.0) unless
noted otherwise. No GPL or AGPL dependencies are used in production code.

### Required (core parsers)

```
pdfplumber          # PDF extraction (MIT)
python-docx         # DOCX reading (MIT)
python-pptx         # PPTX reading (MIT)
openpyxl            # XLSX reading (MIT)
beautifulsoup4      # HTML parsing (MIT)
Pillow              # Image handling (MIT)
pytesseract         # OCR bridge to Tesseract (Apache 2.0)
```

### Required (expanded parsers)

```
ezdxf               # DXF (AutoCAD exchange) parsing (MIT)
numpy-stl           # STL mesh reading (BSD)
striprtf            # RTF text extraction (BSD)
olefile             # OLE2 file reading for .doc/.msg fallback (BSD)
python-oxmsg        # Outlook .msg reading (MIT)
psd-tools           # Photoshop PSD layer extraction (MIT)
vsdx                # Visio .vsdx reading (BSD)
python-evtx         # Windows event log reading (Apache 2.0)
dpkt                # Network packet capture parsing (BSD)
cryptography        # X.509 certificate parsing (Apache/BSD)
access-parser       # Access database reading (Apache 2.0)
```

### Optional (system tools)

```
Tesseract OCR       # External binary for image OCR (Apache 2.0)
antiword            # External binary for .doc extraction (GPL -- optional)
```

> **Non-programmer note:** "MIT", "BSD", and "Apache 2.0" are all permissive
> open-source licenses. They allow you to use the software freely, even in
> commercial projects. "GPL" is a "copyleft" license that requires you to
> release your own source code if you distribute software using it. We avoid
> GPL dependencies in production. The optional `antiword` tool is GPL but
> is only used as a best-effort fallback; the parser works without it.

---

## 5. How to Add a New Format

1. **Create the parser file** in `src/parsers/`:
   - Copy an existing parser as a template (e.g., `rtf_parser.py` for simple ones)
   - Your class must have `parse(file_path) -> str` and
     `parse_with_details(file_path) -> Tuple[str, Dict]`
   - Import dependencies inside the method (not at module level) so import
     errors are caught gracefully

2. **Register the extension** in `src/parsers/registry.py`:
   - Import your parser class at the top
   - Add `self.register(".ext", "YourParser", YourParser)` in `__init__`

3. **Update this document** (docs/FORMAT_SUPPORT.md):
   - Add the format to the appropriate table above
   - Include the dependency and its license

4. **Write a test** in `tests/`:
   - Create a synthetic test file or use a minimal real file
   - Test both `parse()` and `parse_with_details()`
   - Test graceful failure when the dependency is missing

5. **Install the dependency**:
   - Add it to `requirements.txt`
   - Run `pip install -r requirements.txt`

---

## 6. Architecture Notes

### Parser Interface

Every parser must implement:

```python
class MyParser:
    def parse(self, file_path: str) -> str:
        """Return extracted text. Empty string on failure."""
        text, _ = self.parse_with_details(file_path)
        return text

    def parse_with_details(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Return (text, details_dict).
        details_dict always contains: file, parser, total_len.
        On error: details_dict contains 'error' key."""
        ...
```

### Graceful Degradation Pattern

Parsers that need external libraries use lazy imports:

```python
def parse_with_details(self, file_path):
    try:
        import some_library  # Imported HERE, not at file top
    except ImportError as e:
        details["error"] = f"IMPORT_ERROR: {e}. Install with: pip install some-library"
        return "", details
    # ... normal parsing ...
```

This means HybridRAG starts up fine even if some optional libraries are
not installed. Files that need a missing library will simply return empty
text with an error message in the details dict.

### Registry Lookup Flow

```
File discovered on disk
    |
    v
Extract extension (e.g., ".dxf")
    |
    v
registry.get(".dxf") -> ParserInfo(name="DxfParser", parser_cls=DxfParser)
    |
    v
Instantiate: parser = DxfParser()
    |
    v
Call: text, details = parser.parse_with_details(file_path)
    |
    v
If text is non-empty -> index it for search
If text is empty -> log the error from details["error"]
```

---

*Last updated: 2026-02-20*
*Generated as part of the Expanded Parsing feature branch*
