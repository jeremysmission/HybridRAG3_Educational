# ============================================================================
# HybridRAG — Schematic Conversion Schema (docs/schematic_conversion_schema.md)
# ============================================================================
# This document defines the intermediate JSON format for converting
# scanned legacy schematics into machine-readable electronics formats.
#
# STATUS: Planning / Future Phase
# This is a specification document, not working code. It defines the
# data structures that future schematic extraction tools will produce,
# and that future EDA export tools will consume.
#
# WHY DEFINE THIS NOW:
#   During indexing, we can start collecting metadata that maps to this
#   schema (component counts, part numbers found in OCR text, etc.)
#   even before the full vision pipeline exists. This metadata makes
#   the future extraction pipeline dramatically easier to build.
# ============================================================================

# Schematic Conversion Pipeline
# ==============================
#
# Phase 1 (NOW — during indexing):
#   - OCR scanned schematics to extract text
#   - Classify documents as schematics via filename patterns
#   - Extract part numbers, reference designators (R1, C1, U1) from OCR text
#   - Store source image paths for later vision model processing
#
# Phase 2 (FUTURE — vision model integration):
#   - Feed schematic images to a vision model (GPT-4V or fine-tuned YOLO)
#   - Detect component symbols (resistor, capacitor, IC, connector, etc.)
#   - Detect connection lines (nets/wires)
#   - Detect text labels (reference designators, values, net names)
#   - Output: intermediate JSON (schema defined below)
#
# Phase 3 (FUTURE — EDA export):
#   - Convert intermediate JSON to target EDA formats:
#     * EDIF (Electronic Design Interchange Format) — universal
#     * SPICE netlist — simulation
#     * KiCad (.kicad_sch) — open source EDA
#     * CSV BOM (Bill of Materials) — procurement
#   - Validate generated netlist against extracted component specs
#
# Phase 4 (FUTURE — SysML/MBSE integration):
#   - Convert intermediate JSON to SysML XMI
#   - Generate Block Definition Diagrams (component hierarchy)
#   - Generate Internal Block Diagrams (signal flow / connections)
#   - Link to requirements via traceability matrix

# ============================================================================
# INTERMEDIATE JSON SCHEMA
# ============================================================================
#
# This is the "universal translator" format. Vision models produce it,
# EDA exporters consume it. All formats can be generated from this.
#
# {
#   "schema_version": "1.0",
#   "source": {
#     "file_path": "C:/.../legacy_power_supply.tiff",
#     "file_hash": "sha256:abc123...",
#     "page_number": 1,
#     "extraction_method": "vision_model",       # or "manual", "ocr_heuristic"
#     "extraction_model": "gpt-4-vision",
#     "extraction_confidence": 0.87,
#     "extraction_timestamp": "2026-02-06T19:00:00Z"
#   },
#
#   "metadata": {
#     "title": "Power Supply Unit - Main Board",
#     "drawing_number": "PSU-001-SCH-A",
#     "revision": "C",
#     "date": "1995-03-15",
#     "author": "J. Smith",
#     "classification": "INTERNAL",
#     "system": "AN/APQ-164 Radar",
#     "subsystem": "Power Supply"
#   },
#
#   "components": [
#     {
#       "ref_designator": "R1",                  # Reference designator
#       "type": "resistor",                      # Component category
#       "value": "10k",                          # Value with units
#       "part_number": "RC0805FR-0710KL",        # Manufacturer part number
#       "package": "0805",                       # Physical package/footprint
#       "manufacturer": "Yageo",
#       "description": "RES 10K OHM 1% 1/8W",
#       "part_spec": "M55342K06B10K00",           # Part number if applicable
#       "position": {"x": 120, "y": 340},        # Position on schematic (pixels)
#       "confidence": 0.95                        # Extraction confidence
#     },
#     {
#       "ref_designator": "U1",
#       "type": "integrated_circuit",
#       "value": "LM7805",
#       "part_number": "LM7805CT",
#       "package": "TO-220",
#       "manufacturer": "Texas Instruments",
#       "pins": [
#         {"number": 1, "name": "IN",  "type": "power_in"},
#         {"number": 2, "name": "GND", "type": "ground"},
#         {"number": 3, "name": "OUT", "type": "power_out"}
#       ],
#       "confidence": 0.92
#     }
#   ],
#
#   "nets": [
#     {
#       "name": "VCC_5V",
#       "type": "power",                         # power, signal, ground, bus
#       "pins": ["U1.OUT", "R1.1", "C2.1", "J1.1"],
#       "confidence": 0.88
#     },
#     {
#       "name": "GND",
#       "type": "ground",
#       "pins": ["U1.GND", "C1.2", "C2.2", "J1.3"],
#       "confidence": 0.95
#     }
#   ],
#
#   "annotations": [
#     {
#       "type": "note",
#       "text": "All capacitors rated 50V minimum",
#       "position": {"x": 50, "y": 800}
#     },
#     {
#       "type": "revision_block",
#       "text": "Rev C: Changed R3 from 4.7k to 10k per ECO-2024-156"
#     }
#   ]
# }

# ============================================================================
# SUPPORTED EDA OUTPUT FORMATS
# ============================================================================
#
# EDIF (Electronic Design Interchange Format):
#   - Industry standard since 1985, still widely supported
#   - XML-like syntax, verbose but unambiguous
#   - Imported by: Altium, OrCAD, Mentor, Cadence, Synopsys
#   - Best choice for cross-tool compatibility
#   - Python library: none mature — generate from template
#
# SPICE Netlist:
#   - Text-based circuit description
#   - Every EDA tool and simulator reads it
#   - Format: component_name node1 node2 value
#     Example: R1 VCC_5V NODE_A 10k
#   - Python library: PySpice (for simulation), or generate directly
#
# KiCad Schematic (.kicad_sch):
#   - S-expression format (Lisp-like), well-documented
#   - Open source, free, growing adoption in government
#   - Can import into KiCad then export to other formats
#   - Python library: kiutils (MIT license, reads/writes KiCad files)
#     Note: verify compatibility before adopting — check GitHub issues
#
# CSV Bill of Materials (BOM):
#   - Simplest output: ref_designator, type, value, part_number, quantity
#   - Every procurement system accepts CSV
#   - Good first milestone before full netlist generation
#
# SysML XMI:
#   - XML Metadata Interchange format for SysML models
#   - Imported by: Cameo Systems Modeler, Rhapsody, Papyrus
#   - Most complex output — requires understanding SysML metamodel
#   - Recommend starting with Block Definition Diagrams only

# ============================================================================
# REFERENCE DESIGNATOR STANDARDS (for extraction regex)
# ============================================================================
#
# These are the standard prefixes used on schematics.
# During OCR text extraction, matching these patterns helps identify
# component references even before vision model processing.
#
# Prefix  | Component Type          | Example
# --------|-------------------------|--------
# R       | Resistor                | R1, R47
# C       | Capacitor               | C1, C100
# L       | Inductor                | L1, L3
# D       | Diode                   | D1, D12
# Q       | Transistor              | Q1, Q5
# U       | Integrated Circuit      | U1, U32
# J       | Connector               | J1, J15
# P       | Plug                    | P1, P4
# K       | Relay                   | K1, K3
# T       | Transformer             | T1, T2
# F       | Fuse                    | F1, F3
# SW      | Switch                  | SW1, SW5
# TP      | Test Point              | TP1, TP20
# CR      | Crystal                 | CR1, Y1
# LED     | LED                     | LED1, LED5
# TB      | Terminal Block          | TB1
# BT      | Battery                 | BT1
#
# Regex for extraction: r'\b([A-Z]{1,3}\d{1,4})\b'
# This catches most standard reference designators.

# ============================================================================
# IONOGRAM ANALYSIS PIPELINE
# ============================================================================
#
# STATUS: Planning / Future Phase
#
# Ionograms are frequency-vs-virtual-height plots produced by ionosondes
# (ionospheric radar sounders). They are critical for:
#   - HF radio propagation prediction (research comms)
#   - Over-the-horizon radar (OTH-R) performance modeling
#   - GPS/GNSS accuracy estimation (ionospheric delay)
#   - SATCOM link budget planning
#   - Space weather monitoring
#
# DATA FORMATS:
#
#   SAO (Standard Archiving Output) — version 4.x
#     - ASCII text file, max 120 chars per line
#     - Contains: scaled ionospheric characteristics (foF2, foF1, foE,
#       foEs, h'F, h'E, h'Es, etc.), echo traces h'(f), amplitudes,
#       frequency/range spread, electron density profiles
#     - Produced by ARTIST autoscaling software on Digisondes
#     - DIRECTLY INDEXABLE by HybridRAG (it's just text)
#     - Available from: DIDBase, NOAA NGDC, WDC sites
#
#   SAOXML 5 — the modern successor
#     - XML format (also text, also directly parseable)
#     - Richer metadata, better extensibility
#     - URSI-recommended standard for data exchange
#
#   Raw ionogram images (PNG/TIFF)
#     - The visual frequency-vs-virtual-height plot
#     - Traditionally requires expert manual scaling
#     - AI autoscaling: CNNs have achieved RMSE of 0.12 MHz for foF2
#       compared to human operators (published research, 2024)
#
# ANALYSIS PIPELINE:
#
#   Phase 1 (NOW — during indexing):
#     - Index SAO text files as regular documents
#     - Extract ionospheric parameters (foF2, hmF2, etc.) from SAO text
#     - Tag files as "ionogram" via classification rules
#     - Store station location, timestamp, sounding parameters as metadata
#
#   Phase 2 (FUTURE — SAO parser):
#     - Write a dedicated SAO parser that extracts structured data:
#       * Scaled ionospheric characteristics (49 defined parameters)
#       * Echo trace arrays h'(f)
#       * Electron density profiles N(h)
#       * ARTIST confidence scores (C-level: 11=best to 55=worst)
#     - Store as structured JSON for querying:
#       "Show me all ionograms where foF2 > 8 MHz"
#
#   Phase 3 (FUTURE — vision model):
#     - Feed raw ionogram images to vision model
#     - Extract: ordinary/extraordinary trace separation, spread-F detection,
#       sporadic-E identification, interference classification
#     - Compare with ARTIST autoscaling for validation
#     - Output: structured ionospheric parameters matching SAO schema
#
#   Phase 4 (FUTURE — propagation prediction):
#     - Use extracted ionospheric parameters with propagation models
#       (VOACAP, ICEPAC, PropLab Pro) for HF prediction
#     - "Given current ionospheric conditions at station X, what
#       frequencies will support comms between points A and B?"
#     - This is where RAG becomes truly operational for research comms
#
# INTERMEDIATE JSON SCHEMA (ionogram-specific):
#
# {
#   "schema_version": "1.0",
#   "source": {
#     "station_code": "WP937",
#     "station_name": "Wallops Island",
#     "latitude": 37.93,
#     "longitude": -75.47,
#     "instrument": "DPS-4D",
#     "timestamp_utc": "2026-02-06T19:00:00Z",
#     "sao_version": "4.3",
#     "artist_version": "5.2",
#     "confidence_score": 85,
#     "c_level": "12"
#   },
#
#   "scaled_characteristics": {
#     "foF2": 7.8,        # F2 layer critical frequency (MHz)
#     "foF1": 4.2,        # F1 layer critical frequency (MHz)
#     "foE": 3.1,         # E layer critical frequency (MHz)
#     "foEs": 5.5,        # Sporadic E critical frequency (MHz)
#     "hmF2": 280.0,      # F2 layer peak height (km)
#     "hmF1": 200.0,      # F1 layer peak height (km)
#     "hmE": 110.0,       # E layer peak height (km)
#     "h_prime_F": 250.0, # F layer minimum virtual height (km)
#     "h_prime_E": 100.0, # E layer minimum virtual height (km)
#     "fmin": 1.5,        # Minimum observed frequency (MHz)
#     "MUF_3000_F2": 24.5,# Maximum usable frequency at 3000km
#     "TEC": 15.2         # Total electron content (TECU)
#   },
#
#   "traces": {
#     "ordinary": [[1.5, 250], [2.0, 245], [2.5, 240], ...],
#     "extraordinary": [[1.8, 252], [2.3, 247], ...],
#     "units": ["MHz", "km"]
#   },
#
#   "electron_density_profile": {
#     "heights_km": [100, 110, 120, ...],
#     "density_per_cm3": [1e4, 5e4, 1e5, ...],
#     "method": "ARTIST_inversion"
#   },
#
#   "conditions": {
#     "spread_f": false,
#     "sporadic_e": true,
#     "es_type": "c",
#     "interference_level": "low",
#     "solar_conditions": "moderate"
#   }
# }
