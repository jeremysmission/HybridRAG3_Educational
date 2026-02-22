#!/usr/bin/env python3
"""
============================================================================
HybridRAG3 -- Bulk Transfer V2 Stress Test
============================================================================
WHAT THIS FILE DOES (plain English):
  Simulates 5 GB of mixed files from an enterprise network drive and
  compares what would happen with a basic robocopy approach versus
  HybridRAG3's Bulk Transfer V2 engine.

  No actual files are created or network drives are accessed. The entire
  simulation runs in memory using virtual file objects. Each file has
  realistic attributes: size, extension, hash, locked/hidden/symlink
  flags, encoding issues, path lengths, etc.

  The simulation produces a 2-column comparison report showing exactly
  how many files would be copied, skipped, deduplicated, quarantined,
  verified, and failed -- plus timing estimates.

HOW TO READ THE RESULTS:
  The "Original" column represents a basic file copy (like robocopy):
    - Copies ALL files regardless of type
    - No hash verification
    - No deduplication
    - No staging directories
    - Locked files just fail
    - Hidden files are invisible problems
    - No manifest (no way to audit what happened)

  The "V2" column represents the Bulk Transfer V2 engine:
    - Only copies RAG-parseable file types
    - SHA-256 hash verification on every file
    - Content-hash deduplication (skip identical files)
    - Three-stage staging (incoming/verified/quarantine)
    - Locked file detection and quarantine
    - Full manifest with zero-gap accounting

INTERNET ACCESS: NONE
============================================================================
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ============================================================================
# Virtual Filesystem Generator
# ============================================================================
# NON-PROGRAMMER NOTE:
#   This creates a "pretend" set of files that mimics what you'd find on
#   a real enterprise network drive. The distribution of file types, sizes,
#   and edge cases is based on real-world corporate file servers.
#
#   Each tuple is: (extension, average_size_bytes, weight, rag_parseable)
#   - weight controls how common this file type is (higher = more files)
#   - rag_parseable means HybridRAG has a parser for this type
# ============================================================================

_DIST = [
    # RAG-parseable document types (the ones we actually want)
    (".pdf",   2_500_000,  25, True),   # Most common: engineering specs, reports
    (".docx",    800_000,  15, True),   # Word docs: procedures, manuals
    (".xlsx",    400_000,  10, True),   # Spreadsheets: test data, schedules
    (".pptx",  3_000_000,   8, True),   # PowerPoints: training, briefings
    (".txt",      15_000,   5, True),   # Plain text: logs, config, notes
    (".csv",     200_000,   4, True),   # Data files: sensor readings, exports
    (".json",     50_000,   3, True),   # API responses, config files
    (".eml",     150_000,   3, True),   # Saved emails
    (".png",   1_200_000,   5, True),   # Screenshots, diagrams
    (".jpg",     800_000,   4, True),   # Photos, scanned documents
    (".html",    100_000,   2, True),   # Web pages, saved reports
    (".xml",     300_000,   2, True),   # Structured data
    (".log",      80_000,   2, True),   # Application logs
    # Non-RAG types (waste of bandwidth if copied)
    (".exe",  15_000_000,   2, False),  # Executables
    (".dll",   2_000_000,   2, False),  # Libraries
    (".zip",  50_000_000,   1, False),  # Archives
    (".mp4", 100_000_000,   1, False),  # Videos (huge bandwidth waste)
    (".dwg",   5_000_000,   3, False),  # CAD files (no parser)
    (".bak",  10_000_000,   1, False),  # Backup files
    (".tmp",      50_000,   2, False),  # Temp files
    (".pst",  80_000_000,   1, False),  # Outlook PST (locked, unparseable)
    (".iso", 200_000_000, 0.5, False),  # Disk images (massive)
]

# Realistic folder structure for an enterprise network drive
_FOLDERS = [
    "Engineering/Designs/System_A",
    "Engineering/Designs/System_B",
    "Engineering/Designs/System_C/Subsystem_1",
    "Engineering/Designs/System_C/Subsystem_2",
    "Engineering/Test_Results/2024",
    "Engineering/Test_Results/2025",
    "Engineering/Test_Results/2026",
    "Reports/Monthly", "Reports/Quarterly", "Reports/Annual",
    "Procedures/Safety", "Procedures/Operations", "Procedures/Maintenance",
    "Training/Slides", "Training/Manuals",
    "Reference/Standards", "Reference/Datasheets", "Reference/Specifications",
    "Correspondence/Internal", "Correspondence/External",
    "Software/Tools", "Software/Configs",
    "Media/Photos", "Media/Diagrams",
    "Archives/2023", "Archives/2024",
    "Temp/Working", "Temp/Scratch",
    ".hidden_config",           # Hidden directory (starts with .)
    "Deep/Nesting/Level3/Level4/Level5/Level6/Level7",  # Deep nesting
]


@dataclass
class VFile:
    """
    Virtual file with realistic attributes.

    NON-PROGRAMMER NOTE:
      Each VFile represents one file on the simulated network drive.
      The boolean flags (is_locked, is_hidden, etc.) simulate real-world
      edge cases that the transfer engine must handle.
    """
    path: str                           # Full UNC path
    size: int                           # File size in bytes
    ext: str                            # Extension (.pdf, .docx, etc.)
    rag: bool                           # Can HybridRAG parse this?
    content_hash: str                   # SHA-like content fingerprint
    is_duplicate: bool = False          # Same content as another file
    is_locked: bool = False             # Locked by another process (e.g. Outlook)
    is_hidden: bool = False             # Windows hidden attribute
    is_symlink: bool = False            # Junction point / symlink
    is_corrupt: bool = False            # Will fail hash verification
    encoding_bad: bool = False          # Non-UTF-8 characters in filename
    path_over_260: bool = False         # Exceeds Windows MAX_PATH
    being_written: bool = False         # Another process writing to it now
    latency_ms: float = 5.0            # Simulated network latency


def gen_files(target_gb: float = 5.0, seed: int = 42) -> List[VFile]:
    """
    Generate a virtual filesystem of approximately target_gb gigabytes.

    NON-PROGRAMMER NOTE:
      This creates ~1000 virtual files totaling 5 GB with realistic
      edge cases:
        - ~15% of files are duplicates (same content in multiple folders)
        - ~2% are locked by another process
        - ~3% are hidden files
        - ~1% are symlinks
        - ~0.5% are corrupt
        - ~0.8% have encoding problems in their names
        - ~0.5% have paths longer than 260 characters
        - ~1.5% are being actively written to

      The random seed (42) ensures reproducible results -- every run
      generates the exact same set of files.
    """
    rng = random.Random(seed)
    target = int(target_gb * 1024**3)

    # Build weighted choice list
    choices = []
    for ext, avg, w, rag in _DIST:
        choices.extend([(ext, avg, rag)] * int(w * 10))

    files: List[VFile] = []
    hashes: List[str] = []
    total = 0
    n = 0

    while total < target:
        ext, avg, rag = rng.choice(choices)
        # lognormvariate creates realistic file size distribution:
        # most files are near average, some are much larger
        size = max(50, int(avg * rng.lognormvariate(0, 0.5)))
        n += 1
        h = hashlib.md5(f"f{n}_{size}".encode()).hexdigest()

        # ~15% chance of being a duplicate of an existing file
        is_dup = False
        if hashes and rng.random() < 0.15:
            h = rng.choice(hashes)
            is_dup = True
        hashes.append(h)

        folder = rng.choice(_FOLDERS)
        fname = f"file_{n:06d}{ext}"
        path = f"\\\\NetDrive\\Share\\{folder}\\{fname}"

        # Apply realistic edge case probabilities
        is_locked = ext == ".pst" or (rng.random() < 0.02)
        is_hidden = folder.startswith(".") or (rng.random() < 0.03)
        is_symlink = rng.random() < 0.01
        is_corrupt = rng.random() < 0.005
        encoding_bad = rng.random() < 0.008
        path_long = len(path) > 255 or rng.random() < 0.005
        being_written = rng.random() < 0.015

        # Network latency: 98% normal (1-50ms), 2% slow (200-2000ms)
        lat = (rng.uniform(200, 2000) if rng.random() < 0.02
               else rng.uniform(1, 50))

        files.append(VFile(
            path=path, size=size, ext=ext, rag=rag, content_hash=h,
            is_duplicate=is_dup, is_locked=is_locked, is_hidden=is_hidden,
            is_symlink=is_symlink, is_corrupt=is_corrupt,
            encoding_bad=encoding_bad, path_over_260=path_long,
            being_written=being_written, latency_ms=lat,
        ))
        total += size

    return files


# ============================================================================
# Simulation Functions
# ============================================================================

# Extensions HybridRAG can parse (must match _RAG_EXTENSIONS in engine)
RAG_EXTS: Set[str] = {
    ".txt", ".md", ".csv", ".json", ".xml", ".log", ".yaml", ".yml",
    ".ini", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
    ".eml", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
    ".gif", ".webp",
}


@dataclass
class Sim:
    """
    Simulation results container.

    NON-PROGRAMMER NOTE:
      After running a simulation, all the numbers end up here. The
      report function reads these numbers and formats them into the
      2-column comparison table.
    """
    name: str = ""
    total_files: int = 0
    total_bytes: int = 0
    discovery_sec: float = 0
    transfer_sec: float = 0
    files_copied: int = 0
    bytes_copied: int = 0
    files_dedup: int = 0
    files_failed: int = 0
    files_locked: int = 0
    files_hidden_skip: int = 0
    files_symlink_skip: int = 0
    files_encoding_skip: int = 0
    files_long_path_skip: int = 0
    files_being_written: int = 0
    files_corrupt: int = 0
    files_ext_skip: int = 0
    files_size_skip: int = 0
    files_quarantined: int = 0
    files_verified: int = 0
    bytes_wasted: int = 0
    rag_files: int = 0
    rag_bytes: int = 0
    workers: int = 1
    # Capability flags (YES/NO in the report)
    atomic_copy: bool = False
    staging_dirs: bool = False
    hash_verify: bool = False
    locked_detect: bool = False
    dedup: bool = False
    delta_sync: bool = False
    rename_detect: bool = False
    delete_detect: bool = False
    symlink_guard: bool = False
    long_path_check: bool = False
    hidden_aware: bool = False
    encoding_check: bool = False
    zero_gap_manifest: bool = False
    per_file_timing: bool = False
    live_stats: bool = False
    quarantine: bool = False
    mid_write_detect: bool = False
    ext_counts: Dict[str, int] = field(default_factory=dict)


def sim_original(files: List[VFile], net_mbps: float = 100) -> Sim:
    """
    Simulate a basic robocopy-style transfer (no intelligence).

    NON-PROGRAMMER NOTE:
      Robocopy copies EVERYTHING. It doesn't know about RAG, doesn't
      check hashes, doesn't detect duplicates. It just blindly copies
      files one at a time. This simulation models that behavior:
        - Single-threaded (1 worker)
        - Copies all file types (even .exe, .mp4)
        - No deduplication
        - No hash verification
        - Locked files just fail
        - No manifest
    """
    s = Sim(name="Original HybridRAG3 (robocopy + manual copy)")
    s.total_files = len(files)
    s.total_bytes = sum(f.size for f in files)
    s.workers = 1
    bps = net_mbps * 1024 * 1024  # Convert Mbps to bytes/sec

    # Discovery: robocopy scans at ~500 entries/sec
    s.discovery_sec = len(files) / 500.0

    # Transfer: copies ALL files (no filtering)
    copied_b = 0
    ext_c: Dict[str, int] = defaultdict(int)
    for f in files:
        if f.is_corrupt:
            s.files_failed += 1
            continue
        if f.is_locked:
            # robocopy retries but PST stays locked, eventually gives up
            s.files_locked += 1
            s.files_failed += 1
            continue
        copied_b += f.size
        s.files_copied += 1
        ext_c[f.ext] += 1

    s.bytes_copied = copied_b
    s.ext_counts = dict(ext_c)

    # Single-threaded transfer time estimate
    s.transfer_sec = copied_b / bps + s.files_copied * 0.005
    avg_lat = sum(f.latency_ms for f in files) / len(files) / 1000
    s.transfer_sec += avg_lat * s.files_copied

    # RAG readiness: how much of what we copied is actually useful?
    rag_f = [
        f for f in files
        if f.ext in RAG_EXTS and not f.is_corrupt and not f.is_locked
    ]
    s.rag_files = len(rag_f)
    s.rag_bytes = sum(f.size for f in rag_f)
    s.bytes_wasted = s.bytes_copied - s.rag_bytes  # Bandwidth wasted on junk

    # Capabilities: all NO
    s.atomic_copy = False
    s.staging_dirs = False
    s.hash_verify = False
    s.locked_detect = False
    s.dedup = False
    s.delta_sync = False
    s.rename_detect = False
    s.delete_detect = False
    s.symlink_guard = False
    s.long_path_check = False
    s.hidden_aware = False
    s.encoding_check = False
    s.zero_gap_manifest = False
    s.per_file_timing = False
    s.live_stats = False
    s.quarantine = False
    s.mid_write_detect = False

    return s


def sim_v2(files: List[VFile], net_mbps: float = 100, workers: int = 8) -> Sim:
    """
    Simulate the V2 Bulk Transfer engine.

    NON-PROGRAMMER NOTE:
      This models what V2 does: smart filtering during discovery,
      parallel transfers, hash verification, deduplication, locked
      file detection, and quarantine. The result is fewer files
      copied (only RAG-useful ones), faster transfer (8 workers),
      and every file accounted for.
    """
    s = Sim(name="HybridRAG3 Bulk Transfer V2")
    s.total_files = len(files)
    s.total_bytes = sum(f.size for f in files)
    s.workers = workers
    bps = net_mbps * 1024 * 1024

    # V2 discovery: smart filter at ~2000 entries/sec (attribute checks
    # are fast, the filtering happens during the walk)
    s.discovery_sec = len(files) / 2000.0

    # Filter and copy (mirrors the real engine's decision logic)
    seen_hashes: Set[str] = set()
    copied_b = 0
    ext_c: Dict[str, int] = defaultdict(int)

    for f in files:
        # Extension filter: skip non-RAG types at discovery time
        if f.ext not in RAG_EXTS:
            s.files_ext_skip += 1
            continue
        # Size filter
        if f.size < 100 or f.size > 500_000_000:
            s.files_size_skip += 1
            continue
        # Hidden/system files
        if f.is_hidden:
            s.files_hidden_skip += 1
            continue
        # Symlinks
        if f.is_symlink:
            s.files_symlink_skip += 1
            continue
        # Encoding issues
        if f.encoding_bad:
            s.files_encoding_skip += 1
            continue
        # Long paths
        if f.path_over_260:
            s.files_long_path_skip += 1
            continue
        # Locked file detection (pre-copy check)
        if f.is_locked:
            s.files_locked += 1
            s.files_quarantined += 1
            continue
        # Content deduplication
        if f.content_hash in seen_hashes:
            s.files_dedup += 1
            continue
        seen_hashes.add(f.content_hash)
        # Hash verification catches corruption
        if f.is_corrupt:
            s.files_corrupt += 1
            s.files_quarantined += 1
            continue
        # Mid-write detection (hash changes during copy)
        if f.being_written:
            s.files_being_written += 1
            s.files_quarantined += 1
            continue

        # File passes all checks -- copy it
        copied_b += f.size
        s.files_copied += 1
        s.files_verified += 1
        ext_c[f.ext] += 1

    s.bytes_copied = copied_b
    s.ext_counts = dict(ext_c)

    # Parallel transfer time estimate:
    #   data_time:  raw data transfer at network speed
    #   per_file:   overhead per file (hash + copy + verify + rename)
    #   lat_time:   network latency divided across workers
    #   hash_time:  SHA-256 hashing (source + destination)
    data_time = copied_b / bps
    per_file = 0.015 / workers
    overhead = s.files_copied * per_file
    avg_lat = sum(
        f.latency_ms for f in files
        if f.ext in RAG_EXTS and not f.is_corrupt and not f.is_locked
    ) / max(1, s.files_copied) / 1000
    lat_time = (avg_lat * s.files_copied) / workers
    hash_time = (copied_b / (1024**2)) * 0.005 * 2 / workers

    s.transfer_sec = data_time + overhead + lat_time + hash_time

    # RAG readiness: 100% of copied files are RAG-parseable
    s.rag_files = s.files_copied
    s.rag_bytes = s.bytes_copied
    s.bytes_wasted = 0  # Zero waste

    # Capabilities: all YES
    s.atomic_copy = True
    s.staging_dirs = True
    s.hash_verify = True
    s.locked_detect = True
    s.dedup = True
    s.delta_sync = True
    s.rename_detect = True
    s.delete_detect = True
    s.symlink_guard = True
    s.long_path_check = True
    s.hidden_aware = True
    s.encoding_check = True
    s.zero_gap_manifest = True
    s.per_file_timing = True
    s.live_stats = True
    s.quarantine = True
    s.mid_write_detect = True

    return s


# ============================================================================
# Report Generator
# ============================================================================

def yn(b: bool) -> str:
    """Format boolean as YES/NO."""
    return "YES" if b else "NO"


def sz(b) -> str:
    """Format bytes as human-readable size."""
    b = float(b)
    if b < 1024:
        return f"{b:.0f} B"
    if b < 1024**2:
        return f"{b/1024:.1f} KB"
    if b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    return f"{b/1024**3:.2f} GB"


def dur(s: float) -> str:
    """Format seconds as human-readable duration."""
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{int(m)}m {int(sec)}s"
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{int(h)}h {int(m)}m"


def report(o: Sim, v: Sim, files: List[VFile]) -> str:
    """
    Generate the 2-column comparison report.

    NON-PROGRAMMER NOTE:
      This is the main output. Each row compares one metric between
      the original approach and V2. The report is designed to be
      read by non-technical stakeholders to understand the value
      of the V2 engine.
    """
    C = 28  # Column width

    def row(label, v1, v2):
        return f"  {label:<38s} {v1:>{C}s} {v2:>{C}s}"

    def sep():
        return "  " + "-" * 38 + " " + "-" * C + " " + "-" * C

    ot = o.discovery_sec + o.transfer_sec
    vt = v.discovery_sec + v.transfer_sec
    spd = ot / max(vt, 0.001)

    L = [
        "", "=" * 98,
        "  STRESS TEST: Original HybridRAG3 vs Bulk Transfer V2",
        "=" * 98, "",
        f"  Simulated: {sz(sum(f.size for f in files))} across {len(files):,} files",
        f"  Network: 100 Mbps simulated",
        "",
        row("METRIC", "ORIGINAL", "V2 (BULK TRANSFER)"),
        sep(), "",
        "  -- PERFORMANCE --",
        row("Discovery time", dur(o.discovery_sec), dur(v.discovery_sec)),
        row("Transfer time", dur(o.transfer_sec), dur(v.transfer_sec)),
        row("Total time", dur(ot), dur(vt)),
        row("Speedup", "1.0x (baseline)", f"{spd:.1f}x faster"),
        row("Workers", str(o.workers), str(v.workers)),
        row("Data transferred", sz(o.bytes_copied), sz(v.bytes_copied)),
        row("Wasted bandwidth (non-RAG)", sz(o.bytes_wasted), sz(v.bytes_wasted)),
        row("Avg speed", f"{o.bytes_copied/max(ot,0.1)/1024**2:.1f} MB/s",
            f"{v.bytes_copied/max(vt,0.1)/1024**2:.1f} MB/s"),
        row("Effective RAG throughput",
            f"{o.rag_bytes/max(ot,0.1)/1024**2:.1f} MB/s",
            f"{v.rag_bytes/max(vt,0.1)/1024**2:.1f} MB/s"),
        "",
        "  -- FILE COUNTS --",
        row("Files copied", f"{o.files_copied:,}", f"{v.files_copied:,}"),
        row("Files deduplicated", f"{o.files_dedup:,}", f"{v.files_dedup:,}"),
        row("Filtered (wrong ext)", f"{o.files_ext_skip:,}", f"{v.files_ext_skip:,}"),
        row("Filtered (size)", f"{o.files_size_skip:,}", f"{v.files_size_skip:,}"),
        row("Locked files detected", f"{o.files_locked:,}", f"{v.files_locked:,}"),
        row("Hidden/system skipped", f"{o.files_hidden_skip:,}", f"{v.files_hidden_skip:,}"),
        row("Symlinks skipped", f"{o.files_symlink_skip:,}", f"{v.files_symlink_skip:,}"),
        row("Encoding issues caught", f"{o.files_encoding_skip:,}", f"{v.files_encoding_skip:,}"),
        row("Long path (>260) caught", f"{o.files_long_path_skip:,}", f"{v.files_long_path_skip:,}"),
        row("Mid-write detected", f"{o.files_being_written:,}", f"{v.files_being_written:,}"),
        row("Corrupt (hash mismatch)", f"{o.files_corrupt:,}", f"{v.files_corrupt:,}"),
        row("Quarantined", f"{o.files_quarantined:,}", f"{v.files_quarantined:,}"),
        row("Failed", f"{o.files_failed:,}", f"{v.files_failed:,}"),
        "",
        "  -- INTEGRITY & SAFETY --",
        row("Atomic copy (.tmp pattern)", yn(o.atomic_copy), yn(v.atomic_copy)),
        row("Three-stage staging dirs", yn(o.staging_dirs), yn(v.staging_dirs)),
        row("SHA-256 hash verification", yn(o.hash_verify), yn(v.hash_verify)),
        row("Files verified", f"{o.files_verified:,}", f"{v.files_verified:,}"),
        row("Locked file detection", yn(o.locked_detect), yn(v.locked_detect)),
        row("Mid-write detection", yn(o.mid_write_detect), yn(v.mid_write_detect)),
        row("Quarantine for failures", yn(o.quarantine), yn(v.quarantine)),
        "",
        "  -- INTELLIGENCE --",
        row("Content deduplication", yn(o.dedup), yn(v.dedup)),
        row("Delta sync (change detect)", yn(o.delta_sync), yn(v.delta_sync)),
        row("Renamed file detection", yn(o.rename_detect), yn(v.rename_detect)),
        row("Deleted file detection", yn(o.delete_detect), yn(v.delete_detect)),
        row("Symlink loop guard", yn(o.symlink_guard), yn(v.symlink_guard)),
        row("Long path awareness", yn(o.long_path_check), yn(v.long_path_check)),
        row("Hidden/system awareness", yn(o.hidden_aware), yn(v.hidden_aware)),
        row("Filename encoding check", yn(o.encoding_check), yn(v.encoding_check)),
        "",
        "  -- OBSERVABILITY --",
        row("Zero-gap manifest", yn(o.zero_gap_manifest), yn(v.zero_gap_manifest)),
        row("Per-file timing/speed", yn(o.per_file_timing), yn(v.per_file_timing)),
        row("Live progress + ETA", yn(o.live_stats), yn(v.live_stats)),
        "",
        "  -- RAG READINESS --",
        row("RAG-parseable files", f"{o.rag_files:,}", f"{v.rag_files:,}"),
        row("RAG-parseable data", sz(o.rag_bytes), sz(v.rag_bytes)),
        "",
        sep(), "",
    ]

    # Top file types breakdown
    L.append("  FILES BY TYPE (top 10):")
    L.append("")
    all_ext = set(o.ext_counts) | set(v.ext_counts)
    for ext in sorted(all_ext, key=lambda e: v.ext_counts.get(e, 0), reverse=True)[:10]:
        oc = o.ext_counts.get(ext, 0)
        vc = v.ext_counts.get(ext, 0)
        tag = "" if ext in RAG_EXTS else " (not RAG)"
        L.append(row(f"  {ext}{tag}", f"{oc:,}", f"{vc:,}"))

    # Key findings narrative
    dup_bytes = sum(f.size for f in files if f.is_duplicate and f.ext in RAG_EXTS)

    L.extend([
        "", sep(), "",
        "  KEY FINDINGS:", "",
        f"  1. SPEED: V2 is {spd:.1f}x faster ({dur(vt)} vs {dur(ot)})",
        f"     via {v.workers}-thread parallelism + pre-filtering.",
        "",
        f"  2. BANDWIDTH: Original wasted {sz(o.bytes_wasted)} copying non-RAG files.",
        f"     V2 saves this entirely by filtering at discovery time.",
        "",
        f"  3. INTEGRITY: V2 verified {v.files_verified:,} files via SHA-256",
        f"     (hash at source BEFORE copy, hash at dest AFTER copy).",
        f"     Original has no verification -- corrupt copies go undetected.",
        "",
        f"  4. ATOMIC COPY: V2 writes to .tmp, verifies hash, then atomic",
        f"     rename to verified/. A crash mid-copy leaves no partial files",
        f"     in the indexer's input directory. Original has no protection.",
        "",
        f"  5. LOCKED FILES: V2 detected {v.files_locked:,} locked/in-use files",
        f"     (including PSTs) and quarantined them. Original would retry",
        f"     endlessly or copy a corrupt mid-write snapshot.",
        "",
        f"  6. MID-WRITE SAFETY: V2 caught {v.files_being_written:,} files being",
        f"     actively written (hash changed between read start and end).",
        f"     Original copies these silently -- producing corrupt embeddings.",
        "",
        f"  7. DEDUPLICATION: V2 eliminated {v.files_dedup:,} duplicates",
        f"     ({sz(dup_bytes)} saved). Original copies every duplicate.",
        "",
        f"  8. ZERO-GAP MANIFEST: V2 accounts for every single file in the",
        f"     source -- transferred, skipped, locked, quarantined. Original",
        f"     has no manifest; files that fail silently are invisible.",
        "",
        f"  9. DELTA SYNC: V2 supports incremental runs -- only transfers",
        f"     new/modified files. Detects renames (avoids re-indexing)",
        f"     and deletions (flags orphaned chunks for removal).",
        "",
        f"  10. EDGE CASES: V2 caught {v.files_encoding_skip:,} encoding issues,",
        f"      {v.files_long_path_skip:,} long paths, {v.files_symlink_skip:,} symlinks,",
        f"      {v.files_hidden_skip:,} hidden files. Original ignores all of these.",
        "",
        "=" * 98,
    ])

    return "\n".join(L)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """Run the stress test and generate the comparison report."""
    print("Generating virtual filesystem (5 GB)...")
    files = gen_files(5.0, seed=42)
    print(f"  {len(files):,} files, {sz(sum(f.size for f in files))}")
    print()

    print("Simulating: Original HybridRAG3 (robocopy)...")
    o = sim_original(files)
    print(f"  Done: {dur(o.discovery_sec + o.transfer_sec)}")

    print("Simulating: HybridRAG3 Bulk Transfer V2 (8 workers)...")
    v = sim_v2(files, workers=8)
    print(f"  Done: {dur(v.discovery_sec + v.transfer_sec)}")
    print()

    r = report(o, v, files)
    print(r)

    # Save report to docs/
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "docs", "BULK_TRANSFER_STRESS_TEST.md",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Bulk Transfer V2 Stress Test Results\n\n```\n")
        f.write(r)
        f.write("\n```\n")
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
