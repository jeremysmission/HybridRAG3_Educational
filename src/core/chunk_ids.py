# ============================================================================
# HybridRAG -- Chunk ID Generator (src/core/chunk_ids.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Creates a unique, repeatable ID for every chunk of text.
#
#   "Repeatable" means: if you run indexing twice on the same file,
#   the same chunks get the same IDs both times. This is called
#   "deterministic" -- the output is determined by the input, not by
#   randomness or timestamps.
#
# WHY THIS MATTERS:
#   Imagine you're indexing 10,000 files and your laptop crashes at
#   file #7,000. You restart. Without deterministic IDs, the indexer
#   would create DUPLICATE chunks for files #1 through #7,000 because
#   each run would generate new random IDs.
#
#   With deterministic IDs, the database says "INSERT OR IGNORE" -- 
#   it sees the same chunk_id already exists and skips it. No duplicates.
#   Indexing effectively resumes from where it left off.
#
# HOW IT WORKS:
#   We combine five pieces of information into a single string, then
#   hash it with SHA256:
#     1. File path (which file)
#     2. File modification time (which VERSION of the file)
#     3. Chunk start position (where in the file)
#     4. Chunk end position (where in the file)
#     5. Chunk text fingerprint (what the text actually says)
#
#   If ANY of these change, the chunk ID changes. This means:
#     - Edit a file -> new mtime -> new IDs -> file gets re-indexed
#     - Same file, same content -> same IDs -> safely skipped
#
# WHY SHA256 INSTEAD OF SIMPLER OPTIONS:
#   - UUID4: random, not deterministic (defeats the whole purpose)
#   - MD5: technically works but has known collision vulnerabilities
#   - Simple string concat: too long, not fixed-width, messy in SQL
#   - SHA256: 64-char hex string, deterministic, collision-proof,
#     built into Python (no extra dependencies)
#
# WHY NOT HASH THE ENTIRE FILE CONTENT (like xxhash)?
#   Reading entire files is slow on network drives. A 500MB PDF takes
#   seconds to hash. By using mtime (modification timestamp), we get
#   "did this file change?" detection for free -- the OS already tracks
#   it. The tradeoff is that mtime can be wrong in rare cases (copied
#   files, clock skew), but for practical use it's reliable enough.
# ============================================================================

from __future__ import annotations
import hashlib


def make_chunk_id(
    file_path: str,
    file_mtime_ns: int,
    chunk_start: int,
    chunk_end: int,
    chunk_text: str,
) -> str:
    """
    Create a deterministic chunk ID from file identity + position + content.

    Parameters
    ----------
    file_path : str
        Full path of the source file.
        Example: "D:\\HybridRAG2\\docs\\manual.pdf"

    file_mtime_ns : int
        File modification time in nanoseconds.
        Obtained from: Path("file.pdf").stat().st_mtime_ns
        This changes whenever the file is edited, triggering re-indexing.

    chunk_start : int
        Character offset where this chunk starts in the file's text.

    chunk_end : int
        Character offset where this chunk ends in the file's text.

    chunk_text : str
        The actual text content of the chunk.
        We only use the first 2000 characters for the fingerprint
        (performance: hashing 2000 chars is instant, hashing 200K is not).

    Returns
    -------
    str
        A 64-character hex string (SHA256 hash).
        Example: "a1b2c3d4e5f6..."
        Guaranteed unique for different inputs (collision-proof).
        Guaranteed identical for the same inputs (deterministic).
    """
    # Normalize the file path so it's consistent across runs:
    #   - Strip whitespace
    #   - Convert backslashes to forward slashes (Windows paths vary)
    #   - Lowercase (Windows is case-insensitive: "File.PDF" == "file.pdf")
    norm_path = file_path.strip().replace("\\", "/").lower()

    # Fingerprint the chunk text (first 2000 chars only, for speed)
    text_sample = (chunk_text or "")[:2000]
    text_fp = hashlib.sha256(
        text_sample.encode("utf-8", errors="ignore")
    ).hexdigest()

    # Combine all five pieces into one string, separated by pipes
    raw = f"{norm_path}|{file_mtime_ns}|{chunk_start}|{chunk_end}|{text_fp}"

    # Hash the combined string to get a fixed-width, unique ID
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()