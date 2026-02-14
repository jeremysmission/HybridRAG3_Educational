# ============================================================================
# HybridRAG — Chunker (src/core/chunker.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Splits a large block of text into smaller, overlapping "chunks" that
#   can be individually embedded and searched. This is a core step in any
#   RAG system — the LLM can only read so much text at once, so we need
#   to break documents into pieces and only send the relevant ones.
#
# WHY CHUNKING MATTERS:
#   A 500-page PDF might contain 2 million characters. The embedding model
#   (all-MiniLM-L6-v2) works best on passages of ~200–500 words. If you
#   feed it the entire document, the embedding is a vague average of
#   everything and matches nothing well. Small, focused chunks produce
#   sharp embeddings that match specific queries accurately.
#
# KEY DESIGN DECISIONS:
#
#   1. Smart boundary detection (paragraph > sentence > newline)
#      WHY: Splitting at exactly 1200 characters often cuts mid-sentence:
#        "The digisonde operates at frequenc|ies between 1 and 30 MHz."
#      Instead, we look for the best natural break point in the second
#      half of the chunk window:
#        - First choice: paragraph break (double newline)
#        - Second choice: sentence end (". ")
#        - Third choice: any newline
#        - Last resort: hard cut at chunk_size
#      This keeps each chunk as a coherent thought.
#
#   2. Overlapping chunks (default 200 chars)
#      WHY: If a key fact spans the boundary between two chunks, overlap
#      ensures it appears in full in at least one of them. Without overlap,
#      "The system uses 10 MHz" might get split as "The system uses" in
#      chunk N and "10 MHz for transmission" in chunk N+1, and neither
#      chunk alone answers "What frequency does the system use?"
#
#   3. Section heading prepend ("[SECTION] HEADING\n...")
#      WHY: Engineering documents have deep structure. A chunk from page 47
#      saying "Set the value to 5.0" is useless without knowing it's from
#      "3.2.1 Calibration Procedure". We look backward up to 2000 chars
#      to find the nearest heading and prepend it. This gives the LLM
#      context about where the chunk came from.
#
#   4. Heading detection heuristics
#      WHY: We don't have markdown or HTML structure in extracted PDF text.
#      Instead, we use three rules that catch most engineering doc headings:
#        - ALL CAPS lines (like "CALIBRATION PROCEDURE")
#        - Numbered sections (like "3.2.1 Signal Processing")
#        - Lines ending with ":" (like "Installation Steps:")
#      These aren't perfect, but they cover ~90% of real docs.
#
#   ALTERNATIVES CONSIDERED:
#   - Fixed-size chunks (no boundary detection): simpler but produces
#     broken sentences that confuse the LLM.
#   - Recursive/tree-based splitting (LangChain style): over-engineered
#     for our use case and adds a dependency.
#   - Sentence-level chunks: too small — each chunk needs enough context
#     for the embedding to be meaningful.
#   - Token-based splitting: requires a tokenizer, adds complexity, and
#     character-based is close enough for our embedding model.
# ============================================================================

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# Configuration for the chunker
# ---------------------------------------------------------------------------

@dataclass
class ChunkerConfig:
    """
    Settings that control how text is split into chunks.

    Fields:
      chunk_size     — target size of each chunk in characters (default 1200).
                       ~200–300 words, which is the sweet spot for
                       all-MiniLM-L6-v2 embeddings.
      overlap        — how many characters to repeat at the start of the
                       next chunk (default 200). Ensures facts near chunk
                       boundaries aren't lost.
      max_heading_len — maximum length of a line to consider as a heading
                       (default 160). Prevents long paragraphs from being
                       mistaken for headings.
    """
    chunk_size: int = 1200
    overlap: int = 200
    max_heading_len: int = 160


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """
    Splits text into overlapping chunks with smart boundary detection
    and section heading prepend.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : ChunkerConfig (or any object with chunk_size, overlap,
                 max_heading_len attributes)
        """
        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        self.max_heading_len = config.max_heading_len

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text):
        """
        Split text into overlapping chunks.

        Returns a list of strings, each one a chunk of text ready to
        be embedded. Empty/whitespace-only input returns an empty list.

        Algorithm (step by step):
          1. Start at position 0 in the text
          2. Look ahead chunk_size characters for the tentative end
          3. In the second half of that window, search backward for
             the best break point (paragraph > sentence > newline)
          4. Extract the chunk, strip whitespace
          5. Look backward for the nearest section heading and prepend
             it as "[SECTION] heading" if found
          6. Move the start position forward, overlapping by 'overlap' chars
          7. Repeat until we reach the end of the text
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # --- Step 2: Tentative end position ---
            end = min(start + self.chunk_size, text_len)

            # --- Step 3: Find the best break point ---
            # Only search for breaks if we're not at the end of the text
            if end < text_len:
                # Only look in the second half of the window to avoid
                # making chunks too small
                half = start + self.chunk_size // 2

                # First choice: paragraph break (double newline)
                # chr(10) is the newline character "\n"
                para = text.rfind(chr(10)+chr(10), half, end)
                if para != -1:
                    end = para + 2  # Include the double newline
                else:
                    # Second choice: sentence end (period + space)
                    sent = text.rfind(". ", half, end)
                    if sent != -1:
                        end = sent + 2  # Include the ". "
                    else:
                        # Third choice: any newline
                        nl = text.rfind(chr(10), half, end)
                        if nl != -1:
                            end = nl + 1  # Include the newline
                        # Last resort: hard cut at chunk_size (end stays as-is)

            # --- Step 4: Extract and clean the chunk ---
            chunk = text[start:end].strip()

            if chunk:
                # --- Step 5: Find and prepend the nearest section heading ---
                heading = self._find_heading(text, start)
                if heading and not chunk.startswith(heading):
                    # Prepend "[SECTION] heading" so the LLM knows context
                    chunk = "[SECTION] " + heading + chr(10) + chunk
                chunks.append(chunk)

            # If we've reached the end of the text, stop
            if end >= text_len:
                break

            # --- Step 6: Advance with overlap ---
            # Move start forward but keep 'overlap' chars of overlap
            # with the previous chunk. The max(..., start + 1) prevents
            # infinite loops if overlap >= chunk length.
            start = max(end - self.overlap, start + 1)

        return chunks

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _find_heading(self, text, pos):
        """
        Look backward from position 'pos' to find the nearest section heading.

        Searches up to 2000 characters before the current position.
        Returns the heading text, or empty string if none found.

        Heading detection rules (checked in order):
          1. ALL CAPS line longer than 3 chars
             Examples: "CALIBRATION PROCEDURE", "SYSTEM OVERVIEW"
          2. Numbered section line (starts with digits and dots)
             Examples: "3.2.1 Signal Processing", "1 Introduction"
          3. Line ending with ":" under 80 chars
             Examples: "Installation Steps:", "Requirements:"

        WHY ONLY THE FIRST NON-EMPTY LINE (searching backward):
          We check the nearest non-empty line above the chunk start.
          If it matches a heading pattern, great. If not, we stop —
          going further back risks picking up an unrelated heading
          from a completely different section.
        """
        # Don't search more than 2000 chars back
        search_start = max(0, pos - 2000)
        region = text[search_start:pos]

        # Split into lines and check from bottom up (nearest first)
        lines = region.split(chr(10))
        for line in reversed(lines):
            line = line.strip()
            # Skip blank lines
            if not line:
                continue

            # Check if this line looks like a heading
            if len(line) <= self.max_heading_len:
                # Rule 1: ALL CAPS (minimum 4 chars to avoid "OK", "N/A")
                if line.isupper() and len(line) > 3:
                    return line
                # Rule 2: Numbered section (e.g., "3.2.1 Title")
                if re.match(r"^\d+(\.\d+)*\s+", line):
                    return line
                # Rule 3: Ends with colon (e.g., "Procedure:")
                if line.endswith(":") and len(line) < 80:
                    return line

            # If the nearest non-empty line doesn't match any pattern,
            # stop searching. Don't keep going — the next line up is
            # probably body text from the previous paragraph.
            break

        return ""
