# ============================================================================
# HybridRAG — Retriever (src/core/retriever.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   This is the "search engine" of HybridRAG. When a user types a question,
#   the Retriever finds the most relevant chunks from the indexed documents.
#
#   It supports three search strategies:
#     1. Vector search    — finds chunks whose *meaning* is similar to the query
#     2. Keyword search   — finds chunks that contain the query's *exact words*
#     3. Hybrid search    — combines both using Reciprocal Rank Fusion (RRF)
#
#   Hybrid is the default and usually best: vector catches paraphrases and
#   synonyms, while keyword catches exact terms like part numbers or acronyms
#   that vector search might miss.
#
# KEY DESIGN DECISIONS:
#
#   1. Reciprocal Rank Fusion (RRF) for hybrid search
#      WHY: We have two ranked lists (vector hits and keyword hits) that use
#      completely different scoring scales. Vector scores are cosine similarity
#      (0.0–1.0), keyword scores are BM25 weights (unbounded). RRF sidesteps
#      this by only using *rank position*, not raw scores. A chunk ranked #1
#      in both lists gets a higher combined score than one ranked #1 in only
#      one list. This is the same algorithm used by Elasticsearch and other
#      production search engines.
#
#   2. The "* 30" display score scaling
#      WHY: RRF scores are tiny fractions (e.g., 0.016 for a top result).
#      The min_score threshold (default 0.20) is calibrated for cosine
#      similarity (0.0–1.0 range). Multiplying RRF by 30 rescales it into
#      roughly the same 0.0–1.0 range so the same min_score works for both
#      hybrid and vector-only modes. We then cap at 1.0.
#
#   3. Optional cross-encoder reranker
#      WHY: The initial retrieval (vector or hybrid) is fast but approximate.
#      A cross-encoder reads the full query + chunk text together and produces
#      a much more accurate relevance score, but it's ~100x slower. So we
#      use it as a second pass: retrieve 20 candidates fast, then rerank
#      to find the best 5. Disabled by default (needs extra model download).
#
#   4. Lexical boost for vector-only mode
#      WHY: When hybrid mode is off, pure vector search sometimes ranks a
#      chunk high even though none of the query's actual words appear in it.
#      The lexical boost adds a small score bonus (+0.02 per matching word,
#      capped at lex_boost) if query terms appear in the first 250 characters
#      of the chunk. This is a lightweight fallback — hybrid mode with FTS5
#      is the better solution and makes this unnecessary.
#
#   ALTERNATIVES CONSIDERED:
#   - Linear score combination (0.7*vector + 0.3*keyword): requires careful
#     weight tuning per dataset. RRF is parameter-free and robust.
#   - BM25 from scratch: SQLite FTS5 already implements BM25 natively.
#   - Always-on reranker: too slow for interactive use on CPU laptops.
#     We keep it opt-in for batch/evaluation workflows.
# ============================================================================

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .vector_store import VectorStore
from .embedder import Embedder


# ---------------------------------------------------------------------------
# Data class: one search result
# ---------------------------------------------------------------------------

@dataclass
class SearchHit:
    """
    One search result returned by the Retriever.

    Fields:
      score       — relevance score (0.0 to 1.0, higher = more relevant)
      source_path — full file path of the original document
      chunk_index — which chunk within that document (0-based)
      text        — the actual text content of the chunk
    """
    score: float
    source_path: str
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Searches the vector store for chunks relevant to a user's query.

    Designed for:
    - Sub-200ms search on a laptop with 10,000+ chunks
    - Hybrid search combining semantic + keyword matching
    - Tunable via config (top_k, min_score, hybrid on/off, etc.)
    - Optional cross-encoder reranking for higher precision
    """

    def __init__(self, vector_store, embedder, config):
        """
        Parameters
        ----------
        vector_store : VectorStore
            The storage backend containing chunks and embeddings.

        embedder : Embedder
            Converts query text into an embedding vector.

        config : Config
            Master configuration. Retrieval settings come from config.retrieval.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config

        # --- Pull retrieval settings from config (with safe defaults) ---
        # getattr chains handle the case where config.retrieval doesn't exist
        retrieval = getattr(config, "retrieval", None)

        # top_k: How many final results to return to the LLM.
        # More results = more context but slower LLM response and higher cost.
        self.top_k = int(getattr(retrieval, "top_k", 8)) if retrieval else 8

        # block_rows: How many embedding rows to scan per vector search.
        # Higher = more thorough but slower. 25,000 covers most indexes.
        self.block_rows = int(getattr(retrieval, "block_rows", 25000)) if retrieval else 25000

        # min_score: Minimum relevance score to include a result.
        # Chunks scoring below this are filtered out. Prevents low-quality
        # results from polluting the LLM's context window.
        self.min_score = float(getattr(retrieval, "min_score", 0.20)) if retrieval else 0.20

        # lex_boost: Maximum score bonus for keyword matches in vector-only mode.
        # Only used when hybrid_search is False. See design decision #4 above.
        self.lex_boost = float(getattr(retrieval, "lex_boost", 0.06)) if retrieval else 0.06

        # hybrid_search: Whether to combine vector + keyword search.
        # True (default) gives best results for most engineering documents.
        self.hybrid_search = bool(getattr(retrieval, "hybrid_search", True)) if retrieval else True

        # rrf_k: Smoothing constant for Reciprocal Rank Fusion.
        # Standard value is 60 (from the original RRF paper). Higher values
        # reduce the advantage of being ranked #1 vs #2. Don't change unless
        # you have a specific reason.
        self.rrf_k = int(getattr(retrieval, "rrf_k", 60)) if retrieval else 60

        # --- Reranker settings (optional, disabled by default) ---
        self.reranker_enabled = bool(getattr(retrieval, "reranker_enabled", False)) if retrieval else False
        self.reranker_model_name = str(getattr(retrieval, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")) if retrieval else "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # reranker_top_n: How many candidates to retrieve before reranking.
        # Must be >= top_k. We fetch more candidates so the reranker has
        # a larger pool to pick the best results from.
        self.reranker_top_n = int(getattr(retrieval, "reranker_top_n", 20)) if retrieval else 20

        # --- Environment variable overrides (machine-specific tuning) ---
        env_block = os.getenv("HYBRIDRAG_RETRIEVAL_BLOCK_ROWS")
        if env_block:
            self.block_rows = int(env_block)
        env_min = os.getenv("HYBRIDRAG_MIN_SCORE")
        if env_min:
            self.min_score = float(env_min)

        # Lazy-loaded reranker model (only loaded when first needed)
        self._reranker = None

    # ------------------------------------------------------------------
    # Public API — this is what query_engine.py calls
    # ------------------------------------------------------------------

    def search(self, query):
        """
        Search for chunks relevant to the query.

        Pipeline:
          1. Choose search strategy (hybrid or vector-only)
          2. Retrieve candidates (more than top_k if reranker is on)
          3. Optionally rerank with cross-encoder
          4. Filter by min_score
          5. Trim to top_k results

        Returns a list of SearchHit objects, sorted by score descending.
        """
        # If reranker is on, we fetch more candidates so it has a bigger
        # pool to rerank from. Otherwise, just fetch top_k directly.
        candidate_k = self.reranker_top_n if self.reranker_enabled else self.top_k

        # --- Step 1: Retrieve candidates ---
        if self.hybrid_search:
            hits = self._hybrid_search(query, candidate_k)
        else:
            hits = self._vector_search(query, candidate_k)

        # --- Step 2: Optional reranking ---
        if self.reranker_enabled and len(hits) > 0:
            hits = self._rerank(query, hits)

        # --- Step 3: Filter by minimum score ---
        hits = [h for h in hits if h.score >= self.min_score]

        # --- Step 4: Trim to final top_k ---
        hits = hits[:self.top_k]

        return hits

    # ------------------------------------------------------------------
    # Hybrid search (vector + keyword via RRF)
    # ------------------------------------------------------------------

    def _hybrid_search(self, query, candidate_k):
        """
        Run both vector search and keyword search, then merge results
        using Reciprocal Rank Fusion (RRF).

        This is the default and recommended search mode.
        """
        # Vector search: embed the query, find similar embeddings
        q_vec = self.embedder.embed_query(query)
        vector_hits = self.vector_store.search(q_vec, top_k=candidate_k, block_rows=self.block_rows)

        # Keyword search: use SQLite FTS5 full-text index
        fts_hits = self.vector_store.fts_search(query, top_k=candidate_k)

        # Merge the two ranked lists using RRF
        return self._reciprocal_rank_fusion(vector_hits, fts_hits)

    def _reciprocal_rank_fusion(self, vector_hits, fts_hits):
        """
        Merge two ranked lists into one using Reciprocal Rank Fusion.

        HOW RRF WORKS (plain English):
          Each result gets a score based on its rank position:
            rrf_score = 1 / (k + rank + 1)

          where k=60 (smoothing constant) and rank starts at 0.

          So the #1 result gets 1/61 = 0.0164, #2 gets 1/62 = 0.0161, etc.

          If a chunk appears in BOTH lists, its RRF scores are added together.
          This means a chunk ranked well in both vector AND keyword search
          will float to the top, even if it wasn't #1 in either list alone.

        WHY THIS WORKS:
          Vector search finds "digisonde ionospheric measurements" when you
          ask about "ionogram data collection". Keyword search finds chunks
          containing the exact word "ionogram". RRF combines both signals
          so you get the best of both worlds.
        """
        # Dictionary to accumulate RRF scores. Key = (source_path, chunk_index)
        combined = {}

        # Score each vector hit by its rank position
        for rank, hit in enumerate(vector_hits):
            key = (hit["source_path"], hit["chunk_index"])
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if key not in combined:
                combined[key] = {
                    "rrf_score": 0.0,
                    "vector_score": hit.get("score", 0.0),
                    "source_path": hit["source_path"],
                    "chunk_index": hit["chunk_index"],
                    "text": hit["text"],
                }
            combined[key]["rrf_score"] += rrf_score

        # Score each keyword hit by its rank position.
        # If the same chunk was already added from vector hits, its
        # RRF score increases (that's the whole point of fusion).
        for rank, hit in enumerate(fts_hits):
            key = (hit["source_path"], hit["chunk_index"])
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if key not in combined:
                combined[key] = {
                    "rrf_score": 0.0,
                    "vector_score": hit.get("score", 0.0),
                    "source_path": hit["source_path"],
                    "chunk_index": hit["chunk_index"],
                    "text": hit["text"],
                }
            combined[key]["rrf_score"] += rrf_score

        # Sort by combined RRF score (highest first)
        sorted_results = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Convert to SearchHit objects with a display-friendly score
        hits = []
        for item in sorted_results:
            # Raw RRF scores are tiny (0.01–0.03 range). We need a score in
            # the 0.0–1.0 range so min_score filtering works consistently
            # whether we're in hybrid mode or vector-only mode.
            #
            # The "* 30" scaling maps the typical RRF range (~0.016 for a
            # top-ranked-in-both-lists result) into roughly 0.5–0.9.
            # We take the max of the vector score and the scaled RRF score,
            # then cap at 1.0 to keep the display clean.
            display_score = max(item["vector_score"], item["rrf_score"] * 30)
            hits.append(SearchHit(
                score=min(display_score, 1.0),
                source_path=item["source_path"],
                chunk_index=item["chunk_index"],
                text=item["text"],
            ))
        return hits

    # ------------------------------------------------------------------
    # Vector-only search (fallback when hybrid is disabled)
    # ------------------------------------------------------------------

    def _vector_search(self, query, candidate_k):
        """
        Pure vector (semantic) search with optional lexical boost.

        Used when hybrid_search is set to False in config. Less accurate
        than hybrid for queries containing specific terms or part numbers,
        but simpler and faster.
        """
        # Embed the query into a 384-dimensional vector
        q_vec = self.embedder.embed_query(query)

        # Find the closest chunk embeddings by cosine similarity
        raw_hits = self.vector_store.search(q_vec, top_k=candidate_k, block_rows=self.block_rows)

        # Extract query terms for the optional lexical boost
        q_terms = self._query_terms(query)

        # Convert raw dict results to SearchHit objects, applying boost
        hits = []
        for h in raw_hits:
            base = float(h.get("score", 0.0))
            text = str(h.get("text", "") or "")
            # Add a small bonus if query words appear in the chunk's opening text
            boosted = base + self._lexical_boost(text, q_terms)
            hits.append(SearchHit(
                score=boosted,
                source_path=str(h.get("source_path", "")),
                chunk_index=int(h.get("chunk_index", 0)),
                text=text,
            ))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Cross-encoder reranking (optional second pass)
    # ------------------------------------------------------------------

    def _rerank(self, query, hits):
        """
        Re-score hits using a cross-encoder model for higher accuracy.

        A cross-encoder reads the query and chunk text TOGETHER (not
        separately like the bi-encoder used for initial retrieval).
        This produces much more accurate relevance scores but is ~100x
        slower, so we only run it on the top N candidates.

        The raw cross-encoder output is a logit (can be any number).
        We convert it to a 0–1 probability using the sigmoid function:
          sigmoid(x) = 1 / (1 + e^(-x))
        """
        # Lazy-load the reranker model on first use
        if self._reranker is None:
            self._reranker = self._load_reranker()
        if self._reranker is None:
            return hits  # Reranker unavailable, return original order

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, hit.text) for hit in hits]
        try:
            # Get raw logit scores from the cross-encoder
            scores = self._reranker.predict(pairs)
            # Convert logits to 0–1 probabilities using sigmoid
            # (2.718281828 is Euler's number "e")
            for hit, score in zip(hits, scores):
                hit.score = 1.0 / (1.0 + pow(2.718281828, -float(score)))
            # Re-sort by the new scores
            hits.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            import sys
            print(f"Reranker error: {e}", file=sys.stderr)
        return hits

    def _load_reranker(self):
        """Load the cross-encoder model. Returns None if unavailable."""
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(self.reranker_model_name)
        except ImportError:
            import sys
            print("Reranker requires sentence-transformers.", file=sys.stderr)
            return None
        except Exception as e:
            import sys
            print(f"Failed to load reranker: {e}", file=sys.stderr)
            return None

    # ------------------------------------------------------------------
    # Context building — format hits for the LLM prompt
    # ------------------------------------------------------------------

    def build_context(self, hits):
        """
        Format search hits into a text block that gets inserted into
        the LLM prompt as the "retrieved context".

        Each hit is labeled with its source, chunk number, and score
        so the LLM (and the user) can trace answers back to sources.
        """
        blocks = []
        for i, h in enumerate(hits, start=1):
            blocks.append(
                f"[Source {i}] {h.source_path} "
                f"(chunk {h.chunk_index}, score={h.score:.3f})\n{h.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def get_sources(self, hits):
        """
        Summarize which files contributed to the search results.

        Returns a list of dicts with:
          path          — file path
          chunks        — how many chunks came from this file
          avg_relevance — average score of those chunks

        Useful for showing "Sources used" in the UI.
        """
        if not hits:
            return []
        # Group hits by file path
        by_path = {}
        for h in hits:
            by_path.setdefault(h.source_path, []).append(h.score)
        # Build summary for each file
        out = []
        for path, scores in by_path.items():
            out.append({
                "path": path,
                "chunks": len(scores),
                "avg_relevance": sum(scores) / max(1, len(scores)),
            })
        return out

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _query_terms(self, query):
        """
        Extract searchable terms from the query.

        Splits on non-alphanumeric characters, lowercases everything,
        and drops very short words (under 3 chars) which are usually
        stop words like "a", "is", "of", etc.
        """
        terms = re.findall(r"[A-Za-z0-9]+", (query or "").lower())
        return [t for t in terms if len(t) >= 3]

    def _lexical_boost(self, chunk_text, q_terms):
        """
        Small score bonus when query terms appear in the chunk's opening text.

        Only used in vector-only mode (hybrid mode uses FTS5 instead).

        Checks the first 250 characters of the chunk (typically the heading
        or first sentence). Each matching term adds +0.02 to the score,
        capped at self.lex_boost (default 0.06).

        WHY ONLY 250 CHARS:
          A term appearing in the heading/first sentence is much more likely
          to indicate the chunk is *about* that topic, vs. a passing mention
          buried in paragraph 5.
        """
        if not chunk_text or not q_terms:
            return 0.0
        head = chunk_text[:250].lower()
        matches = sum(1 for t in q_terms if t in head)
        if matches == 0:
            return 0.0
        return min(self.lex_boost, 0.02 * matches)
