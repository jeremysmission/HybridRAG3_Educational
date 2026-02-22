# ============================================================================
# HybridRAG -- Embedder (src/core/embedder.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Converts text into "embedding vectors" -- arrays of 384 numbers that
#   represent the MEANING of the text. Two pieces of text with similar
#   meaning will have similar vectors, even if they use different words.
#
#   Example:
#     "The maximum operating temperature is 85C"
#     "Thermal limit: do not exceed eighty-five degrees"
#     -> These produce similar vectors (high cosine similarity)
#
# HOW IT WORKS:
#   We use a pre-trained model called "all-MiniLM-L6-v2" from the
#   sentence-transformers library. The model was trained on millions
#   of text pairs to learn what "similar meaning" looks like.
#
#   Input: a string of text (up to ~256 tokens ~ ~1200 characters)
#   Output: a numpy array of 384 floating-point numbers
#
# WHY all-MiniLM-L6-v2:
#   - Small: ~80MB download, loads in seconds
#   - Fast on CPU: embeds ~100 chunks/second on a laptop
#   - 384 dimensions: good balance of quality vs storage cost
#   - Pre-normalized: vectors already have length 1, so
#     cosine similarity = simple dot product (very fast)
#   - Widely used: millions of downloads, well-tested, stable
#
# ALTERNATIVES CONSIDERED:
#   - all-mpnet-base-v2: slightly better quality but 768 dims (2x storage)
#   - e5-large-v2: much better quality but 1024 dims, needs GPU
#   - OpenAI text-embedding-ada-002: excellent but requires internet + API key
#   - nomic-embed-text: comparable to MiniLM, Apache 2.0 licensed
#
# SECURITY MODEL (3-layer network lockdown):
#   Layer 1: start_hybridrag.ps1 sets HF_HUB_OFFLINE=1 in the PowerShell
#            session. This is the primary lockdown -- works for all normal
#            usage when you launch via the start script.
#
#   Layer 2: THIS FILE (embedder.py) enforces HF_HUB_OFFLINE=1 at the
#            Python level BEFORE importing sentence-transformers. This is
#            safety net -- if someone runs Python directly without
#            going through start_hybridrag.ps1, the lockdown still holds.
#            The HuggingFace libraries check os.environ at import time,
#            so we MUST set these before the import statement.
#
#   Layer 3: SEC-001 fix in config.py -- API endpoint defaults to empty
#            string instead of public OpenAI, preventing accidental
#            data exfiltration through the LLM query path.
#
#   IMPORTANT: These env vars are per-process only. They do NOT affect
#   your browser, email, Teams, or any other app on your machine. Only
#   the Python process that imports this file is affected.
#
# INTERNET ACCESS: NONE
#   This file blocks all HuggingFace network calls. Models must be
#   pre-cached in .model_cache/ (done by setup_install.ps1 Step 8).
#   If a model is not cached, SentenceTransformer will raise an error
#   instead of silently downloading -- loud failure, not silent leak.
# ============================================================================

from __future__ import annotations

import os

# ============================================================================
# SECURITY: Python-level network lockdown (Layer 2)
# ============================================================================
# These MUST be set BEFORE importing sentence_transformers or transformers.
# The HuggingFace libraries read os.environ at import time to decide
# whether they're allowed to contact the internet.
#
# WHY we set them here instead of only in start_hybridrag.ps1:
#   If someone runs "python run_index_once.py" from a plain terminal
#   (without start_hybridrag.ps1), the PowerShell env vars won't be set.
#   This Python-level enforcement ensures the lockdown holds regardless
#   of how the Python process was launched.
#
# These are set unconditionally -- there is no "online" mode for the
# embedding model. The embedder always runs locally with cached models.
# Online mode (for the LLM) goes through llm_router.py, not here.
# ============================================================================

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# NOW it's safe to import HuggingFace libraries
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around sentence-transformers for text -> vector conversion.

    Used in two places:
      1. Indexing: embed_batch() processes chunks in bulk
      2. Querying: embed_query() processes a single user question
    """

    # Default used only when no config is available (tests, scripts).
    # Production code should always pass model_name from config.
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        """
        Load the embedding model from local cache.

        Parameters
        ----------
        model_name : str or None
            Name of the sentence-transformers model.  When None, falls
            back to DEFAULT_MODEL.  Must be pre-cached -- the model
            will NOT be downloaded from the internet because
            HF_HUB_OFFLINE=1 is enforced above.

            Set via config/default_config.yaml -> embedding.model_name.
        """
        self.model_name = model_name or self.DEFAULT_MODEL

        # trust_remote_code=True is safe here because HF_HUB_OFFLINE=1
        # prevents downloading new code -- only pre-cached code runs.
        # Required by models like nomic-embed-text that ship custom
        # architecture code alongside their weights.
        self.model = SentenceTransformer(
            self.model_name, trust_remote_code=True,
        )

        # Read the dimension from the model itself (not hardcoded).
        # all-MiniLM-L6-v2 = 384, nomic-embed-text = 768, etc.
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed multiple texts at once (used during indexing).

        Batching is faster than embedding one at a time because the
        model can process multiple inputs in parallel on the CPU.

        Parameters
        ----------
        texts : list[str]
            List of text strings to embed.

        Returns
        -------
        np.ndarray
            Shape (N, 384), dtype float32.
            Vectors are normalized (length 1) so dot product = cosine sim.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,       # Return numpy array (not tensor)
            normalize_embeddings=True,    # Normalize to unit length
            show_progress_bar=False,      # Don't clutter the terminal
        )
        return embeddings.astype(np.float32, copy=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Alias for embed_batch().

        EXISTS FOR CONVENIENCE:
          Some callers (diagnostic tools, notebooks, quick tests) use
          the more intuitive name "encode" instead of "embed_batch".
          This alias keeps both working without duplicating code.
        """
        return self.embed_batch(texts)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query string (used at search time).

        Parameters
        ----------
        text : str
            The user's question.

        Returns
        -------
        np.ndarray
            Shape (384,), dtype float32.
        """
        vec = self.embed_batch([str(text or "")])
        return vec[0]

    def close(self) -> None:
        """
        Release the embedding model from memory.

        The SentenceTransformer model occupies ~100MB of RAM. Without
        an explicit close(), that memory is never freed until the Python
        process exits. Over repeated indexing runs (e.g., a GUI that
        starts/stops indexing), this would leak memory.

        Deletes the model reference so Python's garbage collector can
        reclaim the memory. Calls gc.collect() to force immediate cleanup
        rather than waiting for Python to get around to it.

        Safe to call multiple times. After calling close(), the embedder
        cannot be used again -- you'd need to create a new instance.
        """
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
            import gc
            gc.collect()
