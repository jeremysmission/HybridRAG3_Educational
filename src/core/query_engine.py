# ============================================================================
# HybridRAG - Query Engine (src/core/query_engine.py)
# ============================================================================
# Orchestrate: search -> retrieve context -> call LLM -> audit log -> return result
#
# Step 4 Notes:
# - Retriever now uses a memory-mapped embedding matrix for fast/low-RAM search.
# - QueryEngine API stays the same.
# ============================================================================

import time
from typing import Optional
from dataclasses import dataclass

from .config import Config
from .vector_store import VectorStore
from .retriever import Retriever
from .embedder import Embedder
from .llm_router import LLMRouter, LLMResponse
from ..monitoring.logger import get_app_logger, QueryLogEntry


@dataclass
class QueryResult:
    """
    Result of a query.

    answer:
      The final model answer.

    sources:
      A list of dicts:
        [{"path": str, "chunks": int, "avg_relevance": float}, ...]

    chunks_used:
      How many chunks were provided as context to the model.

    tokens_in / tokens_out:
      Token accounting for online mode (GPT-3.5).
      Offline mode may report 0 depending on your LLM router.

    cost_usd:
      Estimated API cost (online mode only).

    latency_ms:
      End-to-end latency for the query.
    """
    answer: str
    sources: list
    chunks_used: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    mode: str
    error: Optional[str] = None


class QueryEngine:
    """
    Execute user queries against indexed documents.

    Pipeline:
        1) Retrieve relevant chunks (fast memmap + SQLite fetch)
        2) Build context
        3) Build prompt
        4) Call LLM (offline via Ollama or online via GPT-3.5)
        5) Log query + compute cost
    """

    def __init__(
        self,
        config: Config,
        vector_store: VectorStore,
        embedder: Embedder,
        llm_router: LLMRouter,
    ):
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_router = llm_router

        # Retriever is now memmap-based internally, but QueryEngine doesn't care.
        self.retriever = Retriever(vector_store, embedder, config)

        self.logger = get_app_logger("query_engine")

    def query(self, user_query: str) -> QueryResult:
        """
        Execute a query and return an answer plus metadata.
        """
        start_time = time.time()

        try:
            # ------------------------------------------------------------
            # Step 1: Retrieve (vector search)
            # ------------------------------------------------------------
            search_results = self.retriever.search(user_query)

            if not search_results:
                return QueryResult(
                    answer="No relevant information found in knowledge base.",
                    sources=[],
                    chunks_used=0,
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    mode=self.config.mode,
                )

            # ------------------------------------------------------------
            # Step 2: Build context text
            # ------------------------------------------------------------
            context = self.retriever.build_context(search_results)
            sources = self.retriever.get_sources(search_results)

            if not context.strip():
                # Extremely rare edge case (should not happen if chunks have text),
                # but we handle it gracefully.
                return QueryResult(
                    answer="Relevant documents were found, but no usable context text was available.",
                    sources=sources,
                    chunks_used=len(search_results),
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    mode=self.config.mode,
                    error="empty_context",
                )

            # ------------------------------------------------------------
            # Step 3: Build LLM prompt
            # ------------------------------------------------------------
            prompt = self._build_prompt(user_query, context)

            # ------------------------------------------------------------
            # Step 4: Call the LLM
            # ------------------------------------------------------------
            llm_response = self.llm_router.query(prompt)

            if not llm_response:
                return QueryResult(
                    answer="Error calling LLM. Please try again.",
                    sources=sources,
                    chunks_used=len(search_results),
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    mode=self.config.mode,
                    error="LLM call failed",
                )

            # ------------------------------------------------------------
            # Step 5: Calculate cost (online only)
            # ------------------------------------------------------------
            cost_usd = self._calculate_cost(llm_response)

            # ------------------------------------------------------------
            # Step 6: Format + log
            # ------------------------------------------------------------
            elapsed_ms = (time.time() - start_time) * 1000

            result = QueryResult(
                answer=llm_response.text,
                sources=sources,
                chunks_used=len(search_results),
                tokens_in=llm_response.tokens_in,
                tokens_out=llm_response.tokens_out,
                cost_usd=cost_usd,
                latency_ms=elapsed_ms,
                mode=self.config.mode,
            )

            # Log query summary (structured logging)
            log_entry = QueryLogEntry.build(
                query=user_query,
                mode=self.config.mode,
                chunks_retrieved=len(search_results),
                latency_ms=elapsed_ms,
                cost_usd=cost_usd,
            )
            self.logger.info("query_complete", **log_entry)

            return result

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error("query_error", error=error_msg, query=user_query)

            return QueryResult(
                answer=f"Error processing query: {error_msg}",
                sources=[],
                chunks_used=0,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                mode=self.config.mode,
                error=error_msg,
            )

    def _build_prompt(self, user_query: str, context: str) -> str:
        """
        Build the full prompt for the LLM.

        This is intentionally simple.
        Later we can add:
        - explicit citation instructions
        - guardrails (PII rules)
        - formatting rules (bullets, step-by-step)
        """
        return f"""Use the following context to answer the user's question.

{context}

User Question:
{user_query}

Answer:"""

    def _calculate_cost(self, llm_response: LLMResponse) -> float:
        """
        Calculate cost of LLM call (online only).

        Offline mode cost is 0 because it runs locally.
        """
        if self.config.mode == "offline":
            return 0.0

        input_cost = (llm_response.tokens_in / 1000) * self.config.cost.input_cost_per_1k
        output_cost = (llm_response.tokens_out / 1000) * self.config.cost.output_cost_per_1k
        return input_cost + output_cost
