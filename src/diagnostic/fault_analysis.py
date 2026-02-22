# ============================================================================
# HybridRAG v3 -- Diagnostic: Fault Analysis Engine
# ============================================================================
# FILE: src/diagnostic/fault_analysis.py
#
# WHAT THIS FILE DOES:
#   After all health checks and benchmarks run, this module examines the
#   combined results and produces an intelligent fault analysis:
#
#   1. CORRELATES failures across subsystems (e.g., if embedding fails AND
#      memmap is bad, the root cause is likely the embedding model, not
#      the memmap file itself)
#
#   2. RANKS the top 3 most likely root causes by probability, using a
#      weighted scoring system based on which tests failed and how they
#      relate to each other
#
#   3. RECOMMENDS the next diagnostic step for each ranked cause --
#      the single most useful action to isolate the problem further
#
# WHY THIS MATTERS:
#   When you're troubleshooting at 2am or handing off to another engineer,
#   you don't want to stare at 14 test results and guess. You want the
#   system to tell you: "Here are the 3 most likely problems, ranked by
#   probability, and here's exactly what to check next for each one."
#
# HOW IT WORKS:
#   - Each "fault hypothesis" has a base weight (prior probability)
#   - Test results add or subtract evidence points from each hypothesis
#   - Hypotheses are ranked by total evidence score
#   - Top 3 are reported with next-step recommendations
#
# LOG FORMAT:
#   Results are structured as JSON-serializable dicts for future GUI
#   integration. The admin GUI will display these in a "Fault Analysis"
#   panel with expandable details per hypothesis.
#
# AUTHOR: HybridRAG Team  |  DATE: 2026-02-09
# ============================================================================

from __future__ import annotations

import os
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

from . import DiagnosticReport, TestResult, PerfMetric, PROJ_ROOT


# ============================================================================
# Fault Hypothesis -- one possible root cause
# ============================================================================
# Think of each hypothesis as a "theory" about what's wrong.
# Evidence from test results either supports or contradicts each theory.
# The theory with the most supporting evidence ranks highest.
# ============================================================================

@dataclass
class FaultHypothesis:
    """
    One possible root cause of a system problem.

    Fields:
        fault_id:       Unique ID like "FAULT-001" (for logs and GUI)
        title:          Short human-readable name
        description:    What this fault means in plain English
        subsystem:      Which part of the pipeline is affected
        evidence_score: Running total -- higher = more likely this is the cause
        evidence_items: List of reasons this hypothesis was scored up or down
        severity:       How bad is this if it IS the cause (CRITICAL/HIGH/MEDIUM/LOW)
        next_step:      The ONE thing to do next to confirm or rule out this cause
        next_step_cmd:  Copy-paste-ready command to run (if applicable)
        related_bugs:   Links to known bug IDs (BUG-001, SEC-001, etc.)
        confidence:     Calculated from evidence_score, 0.0 to 1.0
    """
    fault_id: str
    title: str
    description: str
    subsystem: str
    evidence_score: float = 0.0
    evidence_items: List[str] = field(default_factory=list)
    severity: str = "MEDIUM"
    next_step: str = ""
    next_step_cmd: str = ""
    related_bugs: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def add_evidence(self, points: float, reason: str):
        """Add supporting (+) or contradicting (-) evidence."""
        self.evidence_score += points
        direction = "+" if points >= 0 else ""
        self.evidence_items.append(f"[{direction}{points:.1f}] {reason}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON logging / GUI consumption."""
        return {
            "fault_id": self.fault_id,
            "title": self.title,
            "description": self.description,
            "subsystem": self.subsystem,
            "evidence_score": round(self.evidence_score, 2),
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "evidence_items": self.evidence_items,
            "next_step": self.next_step,
            "next_step_cmd": self.next_step_cmd,
            "related_bugs": self.related_bugs,
        }


# ============================================================================
# Fault Analysis Result -- the final output
# ============================================================================

@dataclass
class FaultAnalysisResult:
    """
    Complete fault analysis output.

    This is what gets logged, displayed in the terminal, and eventually
    shown in the admin GUI's "Fault Analysis" panel.
    """
    timestamp: str = ""
    system_healthy: bool = True
    total_hypotheses_evaluated: int = 0
    ranked_faults: List[FaultHypothesis] = field(default_factory=list)
    all_clear_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_healthy": self.system_healthy,
            "total_hypotheses_evaluated": self.total_hypotheses_evaluated,
            "ranked_faults": [f.to_dict() for f in self.ranked_faults],
            "all_clear_message": self.all_clear_message,
        }


# ============================================================================
# The Fault Analyzer -- the "smart mechanic"
# ============================================================================

# ARCHITECTURE NOTE: FaultAnalyzer exceeds 500 lines (656 lines).
# Split into FaultAnalyzer (hypothesis engine) and
# FaultReporter (formatting/output) when next modified.
class FaultAnalyzer:
    """
    Examines all diagnostic results and produces a ranked fault analysis.

    How to use:
        analyzer = FaultAnalyzer(report)
        result = analyzer.analyze()
        # result.ranked_faults = top 3 most likely causes
        # Each has .next_step = what to do next
    """

    def __init__(self, report: DiagnosticReport):
        self.report = report
        # Quick lookup: test_name -> TestResult
        self._results: Dict[str, TestResult] = {
            r.name: r for r in report.results
        }
        # Quick lookup: benchmark_name -> PerfMetric
        self._perfs: Dict[str, PerfMetric] = {
            m.name: m for m in report.perf_metrics
        }
        # Known bug IDs detected
        self._bug_ids: set = {
            b["id"] for b in report.known_bugs
        }

    def _get(self, name: str) -> Optional[TestResult]:
        """Look up a test result by name. Returns None if not found."""
        return self._results.get(name)

    def _status(self, name: str) -> str:
        """Get status of a test, or 'MISSING' if it didn't run."""
        r = self._get(name)
        return r.status if r else "MISSING"

    def _perf_slow(self, name: str, threshold_ms: float) -> bool:
        """Check if a benchmark exceeded a time threshold."""
        m = self._perfs.get(name)
        return m is not None and m.avg_val > threshold_ms

    def analyze(self) -> FaultAnalysisResult:
        """
        Run the full fault analysis.

        This is the main entry point. It:
          1. Creates all possible fault hypotheses
          2. Scores each one against the test results
          3. Calculates confidence levels
          4. Returns the top 3 ranked by evidence score
        """
        result = FaultAnalysisResult()
        result.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build all hypotheses (each one is a "theory" about what's wrong)
        hypotheses = self._build_hypotheses()
        result.total_hypotheses_evaluated = len(hypotheses)

        # Score each hypothesis against the evidence
        for h in hypotheses:
            self._score_hypothesis(h)

        # Calculate confidence (normalized 0-1 scale)
        max_score = max((h.evidence_score for h in hypotheses), default=1.0)
        if max_score > 0:
            for h in hypotheses:
                # Only assign confidence to hypotheses with positive evidence
                if h.evidence_score > 0:
                    h.confidence = h.evidence_score / (max_score * 1.2)
                    # Cap at 0.95 -- never claim 100% certainty
                    h.confidence = min(h.confidence, 0.95)

        # Filter to only hypotheses with positive evidence (actual problems)
        active_faults = [h for h in hypotheses if h.evidence_score > 0]

        # Sort by evidence score descending, take top 3
        active_faults.sort(key=lambda h: h.evidence_score, reverse=True)
        result.ranked_faults = active_faults[:3]

        # Determine overall health
        if not active_faults:
            result.system_healthy = True
            result.all_clear_message = (
                "All subsystems operational. No fault hypotheses triggered. "
                "System is GO for indexing and query operations."
            )
        else:
            result.system_healthy = False

        return result

    # ====================================================================
    # Hypothesis Definitions
    # ====================================================================
    # Each hypothesis represents a distinct failure mode.
    # The _score_hypothesis method examines test results to decide
    # how much evidence supports each one.
    #
    # ADDING NEW HYPOTHESES:
    #   1. Add a new FaultHypothesis in _build_hypotheses()
    #   2. Add scoring logic in _score_hypothesis()
    #   3. That's it -- the ranking and reporting handles the rest
    # ====================================================================

    def _build_hypotheses(self) -> List[FaultHypothesis]:
        """Create all possible fault hypotheses."""
        return [
            # ---- Environment / Setup ----
            FaultHypothesis(
                fault_id="FAULT-ENV-001",
                title="Missing or broken Python dependencies",
                description=(
                    "One or more required Python packages are not installed, "
                    "are the wrong version, or failed to import. This breaks "
                    "whichever pipeline stage needs that package."
                ),
                subsystem="Environment",
                severity="HIGH",
                next_step=(
                    "Run the dependency smoke test to find which packages "
                    "are missing or broken. Compare installed versions "
                    "against requirements-lock.txt."
                ),
                next_step_cmd=(
                    'python -c "import sentence_transformers; import pdfplumber; '
                    'import docx; import pptx; import openpyxl; import httpx; '
                    'import structlog; import yaml; print(\'All imports OK\')"'
                ),
            ),
            FaultHypothesis(
                fault_id="FAULT-ENV-002",
                title="Configuration not loaded or paths wrong",
                description=(
                    "The config file can't be loaded, or environment variables "
                    "point to wrong/missing directories. Everything downstream "
                    "of config loading will fail."
                ),
                subsystem="Environment",
                severity="CRITICAL",
                next_step=(
                    "Verify env vars are set correctly and paths exist on disk. "
                    "Run rag-paths to see current configuration."
                ),
                next_step_cmd="rag-paths",
            ),
            FaultHypothesis(
                fault_id="FAULT-ENV-003",
                title="Model cache incomplete or corrupted",
                description=(
                    "The embedding model or reranker model files are missing "
                    "from the project cache folders. HuggingFace offline mode "
                    "prevents re-downloading them, so loading fails silently."
                ),
                subsystem="Environment",
                severity="HIGH",
                next_step=(
                    "Check model cache folders for expected files. If missing, "
                    "temporarily disable offline mode and re-cache models."
                ),
                next_step_cmd=(
                    'python -c "'
                    "import os; "
                    "mc = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '?'); "
                    "hf = os.environ.get('HF_HOME', '?'); "
                    "print(f'Model cache: {mc}'); "
                    "print(f'HF cache: {hf}'); "
                    "[print(f'  {d}: {len(list(os.scandir(d)))} items') "
                    "for d in [mc, hf] if os.path.isdir(d)]"
                    '"'
                ),
            ),

            # ---- Database ----
            FaultHypothesis(
                fault_id="FAULT-DB-001",
                title="Database missing, corrupt, or schema mismatch",
                description=(
                    "The SQLite database doesn't exist, failed integrity check, "
                    "or is missing required tables/columns. All query and "
                    "indexing operations depend on a healthy database."
                ),
                subsystem="Database",
                severity="CRITICAL",
                next_step=(
                    "Check if the database file exists and passes integrity. "
                    "If corrupt, delete and re-index from source documents."
                ),
                next_step_cmd='rag-status',
                related_bugs=["BUG-001"],
            ),
            FaultHypothesis(
                fault_id="FAULT-DB-002",
                title="FTS5 index missing or out of sync",
                description=(
                    "The full-text search index is empty or has different row "
                    "count than the chunks table. Keyword search will return "
                    "wrong or no results, breaking hybrid search."
                ),
                subsystem="Database",
                severity="HIGH",
                next_step=(
                    "Rebuild the FTS5 index from existing chunks. This is "
                    "non-destructive -- it rebuilds the keyword index without "
                    "touching your source data or embeddings."
                ),
                next_step_cmd=(
                    "python -c \""
                    "import sqlite3, os; "
                    "db = os.path.join(os.environ['HYBRIDRAG_DATA_DIR'], 'hybridrag.sqlite3'); "
                    "c = sqlite3.connect(db); "
                    "c.execute(\\\"INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')\\\"); "
                    "c.commit(); c.close(); "
                    "print('FTS5 rebuilt')"
                    "\""
                ),
            ),

            # ---- Embedding / Search ----
            FaultHypothesis(
                fault_id="FAULT-EMB-001",
                title="Embedding model fails to load",
                description=(
                    "The sentence-transformers embedding model can't be loaded. "
                    "This is usually caused by: missing cache files, HuggingFace "
                    "offline mode blocking a required download, or a version "
                    "mismatch between sentence-transformers and the cached model."
                ),
                subsystem="Embedding",
                severity="CRITICAL",
                next_step=(
                    "Test loading the embedding model directly. If it fails, "
                    "check that offline mode isn't blocking a required file "
                    "and that the model cache has all expected files."
                ),
                next_step_cmd='rag-diag --test-embed',
            ),
            FaultHypothesis(
                fault_id="FAULT-EMB-002",
                title="Memmap embeddings file damaged or mismatched",
                description=(
                    "The memory-mapped embeddings file is the wrong size, "
                    "contains NaN/Inf values, or doesn't match the metadata. "
                    "Vector search will return garbage results or crash."
                ),
                subsystem="Embedding",
                severity="HIGH",
                next_step=(
                    "Validate memmap file size against metadata count. If "
                    "mismatched, rebuild memmap from SQLite (non-destructive)."
                ),
                next_step_cmd='python src/tools/rebuild_memmap_from_sqlite.py',
                related_bugs=[],
            ),

            # ---- Parsers / Indexing ----
            FaultHypothesis(
                fault_id="FAULT-PARSE-001",
                title="Document parsers failing (missing libraries)",
                description=(
                    "One or more document parsers can't import their required "
                    "library (python-docx, pdfplumber, openpyxl, etc.). Files "
                    "of that type will be silently skipped during indexing, "
                    "leaving gaps in your knowledge base."
                ),
                subsystem="Parsers",
                severity="HIGH",
                next_step=(
                    "Run the parser smoke test against a sample file of each "
                    "type. Check which parsers fail to import."
                ),
                next_step_cmd='rag-diag --test-parse "path/to/sample.pdf"',
            ),
            FaultHypothesis(
                fault_id="FAULT-PARSE-002",
                title="Binary garbage in parsed output (BUG-004)",
                description=(
                    "Some parsed files return binary data instead of text. "
                    "This pollutes the embedding space with meaningless vectors "
                    "and degrades search quality for all queries."
                ),
                subsystem="Parsers",
                severity="MEDIUM",
                next_step=(
                    "Run the parse test on suspicious files to check the "
                    "garbage ratio. Files with >10%% non-printable characters "
                    "should be excluded or sent through OCR."
                ),
                next_step_cmd='rag-diag --test-parse "path/to/suspect.pdf" --verbose',
                related_bugs=["BUG-004"],
            ),

            # ---- Indexer Logic ----
            FaultHypothesis(
                fault_id="FAULT-IDX-001",
                title="Re-indexing doesn't detect file changes (BUG-002)",
                description=(
                    "When source documents are updated, re-indexing skips them "
                    "because it only checks if the file exists in the DB, not "
                    "whether it has changed. Stale chunks persist forever."
                ),
                subsystem="Indexer",
                severity="HIGH",
                next_step=(
                    "For now, delete the database and re-index from scratch "
                    "when source files change. Long-term fix: wire "
                    "_file_changed() into the skip logic."
                ),
                next_step_cmd=(
                    '# Delete DB to force clean re-index:\n'
                    '# Remove-Item "$env:HYBRIDRAG_DATA_DIR\\hybridrag.sqlite3"\n'
                    '# Remove-Item "$env:HYBRIDRAG_DATA_DIR\\embeddings.f16.dat"\n'
                    '# rag-index'
                ),
                related_bugs=["BUG-002"],
            ),
            FaultHypothesis(
                fault_id="FAULT-IDX-002",
                title="Resource leak during long indexing runs (BUG-003)",
                description=(
                    "VectorStore and Embedder don't have close() methods. On "
                    "long 24/7 indexing runs, open database connections and "
                    "loaded models accumulate, eventually causing memory "
                    "exhaustion or file handle limits."
                ),
                subsystem="Indexer",
                severity="MEDIUM",
                next_step=(
                    "Monitor RAM usage during indexing with rag-diag. If it "
                    "climbs steadily, restart the indexing process periodically. "
                    "Permanent fix: add close() methods to VectorStore and Embedder."
                ),
                next_step_cmd='rag-diag --verbose',
                related_bugs=["BUG-003"],
            ),

            # ---- LLM / Query ----
            FaultHypothesis(
                fault_id="FAULT-LLM-001",
                title="Ollama not running or model not loaded",
                description=(
                    "The Ollama service isn't running on localhost:11434, or "
                    "the configured model (phi4-mini) isn't pulled. Offline "
                    "queries will fail with a connection error."
                ),
                subsystem="LLM",
                severity="HIGH",
                next_step=(
                    "Check if Ollama is running and the model is available. "
                    "Start Ollama if needed and pull the model."
                ),
                next_step_cmd=(
                    'curl http://localhost:11434/api/tags 2>$null || '
                    'echo "Ollama not running -- start it with: ollama serve"'
                ),
            ),
            FaultHypothesis(
                fault_id="FAULT-LLM-002",
                title="API mode misconfigured or kill switch active",
                description=(
                    "Online/API mode is selected but either: the API endpoint "
                    "is empty (SEC-001 safety), the kill switch is blocking "
                    "all outbound, or no API key is configured."
                ),
                subsystem="LLM",
                severity="MEDIUM",
                next_step=(
                    "Check current mode, endpoint, and kill switch status. "
                    "For online mode: set endpoint, provide API key, and "
                    "disable kill switch."
                ),
                next_step_cmd='rag-paths',
                related_bugs=["SEC-001"],
            ),

            # ---- Security ----
            FaultHypothesis(
                fault_id="FAULT-SEC-001",
                title="Network lockdown preventing model loading",
                description=(
                    "HF_HUB_OFFLINE=1 is blocking HuggingFace connections, "
                    "but the model cache is incomplete. The system needs "
                    "network access once to cache models, then can go offline. "
                    "This is the most common first-run failure on new machines."
                ),
                subsystem="Security",
                severity="HIGH",
                next_step=(
                    "Temporarily disable offline mode, cache all models, "
                    "then re-enable. This only needs to happen once per machine."
                ),
                next_step_cmd=(
                    '$env:HF_HUB_OFFLINE = "0"; $env:TRANSFORMERS_OFFLINE = "0"; '
                    'python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; '
                    "SentenceTransformer('all-MiniLM-L6-v2'); "
                    "CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); "
                    "print('Models cached')\";"
                    ' . .\\start_hybridrag.ps1'
                ),
            ),

            # ---- Performance ----
            FaultHypothesis(
                fault_id="FAULT-PERF-001",
                title="Query latency excessively high",
                description=(
                    "Queries take much longer than expected. The bottleneck "
                    "is almost always the LLM generation step (Ollama on CPU). "
                    "Retrieval itself should be <2 seconds."
                ),
                subsystem="Performance",
                severity="LOW",
                next_step=(
                    "Run the performance benchmarks to isolate which stage "
                    "is slow. If LLM generation is >90%% of total time, "
                    "consider GPU acceleration or switching to API mode."
                ),
                next_step_cmd='rag-diag --test-embed --perf-only --benchmark-iters 5',
            ),
        ]

    def _score_hypothesis(self, h: FaultHypothesis):
        """
        Score a hypothesis against test results.

        Each test that passed/failed/warned adds or removes evidence.
        The scoring uses a simple weighted system:
          +3.0 = strong direct evidence (test for this exact thing failed)
          +2.0 = moderate evidence (related test failed)
          +1.0 = weak evidence (circumstantial)
          -1.0 = contradicting evidence (test for this thing passed)
          -2.0 = strong contradiction (multiple related tests passed)
        """

        # ---- FAULT-ENV-001: Missing dependencies ----
        if h.fault_id == "FAULT-ENV-001":
            s = self._status("parser_registry")
            if s == "WARN":
                h.add_evidence(3.0, "Parser registry has import failures")
            elif s == "PASS":
                h.add_evidence(-1.0, "Parser registry imports all OK")
            if s == "ERROR":
                h.add_evidence(3.0, "Parser registry test crashed (likely import error)")

            s = self._status("embedder")
            if s == "ERROR":
                h.add_evidence(2.0, "Embedder test crashed")
            elif s == "PASS":
                h.add_evidence(-0.5, "Embedder importable")

            s = self._status("chunker")
            if s == "ERROR":
                h.add_evidence(2.0, "Chunker test crashed")
            elif s == "PASS":
                h.add_evidence(-0.5, "Chunker OK")

        # ---- FAULT-ENV-002: Config/paths wrong ----
        elif h.fault_id == "FAULT-ENV-002":
            s = self._status("config_load")
            if s in ("FAIL", "ERROR"):
                h.add_evidence(4.0, "Config load failed -- nothing works without config")
            elif s == "PASS":
                h.add_evidence(-2.0, "Config loads successfully")

            s = self._status("config_paths")
            if s == "FAIL":
                h.add_evidence(3.0, "Configured paths don't exist on disk")
            elif s == "WARN":
                h.add_evidence(1.5, "Some path issues detected")
            elif s == "PASS":
                h.add_evidence(-1.0, "All paths exist")

        # ---- FAULT-ENV-003: Model cache incomplete ----
        elif h.fault_id == "FAULT-ENV-003":
            s = self._status("embedder")
            if s == "ERROR":
                h.add_evidence(3.0, "Embedder failed -- likely can't load cached model")
            elif s == "FAIL":
                h.add_evidence(2.0, "Embedder test failed")
            elif s == "PASS":
                h.add_evidence(-2.0, "Embedder loads fine from cache")

            # If config loads but embedder fails, cache is the prime suspect
            if self._status("config_load") == "PASS" and self._status("embedder") in ("ERROR", "FAIL"):
                h.add_evidence(2.0, "Config OK but embedder broken -- cache problem")

            # Check if the security lockdown is active (increases likelihood)
            r = self._get("security_net")
            if r and r.details.get("kill_switch"):
                if self._status("embedder") in ("ERROR", "FAIL"):
                    h.add_evidence(1.5, "Kill switch active + embedder failing = cache miss")

        # ---- FAULT-DB-001: Database problems ----
        elif h.fault_id == "FAULT-DB-001":
            s = self._status("sqlite_conn")
            if s in ("FAIL", "ERROR"):
                h.add_evidence(4.0, "SQLite connection/integrity failed")
            elif s == "SKIP":
                h.add_evidence(3.0, "Database file not found")
            elif s == "PASS":
                h.add_evidence(-2.0, "SQLite connection healthy")

            s = self._status("schema_chunks")
            if s == "FAIL":
                h.add_evidence(3.0, "Chunks table schema wrong or missing")
            elif s == "PASS":
                h.add_evidence(-1.0, "Schema OK")

            s = self._status("data_integrity")
            if s in ("FAIL", "WARN"):
                h.add_evidence(2.0, "Data integrity issues detected")
            elif s == "PASS":
                h.add_evidence(-1.0, "Data integrity clean")

        # ---- FAULT-DB-002: FTS5 out of sync ----
        elif h.fault_id == "FAULT-DB-002":
            s = self._status("schema_fts5")
            if s == "FAIL":
                h.add_evidence(4.0, "FTS5 table missing or empty")
            elif s == "WARN":
                h.add_evidence(3.0, "FTS5 row count mismatches chunks")
            elif s == "PASS":
                h.add_evidence(-2.0, "FTS5 in sync and queryable")

        # ---- FAULT-EMB-001: Embedding model won't load ----
        elif h.fault_id == "FAULT-EMB-001":
            s = self._status("embedder")
            if s == "ERROR":
                h.add_evidence(4.0, "Embedder test crashed -- model load failure")
            elif s == "FAIL":
                h.add_evidence(3.0, "Embedder test failed (dimension mismatch?)")
            elif s == "PASS":
                h.add_evidence(-3.0, "Embedder loads and produces correct dimensions")

            # Corroborate with memmap -- if both are bad, embedding is root cause
            if self._status("memmap") in ("FAIL", "SKIP"):
                if s in ("ERROR", "FAIL"):
                    h.add_evidence(1.0, "Memmap also bad -- embedding failure is upstream cause")

        # ---- FAULT-EMB-002: Memmap file damaged ----
        elif h.fault_id == "FAULT-EMB-002":
            s = self._status("memmap")
            if s == "FAIL":
                h.add_evidence(4.0, "Memmap size mismatch or contains NaN/Inf")
            elif s == "WARN":
                h.add_evidence(2.0, "Memmap has anomalies")
            elif s == "SKIP":
                h.add_evidence(1.0, "Memmap file missing (not yet indexed)")
            elif s == "PASS":
                h.add_evidence(-2.0, "Memmap healthy")

            # If embedder is fine but memmap is bad, it's a storage issue
            if self._status("embedder") == "PASS" and s in ("FAIL", "WARN"):
                h.add_evidence(2.0, "Embedder OK but memmap broken -- file corruption")

        # ---- FAULT-PARSE-001: Parser import failures ----
        elif h.fault_id == "FAULT-PARSE-001":
            s = self._status("parser_registry")
            if s == "WARN":
                r = self._get("parser_registry")
                fails = r.details.get("failures", {}) if r else {}
                h.add_evidence(3.0, f"Parser instantiation failures: {list(fails.keys())}")
            elif s == "ERROR":
                h.add_evidence(3.0, "Parser registry test crashed")
            elif s == "PASS":
                h.add_evidence(-2.0, "All parsers import and instantiate")

        # ---- FAULT-PARSE-002: Binary garbage ----
        elif h.fault_id == "FAULT-PARSE-002":
            # This is always a possibility -- BUG-004 means no validation exists
            if "BUG-004" in self._bug_ids:
                h.add_evidence(1.5, "BUG-004 active -- no garbage detection in pipeline")
            # If a specific file test showed garbage
            r = self._get("parse_file")
            if r and r.status == "WARN" and r.details.get("garbage_ratio", 0) > 0.10:
                h.add_evidence(3.0, f"Tested file has {r.details['garbage_ratio']:.1%} garbage")

        # ---- FAULT-IDX-001: Stale chunks (BUG-002) ----
        elif h.fault_id == "FAULT-IDX-001":
            if "BUG-002" in self._bug_ids:
                h.add_evidence(2.5, "BUG-002 confirmed: _file_changed() never called")
            s = self._status("change_detection")
            if s in ("FAIL", "WARN"):
                h.add_evidence(2.0, "Change detection test flagged")
            elif s == "PASS":
                h.add_evidence(-2.0, "Change detection wired in")

        # ---- FAULT-IDX-002: Resource leak (BUG-003) ----
        elif h.fault_id == "FAULT-IDX-002":
            if "BUG-003" in self._bug_ids:
                h.add_evidence(2.0, "BUG-003 confirmed: no close() methods")
            s = self._status("resource_cleanup")
            if s == "WARN":
                h.add_evidence(1.5, "Resource cleanup test warned")
            elif s == "PASS":
                h.add_evidence(-2.0, "Cleanup methods present")

        # ---- FAULT-LLM-001: Ollama not running ----
        elif h.fault_id == "FAULT-LLM-001":
            # Check if an e2e query test was run and failed
            r = self._get("e2e_query")
            if r and r.status in ("FAIL", "ERROR"):
                error = r.details.get("error", "")
                if "connect" in str(error).lower() or "timeout" in str(error).lower():
                    h.add_evidence(4.0, "Query failed with connection/timeout error")
                else:
                    h.add_evidence(1.0, "Query failed (may not be Ollama)")
            elif r and r.status == "PASS":
                h.add_evidence(-3.0, "E2E query succeeded -- Ollama is working")

            # If no e2e test, check config mode
            r = self._get("config_load")
            if r and r.details.get("mode") == "offline":
                # Can't directly test Ollama from here, but mode is offline
                # so Ollama is required
                if not self._get("e2e_query"):
                    h.add_evidence(0.5, "Offline mode requires Ollama but no query test ran")

        # ---- FAULT-LLM-002: API misconfigured ----
        elif h.fault_id == "FAULT-LLM-002":
            s = self._status("security_endpoint")
            if s == "FAIL":
                h.add_evidence(2.0, "SEC-001: Endpoint needs configuration")
            elif s == "WARN":
                h.add_evidence(1.0, "Endpoint warnings present")
            elif s == "PASS":
                h.add_evidence(-1.0, "Endpoint config looks safe")

            r = self._get("config_load")
            if r and r.details.get("mode") == "online":
                # Online mode selected -- API config matters more
                h.add_evidence(1.0, "Online mode selected -- API config is critical")

        # ---- FAULT-SEC-001: Offline mode blocking models ----
        elif h.fault_id == "FAULT-SEC-001":
            s = self._status("embedder")
            if s in ("ERROR", "FAIL"):
                h.add_evidence(2.0, "Embedder failing -- could be offline blocking")
                # If security shows kill switch is active, this is very likely
                r = self._get("security_net")
                if r and r.details.get("kill_switch"):
                    h.add_evidence(2.0, "Kill switch active + embedder failing")
            elif s == "PASS":
                h.add_evidence(-3.0, "Embedder loads fine -- offline mode not a problem")

        # ---- FAULT-PERF-001: Slow queries ----
        elif h.fault_id == "FAULT-PERF-001":
            r = self._get("e2e_query")
            if r and r.elapsed_ms > 120000:  # >2 minutes
                h.add_evidence(3.0, f"Query took {r.elapsed_ms/1000:.0f}s (>2 min)")
            elif r and r.elapsed_ms > 60000:  # >1 minute
                h.add_evidence(2.0, f"Query took {r.elapsed_ms/1000:.0f}s (>1 min)")
            elif r and r.elapsed_ms > 0 and r.elapsed_ms < 30000:
                h.add_evidence(-1.0, "Query completed in <30s")

            # Check if vector search itself is slow
            if self._perf_slow("vector_search", 5000):
                h.add_evidence(1.5, "Vector search benchmark >5s (should be <2s)")


# ============================================================================
# Run the analysis and produce structured output
# ============================================================================

def run_fault_analysis(report: DiagnosticReport) -> FaultAnalysisResult:
    """
    Main entry point: analyze a diagnostic report and return ranked faults.

    Usage:
        from src.diagnostic.fault_analysis import run_fault_analysis
        result = run_fault_analysis(report)
        for fault in result.ranked_faults:
            print(f"#{fault.fault_id}: {fault.title} (confidence: {fault.confidence:.0%})")
            print(f"  Next step: {fault.next_step}")
    """
    analyzer = FaultAnalyzer(report)
    return analyzer.analyze()


# ============================================================================
# Terminal output for fault analysis
# ============================================================================

def print_fault_analysis(result: FaultAnalysisResult, verbose: bool = False):
    """
    Print the fault analysis to the terminal in a readable format.

    This is what the user sees after the normal diagnostic report.
    The admin GUI will eventually render the same data in a panel.
    """
    from . import GREEN, RED, YELLOW, CYAN, BOLD, DIM, RESET

    W = 72
    sep = "-" * W

    print(f"\n  {BOLD}{CYAN}FAULT ANALYSIS -- TOP PROBABLE CAUSES{RESET}")
    print(f"  {sep}")

    if result.system_healthy:
        print(f"  {GREEN}{BOLD}  [OK] No faults detected -- system healthy{RESET}")
        print(f"  {DIM}  {result.all_clear_message}{RESET}")
        print(f"  {sep}")
        return

    for rank, fault in enumerate(result.ranked_faults, 1):
        # Color by severity
        sc = RED if fault.severity in ("CRITICAL", "HIGH") else YELLOW
        # Confidence bar: ####.... 65%
        bar_len = 20
        filled = int(fault.confidence * bar_len)
        bar = "#" * filled + "." * (bar_len - filled)

        print(f"\n  {sc}{BOLD}  #{rank}  {fault.title}{RESET}")
        print(f"  {DIM}  Fault ID: {fault.fault_id} | Subsystem: {fault.subsystem} | "
              f"Severity: {fault.severity}{RESET}")
        print(f"  {CYAN}  Confidence: [{bar}] {fault.confidence:.0%}{RESET}")

        # Description (wrapped to fit terminal)
        import textwrap
        for line in textwrap.wrap(fault.description, width=W - 6):
            print(f"  {DIM}  {line}{RESET}")

        # Evidence trail (verbose only)
        if verbose and fault.evidence_items:
            print(f"  {DIM}  Evidence:{RESET}")
            for ev in fault.evidence_items:
                print(f"  {DIM}    {ev}{RESET}")

        # Related bugs
        if fault.related_bugs:
            print(f"  {YELLOW}  Related: {', '.join(fault.related_bugs)}{RESET}")

        # NEXT STEP -- the most important part
        print(f"  {GREEN}{BOLD}  -> NEXT STEP:{RESET} {fault.next_step}")
        if fault.next_step_cmd:
            # Show command on separate line, indented for copy-paste
            cmd_display = fault.next_step_cmd
            if len(cmd_display) > W - 8:
                cmd_display = cmd_display[:W - 11] + "..."
            print(f"  {CYAN}    $ {cmd_display}{RESET}")

    print(f"\n  {sep}")
    print(f"  {DIM}  Evaluated {result.total_hypotheses_evaluated} fault hypotheses. "
          f"Showing top {len(result.ranked_faults)}.{RESET}")
    print(f"  {DIM}  Run with --verbose for evidence trail details.{RESET}")
    print(f"  {sep}")


# ============================================================================
# Log fault analysis to structured JSON file
# ============================================================================

def log_fault_analysis(result: FaultAnalysisResult, log_dir: str = ""):
    """
    Write fault analysis to a structured JSON log file.

    These logs are designed for future GUI consumption. The admin panel
    will read the latest log and display the fault analysis results in
    an interactive panel with expandable details.

    Log files are named: fault_analysis_YYYY-MM-DD_HHMMSS.json
    They accumulate in the logs/ directory for trend analysis.
    """
    if not log_dir:
        log_dir = str(PROJ_ROOT / "logs")

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"fault_analysis_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)

    log_data = {
        "schema_version": "1.0",
        "tool": "HybridRAG Fault Analyzer",
        "generated_at": result.timestamp,
        "system_healthy": result.system_healthy,
        "hypotheses_evaluated": result.total_hypotheses_evaluated,
        "faults_detected": len(result.ranked_faults),
        "ranked_faults": [f.to_dict() for f in result.ranked_faults],
    }

    # Also keep the latest result as a "current" file for easy GUI access
    latest_path = os.path.join(log_dir, "fault_analysis_latest.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    return filepath
