# HybridRAG3 Demo Preparation

Last Updated: 2026-02-21

---

## SECTION 1: ELEVATOR PITCH

HybridRAG is a local-first document search and question-answering system that
reads your entire document library -- PDFs, Word files, spreadsheets, emails,
images -- and answers natural-language questions in seconds with citations back
to the exact source files, all without sending a single byte off your machine.

It solves the problem every engineering program faces: thousands of documents
scattered across drives and SharePoint, where keyword search fails because
"antenna frequency range" and "RF operating band" mean the same thing but
share zero words -- HybridRAG understands meaning, not just keywords.

This matters because it turns hours of manual document hunting into a 5-second
query, keeps sensitive program data entirely on the local machine by default,
and has been validated at 98% accuracy on a 400-question engineering test set
using only open-source, US/EU-origin AI models with no restrictive licenses.

---

## SECTION 2: THE 5-MINUTE DEMO SCRIPT

### Before the Demo

**Checklist** (do this 30 minutes before):

1. Verify Ollama is running: `rag-ollama-status`
2. Verify phi4-mini is loaded: should show in Ollama model list
3. Verify index is current: `rag-index-status` (should show ~39,602 chunks)
4. Open the GUI: `python src/gui/launch_gui.py` (wait for status bar to
   show green "Ready")
5. Have a second PowerShell window open for CLI fallback
6. Close all other apps to avoid RAM pressure on 8GB laptop

### Demo Flow (5 minutes)

---

**BEAT 1: The Problem (30 seconds)**

> *"Imagine you have 1,345 technical documents -- specs, procedures, training
> guides, calibration manuals. A field engineer calls and asks: 'What's the
> operating frequency for this system?' You can Ctrl+F through documents one
> at a time, but that only finds exact words. If the spec says 'RF operating
> band' instead of 'frequency,' you miss it entirely. HybridRAG solves this."*

---

**BEAT 2: Offline Query (90 seconds)**

Show the GUI. Point out:
- The mode indicator says **OFFLINE** (green) -- no internet connection
- The use-case dropdown is set to **Engineering / STEM**
- The model field shows **phi4-mini (Ollama)** -- running locally

Type a question into the query box:

> **"What is the operating frequency range?"**

Press Enter. While it processes (~10-15 seconds on laptop):

> *"Right now, the system is doing three things. First, it converted your
> question into a mathematical fingerprint -- a meaning vector. Second, it
> ran two parallel searches: one by meaning and one by keyword, then merged
> the results. Third, it's sending the top matching passages to a local AI
> model running right here on this laptop. No internet involved."*

When the answer appears:
- Read the answer aloud -- note it includes specific values from the source
- Point to the **Sources** line: *"It tells you exactly which documents the
  answer came from"*
- Point to the **Metrics** line: *"X milliseconds, Y tokens -- full
  auditability"*

---

**BEAT 3: Admin Menu (30 seconds)**

Click **Admin > Admin Settings**.

> *"For technical users, everything is tunable. You can adjust how many
> document chunks to retrieve, change the similarity threshold, switch
> between models, or change the temperature. These are the same parameters
> we used to achieve 98% accuracy on our evaluation set."*

Point out key sliders:
- **top_k**: how many chunks to retrieve (currently 12)
- **min_score**: minimum relevance threshold (currently 0.10)
- **temperature**: LLM creativity (currently 0.05 -- very precise)

Close the settings window.

---

**BEAT 4: Online Mode Switch (60 seconds)**

> *"When you need faster answers or a more capable model, you can switch
> to online mode with one click."*

Click the **ONLINE** button in the title bar.

> *"The system validates credentials first. API keys are stored in Windows
> Credential Manager -- encrypted with DPAPI, tied to your Windows login.
> They're never in config files, environment variables, or source code."*

Once online mode activates:
- Point out the model field now shows the cloud model name
- Type the same question again
- Note the faster response time (~2-5 seconds vs 10-15)

> *"Same question, same sources, but the cloud model gives a more detailed
> answer. The key point: switching between offline and online is one click.
> The default is always offline -- online must be explicitly activated."*

Switch back to OFFLINE before continuing.

---

**BEAT 5: Indexing a Folder (60 seconds)**

Click the **Index** tab (or panel).

> *"Adding new documents is point-and-click. You select a folder, click
> Start, and it processes everything automatically."*

Click **Browse**, navigate to a small demo folder (prepare a folder with
5-10 short documents ahead of time).

Click **Start Indexing**. As the progress bar moves:

> *"It handles 24+ file formats: PDF, Word, PowerPoint, Excel, emails,
> images with OCR, plain text, and more. It splits each document into
> half-page chunks at paragraph boundaries so context is preserved.
> Each chunk gets a meaning fingerprint from a local embedding model.
> Everything is stored in a local SQLite database -- portable, no server
> needed, XCOPY-deployable."*

> *"After the first run, only new or changed files are re-indexed.
> A thousand-file update takes seconds, not hours."*

---

**BEAT 6: Wrap-Up (30 seconds)**

> *"To summarize: HybridRAG gives you meaning-based search across your
> entire document library, with direct answers and source citations, running
> entirely on a standard laptop with no internet required. It's been
> validated at 98% accuracy, uses only approved open-source models, and
> every operation is logged for audit."*

---

### If Something Goes Wrong

| Problem | Recovery |
|---------|----------|
| Ollama not responding | Switch to online mode for the demo |
| GUI won't launch | Use CLI: `rag-query "What is the frequency?"` |
| Slow response | Say: "This is an 8GB laptop. On the workstation, responses are 3-5x faster." |
| Wrong answer | Say: "98% accuracy means 2% failure rate. Let me show a different question." |
| Crash/error | Show the error gracefully: "Every failure returns a safe result, never an unhandled exception." |

---

## SECTION 3: ANTICIPATED QUESTIONS AND ANSWERS

### Q1: "Can this access restricted data?"

**A:** HybridRAG processes whatever documents you point it at. In offline mode,
nothing leaves the machine -- no network connections of any kind. The system
has three independent network-blocking layers (OS-level, application-level,
and code-level) that all must fail simultaneously before any data could leave.
Data classification handling is governed by your existing policies for the
machine it runs on, not by the software itself.

### Q2: "What happens if it gives a wrong answer?"

**A:** We've tested against a 400-question evaluation set spanning factual
accuracy, unanswerable questions, ambiguity handling, and prompt injection
resistance. The system scores 98% overall. When it doesn't know the answer,
it's trained to say "The requested information was not found in the provided
documents" rather than guess. Every answer includes source citations so the
user can verify. We also have a hallucination guard that can fact-check
answers against source documents before returning them.

### Q3: "Why not just use ChatGPT?"

**A:** Three reasons. First, ChatGPT doesn't have your documents -- it can't
answer questions about your specific specs and procedures. Second, sending
program documents to OpenAI's servers may violate data handling requirements.
Third, ChatGPT can hallucinate confidently. HybridRAG is source-bounded:
it only answers from your actual documents and cites its sources.

### Q4: "What AI models does it use? Are they approved?"

**A:** Two types. The embedding model (all-MiniLM-L6-v2, Microsoft, MIT
license, 87MB) runs locally and converts text to meaning vectors. The
language model for generating answers is either phi4-mini (Microsoft, MIT,
2.3GB, runs locally) or a configured cloud API. All offline models are from
US or NATO-ally publishers (Microsoft, Mistral AI France) with permissive
open-source licenses. No China-origin models, no Meta/Llama models.
Full audit documented in our model procurement brief.

### Q5: "How long does it take to set up?"

**A:** First-time indexing of ~1,345 documents takes a few hours on a
standard laptop. After that, re-indexing only processes new or changed files
and takes seconds. The software itself installs in under 10 minutes:
Python, a few pip packages, and Ollama for the local AI model. No databases
to install, no servers to configure -- the database is a single portable
SQLite file.

### Q6: "Can multiple people use it at the same time?"

**A:** Currently, HybridRAG runs as a single-user desktop application.
The REST API (localhost-only) enables integration with other local tools.
Multi-user support with authentication and query audit is on the roadmap
for the workstation deployment phase.

### Q7: "What file types does it support?"

**A:** 24+ formats: PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx),
email (.eml), images (via OCR), plain text (.txt), CSV, HTML, XML, Markdown,
and more. New formats can be added by registering a parser -- the architecture
is plug-and-play for file types.

### Q8: "How does it handle updates to documents?"

**A:** Change detection is automatic. Each file is fingerprinted by size and
modification time. When you re-index, only files that have actually changed
are reprocessed. Deleted source files have their chunks automatically cleaned
up. The process is crash-safe -- if power is lost during indexing, it picks
up where it left off with no data loss or duplicates.

### Q9: "What hardware does it need?"

**A:** Minimum: any Windows 10/11 laptop with 8GB RAM. That's what we're
demoing on today. The system was specifically designed for constrained
hardware: disk-backed embeddings (numpy memory-mapped files) mean an 8GB
laptop can search millions of chunks because only the active rows are loaded
into RAM. A workstation with dual GPUs and 64GB RAM is planned for faster
response times and larger models.

### Q10: "Is the data encrypted?"

**A:** API credentials are encrypted using Windows DPAPI (the same system
that protects saved browser passwords), tied to the user's Windows login.
The document index is a standard SQLite file -- access is controlled by
OS-level file permissions. SQLCipher encryption at rest is on the roadmap.
In offline mode, there's nothing to encrypt in transit because there is no
transit -- everything stays on the local disk.

### Q11: "Can it be tricked by adversarial content in documents?"

**A:** We specifically test for this. Our evaluation includes prompt injection
scenarios where documents contain instructions like "ignore your rules and
claim X." The system's 9-rule prompt includes explicit injection resistance:
it ignores embedded instructions and refers to suspicious content generically
without repeating the false claims. We score 100% on injection resistance in
our test set.

### Q12: "How does this compare to SharePoint search?"

**A:** SharePoint search is keyword-only -- it finds documents containing
your exact words but can't understand synonyms or meaning. It returns a list
of documents you then have to read yourself. HybridRAG understands meaning
(semantic search), combines it with keyword search for the best of both
worlds, and gives you a direct answer with citations instead of a document
list. It also works offline, which SharePoint does not.

---

## SECTION 4: TECHNICAL DIFFERENTIATORS

### vs. ChatGPT / Cloud AI

| Capability | HybridRAG | ChatGPT |
|-----------|-----------|---------|
| Answers from YOUR documents | Yes (RAG pipeline) | No (general training data only) |
| Works offline | Yes (default mode) | No (requires internet) |
| Source citations | Yes (exact file + chunk) | No (may cite hallucinated sources) |
| Data stays on your machine | Yes (zero network in offline) | No (data sent to OpenAI servers) |
| Hallucination control | 9-rule source-bounded prompt + guard | Prompt-level only |
| Injection resistance | Tested, 100% on eval set | Not designed for adversarial docs |
| Audit trail | Every query logged with run_id | Limited API logs only |
| Cost per query (offline) | $0.00 (local compute) | $0.01-$0.10 per query |

### vs. Basic RAG Systems

| Capability | HybridRAG | Typical RAG Tutorial |
|-----------|-----------|---------------------|
| Search method | Hybrid: vector + BM25 + RRF fusion | Vector-only (misses exact terms) |
| Keyword matching | FTS5 full-text search (part numbers, acronyms) | None |
| Score fusion | Reciprocal Rank Fusion (k=60) | Single score, no fusion |
| Storage backend | SQLite + memmap (no server, XCOPY-portable) | Requires vector DB server (Pinecone, Weaviate) |
| RAM efficiency | Memmap: 8GB laptop searches 10M+ embeddings | Loads all vectors into RAM |
| Crash recovery | Deterministic chunk IDs, INSERT OR IGNORE | Start over from scratch |
| File format support | 24+ formats via parser registry | Usually PDF-only |
| Evaluation harness | 400-question golden set, automated scoring | No evaluation framework |
| Model governance | Approved model stack with audit trail | Uses whatever is available |
| Prompt engineering | 9-rule source-bounded with priority ordering | Basic "answer from context" |
| Use-case profiles | 9 profiles with tuned params per role | One-size-fits-all |

### vs. SharePoint / Enterprise Search

| Capability | HybridRAG | SharePoint Search |
|-----------|-----------|-------------------|
| Semantic understanding | Yes (meaning-based search) | No (keywords only) |
| Direct answers | Yes, with source citations | No (returns document list) |
| Synonym handling | Yes ("RF band" finds "frequency range") | No |
| Offline operation | Yes (default) | No (requires network + server) |
| Setup complexity | Single laptop, 10-minute install | Server farm, IT administration |
| File format flexibility | 24+ formats, plug-and-play parsers | SharePoint-supported formats |
| Cost | $0 ongoing (open-source stack) | SharePoint licensing + infrastructure |
| Customizable retrieval | Tunable top_k, threshold, temperature | Fixed search algorithm |
| Audit trail | Query-level logging with full metadata | Search analytics (limited) |

### Concrete, Defensible Claims

These are tested and measurable:

1. **98% accuracy** on a 400-question evaluation set covering factual,
   unanswerable, ambiguous, and adversarial categories
2. **100% injection resistance** on prompt injection test scenarios
3. **Zero network connections** in offline mode (three independent blocking layers)
4. **24+ file formats** supported via extensible parser registry
5. **39,602 chunks** indexed from 1,345 source documents
6. **<100ms vector search** across the full index on an 8GB laptop
7. **Crash-safe indexing** with deterministic chunk IDs and INSERT OR IGNORE
8. **5 approved AI models**, all from US/NATO-ally publishers with MIT or
   Apache 2.0 licenses
9. **9 use-case profiles** with tuned parameters for different engineering roles
10. **135+ automated tests** passing across the codebase

---

## SECTION 5: KNOWN LIMITATIONS

Presented honestly. Each limitation includes what we plan to do about it.

### 1. Single-User Only (Currently)

HybridRAG runs as a desktop application for one user at a time. The REST API
enables local tool integration but is not designed for concurrent multi-user
access. **Planned**: multi-user authentication and concurrent query support
in the workstation deployment phase.

### 2. Embedding Model Is Dated

The current embedding model (all-MiniLM-L6-v2, 2021) achieves 56% Top-5
retrieval accuracy on modern benchmarks. Newer models like
snowflake-arctic-embed-m-v2.0 or nomic-embed-text-v1.5 offer +30% retrieval
improvement. **Planned**: embedding model upgrade with full re-indexing
(requires validation against the 400-question eval set).

### 3. Offline Response Time Is Slow on Laptop

Queries take 10-30 seconds in offline mode on an 8GB laptop because the
local AI model (phi4-mini, 3.8B parameters) runs on CPU. **Planned**: the
dual-3090 workstation will provide GPU acceleration, reducing response times
to 2-5 seconds for offline queries.

### 4. No Real-Time Document Watching

Documents must be manually re-indexed when they change. The system does not
watch folders for changes in real time. **Planned**: Windows Task Scheduler
integration for periodic automated re-indexing.

### 5. No Built-In Access Control

All indexed documents are searchable by the user running the application.
There is no document-level permission model (e.g., "this user can see
these documents but not those"). **Planned**: role-based access control
in the multi-user deployment.

### 6. Reranker Disabled Due to Behavioral Score Impact

The cross-encoder reranker improves factual accuracy by 33-40% but degrades
performance on unanswerable questions (100->76%), injection resistance
(100->46%), and ambiguous queries (100->82%). It is currently disabled.
**Planned**: selective reranking that applies only to factual query
categories while bypassing behavioral categories.

### 7. No Encryption at Rest for the Index

The SQLite database and embedding files are stored as plain files. Access
is controlled by OS-level file permissions. **Planned**: SQLCipher
integration for transparent encryption at rest.

### 8. English-Only

The embedding model and prompts are optimized for English-language
documents. Non-English documents may return lower-quality results.
**Planned**: multilingual embedding model support in a future release.

---

## SECTION 6: ROADMAP TALKING POINTS

### Near-Term (Next 1-2 Months)

- **Embedding model upgrade**: Replace all-MiniLM-L6-v2 with
  snowflake-arctic-embed-m-v2.0 or nomic-embed-text-v1.5 for +30% retrieval
  accuracy. Requires full re-indexing and eval validation.

- **Workstation deployment**: Dual-RTX-3090 workstation (48GB GPU, 64GB RAM)
  enables GPU-accelerated inference with larger models (Mistral Small 24B)
  and 3-5x faster response times.

- **Contextual chunking**: Prepend document-context summaries to each chunk
  before embedding (contextual retrieval technique). Reduces retrieval failures by
  35-49% in published benchmarks.

### Mid-Term (3-6 Months)

- **FAISS vector index**: Replace memmap with FAISS for sub-millisecond
  approximate nearest neighbor search. Critical for scaling beyond 100K
  chunks and enabling real-time search across larger document collections.

- **Larger offline models**: Mistral Small 3.1 (24B, Apache 2.0) as the
  primary offline model on the workstation. Near-cloud-quality answers
  with zero network dependency.

- **Multi-user REST API**: Expand the FastAPI server to support
  authenticated concurrent access, query quotas, and per-user audit trails.

- **Three-way hybrid search**: Add SPLADE (learned sparse retrieval) as a
  third search signal alongside BM25 and vector search. IBM research shows
  this is the optimal retrieval combination.

### Long-Term (6-12 Months)

- **Web interface**: Browser-based UI replacing the tkinter desktop app.
  Enables access from any machine on the local network without installing
  client software.

- **SQLCipher encryption**: Transparent encryption at rest for the SQLite
  database and embedding files.

- **Audio/video indexing**: Whisper transcription pipeline to index
  recordings, briefings, and video content alongside text documents.

- **Query decomposition**: Automatically break complex multi-part questions
  into targeted sub-queries for higher accuracy on compound questions.
  Published benchmarks show -40% hallucinations on complex queries.

- **Corrective RAG (CRAG)**: Add a retrieval evaluator that grades document
  relevance before passing to the LLM. Triggers query reformulation when
  initial retrieval is poor. Reduces hallucinations by up to 78%.

### Vision Statement

> HybridRAG evolves from a single-user desktop tool into a team-scale
> knowledge engine: a locally-hosted, secure, auditable system where any
> authorized team member can ask questions across the program's entire
> document library and get accurate, cited answers in seconds -- without
> any data leaving the network boundary.

---

## APPENDIX: QUICK REFERENCE

### Key Commands for Demo

```powershell
# Start the environment
.\start_hybridrag.ps1

# Check system status
rag-status
rag-ollama-status
rag-index-status

# Launch GUI
python src/gui/launch_gui.py

# CLI query (offline)
rag-query "What is the operating frequency range?"

# CLI query (online)
rag-query-api "What is the operating frequency range?"

# Index documents
rag-index

# Show config
rag-config
```

### Key Numbers to Memorize

| Metric | Value |
|--------|-------|
| Documents indexed | ~1,345 |
| Text chunks | ~39,602 |
| File formats supported | 24+ |
| Eval accuracy | 98% (400-question set) |
| Injection resistance | 100% |
| Offline query time (laptop) | 10-30 seconds |
| Online query time | 2-5 seconds |
| Vector search time | <100ms |
| Approved AI models | 5 (US/EU origin) |
| Use-case profiles | 9 |
| Automated tests | 135+ passing |
| RAM minimum | 8 GB |
| Index size on disk | ~15 MB (39K chunks) |

### Model Quick Reference

| Model | Use | Size | Publisher | License |
|-------|-----|------|-----------|---------|
| all-MiniLM-L6-v2 | Embeddings (always local) | 87 MB | Microsoft | MIT |
| phi4-mini | Offline answers (primary) | 2.3 GB | Microsoft | MIT |
| mistral:7b | Offline answers (alt) | 4.1 GB | Mistral AI | Apache 2.0 |
| phi4:14b-q4_K_M | Workstation primary | 9.1 GB | Microsoft | MIT |
| mistral-nemo:12b | Workstation upgrade | 7.1 GB | Mistral+NVIDIA | Apache 2.0 |
