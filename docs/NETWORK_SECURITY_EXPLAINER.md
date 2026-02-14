# HybridRAG v3 — Network, Security & Architecture Explainer
**Last Updated: 2026-02-08**
**Distribution: Internal / Briefing Use**

---

## What Is HybridRAG?

HybridRAG is a document search and question-answering system. You give it
documents (PDFs, Word files, etc.), it reads and indexes them, and then you can
ask it questions in plain English. It finds the most relevant passages and uses
an AI model to write an answer based ONLY on your documents — not the internet.

---

## The Three Phases (When Things Happen)

```
 PHASE 1: INDEXING              PHASE 2: SEARCHING           PHASE 3: ANSWERING
 (runs once per document set)   (runs every query)           (runs every query)

 Read documents                 Convert your question         Send top passages +
 Break into chunks              into a vector                 question to LLM
 Convert chunks to              Search for similar            LLM writes answer
   numeric vectors                chunk vectors                 using ONLY those
 Store in database              Also keyword search             passages
                                Combine and rerank

 NETWORK: NONE                  NETWORK: NONE                 NETWORK: DEPENDS
 Everything is local            Everything is local            Offline = Ollama
                                                               Online  = API call
```

Key takeaway: Phases 1 and 2 NEVER touch the network. Phase 3 only touches
the network if you explicitly switch to online/API mode.

---

## What Talks to the Internet (and What Doesn't)

| Component | Purpose | Needs Internet? | Details |
|-----------|---------|----------------|---------|
| Document Parsers | Read PDFs, DOCX, etc. | NEVER | Pure file reading |
| Chunker | Split text into pieces | NEVER | Pure text processing |
| Embedding Model | Convert text to vectors | NEVER | Runs 100% locally (~87MB cached) |
| Vector Search | Find similar chunks | NEVER | Math on local vectors |
| FTS5 Search | Keyword matching | NEVER | SQLite full-text search |
| Reranker | Re-score results | NEVER | Cross-encoder runs locally (~80MB cached) |
| Ollama (offline LLM) | Generate answers | NEVER | Runs on localhost:11434 only |
| API LLM (online mode) | Generate answers | YES | Calls configured API endpoint ONLY |
| HuggingFace Hub | Download AI models | BLOCKED | Only needed once to cache models |

---

## How the Network Lockdown Works

### Layer 1: Environment Variables (set by start_hybridrag.ps1)

```
HF_HUB_OFFLINE = 1              Blocks huggingface.co connections
TRANSFORMERS_OFFLINE = 1         Blocks transformers library network calls
HF_HUB_DISABLE_TELEMETRY = 1    Blocks usage tracking / analytics
```

What these do: The HuggingFace and Transformers Python libraries check these
variables before making ANY network request. If set to "1", they refuse to
connect and use only locally cached model files.

Why this matters: Without these, every time you load the embedding model or
reranker, the library tries to check huggingface.co for model updates,
download new tokenizer files, and send telemetry (usage statistics) back
to HuggingFace. This is unacceptable in a restricted environment.

### Layer 2: Application Kill Switch

```
HYBRIDRAG_NETWORK_KILL_SWITCH = true
```

What this does: Our own Python code checks this variable before making any
HTTP request. Even if Layer 1 is somehow bypassed, our code will refuse to
connect. This is the enterprise-in-depth layer.

### Layer 3: Mode Setting (config/default_config.yaml)

```yaml
mode: offline
```

What this does: The LLM router reads this setting. In "offline" mode, it
sends prompts to Ollama (localhost only). In "online" mode, it sends prompts
to the configured API endpoint and NOWHERE ELSE.

### How They Work Together

```
 You run a query
      |
      v
 Embedding model loads --> HF_HUB_OFFLINE=1 --> Uses cached model file
      |                     (no internet check)
      v
 Search runs locally (no network involved)
      |
      v
 Reranker loads --> HF_HUB_OFFLINE=1 --> Uses cached model file
      |               (no internet check)
      v
 LLM Router checks mode
      |
      +-- mode=offline --> Sends to Ollama (localhost:11434)
      |                     Kill switch does not block localhost
      |
      +-- mode=online  --> Checks KILL_SWITCH
                            |-- kill_switch=true  --> BLOCKED, returns error
                            +-- kill_switch=false --> Sends to API endpoint ONLY
                                                      HuggingFace STILL blocked
```

IMPORTANT: Switching to API/online mode does NOT unblock HuggingFace.
The HF lockdown and the API mode are completely independent controls.

---

## Offline Mode vs Online Mode

### Offline Mode (Default)

- LLM: Ollama running locally (llama3, qwen2.5, or llama2)
- Network: Zero outbound connections
- Speed: ~2-3 minutes per query (CPU-bound LLM generation)
- Quality: Good for 7B parameter models
- Use when: On corporate network, restricted environments, no internet

### Online / API Mode

- LLM: Cloud API (OpenAI, Anthropic Claude, etc.)
- Network: ONLY connects to configured API URL
- Speed: ~5-15 seconds per query
- Quality: Much better answers from larger models
- Use when: Home/unrestricted network, need higher quality answers
- HuggingFace: STILL BLOCKED even in online mode

### What Gets Sent to the API (Online Mode Only)

When you ask a question in online mode, ONLY this goes to the API:
1. Your question (the text you typed)
2. The top 5 most relevant text chunks (small excerpts, not full documents)
3. A system prompt telling the LLM to answer from those chunks only

What NEVER gets sent:
- Full documents
- File paths or filenames (unless they appear in chunk text)
- Your database
- Your embedding vectors
- Any other files on your computer

---

## Model Cache: Why and How

### The Problem

AI models (embedding model, reranker) are large files (80-400MB). They must be
downloaded from HuggingFace the first time you use them. After that, they are
stored locally (cached) and never need the internet again.

### Where Models Live

```
{PROJECT_ROOT}\
  .model_cache\     Sentence-transformers models (embedding)
  .hf_cache\        HuggingFace hub cache (reranker, tokenizers)
  .torch_cache\     PyTorch model weights
```

These folders are set by start_hybridrag.ps1 so models stay WITH the project,
not scattered in your user profile. This makes the project portable.

### Models You Need Cached

| Model | Purpose | Size | How to Cache |
|-------|---------|------|--------------|
| all-MiniLM-L6-v2 | Embedding (text to vectors) | ~87MB | Auto-cached on first rag-index |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker (re-score results) | ~80MB | rag-download-models |
| llama3 (Ollama) | Answer generation (offline) | ~4.7GB | ollama pull llama3 |

### First-Time Setup (One Time, With Internet)

```powershell
# 1. Temporarily allow downloads
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"

# 2. Cache all HuggingFace models
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('All models cached')
"

# 3. Re-enable lockdown (or just re-source the startup script)
. .\start_hybridrag.ps1

# 4. Cache Ollama model (separate from HuggingFace)
ollama pull llama3
```

After this, the system never needs internet again unless you switch to API mode.

---

## Available Offline LLM Models

These are already downloaded and ready to use with Ollama:

| Model | Size | Best For | How to Select |
|-------|------|----------|---------------|
| llama3 (8B) | 4.7 GB | General Q&A, best quality | Default in config |
| qwen2.5:7b-instruct | 5.4 GB | Follows instructions precisely | Change ollama_model in YAML |
| llama2 (7B) | 3.8 GB | Legacy, not recommended | Smallest but worst quality |

To change which model Ollama uses, edit config/default_config.yaml:
```yaml
ollama_model: "llama3"          # or "qwen2.5:7b-instruct"
```

---

## Security Audit Checklist

For anyone reviewing this system's security posture:

| Check | Status | Evidence |
|-------|--------|----------|
| No hardcoded API keys | PASS | rag-diag scans all files |
| No public endpoints | PASS | Ollama binds to localhost only |
| HuggingFace blocked | PASS | HF_HUB_OFFLINE=1 in startup script |
| Telemetry disabled | PASS | HF_HUB_DISABLE_TELEMETRY=1 |
| Kill switch exists | PASS | HYBRIDRAG_NETWORK_KILL_SWITCH=true |
| Network-capable files identified | PASS | rag-diag lists them (only llm_router.py) |
| Source documents stay local | PASS | Never sent to any API |
| Only chunks sent to LLM | PASS | 5 chunks max per query, not full documents |
| Audit logging | PASS | All queries logged with timestamps |
| Mode visible at startup | PASS | start_hybridrag.ps1 prints lockdown status |

---

## Common Questions

**Q: Do my documents get sent to the internet?**
A: In offline mode, never. In online/API mode, only small text snippets (chunks)
are sent to the API, never full documents. The API only sees the ~5 most
relevant passages plus your question.

**Q: Can I run this with zero internet forever?**
A: Yes. Once models are cached and you stay in offline mode, no internet is
needed. Ollama runs entirely on your CPU/GPU.

**Q: What if I switch to API mode by accident?**
A: The kill switch (HYBRIDRAG_NETWORK_KILL_SWITCH=true) blocks all API calls
even if mode is set to online. You must explicitly set kill_switch to false
AND set mode to online for API calls to work. Two independent safeties.

**Q: Why is the query slow (2-3 minutes)?**
A: In offline mode, Ollama generates text using your laptop's CPU. The
retrieval/search part takes ~1-2 seconds. The remaining time is Ollama
thinking. A GPU would make this 10-20x faster. Online/API mode would
make it ~5-15 seconds total.

**Q: What is the embedding model?**
A: It is a small AI model (all-MiniLM-L6-v2, 87MB) that converts text into
a list of 384 numbers (a "vector"). Similar texts get similar numbers. This
is how the system finds relevant passages without keyword matching. It runs
100% locally and never contacts the internet.

**Q: What is the reranker?**
A: After the initial search finds ~20 candidate passages, the reranker
(cross-encoder/ms-marco-MiniLM-L-6-v2) re-scores each one by looking at
the query AND the passage together. This catches passages that are relevant
but were missed by simple vector similarity. It runs 100% locally.

**Q: What is "hybrid search"?**
A: We search two ways simultaneously:
  1. Vector search - finds passages with similar MEANING (semantic)
  2. FTS5/BM25 search - finds passages with matching KEYWORDS (exact words)
Then we merge results using Reciprocal Rank Fusion (RRF), which combines
both rankings into one. This catches things that either method alone would miss.

**Q: How do I explain this to my manager?**
A: "It is a local document search system that reads our files, indexes them,
and answers questions using only the content in those files. In offline mode,
nothing leaves the computer. In online mode, only small text excerpts are
sent to a cloud API for better answer quality. All AI models for search run
locally. The system has multiple network safety layers and audit logging."

---

## Architecture Summary (One-Page View)

```
YOUR DOCUMENTS (PDFs, DOCX, etc.)
       |
       v
  [Parsers] --> Extract text from each file type
       |
       v
  [Chunker] --> Break into ~500-word pieces with overlap
       |
       v
  [Embedding Model] --> Convert each chunk to 384-dim vector
       |                 (all-MiniLM-L6-v2, runs locally)
       |
       v
  [SQLite Database] --> Store chunks + vectors + FTS5 index
       |
       |  (indexing complete, database ready)
       |
       v
  YOU ASK A QUESTION
       |
       v
  [Embedding Model] --> Convert question to vector
       |
       +---> [Vector Search] --> Top 20 by meaning
       |
       +---> [FTS5 Search] ----> Top 20 by keywords
       |
       v
  [RRF Fusion] --> Merge and deduplicate both result sets
       |
       v
  [Reranker] --> Re-score top candidates for true relevance
       |          (cross-encoder, runs locally)
       v
  Top 5 chunks selected
       |
       v
  [LLM] --> Generate answer from chunks + question
       |
       +-- Offline: Ollama (localhost, no network)
       +-- Online:  API call (configured endpoint only)
       |
       v
  ANSWER + SOURCE CITATIONS
```

---

## Quick Reference Commands

| Command | What It Does |
|---------|-------------|
| `. .\start_hybridrag.ps1` | Activate environment (run first, every session) |
| `rag-index` | Index all documents in source folder |
| `rag-query "your question"` | Search and get an answer |
| `rag-diag` | Run system health check |
| `rag-diag --verbose` | Detailed health check with all test results |
| `rag-status` | Quick DB stats and network status |
| `rag-paths` | Show all configured paths and env vars |
