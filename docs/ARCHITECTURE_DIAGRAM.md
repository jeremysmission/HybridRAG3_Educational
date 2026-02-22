# HybridRAG3 -- System Architecture

> Block diagrams showing data flow through the system.
> All diagrams read top to bottom.
>
> **Color key (Query and Indexing diagrams):**
> - Blue = **Retrieval** (finding relevant information)
> - Orange = **AI Generation** (creating the answer)
> - Purple = **Handoff** (where retrieval output meets generation input)
> - Gray = **Infrastructure** (security, routing)

---

## Boot Sequence (runs once at startup)

```
              +-------------------+
              |       BOOT        |
              |     pipeline      |
              +-------------------+
                       |
                       v
              +-------------------+
              |  1. Load config   |
              |     (YAML)        |
              +-------------------+
                       |
                       v
              +-------------------+
              |  2. Resolve       |
              |     credentials   |
              |     (keyring/env) |
              +-------------------+
                       |
                       v
              +-------------------+
              |  3. Configure     |
              |     network gate  |
              |     (set mode)    |
              +-------------------+
                       |
                       v
              +-------------------+
              |  4. Probe         |
              |     backends      |
              |     (Ollama+API)  |
              +-------------------+
                       |
                       v
              +-------------------+
              |    BOOT RESULT    |
              |   success flag    |
              |   api_client      |
              |   warnings[]      |
              +-------------------+
                       |
                       v
             System ready for use
```

---

## Query Path (user asks a question)

```mermaid
flowchart TD
    Q(["User asks a question"]):::user

    Q --> QE["QUERY ENGINE<br/><i>orchestrates full pipeline</i>"]:::handoff

    %% ── Retrieval phase (blue) ──────────────────────
    QE --> EMB["EMBEDDER<br/>MiniLM-L6-v2<br/>query → 384-dim vector"]:::retrieval
    EMB --> RET["RETRIEVER<br/>hybrid search"]:::retrieval

    RET --> BM25["BM25 keyword<br/>search (FTS5)"]:::retrieval
    RET --> VEC["Vector cosine<br/>search (memmap)"]:::retrieval

    BM25 --> RRF["Reciprocal Rank Fusion<br/>+ min_score filter"]:::retrieval
    VEC --> RRF

    %% ── Handoff: retrieval output → generation input ─
    RRF --> CHUNKS(["top_k chunks<br/><i>retrieval output → generation input</i>"]):::handoff

    %% ── Generation phase (orange) ───────────────────
    CHUNKS --> PB["PROMPT BUILDER<br/>9-rule injection-resistant template"]:::generation
    PB --> LLM["LLM ROUTER"]:::generation

    LLM --> OFF["OFFLINE<br/>Ollama<br/>localhost"]:::generation
    LLM --> ON["ONLINE<br/>API endpoint"]:::generation

    ON --> GATE["NETWORK GATE<br/>check + audit log"]:::infra

    OFF --> RESULT
    GATE --> RESULT

    RESULT["QUERY RESULT<br/>answer, sources<br/>tokens, cost, latency"]:::handoff
    RESULT --> USER(["User sees answer"]):::user

    %% ── Styles ──────────────────────────────────────
    classDef retrieval fill:#1976D2,stroke:#0D47A1,color:#fff
    classDef generation fill:#F57C00,stroke:#E65100,color:#fff
    classDef handoff fill:#7B1FA2,stroke:#4A148C,color:#fff
    classDef infra fill:#546E7A,stroke:#37474F,color:#fff
    classDef user fill:#FAFAFA,stroke:#BDBDBD,color:#333
```

---

## Indexing Path (building the search index)

```mermaid
flowchart TD
    SRC(["Source document folder<br/>PDF, DOCX, TXT, ..."]):::user

    SRC --> IDX["INDEXER"]:::retrieval

    IDX --> SCAN["File scan + validator<br/><i>skip corrupt files</i>"]:::retrieval
    SCAN --> HASH["Hash check<br/><i>skip unchanged files</i>"]:::retrieval
    HASH --> PARSE["File parser<br/>24+ formats"]:::retrieval
    PARSE --> CHUNK["CHUNKER<br/>1200 chars, 200 overlap<br/>smart boundary split"]:::retrieval
    CHUNK --> EMBED["EMBEDDER<br/>MiniLM-L6-v2<br/>batch embed → 384d"]:::retrieval

    EMBED --> STORE["VECTOR STORE"]:::handoff

    STORE --> SQL[("SQLite<br/>chunks, metadata<br/>FTS5 index, hashes")]:::data
    STORE --> MEM[("Memmap<br/>float16 vectors<br/>shape N x 384")]:::data

    %% ── Styles ──────────────────────────────────────
    classDef retrieval fill:#1976D2,stroke:#0D47A1,color:#fff
    classDef handoff fill:#7B1FA2,stroke:#4A148C,color:#fff
    classDef data fill:#388E3C,stroke:#1B5E20,color:#fff
    classDef user fill:#FAFAFA,stroke:#BDBDBD,color:#333
```

> The entire indexing pipeline is **retrieval-side** (blue) -- it builds
> the search index that the query path reads from. The green database
> icons at the bottom are the **shared storage** that connects the two paths.

---

## Storage Layer

```
     <indexed data directory>/
            |
            +-- hybridrag.sqlite3       SQLite: chunks, metadata, FTS5, file hashes
            |
            +-- embeddings.f16.dat      Memmap: float16 vectors, shape [N, 384]
            |
            +-- embeddings_meta.json    Bookkeeping: dim, count, dtype
```

---

## Security Layers

```
     +----------------------------------------------------+
     |                 NETWORK GATE                        |
     |                                                    |
     |   OFFLINE    localhost:11434 (Ollama) only          |
     |   ONLINE     localhost + approved API endpoint      |
     +----------------------------------------------------+
                          |
                          v
     +----------------------------------------------------+
     |              CREDENTIAL MANAGER                     |
     |                                                    |
     |   1st  Windows Credential Manager (DPAPI)          |
     |   2nd  Environment variables                        |
     |   3rd  Config file (not recommended)                |
     +----------------------------------------------------+
                          |
                          v
     +----------------------------------------------------+
     |              EMBEDDING LOCKDOWN                     |
     |                                                    |
     |   HF_HUB_OFFLINE=1 enforced at startup             |
     |   Model loaded from local cache only                |
     +----------------------------------------------------+
```

---

## User Interfaces

```mermaid
flowchart TD
    PS["PowerShell<br/>start_hybridrag<br/>rag-query, rag-index"]:::ui
    GUI["GUI<br/>launch_gui.ps1<br/>tkinter, dark/light"]:::ui
    API["REST API<br/>rag-server<br/>localhost:8000"]:::ui

    PS --> ENGINE
    GUI --> ENGINE
    API --> ENGINE

    ENGINE["QUERY ENGINE<br/><i>same pipeline for all three</i>"]:::handoff

    classDef ui fill:#FAFAFA,stroke:#BDBDBD,color:#333
    classDef handoff fill:#7B1FA2,stroke:#4A148C,color:#fff
```

---

## Color Legend

| Color | Meaning | Examples |
|-------|---------|---------|
| Blue | **Retrieval** -- finding relevant information | Embedder, Retriever, BM25, Vector search, RRF |
| Orange | **Generation** -- AI creates the answer | Prompt Builder, LLM Router, Ollama, API call |
| Purple | **Handoff** -- retrieval meets generation | Query Engine, top_k chunks, Query Result |
| Green | **Storage** -- persistent data | SQLite, Memmap files |
| Gray | **Infrastructure** -- security and routing | Network Gate |
