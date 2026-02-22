# HybridRAG3 -- System Architecture

> Block diagram showing data flow through the system.
> Read top to bottom. Left column = query path, right column = indexing path.

---

## Boot Sequence (runs once at startup)

```
  config/default_config.yaml
          |
          v
    +------------+      +----------------+      +--------------+
    |  CONFIG    |----->| CREDENTIALS    |----->| NETWORK GATE |
    |  loader    |      | keyring / env  |      | offline lock |
    +------------+      +----------------+      +--------------+
          |                     |                       |
          +----------+----------+-----------+-----------+
                     |
                     v
              +-----------+
              |   BOOT    |
              |  pipeline |
              +-----------+
                     |
          +----------+----------+
          |                     |
          v                     v
    +-----------+         +-----------+
    | Ollama    |         | API Client|
    | check     |         | factory   |
    +-----------+         +-----------+
```

---

## Query Path (user asks a question)

```
    "What is the calibration procedure?"
                     |
                     v
            +----------------+
            | QUERY ENGINE   |
            +----------------+
                     |
                     v
            +----------------+
            |   EMBEDDER     |
            | MiniLM-L6-v2   |
            | query -> 384d  |
            +----------------+
                     |
                     v
            +----------------+
            |   RETRIEVER    |
            |  hybrid search |
            +----------------+
                /          \
               v            v
      +-----------+   +-----------+
      | BM25      |   | Vector    |
      | keyword   |   | cosine    |
      | (FTS5)    |   | (memmap)  |
      +-----------+   +-----------+
               \          /
                v        v
            +----------------+
            | Reciprocal     |
            | Rank Fusion    |
            | + min_score    |
            +----------------+
                     |
                     v
            top_k chunks (scored, ranked)
                     |
                     v
            +----------------+
            | PROMPT BUILDER |
            | 9-rule template|
            | (injection-    |
            |  resistant)    |
            +----------------+
                     |
                     v
            +----------------+
            |  LLM ROUTER    |
            +----------------+
                /          \
               v            v
      +-----------+   +-----------+
      | OFFLINE   |   | ONLINE    |
      | Ollama    |   | API call  |
      | localhost  |   | (gated)  |
      | phi4-mini |   +-----------+
      +-----------+        |
               \           v
                \   +-----------+
                 \  | NETWORK   |
                  \ | GATE      |
                   \| audit log |
                    +-----------+
                         /
                v-------'
            +----------------+
            | QUERY RESULT   |
            | answer, sources|
            | tokens, cost,  |
            | latency        |
            +----------------+
                     |
                     v
              User sees answer
```

---

## Indexing Path (building the search index)

```
    {SOURCE_DIR}
    (1,345 files)
           |
           v
    +-------------+
    |   INDEXER   |
    +-------------+
           |
           v
    +-------------+
    | File scan   |
    | + validator |
    | (skip bad)  |
    +-------------+
           |
           v
    +-------------+
    | Hash check  |
    | (skip if    |
    |  unchanged) |
    +-------------+
           |
           v
    +-------------+
    | File parser |
    | PDF, DOCX,  |
    | TXT, MD     |
    +-------------+
           |
           v
    +-------------+
    |  CHUNKER    |
    | 1200 chars  |
    | 200 overlap |
    | smart split |
    +-------------+
           |
           v
    +-------------+
    |  EMBEDDER   |
    | MiniLM-L6-v2|
    | batch embed |
    | -> 384d     |
    +-------------+
           |
           v
    +-------------+
    | VECTOR STORE|
    +-------------+
        /      \
       v        v
  +--------+ +--------+
  | SQLite | | Memmap |
  | chunks | | vectors|
  | meta   | | 384d   |
  | FTS5   | | float32|
  +--------+ +--------+

  39,602 chunks indexed
```

---

## Storage Layer

```
  {DATA_DIR}\
       |
       +-- hybridrag.db          SQLite: chunks, metadata, FTS5 index
       |
       +-- embeddings.npy        Memmap: embedding vectors (384d float32)
       |
       +-- file_hashes.json      Change detection for incremental indexing
```

---

## Security Layers

```
  +--------------------------------------------------+
  |              NETWORK GATE                        |
  |  Mode        Allowed destinations                |
  |  --------    --------------------------------    |
  |  OFFLINE     localhost:11434 (Ollama) only       |
  |  ONLINE      localhost + approved API endpoint   |
  +--------------------------------------------------+
          |
          v
  +--------------------------------------------------+
  |          CREDENTIAL MANAGER                      |
  |  Priority    Source                               |
  |  --------    --------------------------------    |
  |  1st         Windows Credential Manager (DPAPI)  |
  |  2nd         Environment variables               |
  |  3rd         Config file (not recommended)       |
  +--------------------------------------------------+
          |
          v
  +--------------------------------------------------+
  |          EMBEDDING LOCKDOWN                      |
  |  HF_HUB_OFFLINE=1 enforced at startup           |
  |  Model loaded from local cache only              |
  +--------------------------------------------------+
```

---

## User Interfaces

```
  +-------------------+    +-------------------+    +-------------------+
  |    PowerShell     |    |       GUI         |    |     REST API      |
  |                   |    |                   |    |                   |
  | . .\start_        |    | .\tools\          |    | rag-server        |
  |   hybridrag.ps1   |    |   launch_gui.ps1  |    |                   |
  |                   |    |                   |    | localhost:8000    |
  | rag-query "..."   |    | tkinter window    |    | /query            |
  | rag-index         |    | dark/light theme  |    | /index            |
  | rag-status        |    | mode toggle       |    | /health           |
  +-------------------+    +-------------------+    +-------------------+
           \                       |                       /
            +-----------+----------+-----------+----------+
                        |
                        v
                 QUERY ENGINE
                 (same pipeline)
```
