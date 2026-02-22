# HybridRAG3 -- Glossary of Terms and Acronyms

Last Updated: 2026-02-21

---

## Acronyms

| Acronym | Full Name | Meaning |
|---------|-----------|---------|
| API | Application Programming Interface | A defined way for software programs to communicate with each other |
| ASGI | Asynchronous Server Gateway Interface | Python standard for async web servers (used by FastAPI/uvicorn) |
| BM25 | Best Matching 25 | A ranking algorithm for keyword-based text search; the standard behind most search engines |
| BOM | Byte Order Mark | A special character at the start of a file that identifies its encoding (required for PowerShell UTF-8 files) |
| CRLF | Carriage Return + Line Feed | Windows-style line endings (\r\n) |
| CPU | Central Processing Unit | The main processor in a computer |
| DPAPI | Data Protection Application Programming Interface | Windows built-in encryption that ties data to a specific user login |
| FTS5 | Full-Text Search version 5 | SQLite's built-in full-text search engine |
| GPU | Graphics Processing Unit | A specialized processor that can run AI models much faster than a CPU |
| GUI | Graphical User Interface | A visual application with windows, buttons, and menus (as opposed to a command line) |
| HTTP | Hypertext Transfer Protocol | The protocol used for web communication |
| HTTPS | HTTP Secure | Encrypted version of HTTP |
| IVF | Inverted File Index | A FAISS index structure that partitions vectors into clusters for faster approximate search |
| JSON | JavaScript Object Notation | A lightweight data format used for configuration and data exchange |
| KV Cache | Key-Value Cache | Memory used by LLMs during text generation to store intermediate computation results |
| LLM | Large Language Model | An AI model trained on text that can understand and generate human language (e.g., GPT, phi4-mini, Mistral) |
| MIT | MIT License | A permissive open-source license allowing free use, modification, and distribution |
| NLI | Natural Language Inference | An AI task that determines if one statement supports, contradicts, or is neutral toward another |
| OCR | Optical Character Recognition | Technology that converts images of text into machine-readable text |
| PDF | Portable Document Format | Adobe's document format |
| PII | Personally Identifiable Information | Data that can identify a specific individual (names, SSNs, etc.) |
| PyPI | Python Package Index | The official repository for Python software packages (pypi.org) |
| RAG | Retrieval-Augmented Generation | A technique where an AI finds relevant documents first, then generates an answer using only those documents |
| RAM | Random Access Memory | A computer's short-term working memory |
| REST | Representational State Transfer | An architectural style for web APIs using standard HTTP methods |
| RRF | Reciprocal Rank Fusion | An algorithm that combines multiple ranked lists into one by scoring items based on their rank position |
| SQ8 | Scalar Quantization 8-bit | A compression method that stores each vector dimension as an 8-bit integer instead of a 32-bit float |
| SQL | Structured Query Language | The standard language for database queries |
| SQLite | SQL Lite | A self-contained, file-based database engine requiring no server |
| VRAM | Video RAM | Memory on a GPU, used for loading AI models and running inference |
| YAML | YAML Ain't Markup Language | A human-readable configuration file format |

---

## Technical Terms

| Term | Definition |
|------|------------|
| **offline** | A computer or network that is physically isolated from the internet and other unsecured networks |
| **Audit trail** | A chronological record of system activities that allows reconstruction of what happened, when, and by whom |
| **Batch processing** | Processing multiple items at once instead of one at a time, for efficiency |
| **Boot pipeline** | The sequence of startup checks and initialization steps that run when HybridRAG starts |
| **Chunk** | A small piece of text (about 1,200 characters / half a printed page) that a document is split into for indexing and searching |
| **Cosine similarity** | A mathematical measure of how similar two vectors are, based on the angle between them. Ranges from -1 (opposite) to 1 (identical). Used to compare meaning |
| **Cross-encoder** | An AI model that takes two texts as input and directly scores their relevance to each other. More accurate than embedding comparison but much slower |
| **Crash-safe** | Designed so that an unexpected shutdown (power failure, crash) does not corrupt data or require starting over |
| **Dataclass** | A Python feature for creating structured data objects with typed fields and automatic validation |
| **Deterministic** | Producing the same output every time given the same input. HybridRAG uses deterministic chunk IDs so re-indexing the same file produces the same IDs |
| **Dot product** | A mathematical operation that multiplies corresponding elements of two vectors and sums the results. Used for fast similarity comparison |
| **Embedding** | A list of numbers (vector) that represents the meaning of a piece of text. Texts with similar meanings have similar embeddings |
| **Embedding model** | An AI model that converts text into embeddings. HybridRAG uses all-MiniLM-L6-v2 (384 dimensions, ~87 MB) |
| **Endpoint** | A specific URL that a service listens on for requests (e.g., `http://localhost:8000/query`) |
| **FAISS** | Facebook AI Similarity Search -- a library for fast approximate nearest-neighbor search on large vector collections |
| **FastAPI** | A modern Python web framework for building REST APIs |
| **Faithfulness** | How accurately an AI's answer reflects the source documents. A faithfulness score of 1.0 means every claim is supported by the sources |
| **Float16 / float32** | Numeric precision formats. Float16 uses 2 bytes per number (half precision), float32 uses 4 bytes (full precision). Float16 saves storage with minimal quality loss |
| **FTS5** | SQLite's full-text search engine. Creates a searchable keyword index that supports BM25 ranking |
| **Graceful degradation** | The ability to continue operating with reduced functionality when a component fails, rather than crashing entirely |
| **Hallucination** | When an AI generates information that sounds plausible but is not in the source documents. A key risk in RAG systems |
| **Hash-based change detection** | Comparing a file's current fingerprint (size + modification time) to a stored fingerprint to determine if it has changed |
| **Hybrid search** | Combining two or more search methods (typically semantic vector search and keyword search) for better results than either alone |
| **Indexing** | The process of reading documents, splitting them into chunks, computing embeddings, and storing everything in a searchable database |
| **Inference** | Running an AI model to produce output (as opposed to training it). In HybridRAG, inference means generating an answer from a prompt |
| **Lifespan** | In FastAPI, the startup/shutdown lifecycle of the application. Used to initialize and clean up resources |
| **Localhost** | The network address for "this computer" (127.0.0.1). Connections to localhost never leave the machine |
| **Memmap** | Memory-mapped file. A technique where a file on disk is mapped into memory so it can be read as if it were in RAM, without loading the entire file |
| **Network Gate** | HybridRAG's centralized access control system that checks every outbound network connection against an allowlist |
| **Ollama** | A program that runs LLMs locally on your computer. HybridRAG uses it for offline answer generation |
| **Overlap** | The number of characters shared between adjacent chunks. Ensures important information near chunk boundaries appears in at least one complete chunk |
| **Parser** | A module that reads a specific file format and extracts its text content |
| **Pipeline** | A sequence of processing steps where each step's output becomes the next step's input |
| **Profile** | A pre-configured set of performance parameters tuned for specific hardware (laptop_safe, desktop_power, server_max) |
| **Prompt** | The instructions and context sent to an LLM. Includes a system prompt (rules), context (retrieved chunks), and the user's question |
| **Quantization** | Reducing the precision of model weights (e.g., from 32-bit to 4-bit) to reduce file size and memory usage. Q4_K_M is a common quantization level |
| **Query** | A natural language question that a user asks the system |
| **Recall** | The percentage of relevant results that are actually found by a search. 95% recall means 5% of relevant results are missed |
| **Reciprocal Rank Fusion** | An algorithm for merging ranked result lists: `score = 1/(k + rank)`. Chunks ranked highly in multiple lists get the highest combined scores |
| **Reranker** | A second-pass model that re-scores search results for better accuracy. Slower but more precise than first-pass retrieval |
| **Retriever** | The component that searches the indexed database and returns the most relevant chunks for a given query |
| **Run tracker** | A logging system that records metadata about each indexing run (start time, file counts, errors) for auditing |
| **Semantic search** | Searching by meaning rather than exact keywords. "RF operating band" matches "antenna frequency range" because they mean similar things |
| **Sentence-transformers** | A Python library that provides pre-trained embedding models for converting text to vectors |
| **Source-bounded generation** | The requirement that an LLM must only use information from the retrieved source documents, never its pre-trained knowledge |
| **Structured logging** | Logging where each entry is a data object with typed fields (timestamp, level, module, message) rather than plain text |
| **Tkinter** | Python's built-in GUI toolkit. Included with Python, no additional installation needed |
| **Token** | The smallest unit of text that an LLM processes. Roughly 1 token = 0.75 words. Used for cost estimation |
| **Top-K** | The number of highest-scoring search results to return. top_k=12 means the 12 most relevant chunks are sent to the LLM |
| **Vector** | An ordered list of numbers. In RAG, a 384-dimensional vector represents the meaning of a text chunk |
| **Vector store** | A database optimized for storing and searching embedding vectors |
| **Virtual environment** | An isolated Python installation that keeps project dependencies separate from the system Python |
| **Zero-trust** | A security model that assumes no network connection is safe by default. Everything must be explicitly allowed |
