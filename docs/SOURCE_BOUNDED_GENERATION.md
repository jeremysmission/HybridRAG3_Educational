# ============================================================================
# HybridRAG — Source-Bounded Generation Requirements
# ============================================================================
# STATUS: Approved requirement, implementation deferred to query pipeline phase
# PHASE: Post-indexing (after first successful index run)
# PRIORITY: Band 4 critical — this is the #1 differentiator for production RAG
# ============================================================================

## Objective

The RAG system must ensure that all generated answers are derived **only**
from retrieved indexed documents (manuals, reports, engineering documentation).
The LLM must function as a **summarizer of retrieved content**, not as an
independent knowledge source.

This requirement applies equally to:
- **Offline mode** (local Llama3 model)
- **Online mode** (API GPT model)

The only difference between modes is the generator model.
Retrieval behavior must remain identical.

---

## 1. System Architecture Requirement

Pipeline must follow this sequence:

```
User Query
    ↓
Retriever
    ↓
Retrieval Validation (Gate)
    ↓
Context Builder
    ↓
LLM Generation (Summarization Only)
    ↓
Answer + Source Citations
```

The LLM must **never** determine whether sufficient information exists.
The **retriever** determines this.

---

## 2. Retrieval Gate (MANDATORY)

The system must prevent LLM generation when retrieval evidence is insufficient.

### Required Logic

```python
chunks = retrieve(query)

if max_similarity < SIMILARITY_THRESHOLD:
    return "No supporting documentation found in indexed sources."
```

### Minimum requirements:
- Similarity threshold (configurable via `config.retrieval.min_score`)
- Minimum number of retrieved chunks (recommended: 2–3)
- **No LLM call if retrieval fails**

### Purpose
Prevent model answers based on pretrained knowledge.

### Implementation notes:
- `config.retrieval.min_score` already exists (default 0.20)
- Add `config.retrieval.min_chunks` (default 2)
- Gate logic goes in query_engine.py BEFORE the LLM call

---

## 3. Context Isolation Requirement

The LLM must be instructed to operate only on retrieved context.

### Required System Instruction Pattern

```
You must answer using ONLY the provided documentation.
If the answer is not contained in the context, respond:
"The indexed documentation does not contain this information."
Do not use outside knowledge.
Do not infer or guess missing steps.
```

### Implementation notes:
- System prompt template stored in config or a prompt template file
- Same template used for BOTH offline (Ollama) and online (GPT) modes
- Template includes `{context}` placeholder filled by Context Builder

---

## 4. Citation Requirement (MANDATORY)

Every answer must include traceable sources.

### Required Output Structure

```
Answer:
<generated response>

Sources:
- Document_Name, Section/Page (score: 0.83)
- Document_Name, Section/Page (score: 0.79)
```

### Preferred implementation:
- Each retrieved chunk assigned an ID (S1, S2, S3…)
- Model references chunk IDs or document metadata
- GUI displays sources alongside answer

### What's already captured at index time:
- `source_path` → document name
- `heading` → section hierarchy
- `page_number` → page reference (PDFs)
- `chunk_index` → stable position reference

---

## 5. Online Mode Constraint

Online GPT models must behave identically to offline models with respect to:
- Retrieval gate
- Context isolation
- Citation requirement
- Refusal behavior

Online mode must **NOT**:
- Answer without retrieval
- Supplement answers with general knowledge
- Add external information

The generator model changes; system behavior does not.

---

## 6. Post-Generation Validation (Recommended)

After generation:
- Verify that each major statement is supported by at least one retrieved chunk
- If unsupported content is detected:
  - Regenerate with stricter instructions, OR
  - Remove unsupported statements

### Optional metric:

```
attribution_coverage = sentences_with_citations / total_sentences
```

**Target: ≥ 90%**

### Implementation notes:
- Simple heuristic: check if answer sentences contain terms from retrieved chunks
- Advanced: use a small classifier model to verify grounding
- Log metric in `query_log` table for evaluation

---

## 7. Failure Modes to Enforce

The system must explicitly support:

### A) No Evidence
```
No supporting documentation found in indexed sources.
```

### B) Weak Evidence (Optional)
Allow answer but indicate limited support:
```
Answer (LOW CONFIDENCE — limited source coverage):
<response>
```

### C) Normal Operation
Answer with citations.

**The system must never silently fall back to model knowledge.**

---

## 8. Design Principle (Non-Negotiable)

The LLM is not an authority.
The retriever determines available knowledge.
The LLM only formats and summarizes retrieved content.

```
Answer = f(retrieved_chunks)
```

NOT:

```
Answer = f(model_knowledge + retrieved_chunks)
```

---

## 9. Indexer Requirements Supporting This Behavior

Indexer must preserve:
- Document name ✓ (source_path)
- Section or heading hierarchy ✓ (heading field)
- Page or stable location ✓ (page_number field)
- Revision/date if available (future: metadata extraction)

This metadata must be passed to the retriever and included in citations.

---

## Implementation Phases

### Phase 1 (During indexing pipeline — CURRENT)
- [x] Chunk metadata preserved (source_path, heading, page_number)
- [x] Similarity threshold in config
- [x] Run audit trail in SQLite

### Phase 2 (Query pipeline — NEXT)
- [ ] Retrieval Gate in query_engine.py
- [ ] Context-only system prompt template
- [ ] Citation formatting in LLM response
- [ ] "No evidence" refusal behavior
- [ ] `query_log` table (question, chunks, answer, timestamp)

### Phase 3 (Evaluation — AFTER QUERY WORKS)
- [ ] Attribution coverage metric
- [ ] Post-generation validation
- [ ] Evaluation dashboard or report script
- [ ] `min_chunks` config parameter

### Phase 4 (GUI — FINAL)
- [ ] Sources displayed alongside answer
- [ ] Confidence indicators
- [ ] Click-to-open source document
- [ ] Feedback mechanism (correct/incorrect)
