# RAG Chatbot — Design Doc

**Course:** CS4241 — Introduction to Artificial Intelligence (Academic City University)
**Exam due:** 2026-04-25 (10-day window from 2026-04-15)
**Student:** `<STUDENT_NAME>` — Index `<INDEX_NUMBER>`
**Repo name (required by exam):** `ai_<INDEX_NUMBER>`

## 1. Scope

Build a Retrieval-Augmented-Generation chatbot that answers questions over two
Academic City-provided corpora:

- `Ghana_Election_Result.csv` — tabular election results.
- `2025-Budget-Statement-and-Economic-Policy_v4.pdf` — long-form government budget document.

**Hard constraint from the exam:** no LangChain, no LlamaIndex, no pre-built
RAG pipelines. Core RAG components (chunking, embedding, retrieval, prompting)
must be hand-implemented. Vector databases and tokenisers are allowed;
end-to-end orchestration frameworks are not.

## 2. Stack

- **Language:** Python 3.11
- **UI:** Streamlit (required-option: Streamlit / Flask / Next.js)
- **Embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, ~80MB, runs locally)
- **Vector store:** custom — `numpy` matrix + pickle metadata list, cosine similarity
- **Keyword index:** `rank_bm25` (BM25Okapi) — BM25 arithmetic is not what the exam grades
- **LLM:** Gemini 1.5 Flash via `google-generativeai` (free tier)
- **Ingestion:** `pypdf` (budget PDF), `pandas` (election CSV)
- **Deployment:** Streamlit Community Cloud, GitHub-linked
- **Tests:** `pytest`

## 3. Repo Layout

```
ai_<index_number>/
├── app.py                    # Streamlit entrypoint
├── rag/
│   ├── __init__.py
│   ├── ingest.py             # load + clean CSV and PDF
│   ├── chunking.py           # row-based (CSV) + recursive-char (PDF)
│   ├── embeddings.py         # sentence-transformers wrapper
│   ├── vector_store.py       # numpy matrix + cosine search
│   ├── bm25_store.py         # BM25 index
│   ├── retrieval.py          # hybrid search + RRF
│   ├── decomposer.py         # query decomposition (Part G)
│   ├── prompts.py            # prompt templates + context budgeter
│   ├── generator.py          # Gemini client
│   ├── pipeline.py           # orchestration + per-stage logging
│   └── logging_utils.py
├── scripts/
│   └── build_index.py        # ingest → chunk → embed → persist
├── evaluation/
│   ├── adversarial_queries.py
│   └── run_eval.py           # RAG vs pure-LLM comparison
├── data/                     # raw CSV + PDF
├── index/                    # vectors.npy, metadata.pkl, bm25.pkl
├── logs/                     # per-query JSON logs
├── docs/
│   ├── architecture.md
│   ├── experiment_logs.md    # hand-written; required deliverable
│   ├── design_decisions.md
│   └── video_script.md
├── tests/
├── README.md
├── requirements.txt
└── .streamlit/config.toml
```

Every `.py` file has a header comment with the student's name + index number.

## 4. Data Pipeline (Exam Part A — 4 marks)

### 4.1 Cleaning

- **CSV:** strip whitespace, normalise party codes (NPP / NDC / CPP / other),
  coerce numeric vote columns, drop fully-null rows, lower-case constituency
  names for join-friendliness, keep a display-cased column for output.
- **PDF:** `pypdf` page-by-page extraction; collapse repeated whitespace,
  drop running headers/footers by regex (page numbers, ministry header).

### 4.2 Chunking

Two chunkers — the corpora are structurally different:

- **CSV row chunker:** one chunk per row, serialised to a natural-language
  sentence so embeddings encode it well. Example:
  `"In the 2020 Ghana general election, Ayawaso West Wuogon constituency
  (Greater Accra region): NPP received 47,201 votes (53.2%), NDC received
  39,950 votes (45.0%), …"`
  **Rationale:** each row is one constituency-year-party observation, the
  natural semantic unit for a "who won where" query. Chunking multiple rows
  together dilutes retrieval signal.

- **PDF recursive-character chunker:** hand-rolled. Splits on
  `\n\n` → `\n` → `. ` → space, never mid-word. **Chunk size 800 chars,
  overlap 150 chars (~18%).**
  **Rationale:** 800 chars ≈ 150 words ≈ 1–2 paragraphs; preserves one
  complete idea without diluting semantic density. 150-char overlap
  preserves sentences that cross boundaries.

### 4.3 Comparative analysis (required deliverable)

Run retrieval with chunk sizes **400 / 800 / 1500** across a hand-written
10-question eval set; log `recall@5` and subjective answer quality in
`docs/experiment_logs.md`. Expected pattern: 400 hurts recall for
multi-sentence questions, 1500 hurts precision for narrow facts, 800 wins.

## 5. Retrieval System (Exam Part B — 6 marks)

### 5.1 Embeddings

- `all-MiniLM-L6-v2`, batch size 32.
- Unit-normalise at encoding time → cosine = dot product.
- Written once by `scripts/build_index.py`.

### 5.2 Vector store (custom)

- `vectors.npy` — `(N, 384)` float32 matrix.
- `metadata.pkl` — list of dicts: `{id, source, page_or_row, text, tokens}`.
- Cosine search: `scores = vectors @ q`; top-k via `np.argpartition`.
- `<50ms` on this corpus. FAISS noted in docs as scaling path but not used.

### 5.3 BM25 keyword store

- `rank_bm25.BM25Okapi` over tokenised chunks (lower-case, strip punctuation,
  whitespace split). Persisted via pickle.

### 5.4 Hybrid search + RRF (the "extend retrieval" requirement)

1. Dense: top-20 by cosine.
2. Sparse: top-20 by BM25.
3. **Reciprocal Rank Fusion:** `score(doc) = Σ 1 / (k + rank)`, `k = 60`.
4. Final top-5 to the LLM.

Each candidate carries dense score, BM25 score, and RRF score — shown in
Streamlit debug panel.

### 5.5 Failure cases + fix (Part B "critical task")

Three scripted queries where vector-only retrieval misses:
- Acronym only (e.g., "NHIL") — lexical hit, weak dense.
- Misspelled constituency (e.g., "Ayawso West") — dense should rescue.
- Rare proper noun absent from embedding model's training distribution.

For each: log dense-only result → log BM25-only result → log hybrid result.
The hybrid IS the fix; the artefact is a before/after table in
`docs/experiment_logs.md`.

## 6. Prompting & Generation (Exam Part C — 4 marks)

### 6.1 Prompt template

```
You are an assistant answering questions about Ghana's recent elections and
the 2025 national budget. Use ONLY the numbered context blocks below.
If the answer is not present, reply EXACTLY:
"I don't have that information in the provided sources."

Context:
[1] (Source: <doc>, <page|row>) <chunk text>
[2] ...

Conversation so far:
<last-N-turns>

User question: <query>

Answer — cite each claim with its context number, e.g., [1]:
```

### 6.2 Iterations (required deliverable)

Three versions logged in `docs/experiment_logs.md` with the same query and
annotated output:

- **v1** — bare (above, minus the "reply EXACTLY" line).
- **v2** — adds explicit hallucination guard + forced `[n]` citations.
- **v3** — adds "If the question spans both corpora, address each separately".

### 6.3 Context window management

- Hard cap: 6 chunks or 3000 tokens (whichever hits first).
- Drop lowest-RRF chunks when exceeded.
- If RRF gap between rank 3 and 4 exceeds 40%, truncate at 3.

## 7. Full Pipeline & Logging (Exam Part D — 10 marks)

Flow: `User Query → Decomposer → Retrieval → Context Selection → Prompt →
LLM → Response`.

Per-query log (JSON lines, one file per query in `logs/`) with these stages:
`query_received`, `decomposition`, `retrieval(dense|bm25|hybrid)`,
`context_selection`, `prompt_constructed`, `llm_response`.
Each stage records latency, inputs, outputs, and scores.

Streamlit sidebar surfaces all of this live: retrieved chunks, similarity
scores, final prompt, raw LLM response.

## 8. Innovation — Query Decomposition (Exam Part G — 6 marks)

Cross-dataset questions ("compare the NPP's Ashanti vote share with budget
allocations to Ashanti") require reasoning across two corpora. Flat RAG
cannot do this — it retrieves the closest chunks from one distribution.

**Approach:**

1. **Decompose.** A Gemini call with a few-shot prompt returns JSON:
   `{sub_queries: [...], sources: ['elections'|'budget'|'both']}`.
2. **Retrieve per sub-query.** Each sub-query runs hybrid search, scoped to
   its source when the classifier specifies one.
3. **Merge.** Deduplicate by chunk id; re-rank merged candidates by RRF
   across sub-queries.
4. **Synthesise.** Second Gemini call answers using merged context.

The decomposition tree is rendered in the UI so the grader sees it working.

**Novelty argument:** Flat RAG would answer only half the question; multi-step
decomposition with per-sub-query scoping is a well-known research pattern
(HyDE / Self-Ask) implemented manually here.

## 9. Evaluation (Exam Part E — 6 marks)

### 9.1 Adversarial queries (minimum 2)

1. **Ambiguous** — *"Which party won more?"* (no year, no metric, no place).
2. **Misleading** — *"What was the 2025 budget allocation for the 2024
   elections?"* — crosses both corpora; the answer may exist in one, neither,
   or require inference.

### 9.2 Metrics (manual, logged by hand)

- **Accuracy** — correct / partial / wrong / refused (judged against source).
- **Hallucination rate** — count of fabricated claims per response.
- **Consistency** — each query run 3× at identical params; variance noted.

### 9.3 RAG vs pure-LLM baseline

Same queries asked with and without retrieval. Side-by-side table in
`docs/experiment_logs.md`. Per the exam: "evidence-based comparison, not
opinion."

## 10. Architecture (Exam Part F — 8 marks)

Mermaid diagram in `docs/architecture.md`:

```
┌─ Build time ─────────────────────┐
│  CSV + PDF → Ingest → Chunk →    │
│  Embed (MiniLM) → vectors.npy    │
│                  → metadata.pkl  │
│                  → bm25.pkl      │
└──────────────────────────────────┘

┌─ Runtime ────────────────────────────────────────────────┐
│  UI (Streamlit)                                          │
│    │                                                     │
│    ▼                                                     │
│  Decomposer (Gemini) → sub-queries                       │
│    │                                                     │
│    ▼                                                     │
│  Hybrid Retriever = Dense (cosine) ∪ Sparse (BM25)       │
│    │         then Reciprocal Rank Fusion                 │
│    ▼                                                     │
│  Context Selector (budget + RRF cutoffs)                 │
│    │                                                     │
│    ▼                                                     │
│  Prompt Builder → Gemini → Answer                        │
│    │                                                     │
│    ▼                                                     │
│  Logger → logs/*.json + UI debug panel                   │
└──────────────────────────────────────────────────────────┘
```

**Justification (condensed):** dual-corpus domain demands (a) exact-term
matching for proper nouns & budget-section numbers → BM25; (b) semantic
matching for prose budget questions → dense vectors; (c) cross-corpus
queries → decomposition. Each component earns its place.

## 11. Deployment

- Streamlit Community Cloud, linked to GitHub.
- Pre-built index committed to `index/` (few MB).
- Gemini API key in Streamlit secrets.
- Cold start ~30s (model load); `@st.cache_resource` keeps it warm.

## 12. Marks Map (sanity — all 60 accounted for)

| Part | Marks | Primary artefact |
|---|---|---|
| A — Data + chunking | 4 | `rag/chunking.py` + chunk-size comparison in experiment_logs |
| B — Retrieval | 6 | `rag/retrieval.py` + failure-case table |
| C — Prompting | 4 | `rag/prompts.py` + 3 prompt versions logged |
| D — Full pipeline + logging | 10 | `rag/pipeline.py` + Streamlit debug panel |
| E — Adversarial eval | 6 | `evaluation/` + RAG-vs-LLM table |
| F — Architecture | 8 | `docs/architecture.md` |
| G — Innovation | 6 | `rag/decomposer.py` |
| Application | 4 | `app.py` + deployed URL |
| Video walkthrough | 4 | `docs/video_script.md` (recorded by student) |
| Manual experiment logs | 4 | `docs/experiment_logs.md` (hand-filled) |
| Documentation | 4 | `docs/design_decisions.md` + `README.md` |
| **Total** | **60** | |

## 13. Required housekeeping

- Repo named `ai_<INDEX_NUMBER>` on GitHub.
- `README.md` leads with student name + index.
- Every `.py` has a header comment with name + index.
- Add `GodwinDansoAcity` as a GitHub collaborator.
- Email `godwin.danso@acity.edu.gh` with GitHub + deployed URL.
  Subject: `CS4241-Introduction to Artificial Intelligence-2026:[INDEX NAME]`.
- Record ≤2-min walkthrough per `docs/video_script.md`.
