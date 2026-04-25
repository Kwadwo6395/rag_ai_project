# Design Decisions — CS4241 RAG Assistant

**Student:** <STUDENT_NAME>  |  **Index:** <INDEX_NUMBER>  |  **Course:** CS4241 — Introduction to Artificial Intelligence — 2026

This document records the non-obvious choices made while building the system and, for each, the alternatives considered and the reason they were rejected. The goal is to make the reasoning transparent to the grader without expecting them to re-derive it from the code.

---

## 1. Language + framework — Python + Streamlit

Python is the non-negotiable choice on the model side: `sentence-transformers`, `rank_bm25`, `google-generativeai`, and the PDF tooling are all first-class in Python and either absent or second-class in any JavaScript runtime. That fixes the back end. For the UI I considered a Next.js front end calling a Flask or FastAPI back end, but the exam rewards **transparency over polish** — specifically, "show retrieved chunks, similarity scores, prompts, and model output" (Part D). Streamlit collapses each of those from "build a panel, wire it to an endpoint, style it" down to an `st.expander` containing a `st.json(...)`, which means the time saved on plumbing goes directly into retrieval quality and prompt iteration. A decorative SPA would have cost a day and added nothing the grader is graded on.

## 2. Embedding model — `sentence-transformers/all-MiniLM-L6-v2`

MiniLM-L6 is 22 M parameters, 384-dim, indexes the full 1 812 chunks in under 30 s on a laptop, and produces cosine-ready L2-normalised vectors out of the box. Two alternatives were on the table. The first was Google's embedding API (`text-embedding-004`): rejected because the free tier is rate-limited and the indexing pass would be brittle under burst — a single 429 mid-build leaves the corpus half-indexed and the experiment non-reproducible. The second was a larger BGE model (`bge-large-en-v1.5`, 335 M, 1024-dim): rejected because on a corpus this small the accuracy delta is invisible in practice, while the memory and build-time costs are real and the audit story ("why this model?") gets longer. MiniLM is the defensible floor.

## 3. Vector store — custom numpy matrix + pickle metadata

This is the single most deliberate choice in the project. FAISS and Chroma are both one-line drop-ins and both were rejected, for three reasons. (a) The exam explicitly grades **"implement core RAG components manually"** — a `np.ndarray` with a `search(q, k)` doing `vectors @ q` and `argpartition` is the clearest possible evidence that the grader can read in thirty seconds and verify the arithmetic. (b) At 1 812 chunks, FAISS adds ~50 ms of index-load time for exactly zero measurable query-time win over numpy cosine (which is <50 ms total). (c) Chroma's collection abstraction blurs the line between "my retrieval code" and "library retrieval code", which is the wrong side of the "no pre-built pipeline" rule. The trade-off is that we lose incremental-update and persistence-format robustness — acceptable at this scale.

## 4. Keyword index — BM25 via `rank_bm25`

I kept `rank_bm25.BM25Okapi` rather than re-implementing BM25 from scratch, on the reading that the exam's ban is on **RAG pipelines** (LangChain, LlamaIndex), not on textbook retrieval algorithms (BM25, TF-IDF). BM25 has been around since 1994 and `rank_bm25` is a 200-line library; using it is no different from using numpy. That said, the tokeniser (lowercase + `[a-z0-9]+` regex), the corpus-build step, and the integration into `HybridRetriever` are all our code, which is where the interesting design lives anyway.

## 5. Hybrid retrieval with Reciprocal Rank Fusion

Pure dense retrieval fails exactly where this corpus bites hardest: acronyms ("NPP", "NDC", "NHIL"), exact proper nouns (constituency names), and budget section IDs. Pure sparse retrieval fails on the inverse — "What does the budget say about monetary stability?" has no keyword a BM25 tokeniser can latch onto. A cross-encoder re-ranker (e.g. `bge-reranker-base`) would push precision@5 higher but adds ~200 ms per query and another model weight to load, and on this corpus size the RRF-merged top-5 is already clean. I've kept the re-ranker on the deferred list in `architecture.md` as a scale-path rather than a load-bearing component. **RRF specifically** (rather than weighted score mixing) because cosine scores and BM25 scores live on incomparable scales — rank-only fusion is robust to that by construction.

## 6. Query decomposition (innovation — Part G)

The innovation axis for Part G was chosen deliberately against three alternatives: a feedback loop (let the LLM critique its own answer and re-retrieve), conversational memory (use chat history as additional context), and domain-specific scoring (boost elections-tagged chunks for election-sounding queries). All three solve smaller problems than **cross-corpus decomposition** does. The decisive scenario is a query like "Compare NPP's 2020 Ashanti vote share with the 2025 budget's allocation to Ashanti": a flat top-5 on that query returns either all election rows or all budget pages, never both, because the two corpora live in different embedding neighbourhoods. Decomposition turns one such question into two retrieval passes with explicit source tags, merges results, and answers both halves with per-source citations. It is also the single feature that **visually proves itself in the 2-minute video** — the Decomposition expander shows the split in one screenshot. The alternatives, by contrast, are either invisible or inferable only from careful A/B observation.

## 7. LLM — Gemini 1.5 Flash

Free tier, low latency (usually sub-second on these prompt sizes), and — most importantly for the decomposer — reliable structured JSON output at `temperature=0.2`. GPT-4o was considered but its API isn't free and the marginal quality improvement at this prompt complexity isn't visible in manual testing. Claude Sonnet was considered and rejected on the same cost basis. If the Gemini free-tier quota becomes a problem the `GeminiGenerator` wrapper is narrow enough that swapping in a different provider is a single-file change.

## 8. Prompt iteration — V1 → V2 → V3 rather than writing V3 first

Part C explicitly asks for **evidence of iteration** and **evidence of improvement**, so shipping only the final prompt would forfeit marks even if it's the best prompt. V1 is the naïve baseline ("answer the question using the context"). V2 adds the two most common failure modes observed during manual testing: a hallucination guard ("if the context doesn't contain the answer, say so; do not invent facts") and forced `[n]` citation markers. V3 adds the cross-corpus split rule — when the decomposition produces multiple sub-queries with different source tags, the answer must be structured per-source rather than blended. Keeping all three in `rag/prompts.py` (not git-only) means the grader can diff them in one file, and the experiment log in `docs/experiment_logs.md` E3 captures the concrete before/after on a single query.

## 9. Context-window management — drop by RRF score, not insertion order

`trim_context` enforces two soft limits (≤3 000 chars, ≤6 chunks) and, when over budget, drops chunks in ascending order of RRF score. The obvious alternative is to drop by insertion order or by original retriever rank, but RRF already encodes "how strongly do both retrievers agree this chunk is relevant" — it's the richest signal available at trim time, so using anything else would be throwing away information. A future version could swap RRF for a re-ranker cross-encoder score, which is strictly more informative; RRF is the right choice until that re-ranker exists.

## 10. Logging — one JSON file per query, not a combined log

Graders and debuggers both want the same thing: "show me one full interaction". A combined log forces them to grep a timestamp and hope they caught the right boundaries. One file per query (`logs/query_<ts>_<id>.json`) with ordered stage payloads means replay is a single `open`, and the file doubles as the source of truth for the Streamlit Pipeline Trace expander. The trade-off is directory clutter after many queries, which is fine — this is an exam submission, not a production system, and the directory is git-ignored.

## 11. Deployment — Streamlit Community Cloud

Streamlit Cloud deploys straight from a GitHub repo with a single `requirements.txt`, supports env-var secrets for `GEMINI_API_KEY`, and is free. Render and Railway were both considered: both work, both require either a `Procfile` or a `Dockerfile`, and neither is meaningfully better for a Streamlit app. The one real downside of Streamlit Cloud is the cold-start delay on the first request after idle (worker spins up, model loads) — for a grader clicking the deployed URL once to verify the app works, that's a ~20 s wait on first query and then sub-second after. Acceptable.

---

## What we would build next

- **Cross-encoder re-ranker** — insert `bge-reranker-base` between RRF and `trim_context` once adversarial-eval precision@5 drops below ~0.9. Gives a real boost on the hardest queries.
- **Feedback loop** — after generation, have the LLM score whether the context was sufficient; if not, re-decompose with more specific sub-queries and retrieve again. One retrieval round is usually enough, but this would be the next step up on ambiguous queries.
- **PDF header/footer stripping** — the 2025 Budget Statement repeats the header "Resetting the Economy for the Ghana We Want 2025 Budget" on every page, which currently leaks into chunks and occasionally pulls irrelevant pages into the top-5 on generic queries. A cheap regex pass in `rag/ingest.py`'s `load_pdf_pages` would fix this.
- **Incremental indexing** — today `scripts/build_index.py` rewrites the whole index; a `scripts/add_documents.py` that appends to the numpy matrix + metadata would let the assistant absorb new documents (e.g. a mid-year budget review) without a full rebuild.
- **Per-citation chunk previews in the UI** — when the answer cites `[3]`, hovering should surface chunk 3 inline rather than requiring the user to scroll the Retrieved Context expander. Small UX win; nothing blocking it.
