# Experiment Logs — Manual (not AI-generated)

**Student:** <STUDENT_NAME>  |  **Index:** <INDEX_NUMBER>

> **Instructions to self:** fill in each blank after running the corresponding experiment. Do not paste AI-written summaries. Short notes / observations in your own words are the point — this is the manual-logs deliverable (4 marks).

---

## E1. Chunking — size & overlap comparison (Part A, 4 marks)

**What I did:** Rebuilt the PDF index three times with chunk sizes 400, 800, 1500 (overlap fixed at 150). Ran the same 10 hand-written questions against each and scored recall@5 — a question counts as "hit" if at least one of the top-5 retrieved chunks contains the ground-truth answer text.

**How to reproduce:** temporarily change `CHUNK_SIZE` in `rag/chunking.py`, rerun `python scripts/build_index.py`, then paste each question into the Streamlit app and check the Retrieved Context expander.

**Eval questions** (write these BEFORE running — mix fact-lookup questions with multi-sentence/inferential ones so the chunk size actually matters):

1. _______________________________________________________________
2. _______________________________________________________________
3. _______________________________________________________________
4. _______________________________________________________________
5. _______________________________________________________________
6. _______________________________________________________________
7. _______________________________________________________________
8. _______________________________________________________________
9. _______________________________________________________________
10. ______________________________________________________________

| Chunk size | Recall@5 | Notes on answer quality |
|---|---|---|
| 400  | ___ / 10 | |
| 800  | ___ / 10 | |
| 1500 | ___ / 10 | |

**Conclusion (which size I shipped and why, in my own words):**
______________________________________________________________
______________________________________________________________
______________________________________________________________

---

## E2. Retrieval — failure cases & hybrid fix (Part B, 6 marks)

**What I did:** Ran three deliberately hard queries that each target a known weakness of one retriever. For each query I compared the dense-only top hit, the BM25-only top hit, and the hybrid (RRF) top hit. I toggled retrievers by temporarily disabling one branch in `HybridRetriever` and reading scores out of the Retrieved Context expander.

| Query | Dense-only top hit (source, score) | BM25-only top hit (source, score) | Hybrid top hit (source, RRF) | Which wins + why |
|---|---|---|---|---|
| `<acronym e.g. "What is NHIL?">` | | | | |
| `<misspelled constituency e.g. "Who won Ashnti 2020?">` | | | | |
| `<rare proper noun I found in the corpus>` | | | | |

**Fix summary (in my own words — why hybrid beat each single retriever, and any surprises):**
______________________________________________________________
______________________________________________________________
______________________________________________________________

---

## E3. Prompt iteration (Part C, 4 marks)

**Query used** (pick ONE query and reuse it across V1, V2, V3 so the comparison is apples-to-apples):
______________________________________________________________

### v1 (baseline — `SYSTEM_RULES_V1`, bare instructions)

Output:
> _______________________________________________________________
> _______________________________________________________________
> _______________________________________________________________

Observations (hallucinations / citations / tone):
______________________________________________________________
______________________________________________________________

### v2 (+ hallucination guard + forced `[n]` citations — `SYSTEM_RULES_V2`)

Output:
> _______________________________________________________________
> _______________________________________________________________
> _______________________________________________________________

Observations (what changed from v1):
______________________________________________________________
______________________________________________________________

### v3 (+ cross-corpus split rule — `SYSTEM_RULES_V3`)

Output:
> _______________________________________________________________
> _______________________________________________________________
> _______________________________________________________________

Observations (what changed from v2):
______________________________________________________________
______________________________________________________________

**Evidence of improvement (the exam explicitly asks for this):**
______________________________________________________________
______________________________________________________________
______________________________________________________________

---

## E4. Adversarial queries — RAG vs pure LLM (Part E, 6 marks)

Each query run **3×** for RAG and 3× for the pure-LLM baseline (no retrieval). Source data is `evaluation/results.json` after running `python evaluation/run_eval.py`.

Rubric for the accuracy / hallucination cells:
- **Accuracy:** ✓ (correct or correctly refused), ✗ (wrong answer), ~ (hedged / partially correct)
- **Hallucination:** yes / no (did the model invent a fact not in the sources?)

### Ambiguous: "Which party won more?"

| Run | RAG answer (short) | Pure-LLM answer (short) | RAG accuracy | Pure-LLM accuracy | RAG halluc. | Pure-LLM halluc. |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

**Consistency notes (RAG):** __________________________________________
**Consistency notes (pure LLM):** _____________________________________

### Misleading: "What was the 2025 budget allocation for the 2024 elections?"

| Run | RAG answer (short) | Pure-LLM answer (short) | RAG accuracy | Pure-LLM accuracy | RAG halluc. | Pure-LLM halluc. |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

**Consistency notes (RAG):** __________________________________________
**Consistency notes (pure LLM):** _____________________________________

---

## Summary (evidence-based, per exam instructions)

- **Accuracy delta (RAG − pure LLM):** __________________________________
- **Hallucination-rate delta:** _________________________________________
- **Consistency observations (variance across 3 runs):** ________________
  ______________________________________________________________
- **When pure LLM beats RAG (if ever):** ________________________________
  ______________________________________________________________
- **When RAG beats pure LLM:** _________________________________________
  ______________________________________________________________
- **One-line takeaway:** _______________________________________________
