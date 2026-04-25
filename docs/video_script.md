# Video Walkthrough Script — 2 minutes

**Student:** <KWADWO AGYEI NKANSAH>  |  **Index:** <10022200155>  |  **Course:** CS4241 — Introduction to Artificial Intelligence — 2026

> **Pre-recording checklist**
> - Open the deployed Streamlit app at `<DEPLOYED_URL>` in one browser tab, already logged in / warmed up (send one throwaway query first so the cold start is out of the way).
> - Keep this script in a second tab (or on paper) for the teleprompter.
> - Have these three queries ready to paste:
>   1. `What was the total education allocation in the 2025 budget?`
>   2. `Compare NPP's performance in Ashanti in 2020 with the 2025 budget's priorities for that region.`
>   3. `What was the 2025 budget allocation for the 2024 elections?`
> - Screen-record at 1080p, mic close, quiet room. Aim for 1:55–2:05.

---

## 0:00–0:15 — Intro

**[Camera on face OR on the app title bar]**

> "Hi, I'm <STUDENT_NAME>, index <INDEX_NUMBER>. This is my RAG chat assistant for CS4241. It answers questions over two sources — Ghana's recent presidential election results and the 2025 national budget statement. Everything — chunking, embeddings, retrieval, prompts — is implemented manually. No LangChain. No LlamaIndex."

---

## 0:15–0:45 — Demo 1: basic query + transparency

**[Paste query 1: `What was the total education allocation in the 2025 budget?` — hit Enter]**

**[While it runs — keep talking, don't wait]**

> "Under the hood: I decompose the query into source-tagged sub-queries, run hybrid retrieval — both dense cosine similarity on sentence-transformer embeddings and BM25 keyword search — and fuse them with Reciprocal Rank Fusion. Then I build a prompt with explicit citation rules and send it to Gemini."

**[Answer lands. Point to each expander in turn, one-to-two seconds each]**

> "Here are the retrieved chunks with their RRF scores. Here's the decomposition — one sub-query, tagged `budget`. Here's the full prompt that went to Gemini. And here's the stage-by-stage trace that was also written to `logs/query_…json`."

---

## 0:45–1:15 — Demo 2: cross-corpus query (Part G innovation)

**[Paste query 2: `Compare NPP's performance in Ashanti in 2020 with the 2025 budget's priorities for that region.`]**

**[While it runs]**

> "This is where query decomposition earns its keep. A flat RAG can't retrieve from two corpora simultaneously — whichever one dominates the top-k starves the other. My decomposer splits this into an elections sub-query and a budget sub-query, retrieves each, merges the results, and the prompt forces a per-source answer with citations."

**[Open the Decomposition expander — point to the two sub-queries and their source tags]**

> "You can see the split right here — `elections` for the Ashanti 2020 question, `budget` for the 2025 priorities question. The answer cites both."

---

## 1:15–1:45 — Demo 3: adversarial query

**[Paste query 3: `What was the 2025 budget allocation for the 2024 elections?`]**

**[Show the refusal or hedged answer]**

> "This is a misleading query — the 2025 budget doesn't allocate to past elections. The system refuses, or hedges with explicit citations, because the prompt rules I iterated in V2 and V3 forbid invented facts. The pure-Gemini baseline, without retrieval, tends to hallucinate a confident number here. I documented the three-run RAG-vs-LLM comparison for this exact query in `docs/experiment_logs.md`."

---

## 1:45–2:00 — Close

**[Back to the app title or terminal showing the repo]**

> "Repo: `github.com/<USERNAME>/ai_<INDEX_NUMBER>`. Deployed at `<DEPLOYED_URL>`. Full documentation — architecture, design decisions, experiment logs — in `docs/`. Thank you."

---

## Recovery lines (use if a query stalls or misfires)

- **App is slow / cold start:** "Streamlit Cloud spin-up — one moment. While we wait, let me show you the retrieved-context expander from the previous query…"
- **Refusal on query 1:** "Interesting — the retriever didn't surface a single total-allocation figure. Let me show you the chunks it did find and why."
- **Hallucination on query 3 RAG side:** "The model slipped past the refusal rule here — this is exactly the kind of case documented in E4 of the experiment log, which is why the three-run average matters."
