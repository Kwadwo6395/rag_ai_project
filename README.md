# ai_<10022200155> — Academic City RAG Assistant

**Student:** <KWADWO AGYEI NKANSAH>
**Index Number:** <10022200155>
**Course:** CS4241 — Introduction to Artificial Intelligence — 2026
**Lecturer:** Godwin N. Danso
**Examination Date:** 15 April 2026 

RAG chat assistant that answers questions over Ghana's 2020/2024 presidential election results and the 2025 Budget Statement. Built without LangChain, LlamaIndex, or any pre-built RAG pipeline — all core components (chunking, embedding, retrieval, prompting, pipeline) are hand-implemented.

## Live demo

- **Deployed URL:** <https://kwadwo6395-rag-ai-project-app-izbewx.streamlit.app>
- **2-minute video walkthrough:** https://drive.google.com/file/d/1e-tGLQM_qa0mYYBEAsZl-C8y_lc1R4wy/view?usp=share_link

## Local setup

> **New to Python / this project?** Follow the step-by-step guide:
> - Mac or Linux → [`docs/setup.md`](docs/setup.md)
> - Windows → [`docs/setup-windows.md`](docs/setup-windows.md)
>
> 



```bash
# 1. Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2. Data
curl -L -o data/Ghana_Election_Result.csv \
  https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv
curl -L -o data/2025-Budget-Statement-and-Economic-Policy_v4.pdf \
  https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf

# 3. Gemini key
cp .env.example .env


# 4. Build the index (~30s)
python scripts/build_index.py

# 5. Run the app
streamlit run app.py
```

## Running tests

```bash
source .venv/bin/activate
pytest
```

## Evaluation (manual, Part E)

```bash
python evaluation/run_eval.py
# writes evaluation/results.json
```

Then fill in `docs/experiment_logs.md` by hand.

## Repo structure

```
ai_<INDEX_NUMBER>/
├── app.py
├── rag/
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_store.py
│   ├── retrieval.py
│   ├── decomposer.py
│   ├── prompts.py
│   ├── generator.py
│   ├── pipeline.py
│   └── logging_utils.py
├── scripts/build_index.py
├── evaluation/
│   ├── adversarial_queries.py
│   └── run_eval.py
├── tests/
├── data/
├── index/
├── logs/
├── docs/
│   ├── architecture.md
│   ├── design_decisions.md
│   ├── experiment_logs.md
│   ├── video_script.md
│   └── plans/  # original design + implementation plan
├── README.md
└── requirements.txt
```

## How each exam Part is addressed

| Part | Marks | Where |
|---|---|---|
| A — Data + chunking | 4 | `rag/chunking.py` + `docs/experiment_logs.md` E1 |
| B — Retrieval (+ failure cases) | 6 | `rag/retrieval.py`, `rag/bm25_store.py`, `rag/vector_store.py` + `docs/experiment_logs.md` E2 |
| C — Prompting (3 iterations) | 4 | `rag/prompts.py` (V1/V2/V3) + `docs/experiment_logs.md` E3 |
| D — Full pipeline + logging | 10 | `rag/pipeline.py`, `rag/logging_utils.py`, Streamlit debug panel |
| E — Adversarial eval | 6 | `evaluation/` + `docs/experiment_logs.md` E4 |
| F — Architecture | 8 | `docs/architecture.md` |
| G — Innovation (query decomposition) | 6 | `rag/decomposer.py` |
| Application | 4 | `app.py` + deployed URL |
| Video walkthrough | 4 | `docs/video_script.md` (recorded separately) |
| Manual experiment logs | 4 | `docs/experiment_logs.md` |
| Documentation | 4 | this README + `docs/design_decisions.md` |


## Design references

Full design + plan (from brainstorming): `docs/plans/`
