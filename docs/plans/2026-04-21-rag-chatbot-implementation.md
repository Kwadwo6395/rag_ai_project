# RAG Chatbot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a deployed RAG chat assistant for Academic City's CS4241 exam that answers questions over the Ghana Election Result CSV and the 2025 Budget Statement PDF, with all core RAG components implemented manually (no LangChain / LlamaIndex).

**Architecture:** Python + Streamlit. Two chunkers (row-based for CSV, recursive-char for PDF) → local `all-MiniLM-L6-v2` embeddings → custom numpy vector store + BM25 keyword store → Reciprocal Rank Fusion → Gemini 1.5 Flash. Query decomposition routes cross-corpus questions to per-source retrieval then synthesises.

**Tech Stack:** Python 3.11, `sentence-transformers`, `numpy`, `rank_bm25`, `pypdf`, `pandas`, `google-generativeai`, `streamlit`, `pytest`.

**Design reference:** `docs/plans/2026-04-21-rag-chatbot-design.md`

**Conventions for every task:**
- Every new `.py` starts with a header comment:
  `# Student: <STUDENT_NAME>  |  Index: <INDEX_NUMBER>  |  File: <relative path>`
  (Keep the placeholders as literal strings; they'll be filled in once before submission.)
- Follow TDD where noted. Skip tests only where the task is pure wiring (e.g. Streamlit UI, docs).
- Commits are bundled at the end of each task. User has requested no commits for now — treat the `git` steps as optional until user says otherwise. Still run `git status` at task boundaries so progress is visible.

---

## Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `README.md` (stub)
- Create: `.gitignore`
- Create: `.streamlit/config.toml`
- Create: `rag/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create directories: `data/`, `index/`, `logs/`, `scripts/`, `evaluation/`, `docs/`

**Step 1: Create directory tree**

Run:
```bash
mkdir -p rag tests scripts evaluation docs data index logs .streamlit
touch rag/__init__.py tests/__init__.py
```

**Step 2: Write `requirements.txt`**

```
streamlit==1.39.0
sentence-transformers==3.3.1
numpy==1.26.4
rank-bm25==0.2.2
pypdf==5.1.0
pandas==2.2.3
google-generativeai==0.8.3
python-dotenv==1.0.1
pytest==8.3.3
```

**Step 3: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
logs/*.json
!logs/.gitkeep
data/*.csv
data/*.pdf
!data/.gitkeep
.DS_Store
```

Also create `data/.gitkeep`, `logs/.gitkeep`.

**Step 4: Write `.streamlit/config.toml`**

```toml
[server]
headless = true

[theme]
base = "light"
```

**Step 5: Write `README.md` stub**

```markdown
# ai_<INDEX_NUMBER>

**Student:** <STUDENT_NAME>
**Index Number:** <INDEX_NUMBER>
**Course:** CS4241 — Introduction to Artificial Intelligence — 2026

RAG chat assistant for Academic City, built for the end-of-semester exam.

## Setup
(TBD — filled in at the end.)

## Deployed URL
(TBD)
```

**Step 6: Write `tests/conftest.py`** (shared fixtures placeholder)

```python
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
```

**Step 7: Create virtualenv and install**

Run:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Expected: clean install; `pip list` shows streamlit, sentence-transformers, etc.

**Step 8 (optional commit):** skip for now per user instruction.

---

## Task 2: Download datasets

**Files:**
- Download: `data/Ghana_Election_Result.csv`
- Download: `data/2025-Budget-Statement-and-Economic-Policy_v4.pdf`

**Step 1: Fetch the CSV**

Run:
```bash
curl -L -o data/Ghana_Election_Result.csv \
  https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv
```

Expected: non-empty CSV. Verify with `head -5 data/Ghana_Election_Result.csv`.

**Step 2: Fetch the PDF**

Run:
```bash
curl -L -o data/2025-Budget-Statement-and-Economic-Policy_v4.pdf \
  https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf
```

Expected: PDF file > 1MB.
Verify: `file data/2025-Budget-Statement-and-Economic-Policy_v4.pdf` reports "PDF document".

**Step 3: Inspect CSV schema**

Run: `head -3 data/Ghana_Election_Result.csv` and record the real columns in a scratch note — chunker logic (Task 4) depends on actual column names. If columns differ from `Constituency, Region, Year, Party, Votes, Percentage`, adapt Task 4 accordingly.

---

## Task 3: Data ingestion module

**Files:**
- Create: `rag/ingest.py`
- Create: `tests/test_ingest.py`

**Step 1: Write failing test**

```python
# tests/test_ingest.py
from rag.ingest import load_csv, load_pdf_pages

def test_load_csv_returns_cleaned_dataframe():
    df = load_csv("data/Ghana_Election_Result.csv")
    assert len(df) > 0
    assert df.isnull().all(axis=1).sum() == 0  # no fully-null rows
    # every column name is lower-snake-case
    assert all(c == c.lower().replace(" ", "_") for c in df.columns)

def test_load_pdf_pages_returns_list_of_nonempty_strings():
    pages = load_pdf_pages("data/2025-Budget-Statement-and-Economic-Policy_v4.pdf")
    assert len(pages) > 10
    assert all(isinstance(p, str) for p in pages)
    assert any(len(p) > 200 for p in pages)
```

**Step 2: Run — expect failure**

Run: `pytest tests/test_ingest.py -v`
Expected: `ModuleNotFoundError: No module named 'rag.ingest'`.

**Step 3: Implement `rag/ingest.py`**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/ingest.py
import re
import pandas as pd
from pypdf import PdfReader

_HEADER_FOOTER_RE = re.compile(r"^\s*(Page\s+\d+|\d+\s*\|.*)$", re.IGNORECASE)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_pdf_pages(path: str) -> list[str]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not _HEADER_FOOTER_RE.match(ln.strip())
        ]
        cleaned = re.sub(r"[ \t]+", " ", "\n".join(lines))
        pages.append(cleaned.strip())
    return pages
```

**Step 4: Run — expect pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 2 passed.

---

## Task 4: CSV row chunker

**Files:**
- Create: `rag/chunking.py`
- Create: `tests/test_chunking.py`

**Step 1: Write failing test**

```python
# tests/test_chunking.py
import pandas as pd
from rag.chunking import chunk_csv_rows, Chunk

def test_chunk_csv_rows_produces_one_chunk_per_row():
    df = pd.DataFrame([
        {"constituency": "Ayawaso West", "region": "Greater Accra",
         "year": 2020, "party": "NPP", "votes": 47201, "percentage": 53.2},
        {"constituency": "Ayawaso West", "region": "Greater Accra",
         "year": 2020, "party": "NDC", "votes": 39950, "percentage": 45.0},
    ])
    chunks = chunk_csv_rows(df, source="elections.csv")
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    # Chunk text is natural-language, contains the values
    assert "Ayawaso West" in chunks[0].text
    assert "NPP" in chunks[0].text
    assert "47,201" in chunks[0].text or "47201" in chunks[0].text
    assert chunks[0].metadata["source"] == "elections.csv"
    assert chunks[0].metadata["row"] == 0
```

**Step 2: Run — expect ImportError.**

Run: `pytest tests/test_chunking.py::test_chunk_csv_rows_produces_one_chunk_per_row -v`

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/chunking.py
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _row_to_sentence(row: pd.Series) -> str:
    parts = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        if isinstance(v, (int, float)) and float(v).is_integer():
            parts.append(f"{k.replace('_', ' ')}: {int(v):,}")
        elif isinstance(v, float):
            parts.append(f"{k.replace('_', ' ')}: {v:.2f}")
        else:
            parts.append(f"{k.replace('_', ' ')}: {v}")
    return "; ".join(parts) + "."


def chunk_csv_rows(df: pd.DataFrame, source: str) -> list[Chunk]:
    chunks = []
    for i, row in df.iterrows():
        text = _row_to_sentence(row)
        chunks.append(Chunk(text=text, metadata={"source": source, "row": int(i)}))
    return chunks
```

**Step 4: Run — expect pass.**

---

## Task 5: PDF recursive-character chunker

**Files:**
- Modify: `rag/chunking.py`
- Modify: `tests/test_chunking.py`

**Step 1: Add failing test**

```python
# append to tests/test_chunking.py
from rag.chunking import chunk_pdf_pages

def test_chunk_pdf_pages_respects_size_and_overlap():
    text = ("Paragraph one sentence one. Paragraph one sentence two. "
            * 40)  # long single paragraph
    pages = [text]
    chunks = chunk_pdf_pages(pages, source="budget.pdf",
                             chunk_size=400, overlap=80)
    assert all(len(c.text) <= 400 + 80 for c in chunks)
    assert len(chunks) >= 2
    # overlap exists: some substring at end of chunk[0] appears at start of chunk[1]
    assert chunks[0].text[-40:] in chunks[1].text or \
           chunks[1].text[:40] in chunks[0].text
    assert chunks[0].metadata["source"] == "budget.pdf"
    assert chunks[0].metadata["page"] == 0

def test_chunk_pdf_pages_does_not_split_mid_word():
    pages = ["supercalifragilistic " * 50]
    chunks = chunk_pdf_pages(pages, source="x.pdf",
                             chunk_size=50, overlap=10)
    for c in chunks:
        # chunk text boundaries shouldn't be inside a word
        assert not c.text.startswith("cali")
        assert not c.text.endswith("califr")
```

**Step 2: Run — expect failure (function missing).**

**Step 3: Implement**

```python
# append to rag/chunking.py
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back up to the nearest separator to avoid mid-word splits
            window = text[start:end]
            split_at = -1
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx > size * 0.5:  # don't back up too far
                    split_at = idx + len(sep)
                    break
            if split_at > 0:
                end = start + split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pdf_pages(
    pages: list[str],
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    out = []
    for page_idx, page_text in enumerate(pages):
        for piece in _split_with_overlap(page_text, chunk_size, overlap):
            out.append(Chunk(
                text=piece,
                metadata={"source": source, "page": page_idx},
            ))
    return out
```

**Step 4: Run — expect pass.**

---

## Task 6: Embeddings wrapper

**Files:**
- Create: `rag/embeddings.py`
- Create: `tests/test_embeddings.py`

**Step 1: Write failing test**

```python
# tests/test_embeddings.py
import numpy as np
from rag.embeddings import Embedder

def test_embed_returns_normalised_matrix():
    emb = Embedder()
    texts = ["hello world", "the budget allocates funds to education"]
    vecs = emb.embed(texts)
    assert vecs.shape == (2, emb.dim)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

def test_embed_query_is_unit_vector():
    emb = Embedder()
    v = emb.embed_query("who won Ayawaso West?")
    assert v.shape == (emb.dim,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4
```

**Step 2: Run — expect ImportError.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
```

**Step 4: Run — expect pass.** First run downloads the model (~80MB); allow ~60s.

---

## Task 7: Custom numpy vector store

**Files:**
- Create: `rag/vector_store.py`
- Create: `tests/test_vector_store.py`

**Step 1: Write failing test**

```python
# tests/test_vector_store.py
import numpy as np
from rag.vector_store import VectorStore

def test_top_k_returns_highest_cosine_scores():
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ], dtype=np.float32)
    # normalise
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = [{"id": i} for i in range(3)]
    vs = VectorStore(vectors=vectors, metadata=metadata)

    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = vs.top_k(q, k=2)
    assert [h["id"] for h in hits] == [0, 2]
    assert hits[0]["score"] > hits[1]["score"]
    assert all(0.0 <= h["score"] <= 1.0 + 1e-5 for h in hits)

def test_save_and_load_roundtrip(tmp_path):
    vectors = np.random.rand(5, 8).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    meta = [{"id": i, "text": f"t{i}"} for i in range(5)]
    vs = VectorStore(vectors=vectors, metadata=meta)
    vs.save(tmp_path / "idx")
    reloaded = VectorStore.load(tmp_path / "idx")
    np.testing.assert_array_equal(reloaded.vectors, vectors)
    assert reloaded.metadata == meta
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/vector_store.py
import pickle
from pathlib import Path
import numpy as np


class VectorStore:
    def __init__(self, vectors: np.ndarray, metadata: list[dict]):
        assert len(vectors) == len(metadata)
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata

    def top_k(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        q = query_vec.astype(np.float32)
        if q.ndim == 1:
            q = q[None, :]
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        scores = (self.vectors @ q.T).flatten()
        k = min(k, len(scores))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [
            {**self.metadata[i], "score": float(scores[i]), "chunk_id": int(i)}
            for i in idxs
        ]

    def save(self, prefix: str | Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.save(prefix.with_suffix(".vectors.npy"), self.vectors)
        with open(prefix.with_suffix(".meta.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, prefix: str | Path) -> "VectorStore":
        prefix = Path(prefix)
        vecs = np.load(prefix.with_suffix(".vectors.npy"))
        with open(prefix.with_suffix(".meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        return cls(vectors=vecs, metadata=meta)
```

**Step 4: Run — expect pass.**

---

## Task 8: BM25 keyword store

**Files:**
- Create: `rag/bm25_store.py`
- Create: `tests/test_bm25_store.py`

**Step 1: Write failing test**

```python
# tests/test_bm25_store.py
from rag.bm25_store import BM25Store

def test_bm25_retrieves_exact_term_match():
    docs = [
        "The budget allocates funds to education.",
        "NHIL was revised in 2024.",
        "The election results in Ayawaso West favoured NPP.",
    ]
    meta = [{"id": i} for i in range(3)]
    store = BM25Store.build(docs, meta)
    hits = store.top_k("NHIL", k=2)
    assert hits[0]["id"] == 1
    assert hits[0]["score"] > hits[1]["score"]

def test_bm25_save_load_roundtrip(tmp_path):
    docs = ["apple banana", "banana cherry", "cherry date"]
    meta = [{"id": i} for i in range(3)]
    store = BM25Store.build(docs, meta)
    path = tmp_path / "bm25.pkl"
    store.save(path)
    loaded = BM25Store.load(path)
    a = [h["id"] for h in store.top_k("banana", k=3)]
    b = [h["id"] for h in loaded.top_k("banana", k=3)]
    assert a == b
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/bm25_store.py
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    def __init__(self, bm25: BM25Okapi, metadata: list[dict]):
        self.bm25 = bm25
        self.metadata = metadata

    @classmethod
    def build(cls, docs: list[str], metadata: list[dict]) -> "BM25Store":
        assert len(docs) == len(metadata)
        tokenised = [_tokenise(d) for d in docs]
        return cls(bm25=BM25Okapi(tokenised), metadata=metadata)

    def top_k(self, query: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(_tokenise(query))
        import numpy as np
        k = min(k, len(scores))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [
            {**self.metadata[i], "score": float(scores[i]), "chunk_id": int(i)}
            for i in idxs
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Store":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(bm25=data["bm25"], metadata=data["metadata"])
```

**Step 4: Run — expect pass.**

---

## Task 9: Hybrid retrieval with RRF

**Files:**
- Create: `rag/retrieval.py`
- Create: `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
# tests/test_retrieval.py
from rag.retrieval import reciprocal_rank_fusion

def test_rrf_promotes_docs_ranked_high_by_multiple_sources():
    dense = [{"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 3}]
    sparse = [{"chunk_id": 2}, {"chunk_id": 4}, {"chunk_id": 1}]
    fused = reciprocal_rank_fusion([dense, sparse], k_const=60, top_k=3)
    ids = [f["chunk_id"] for f in fused]
    # chunk 2 appears at rank 1 in sparse, rank 2 in dense -> should win
    assert ids[0] == 2
    # chunk 1 is top of dense, rank 3 of sparse -> second
    assert 1 in ids[:2]
    assert all("rrf_score" in f for f in fused)
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/retrieval.py
from collections import defaultdict
import numpy as np
from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k_const: int = 60,
    top_k: int = 5,
) -> list[dict]:
    scores = defaultdict(float)
    latest = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cid = item["chunk_id"]
            scores[cid] += 1.0 / (k_const + rank)
            # keep a reference so we can surface the original metadata
            latest[cid] = item
    order = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for cid, s in order:
        merged = {**latest[cid], "rrf_score": float(s)}
        out.append(merged)
    return out


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        dense_k: int = 20,
        sparse_k: int = 20,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.dense_k = dense_k
        self.sparse_k = sparse_k

    def search(self, query: str, top_k: int = 5) -> dict:
        q_vec = self.embedder.embed_query(query)
        dense = self.vector_store.top_k(q_vec, k=self.dense_k)
        sparse = self.bm25_store.top_k(query, k=self.sparse_k)
        fused = reciprocal_rank_fusion([dense, sparse], top_k=top_k)
        return {
            "dense": dense,
            "sparse": sparse,
            "hybrid": fused,
        }
```

**Step 4: Run — expect pass.**

---

## Task 10: Build-index script

**Files:**
- Create: `scripts/build_index.py`

**Step 1: Implement (no test; it's a CLI)**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: scripts/build_index.py
"""Build the vector + BM25 index from the two source datasets."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag.ingest import load_csv, load_pdf_pages
from rag.chunking import chunk_csv_rows, chunk_pdf_pages
from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store

CSV_PATH = "data/Ghana_Election_Result.csv"
PDF_PATH = "data/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
INDEX_PREFIX = "index/store"
BM25_PATH = "index/bm25.pkl"


def main():
    print("Loading CSV...")
    df = load_csv(CSV_PATH)
    print(f"  {len(df)} rows")

    print("Loading PDF...")
    pages = load_pdf_pages(PDF_PATH)
    print(f"  {len(pages)} pages")

    print("Chunking...")
    csv_chunks = chunk_csv_rows(df, source="elections.csv")
    pdf_chunks = chunk_pdf_pages(pages, source="budget.pdf")
    all_chunks = csv_chunks + pdf_chunks
    print(f"  {len(csv_chunks)} CSV chunks + {len(pdf_chunks)} PDF chunks "
          f"= {len(all_chunks)} total")

    print("Embedding...")
    emb = Embedder()
    texts = [c.text for c in all_chunks]
    vecs = emb.embed(texts)

    print("Building metadata...")
    metadata = [
        {"chunk_id": i, "text": c.text, **c.metadata}
        for i, c in enumerate(all_chunks)
    ]

    print("Saving vector store...")
    VectorStore(vectors=vecs, metadata=metadata).save(INDEX_PREFIX)

    print("Saving BM25 store...")
    BM25Store.build(texts, metadata).save(BM25_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
```

**Step 2: Run it**

Run: `python scripts/build_index.py`
Expected: outputs chunk counts, writes `index/store.vectors.npy`, `index/store.meta.pkl`, `index/bm25.pkl`. Time: 1–3 minutes.

**Step 3: Sanity check the index**

Run an interactive smoke test:
```bash
python -c "
from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store
from rag.retrieval import HybridRetriever
emb = Embedder()
vs = VectorStore.load('index/store')
bm = BM25Store.load('index/bm25.pkl')
r = HybridRetriever(emb, vs, bm)
for h in r.search('education budget', top_k=3)['hybrid']:
    print(h['source'], h.get('page') or h.get('row'), '->', h['text'][:100])
"
```

Expected: three results, at least two from `budget.pdf`, top results mention education or budget.

---

## Task 11: Gemini generator

**Files:**
- Create: `rag/generator.py`
- Create: `.env.example`
- Create: `tests/test_generator.py`

**Step 1: Write `.env.example`**

```
GEMINI_API_KEY=your_key_here
```

**Step 2: Write failing test (mocked; no live API)**

```python
# tests/test_generator.py
from unittest.mock import patch, MagicMock
from rag.generator import GeminiGenerator

def test_generate_calls_model_with_prompt():
    fake_model = MagicMock()
    fake_model.generate_content.return_value = MagicMock(text="answer")
    with patch("rag.generator.genai.GenerativeModel", return_value=fake_model):
        gen = GeminiGenerator(api_key="fake-key")
        out = gen.generate("hello")
    assert out == "answer"
    fake_model.generate_content.assert_called_once()
    args, _ = fake_model.generate_content.call_args
    assert args[0] == "hello"
```

**Step 3: Run — expect failure.**

**Step 4: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/generator.py
import os
import google.generativeai as genai


class GeminiGenerator:
    def __init__(self, api_key: str | None = None,
                 model_name: str = "gemini-1.5-flash"):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        return (resp.text or "").strip()
```

**Step 5: Run — expect pass.**

---

## Task 12: Prompt templates + context budgeter

**Files:**
- Create: `rag/prompts.py`
- Create: `tests/test_prompts.py`

**Step 1: Write failing test**

```python
# tests/test_prompts.py
from rag.prompts import build_prompt, trim_context, SYSTEM_RULES_V3

def test_build_prompt_includes_numbered_citations_and_rules():
    chunks = [
        {"text": "NPP won Ayawaso.", "source": "elections.csv", "row": 12},
        {"text": "Education received GHS 5bn.", "source": "budget.pdf", "page": 42},
    ]
    prompt = build_prompt(
        query="Who won Ayawaso?",
        chunks=chunks,
        chat_history=[],
        rules=SYSTEM_RULES_V3,
    )
    assert "[1]" in prompt and "[2]" in prompt
    assert "NPP won Ayawaso" in prompt
    assert "Who won Ayawaso?" in prompt
    assert "elections.csv" in prompt

def test_trim_context_drops_by_rrf_score_when_over_budget():
    chunks = [
        {"text": "a" * 1000, "rrf_score": 0.9, "source": "x", "row": 1},
        {"text": "b" * 1000, "rrf_score": 0.5, "source": "x", "row": 2},
        {"text": "c" * 1000, "rrf_score": 0.1, "source": "x", "row": 3},
    ]
    kept, dropped = trim_context(chunks, max_chars=2500)
    assert len(kept) == 2
    assert kept[0]["rrf_score"] == 0.9
    assert dropped[0]["rrf_score"] == 0.1
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/prompts.py

SYSTEM_RULES_V1 = (
    "You are an assistant answering questions about Ghana's elections and "
    "the 2025 national budget. Use the context below to answer."
)

SYSTEM_RULES_V2 = (
    "You are an assistant answering questions about Ghana's elections and "
    "the 2025 national budget. Use ONLY the numbered context blocks. "
    "If the answer is not present, reply EXACTLY: "
    '"I don\'t have that information in the provided sources." '
    "Cite every claim with its block number like [1]."
)

SYSTEM_RULES_V3 = SYSTEM_RULES_V2 + (
    " If the question spans both corpora (elections and budget), "
    "address each separately and make the split explicit."
)


def _locator(chunk: dict) -> str:
    if "page" in chunk:
        return f"{chunk['source']}, page {chunk['page']}"
    if "row" in chunk:
        return f"{chunk['source']}, row {chunk['row']}"
    return chunk.get("source", "unknown")


def build_prompt(
    query: str,
    chunks: list[dict],
    chat_history: list[tuple[str, str]] | None = None,
    rules: str = SYSTEM_RULES_V3,
) -> str:
    blocks = [
        f"[{i+1}] (Source: {_locator(c)})\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    history_text = ""
    if chat_history:
        lines = [f"{role.upper()}: {text}" for role, text in chat_history[-6:]]
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    return (
        f"{rules}\n\n"
        f"Context:\n" + "\n\n".join(blocks) + "\n\n"
        f"{history_text}"
        f"User question: {query}\n\n"
        f"Answer (cite as [n]):"
    )


def trim_context(
    chunks: list[dict],
    max_chars: int = 3000,
    max_chunks: int = 6,
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(chunks, key=lambda c: -c.get("rrf_score", 0.0))
    kept, total = [], 0
    for c in ordered[:max_chunks]:
        if total + len(c["text"]) > max_chars and kept:
            break
        kept.append(c)
        total += len(c["text"])
    dropped = [c for c in ordered if c not in kept]
    return kept, dropped
```

**Step 4: Run — expect pass.**

---

## Task 13: Query decomposer (innovation)

**Files:**
- Create: `rag/decomposer.py`
- Create: `tests/test_decomposer.py`

**Step 1: Write failing test (mocked LLM)**

```python
# tests/test_decomposer.py
from unittest.mock import MagicMock
from rag.decomposer import QueryDecomposer

def test_decomposer_returns_subqueries_and_sources():
    fake_gen = MagicMock()
    fake_gen.generate.return_value = (
        '{"sub_queries": ["NPP Ashanti vote share", '
        '"Ashanti budget allocation"], '
        '"sources": ["elections", "budget"]}'
    )
    dec = QueryDecomposer(generator=fake_gen)
    out = dec.decompose("Compare NPP Ashanti vote share with Ashanti budget")
    assert out["sub_queries"] == [
        "NPP Ashanti vote share",
        "Ashanti budget allocation",
    ]
    assert out["sources"] == ["elections", "budget"]

def test_decomposer_falls_back_when_llm_json_malformed():
    fake_gen = MagicMock()
    fake_gen.generate.return_value = "not json at all"
    dec = QueryDecomposer(generator=fake_gen)
    out = dec.decompose("who won Ayawaso?")
    assert out["sub_queries"] == ["who won Ayawaso?"]
    assert out["sources"] == ["both"]
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/decomposer.py
import json
import re

_SYSTEM = """You break down user questions for a RAG system with two corpora:
- "elections": Ghana election results (per-constituency voting data).
- "budget": Ghana 2025 national budget statement (fiscal policy prose).

Return STRICT JSON with this shape:
{"sub_queries": ["..."], "sources": ["elections"|"budget"|"both", ...]}

Rules:
- If the query is simple and single-topic, return one sub-query.
- If the query spans both corpora, split into per-corpus sub-queries and
  label each source accordingly.
- sources[i] corresponds to sub_queries[i].
- Output ONLY the JSON. No prose, no markdown."""


class QueryDecomposer:
    def __init__(self, generator):
        self.generator = generator

    def decompose(self, query: str) -> dict:
        prompt = f"{_SYSTEM}\n\nUser question: {query}\n\nJSON:"
        raw = self.generator.generate(prompt, temperature=0.0)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if "sub_queries" in data and "sources" in data:
                    return {
                        "sub_queries": data["sub_queries"],
                        "sources": data["sources"],
                    }
            except json.JSONDecodeError:
                pass
        return {"sub_queries": [query], "sources": ["both"]}
```

**Step 4: Run — expect pass.**

---

## Task 14: Structured logging utility

**Files:**
- Create: `rag/logging_utils.py`
- Create: `tests/test_logging_utils.py`

**Step 1: Write failing test**

```python
# tests/test_logging_utils.py
import json
from pathlib import Path
from rag.logging_utils import QueryLogger

def test_query_logger_writes_ordered_stages(tmp_path):
    log = QueryLogger(log_dir=tmp_path)
    log.stage("query_received", {"query": "hello"})
    log.stage("retrieval", {"top_k": 3})
    path = log.flush()
    data = json.loads(Path(path).read_text())
    assert data["query_id"]
    assert [s["stage"] for s in data["stages"]] == ["query_received", "retrieval"]
    assert data["stages"][0]["payload"]["query"] == "hello"
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/logging_utils.py
import json
import time
import uuid
from pathlib import Path


class QueryLogger:
    def __init__(self, log_dir: str | Path = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.query_id = uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.stages: list[dict] = []

    def stage(self, name: str, payload: dict) -> None:
        self.stages.append({
            "stage": name,
            "t": time.time() - self.started_at,
            "payload": payload,
        })

    def flush(self) -> str:
        path = self.log_dir / f"query_{int(self.started_at)}_{self.query_id}.json"
        path.write_text(json.dumps({
            "query_id": self.query_id,
            "started_at": self.started_at,
            "stages": self.stages,
        }, indent=2, default=str))
        return str(path)
```

**Step 4: Run — expect pass.**

---

## Task 15: Full pipeline

**Files:**
- Create: `rag/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write failing test (mocks retriever, decomposer, generator)**

```python
# tests/test_pipeline.py
from unittest.mock import MagicMock
from rag.pipeline import RAGPipeline

def test_pipeline_end_to_end_logs_every_stage(tmp_path):
    retriever = MagicMock()
    retriever.search.return_value = {
        "dense": [{"chunk_id": 1, "score": 0.9}],
        "sparse": [{"chunk_id": 2, "score": 3.1}],
        "hybrid": [
            {"chunk_id": 1, "text": "NPP won.", "source": "elections.csv",
             "row": 0, "rrf_score": 0.5},
        ],
    }
    decomposer = MagicMock()
    decomposer.decompose.return_value = {
        "sub_queries": ["who won?"], "sources": ["elections"],
    }
    generator = MagicMock()
    generator.generate.return_value = "NPP won [1]."

    pipe = RAGPipeline(
        retriever=retriever,
        decomposer=decomposer,
        generator=generator,
        log_dir=tmp_path,
    )
    result = pipe.answer("who won?", chat_history=[])
    assert "NPP" in result["answer"]
    assert result["log_path"]
    stages = [s["stage"] for s in result["trace"]["stages"]]
    for required in [
        "query_received", "decomposition", "retrieval",
        "context_selection", "prompt_constructed", "llm_response",
    ]:
        assert required in stages
```

**Step 2: Run — expect failure.**

**Step 3: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/pipeline.py
from pathlib import Path
from rag.prompts import build_prompt, trim_context, SYSTEM_RULES_V3
from rag.retrieval import reciprocal_rank_fusion
from rag.logging_utils import QueryLogger


class RAGPipeline:
    def __init__(self, retriever, decomposer, generator,
                 log_dir: str | Path = "logs", top_k: int = 5):
        self.retriever = retriever
        self.decomposer = decomposer
        self.generator = generator
        self.log_dir = log_dir
        self.top_k = top_k

    def answer(self, query: str, chat_history: list[tuple[str, str]]) -> dict:
        log = QueryLogger(log_dir=self.log_dir)
        log.stage("query_received", {"query": query,
                                     "history_len": len(chat_history)})

        plan = self.decomposer.decompose(query)
        log.stage("decomposition", plan)

        sub_results = []
        for sq, src in zip(plan["sub_queries"], plan["sources"]):
            r = self.retriever.search(sq, top_k=self.top_k)
            sub_results.append(r["hybrid"])
            log.stage("retrieval", {
                "sub_query": sq, "source_hint": src,
                "dense_scores": [h.get("score") for h in r["dense"][:5]],
                "sparse_scores": [h.get("score") for h in r["sparse"][:5]],
                "hybrid": [{"chunk_id": h["chunk_id"],
                            "rrf_score": h["rrf_score"],
                            "source": h.get("source")}
                           for h in r["hybrid"]],
            })

        merged = reciprocal_rank_fusion(sub_results, top_k=self.top_k * 2) \
            if len(sub_results) > 1 else sub_results[0]
        kept, dropped = trim_context(merged)
        log.stage("context_selection", {
            "kept": [{"chunk_id": c["chunk_id"], "rrf_score": c.get("rrf_score")}
                     for c in kept],
            "dropped": [{"chunk_id": c["chunk_id"], "rrf_score": c.get("rrf_score")}
                        for c in dropped],
        })

        prompt = build_prompt(query, kept, chat_history, rules=SYSTEM_RULES_V3)
        log.stage("prompt_constructed", {"chars": len(prompt), "prompt": prompt})

        answer = self.generator.generate(prompt)
        log.stage("llm_response", {"chars": len(answer), "response": answer})

        log_path = log.flush()
        return {
            "answer": answer,
            "chunks": kept,
            "dropped": dropped,
            "prompt": prompt,
            "decomposition": plan,
            "log_path": log_path,
            "trace": {"stages": log.stages},
        }
```

**Step 4: Run — expect pass.**

---

## Task 16: Streamlit UI

**Files:**
- Create: `app.py`

**Step 1: Implement**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: app.py
import os
import streamlit as st
from dotenv import load_dotenv

from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store
from rag.retrieval import HybridRetriever
from rag.decomposer import QueryDecomposer
from rag.generator import GeminiGenerator
from rag.pipeline import RAGPipeline

load_dotenv()
st.set_page_config(page_title="Academic City RAG", layout="wide")


@st.cache_resource(show_spinner="Loading models and indices...")
def load_pipeline():
    emb = Embedder()
    vs = VectorStore.load("index/store")
    bm = BM25Store.load("index/bm25.pkl")
    retriever = HybridRetriever(emb, vs, bm)
    generator = GeminiGenerator(
        api_key=os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    )
    decomposer = QueryDecomposer(generator=generator)
    return RAGPipeline(retriever, decomposer, generator)


def main():
    st.title("Academic City RAG Assistant")
    st.caption("Ghana elections + 2025 budget statement")

    if "history" not in st.session_state:
        st.session_state.history = []

    pipeline = load_pipeline()

    # chat transcript
    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.write(text)

    query = st.chat_input("Ask a question...")
    if query:
        st.session_state.history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.answer(query, st.session_state.history[:-1])
            st.write(result["answer"])
            with st.expander("Retrieved context"):
                for i, c in enumerate(result["chunks"]):
                    loc = c.get("page", c.get("row", "?"))
                    st.markdown(
                        f"**[{i+1}] {c.get('source', '?')}** "
                        f"(loc {loc}, rrf={c.get('rrf_score', 0):.3f})"
                    )
                    st.write(c["text"])
            with st.expander("Decomposition"):
                st.json(result["decomposition"])
            with st.expander("Full prompt sent to LLM"):
                st.code(result["prompt"])
            with st.expander("Pipeline trace"):
                st.json(result["trace"])
            st.caption(f"Log: `{result['log_path']}`")
        st.session_state.history.append(("assistant", result["answer"]))


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test locally**

Run: `streamlit run app.py`
Expected: page loads, first query takes ~30s (cold model), subsequent queries <5s, all expanders populate.

---

## Task 17: Adversarial evaluation

**Files:**
- Create: `evaluation/adversarial_queries.py`
- Create: `evaluation/run_eval.py`

**Step 1: Write the query set**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: evaluation/adversarial_queries.py

ADVERSARIAL = [
    {
        "id": "ambiguous_1",
        "type": "ambiguous",
        "query": "Which party won more?",
        "notes": "Missing year, metric, and place. Good RAG should refuse or ask for clarification; pure LLM may hallucinate.",
    },
    {
        "id": "misleading_1",
        "type": "misleading",
        "query": "What was the 2025 budget allocation for the 2024 elections?",
        "notes": "Mixes corpora; may not exist. Tests hallucination resistance.",
    },
]
```

**Step 2: Write the comparison runner**

```python
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: evaluation/run_eval.py
"""Run RAG vs pure-LLM on the adversarial set, 3x each, log results."""
import sys
import json
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store
from rag.retrieval import HybridRetriever
from rag.decomposer import QueryDecomposer
from rag.generator import GeminiGenerator
from rag.pipeline import RAGPipeline
from evaluation.adversarial_queries import ADVERSARIAL

load_dotenv()
OUT = pathlib.Path("evaluation/results.json")


def main():
    generator = GeminiGenerator()
    retriever = HybridRetriever(
        Embedder(), VectorStore.load("index/store"), BM25Store.load("index/bm25.pkl")
    )
    decomposer = QueryDecomposer(generator=generator)
    pipe = RAGPipeline(retriever, decomposer, generator)

    results = []
    for item in ADVERSARIAL:
        for run in range(3):
            rag_out = pipe.answer(item["query"], chat_history=[])
            pure_out = generator.generate(item["query"])
            results.append({
                "id": item["id"],
                "type": item["type"],
                "run": run,
                "query": item["query"],
                "rag_answer": rag_out["answer"],
                "rag_chunks": [
                    {"source": c.get("source"),
                     "loc": c.get("page", c.get("row")),
                     "rrf": c.get("rrf_score")}
                    for c in rag_out["chunks"]
                ],
                "pure_llm_answer": pure_out,
            })
            print(f"[{item['id']} run {run}] done")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

**Step 3: Run once**

Run: `python evaluation/run_eval.py`
Expected: prints 6 "done" lines (2 queries × 3 runs), writes `evaluation/results.json`.

---

## Task 18: Architecture doc with diagram

**Files:**
- Create: `docs/architecture.md`

Write the architecture document with:

1. Mermaid diagram (use the layout from the design doc §10).
2. Component-by-component explanation: why each exists, what it produces, what it consumes.
3. Domain-fit justification (Part F graded on this): dual-corpus → hybrid; exact terms → BM25; semantic prose → dense; cross-corpus → decomposition.
4. Scaling notes: "FAISS when N > 100k chunks; currently numpy suffices."

Use the design doc §10 as the backbone. Expand each component into 3–5 sentences.

---

## Task 19: Experiment logs template

**Files:**
- Create: `docs/experiment_logs.md`

This is the HAND-FILLED deliverable (4 marks). Write a template with named sections and blank tables; student fills values during testing.

```markdown
# Experiment Logs (Manual — not AI-generated)

## E1. Chunking — size & overlap comparison (Part A)

Chunk sizes tested: 400, 800, 1500. Overlap fixed at 150.
Eval questions (hand-written, 10):
1. ...
2. ...

| Chunk size | Recall@5 | Notes on answer quality |
|---|---|---|
| 400 | __ / 10 | |
| 800 | __ / 10 | |
| 1500 | __ / 10 | |

Conclusion: (fill in after runs)

## E2. Retrieval — failure cases & hybrid fix (Part B)

| Query | Dense-only top hit | BM25-only top hit | Hybrid top hit | Winner |
|---|---|---|---|---|
| "NHIL" | | | | |
| "Ayawso West" (misspelled) | | | | |
| "<a rare proper noun>" | | | | |

## E3. Prompt iteration (Part C)

Query used: ________

v1 output (baseline, no hallucination guard):
> ...
Hallucinations: __ / Citations: __

v2 output (+ hallucination rule + forced citations):
> ...
Hallucinations: __ / Citations: __

v3 output (+ cross-corpus rule):
> ...
Hallucinations: __ / Citations: __

## E4. Adversarial queries — RAG vs pure LLM (Part E)

### Ambiguous: "Which party won more?"

| Run | RAG answer | Pure LLM answer | RAG accuracy | Pure-LLM accuracy | RAG halluc. count | Pure-LLM halluc. count |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

### Misleading: "What was the 2025 budget allocation for the 2024 elections?"

(same table)

### Summary

- Consistency (RAG): ...
- Consistency (pure LLM): ...
- Hallucination rate delta: ...
```

---

## Task 20: Design decisions doc

**Files:**
- Create: `docs/design_decisions.md`

Transcribe the reasoning from the design doc in narrative form: why Streamlit (vs Next.js/Flask), why custom numpy (vs FAISS), why all-MiniLM-L6-v2 (vs Gemini embeddings), why hybrid search (vs pure dense), why query decomposition (novelty argument), why Gemini 1.5 Flash. One paragraph per decision.

This satisfies the "Documentation (detailed, not simple AI-generated summaries)" deliverable.

---

## Task 21: Video script

**Files:**
- Create: `docs/video_script.md`

Write a 2-minute script the student will read while demoing. Structure (rough timing):

- **0:00–0:15** — intro: name, index, course, problem statement.
- **0:15–0:45** — one live query showing retrieved chunks, similarity scores, final prompt in the debug panel.
- **0:45–1:15** — one cross-corpus query to showcase query decomposition (Part G).
- **1:15–1:45** — one adversarial query showing the RAG refusal vs pure-LLM hallucination.
- **1:45–2:00** — deployment URL + closing.

---

## Task 22: Final README

**Files:**
- Modify: `README.md`

Expand to include:

- Student name + index (top).
- Project summary (2 sentences).
- Deployed URL.
- Local setup: venv, install, env vars, `python scripts/build_index.py`, `streamlit run app.py`.
- Repo structure (tree output).
- How each exam Part is addressed, with file references.
- How to run tests: `pytest`.
- Link to `docs/architecture.md`, `docs/design_decisions.md`, `docs/experiment_logs.md`, `docs/video_script.md`.

---

## Task 23: Streamlit Cloud deployment

**Files:**
- None created (platform config).

**Step 1: Push repo to GitHub**

- Create GitHub repo named exactly `ai_<INDEX_NUMBER>`.
- Add `GodwinDansoAcity` as a collaborator.
- Push all branches.

**Step 2: Deploy on Streamlit Community Cloud**

- Go to share.streamlit.io, connect the GitHub repo, select `app.py`.
- Add secret: `GEMINI_API_KEY = "<your key>"`.
- Wait for first boot (installs deps + loads model: 3–5 minutes).

**Step 3: Verify live URL**

- Open the deployed URL, run one query, confirm chunks + response appear.
- Paste the URL into `README.md` under "Deployed URL".

**Step 4: Email**

- Subject: `CS4241-Introduction to Artificial Intelligence-2026:[<INDEX> <NAME>]`
- To: `godwin.danso@acity.edu.gh`
- Body: GitHub URL, deployed URL, any notes. Attach nothing.

---

## Execution order summary

```
1 (scaffold) → 2 (data) → 3 (ingest) → 4 (csv chunker) → 5 (pdf chunker)
  → 6 (embeddings) → 7 (vector store) → 8 (bm25) → 9 (retrieval)
  → 10 (build index) → 11 (generator) → 12 (prompts) → 13 (decomposer)
  → 14 (logging) → 15 (pipeline) → 16 (UI) → 17 (eval) → 18–22 (docs)
  → 23 (deploy)
```

Each task (except docs + deploy) has tests that must pass before moving on.
