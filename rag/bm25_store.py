# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/bm25_store.py
import pickle
import re
from pathlib import Path
import numpy as np
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
