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
