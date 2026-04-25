# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/retrieval.py
from collections import defaultdict
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
