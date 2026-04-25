import numpy as np
from rag.vector_store import VectorStore


def test_top_k_returns_highest_cosine_scores():
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.1, 0.0],
    ], dtype=np.float32)
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
