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
