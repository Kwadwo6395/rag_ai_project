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
