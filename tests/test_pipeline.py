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


def test_pipeline_no_chunks_skips_llm(tmp_path):
    retriever = MagicMock()
    retriever.search.return_value = {
        "dense": [],
        "sparse": [],
        "hybrid": [],
    }
    decomposer = MagicMock()
    decomposer.decompose.return_value = {
        "sub_queries": ["zzz"], "sources": ["both"],
    }
    generator = MagicMock()

    pipe = RAGPipeline(
        retriever=retriever,
        decomposer=decomposer,
        generator=generator,
        log_dir=tmp_path,
    )
    result = pipe.answer("zzz", chat_history=[])
    generator.generate.assert_not_called()
    assert "No text passages were retrieved" in result["answer"]
    assert result["chunks"] == []
