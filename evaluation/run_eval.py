# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: evaluation/run_eval.py
"""Run RAG vs pure-LLM on the adversarial set, 3x each, log results."""
import os
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
from rag.generator import build_text_generator
from rag.pipeline import RAGPipeline
from evaluation.adversarial_queries import ADVERSARIAL

load_dotenv(override=True)
OUT = pathlib.Path("evaluation/results.json")


def main():
    provider = (
        "groq"
        if os.environ.get("LLM_PROVIDER", "").strip().lower() == "groq"
        else "gemini"
    )
    if provider == "groq":
        model_id = (os.environ.get("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
    else:
        model_id = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()
    generator = build_text_generator(provider=provider, model_id=model_id)
    retriever = HybridRetriever(
        Embedder(),
        VectorStore.load("index/store"),
        BM25Store.load("index/bm25.pkl"),
    )
    # Default on for evals to match full RAG design; set RAG_LLM_DECOMPOSE=0 for a faster run.
    use_llm = os.environ.get("RAG_LLM_DECOMPOSE", "1").strip().lower() not in (
        "0", "false", "no",
    )
    decomposer = QueryDecomposer(
        generator=generator,
        use_llm_decomposition=use_llm,
    )
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
