# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/pipeline.py
from collections.abc import Callable
from pathlib import Path
from rag.prompts import build_prompt, trim_context, SYSTEM_RULES_V4
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

    def answer(
        self,
        query: str,
        chat_history: list[tuple[str, str]],
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        log = QueryLogger(log_dir=self.log_dir)
        log.stage("query_received", {"query": query,
                                     "history_len": len(chat_history)})

        if progress:
            if getattr(self.decomposer, "use_llm_decomposition", False):
                progress("Understanding your question (extra LLM call, slower)…")
            else:
                progress("Routing your question (no extra LLM)…")
        plan = self.decomposer.decompose(query)
        log.stage("decomposition", plan)

        sub_results = []
        if progress:
            progress("Searching your documents…")
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

        merged = (
            reciprocal_rank_fusion(sub_results, top_k=self.top_k * 2)
            if len(sub_results) > 1 else sub_results[0]
        )
        kept, dropped = trim_context(merged)
        log.stage("context_selection", {
            "kept": [{"chunk_id": c["chunk_id"], "rrf_score": c.get("rrf_score")}
                     for c in kept],
            "dropped": [{"chunk_id": c["chunk_id"], "rrf_score": c.get("rrf_score")}
                        for c in dropped],
        })

        if not kept:
            msg = (
                "No text passages were retrieved for this question (try terms from the "
                "election CSV—parties, constituencies, regions—or the 2025 budget: "
                "allocations, ministries, programmes)."
            )
            log.stage("prompt_constructed", {"chars": 0, "prompt": ""})
            log.stage("llm_response", {"chars": len(msg), "response": msg})
            log_path = log.flush()
            return {
                "answer": msg,
                "chunks": [],
                "dropped": dropped,
                "prompt": "",
                "decomposition": plan,
                "log_path": log_path,
                "trace": {"stages": log.stages},
            }

        prompt = build_prompt(query, kept, chat_history, rules=SYSTEM_RULES_V4)
        log.stage("prompt_constructed", {"chars": len(prompt), "prompt": prompt})

        if progress:
            progress("Writing the answer (LLM)…")
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
