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

_BUDGET_HINTS = re.compile(
    r"\b(budget|allocation|mda|fiscal|revenue|expenditure|mofep|mof\b|minister"
    r"|appropriat|gdp|deficit|tax policy|capital expenditure|recurrent)\b",
    re.I,
)
_ELECTION_HINTS = re.compile(
    r"\b(election|vote|votes|ballot|constituenc|ndc|npp|cpp|gum|pnc"
    r"|parliamentary|presidential|polling|turnout|registered voters"
    r"|2020|2024)\b",
    re.I,
)


class QueryDecomposer:
    """Splits queries for retrieval. LLM path is accurate but adds a full Gemini round-trip."""

    def __init__(self, generator, *, use_llm_decomposition: bool = False):
        self.generator = generator
        self.use_llm_decomposition = use_llm_decomposition

    def _heuristic_decompose(self, query: str) -> dict:
        """No network call — routes with simple keywords (good enough for most exam queries)."""
        q = query.strip()
        b = bool(_BUDGET_HINTS.search(q))
        e = bool(_ELECTION_HINTS.search(q))
        if b and not e:
            return {"sub_queries": [q], "sources": ["budget"]}
        if e and not b:
            return {"sub_queries": [q], "sources": ["elections"]}
        return {"sub_queries": [q], "sources": ["both"]}

    def decompose(self, query: str) -> dict:
        if not self.use_llm_decomposition:
            return self._heuristic_decompose(query)

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
