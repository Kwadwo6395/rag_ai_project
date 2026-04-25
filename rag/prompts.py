# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/prompts.py

SYSTEM_RULES_V1 = (
    "You are an assistant answering questions about Ghana's elections and "
    "the 2025 national budget. Use the context below to answer."
)

SYSTEM_RULES_V2 = (
    "You are an assistant answering questions about Ghana's elections and "
    "the 2025 national budget. Use ONLY the numbered context blocks. "
    "If the answer is not present, reply EXACTLY: "
    '"I don\'t have that information in the provided sources." '
    "Cite every claim with its block number like [1]."
)

SYSTEM_RULES_V3 = SYSTEM_RULES_V2 + (
    " If the question spans both corpora (elections and budget), "
    "address each separately and make the split explicit."
)

# Default for the app: still grounded + cited, but avoids the model parroting a hard refusal
# whenever a question is only partially covered (V2/V3 were too strict for real use).
SYSTEM_RULES_V4 = (
    "You answer using Ghana election results and/or the 2025 national budget excerpts "
    "given in numbered context blocks below.\n"
    "- Ground factual claims in those blocks and cite each claim with [n] (block number).\n"
    "- If a block is an explicit 'national totals' / 'aggregated from all regions' roll-up, "
    "prefer it for nationwide questions such as who won a given presidential year.\n"
    "- If the blocks only partially answer the question, state what the sources *do* say "
    "and what is not covered—do not reply with a single canned refusal if any excerpt is relevant.\n"
    "- If the excerpts are clearly unrelated or empty of usable facts for the question, "
    "say briefly that the indexed sources do not cover it and suggest rephrasing around "
    "constituency/party votes, regions, or 2025 budget line items.\n"
    "- Never invent vote counts, candidates, or budget figures not supported by the context.\n"
    "If the question clearly spans both elections data and the budget, address each part separately."
)


def _locator(chunk: dict) -> str:
    if "page" in chunk:
        return f"{chunk['source']}, page {chunk['page']}"
    if "row" in chunk:
        return f"{chunk['source']}, row {chunk['row']}"
    return chunk.get("source", "unknown")


def build_prompt(
    query: str,
    chunks: list[dict],
    chat_history: list[tuple[str, str]] | None = None,
    rules: str = SYSTEM_RULES_V4,
) -> str:
    blocks = [
        f"[{i+1}] (Source: {_locator(c)})\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    history_text = ""
    if chat_history:
        lines = [f"{role.upper()}: {text}" for role, text in chat_history[-6:]]
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    return (
        f"{rules}\n\n"
        f"Context:\n" + "\n\n".join(blocks) + "\n\n"
        f"{history_text}"
        f"User question: {query}\n\n"
        f"Answer (cite as [n]):"
    )


def trim_context(
    chunks: list[dict],
    max_chars: int = 3000,
    max_chunks: int = 6,
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(chunks, key=lambda c: -c.get("rrf_score", 0.0))
    kept, total = [], 0
    for c in ordered[:max_chunks]:
        if total + len(c["text"]) > max_chars and kept:
            break
        kept.append(c)
        total += len(c["text"])
    dropped = [c for c in ordered if c not in kept]
    return kept, dropped
