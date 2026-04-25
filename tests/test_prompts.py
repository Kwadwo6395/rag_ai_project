from rag.prompts import build_prompt, trim_context, SYSTEM_RULES_V4


def test_build_prompt_includes_numbered_citations_and_rules():
    chunks = [
        {"text": "NPP won Ayawaso.", "source": "elections.csv", "row": 12},
        {"text": "Education received GHS 5bn.", "source": "budget.pdf", "page": 42},
    ]
    prompt = build_prompt(
        query="Who won Ayawaso?",
        chunks=chunks,
        chat_history=[],
        rules=SYSTEM_RULES_V4,
    )
    assert "[1]" in prompt and "[2]" in prompt
    assert "NPP won Ayawaso" in prompt
    assert "Who won Ayawaso?" in prompt
    assert "elections.csv" in prompt


def test_trim_context_drops_by_rrf_score_when_over_budget():
    chunks = [
        {"text": "a" * 1000, "rrf_score": 0.9, "source": "x", "row": 1},
        {"text": "b" * 1000, "rrf_score": 0.5, "source": "x", "row": 2},
        {"text": "c" * 1000, "rrf_score": 0.1, "source": "x", "row": 3},
    ]
    kept, dropped = trim_context(chunks, max_chars=2500)
    assert len(kept) == 2
    assert kept[0]["rrf_score"] == 0.9
    assert dropped[0]["rrf_score"] == 0.1
