from unittest.mock import MagicMock
from rag.decomposer import QueryDecomposer


def test_decomposer_returns_subqueries_and_sources():
    fake_gen = MagicMock()
    fake_gen.generate.return_value = (
        '{"sub_queries": ["NPP Ashanti vote share", '
        '"Ashanti budget allocation"], '
        '"sources": ["elections", "budget"]}'
    )
    dec = QueryDecomposer(generator=fake_gen, use_llm_decomposition=True)
    out = dec.decompose("Compare NPP Ashanti vote share with Ashanti budget")
    assert out["sub_queries"] == [
        "NPP Ashanti vote share",
        "Ashanti budget allocation",
    ]
    assert out["sources"] == ["elections", "budget"]


def test_decomposer_falls_back_when_llm_json_malformed():
    fake_gen = MagicMock()
    fake_gen.generate.return_value = "not json at all"
    dec = QueryDecomposer(generator=fake_gen, use_llm_decomposition=True)
    out = dec.decompose("who won Ayawaso?")
    assert out["sub_queries"] == ["who won Ayawaso?"]
    assert out["sources"] == ["both"]


def test_decomposer_heuristic_skips_llm_and_detects_budget():
    fake_gen = MagicMock()
    dec = QueryDecomposer(generator=fake_gen, use_llm_decomposition=False)
    out = dec.decompose("What was the education allocation in the 2025 budget?")
    fake_gen.generate.assert_not_called()
    assert out["sub_queries"][0].startswith("What was")
    assert out["sources"] == ["budget"]


def test_decomposer_heuristic_detects_elections():
    fake_gen = MagicMock()
    dec = QueryDecomposer(generator=fake_gen, use_llm_decomposition=False)
    out = dec.decompose("Who won the NPP votes in Ashanti in 2020?")
    fake_gen.generate.assert_not_called()
    assert out["sources"] == ["elections"]
