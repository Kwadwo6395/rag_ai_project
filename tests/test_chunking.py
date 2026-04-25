# tests/test_chunking.py
import pandas as pd
from rag.chunking import (
    Chunk,
    chunk_csv_rows,
    chunk_election_national_summaries,
)


def test_chunk_csv_rows_produces_one_chunk_per_row():
    df = pd.DataFrame([
        {"constituency": "Ayawaso West", "region": "Greater Accra",
         "year": 2020, "party": "NPP", "votes": 47201, "percentage": 53.2},
        {"constituency": "Ayawaso West", "region": "Greater Accra",
         "year": 2020, "party": "NDC", "votes": 39950, "percentage": 45.0},
    ])
    chunks = chunk_csv_rows(df, source="elections.csv")
    assert len(chunks) == 2
    assert all(isinstance(c, Chunk) for c in chunks)
    # Chunk text is natural-language, contains the values
    assert "Ayawaso West" in chunks[0].text
    assert "NPP" in chunks[0].text
    assert "47,201" in chunks[0].text or "47201" in chunks[0].text
    assert chunks[0].metadata["source"] == "elections.csv"
    assert chunks[0].metadata["row"] == 0


def test_national_summaries_roll_up_npp_ndc():
    df = pd.DataFrame(
        [
            {"year": 2020, "code": "NPP", "votes": 100, "x": 1},
            {"year": 2020, "code": "NDC", "votes": 60, "x": 1},
            {"year": 2020, "code": "Others", "votes": 5, "x": 1},
            {"year": 2020, "code": "NPP", "votes": 50, "x": 2},
            {"year": 2020, "code": "NDC", "votes": 40, "x": 2},
        ]
    )
    out = chunk_election_national_summaries(df, "elections.csv")
    assert len(out) == 1
    assert "150" in out[0].text.replace(",", "") or "150" in out[0].text
    assert "100" in out[0].text or "150" in out[0].text
    assert "NPP won the national" in out[0].text
    assert out[0].metadata["row"] == "national_total_2020"


from rag.chunking import chunk_pdf_pages


def test_chunk_pdf_pages_respects_size_and_overlap():
    text = ("Paragraph one sentence one. Paragraph one sentence two. " * 40)
    pages = [text]
    chunks = chunk_pdf_pages(pages, source="budget.pdf",
                             chunk_size=400, overlap=80)
    assert all(len(c.text) <= 400 + 80 for c in chunks)
    assert len(chunks) >= 2
    # overlap exists: some substring at end of chunk[0] appears at start of chunk[1]
    assert chunks[0].text[-40:] in chunks[1].text or \
           chunks[1].text[:40] in chunks[0].text
    assert chunks[0].metadata["source"] == "budget.pdf"
    assert chunks[0].metadata["page"] == 0


def test_chunk_pdf_pages_does_not_split_mid_word():
    pages = ["supercalifragilistic " * 50]
    chunks = chunk_pdf_pages(pages, source="x.pdf",
                             chunk_size=50, overlap=10)
    for c in chunks:
        assert not c.text.startswith("cali")
        assert not c.text.endswith("califr")
