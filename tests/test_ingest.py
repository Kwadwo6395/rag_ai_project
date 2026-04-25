# tests/test_ingest.py
from rag.ingest import load_csv, load_pdf_pages


def test_load_csv_returns_cleaned_dataframe():
    df = load_csv("data/Ghana_Election_Result.csv")
    assert len(df) > 0
    assert df.isnull().all(axis=1).sum() == 0  # no fully-null rows
    # every column name is lower-snake-case style
    assert all(c == c.lower().replace(" ", "_") for c in df.columns)


def test_load_pdf_pages_returns_list_of_nonempty_strings():
    pages = load_pdf_pages("data/2025-Budget-Statement-and-Economic-Policy_v4.pdf")
    assert len(pages) > 10
    assert all(isinstance(p, str) for p in pages)
    assert any(len(p) > 200 for p in pages)
