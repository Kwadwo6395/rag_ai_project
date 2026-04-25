# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: scripts/build_index.py
"""Build the vector + BM25 index from the two source datasets."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rag.ingest import load_csv, load_pdf_pages
from rag.chunking import (
    chunk_csv_rows,
    chunk_election_national_summaries,
    chunk_pdf_pages,
)
from rag.embeddings import Embedder
from rag.vector_store import VectorStore
from rag.bm25_store import BM25Store

CSV_PATH = "data/Ghana_Election_Result.csv"
PDF_PATH = "data/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
INDEX_PREFIX = "index/store"
BM25_PATH = "index/bm25.pkl"


def main():
    print("Loading CSV...")
    df = load_csv(CSV_PATH)
    print(f"  {len(df)} rows")

    print("Loading PDF...")
    pages = load_pdf_pages(PDF_PATH)
    print(f"  {len(pages)} pages")

    print("Chunking...")
    national = chunk_election_national_summaries(df, source="elections.csv")
    csv_chunks = national + chunk_csv_rows(df, source="elections.csv")
    pdf_chunks = chunk_pdf_pages(pages, source="budget.pdf")
    all_chunks = csv_chunks + pdf_chunks
    print(f"  {len(csv_chunks)} CSV chunks + {len(pdf_chunks)} PDF chunks = {len(all_chunks)} total")

    print("Embedding...")
    emb = Embedder()
    texts = [c.text for c in all_chunks]
    vecs = emb.embed(texts)

    print("Building metadata...")
    metadata = [
        {"chunk_id": i, "text": c.text, **c.metadata}
        for i, c in enumerate(all_chunks)
    ]

    print("Saving vector store...")
    VectorStore(vectors=vecs, metadata=metadata).save(INDEX_PREFIX)

    print("Saving BM25 store...")
    BM25Store.build(texts, metadata).save(BM25_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
