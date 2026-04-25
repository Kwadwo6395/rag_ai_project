# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/ingest.py
import re
import pandas as pd
from pypdf import PdfReader

_HEADER_FOOTER_RE = re.compile(r"^\s*(Page\s+\d+|\d+\s*\|.*)$", re.IGNORECASE)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_pdf_pages(path: str) -> list[str]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not _HEADER_FOOTER_RE.match(ln.strip())
        ]
        cleaned = re.sub(r"[ \t]+", " ", "\n".join(lines))
        pages.append(cleaned.strip())
    return pages
