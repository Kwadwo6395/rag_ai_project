# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: rag/chunking.py
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _row_to_sentence(row: pd.Series) -> str:
    parts = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        if isinstance(v, (int, float)) and float(v).is_integer():
            parts.append(f"{k.replace('_', ' ')}: {int(v):,}")
        elif isinstance(v, float):
            parts.append(f"{k.replace('_', ' ')}: {v:.2f}")
        else:
            parts.append(f"{k.replace('_', ' ')}: {v}")
    return "; ".join(parts) + "."


def chunk_csv_rows(df: pd.DataFrame, source: str) -> list[Chunk]:
    chunks = []
    for i, row in df.iterrows():
        text = _row_to_sentence(row)
        chunks.append(Chunk(text=text, metadata={"source": source, "row": int(i)}))
    return chunks


def chunk_election_national_summaries(df: pd.DataFrame, source: str) -> list[Chunk]:
    """Synthetic chunks: national vote roll-ups per year (sum of regional rows).

    The CSV has no single 'national total' row, so broad questions like 'who won 2020'
    never retrieved a decisive passage. These summaries fix that while staying
    derived only from the same table.
    """
    cols = {str(c).lower() for c in df.columns}
    if not {"year", "code", "votes"}.issubset(cols):
        return []
    work = df.copy()
    work.columns = [str(c).lower() for c in work.columns]
    work["_code"] = work["code"].astype(str).str.strip().str.upper()
    work["_votes"] = pd.to_numeric(work["votes"], errors="coerce").fillna(0).astype(int)
    chunks: list[Chunk] = []
    for year in sorted(work["year"].unique()):
        y = work[work["year"] == year]
        npp = int(y.loc[y["_code"] == "NPP", "_votes"].sum())
        ndc = int(y.loc[y["_code"] == "NDC", "_votes"].sum())
        oth = int(y.loc[~y["_code"].isin(("NPP", "NDC")), "_votes"].sum())
        total = npp + ndc + oth
        if total <= 0:
            continue
        npp_pct = 100.0 * npp / total
        ndc_pct = 100.0 * ndc / total
        if npp > ndc:
            winner = (
                f"In {year}, NPP won the national presidential popular vote over NDC "
                f"in this dataset (regional rows summed)."
            )
        elif ndc > npp:
            winner = (
                f"In {year}, NDC won the national presidential popular vote over NPP "
                f"in this dataset (regional rows summed)."
            )
        else:
            winner = f"In {year}, NPP and NDC are tied on summed national votes in this dataset."
        text = (
            f"Ghana presidential election {year}: national totals aggregated from all regions "
            f"in elections.csv. NPP total votes: {npp:,} ({npp_pct:.2f}% of all votes in this sum); "
            f"NDC total votes: {ndc:,} ({ndc_pct:.2f}%); other candidates combined: {oth:,}. "
            f"{winner} "
            f"When the user asks who won the {year} election nationally, use this roll-up."
        )
        chunks.append(
            Chunk(
                text=text,
                metadata={"source": source, "row": f"national_total_{year}"},
            )
        )
    return chunks


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Back up to the nearest separator to avoid mid-word splits
            window = text[start:end]
            split_at = -1
            for sep in _SEPARATORS:
                idx = window.rfind(sep)
                if idx > size * 0.5:  # don't back up too far
                    split_at = idx + len(sep)
                    break
            if split_at > 0:
                end = start + split_at
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pdf_pages(
    pages: list[str],
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    out = []
    for page_idx, page_text in enumerate(pages):
        for piece in _split_with_overlap(page_text, chunk_size, overlap):
            out.append(Chunk(
                text=piece,
                metadata={"source": source, "page": page_idx},
            ))
    return out
