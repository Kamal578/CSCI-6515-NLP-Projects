# src/load_data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_corpus_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found. Columns: {df.columns.tolist()}")
    df["text"] = df["text"].fillna("").astype(str)
    return df