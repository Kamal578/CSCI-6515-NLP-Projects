# src/apply_cleaning.py
from pathlib import Path
import pandas as pd

from src.clean_corpus import clean_wiki_page

def clean_all_articles(
    input_csv: str = "data/raw/corpus.csv",
    output_csv: str = "data/processed/corpus_clean.csv"
):
    df = pd.read_csv(input_csv)
    df["text_clean"] = df["text"].fillna("").apply(clean_wiki_page)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Cleaned corpus saved to {output_csv}")

if __name__ == "__main__":
    clean_all_articles()
