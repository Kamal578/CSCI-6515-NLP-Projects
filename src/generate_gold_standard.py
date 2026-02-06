# src/generate_gold_standard.py
import random
from pathlib import Path
import pandas as pd
from src.sentence_segment import sentence_segment

def generate_gold_standard(
    corpus_path: str = "data/raw/corpus.csv",
    output_path: str = "data/processed/sent_gold.txt",
    num_samples: int = 100, # Number of random texts to sample from the corpus for gold standard generation
    seed: int = 42,
):
    random.seed(seed)

    df = pd.read_csv(corpus_path)
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {corpus_path}")

    texts = df["text"].dropna().astype(str).tolist()
    if num_samples > len(texts):
        num_samples = len(texts)

    sampled_texts = random.sample(texts, num_samples)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for text in sampled_texts:
            sentences = sentence_segment(text)
            for sentence in sentences:
                f.write(sentence + "\n")

    print(f"Gold standard sentences written to {out_path} (seed={seed}).")

if __name__ == "__main__":
    generate_gold_standard()