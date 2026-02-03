# src/task1_stats.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .load_data import load_corpus_csv
from .tokenize import iter_tokens


def task1_stats(
    corpus_path: str = "data/raw/corpus.csv",
    out_dir: str = "outputs/stats",
    plots_dir: str = "outputs/plots",
    lowercase: bool = True,
    top_n: int = 2000,
    make_zipf_plot: bool = True,
) -> None:
    out_dir = Path(out_dir)
    plots_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_corpus_csv(corpus_path)

    freqs = Counter(iter_tokens(df["text"].tolist(), lowercase=lowercase))

    num_tokens = sum(freqs.values())
    num_types = len(freqs)

    # Save frequency table
    freq_df = (
        pd.DataFrame(freqs.items(), columns=["token", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    freq_df.to_csv(out_dir / "token_freq.csv", index=False, encoding="utf-8")

    # Summary
    summary = {
        "documents": int(len(df)),
        "num_tokens": int(num_tokens),
        "num_types": int(num_types),
        "lowercase": bool(lowercase),
        "top_20": freq_df.head(20).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional Zipf plot (rank vs frequency)
    if make_zipf_plot:
        top = freq_df.head(top_n)
        ranks = list(range(1, len(top) + 1))
        counts = top["count"].tolist()

        plt.figure()
        plt.loglog(ranks, counts)
        plt.xlabel("Rank (log)")
        plt.ylabel("Frequency (log)")
        plt.title("Zipf-like Frequency Plot (Top tokens)")
        plt.tight_layout()
        plt.savefig(plots_dir / "zipf.png", dpi=200)
        plt.close()

    print(f"[Task 1] docs={len(df)} tokens={num_tokens} types={num_types}")
    print(f"Saved: {out_dir / 'summary.json'}")
    print(f"Saved: {out_dir / 'token_freq.csv'}")
    if make_zipf_plot:
        print(f"Saved: {plots_dir / 'zipf.png'}")


if __name__ == "__main__":
    task1_stats()
