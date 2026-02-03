# src/task3_bpe.py
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

from .load_data import load_corpus_csv
from .tokenize import iter_tokens
from .bpe import train_bpe, encode_word_bpe


def task3_bpe(
    corpus_path: str = "data/raw/corpus.csv",
    out_dir: str = "outputs/bpe",
    lowercase: bool = True,
    num_merges: int = 5000,
    min_word_freq: int = 2,
    sample_words: int = 30,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_corpus_csv(corpus_path)
    words = list(iter_tokens(df["text"].tolist(), lowercase=lowercase))

    model = train_bpe(words, num_merges=num_merges, min_freq=min_word_freq)

    # Save merges
    merges_path = out_dir / "merges.txt"
    merges_path.write_text("\n".join([f"{a} {b}" for a, b in model.merges]), encoding="utf-8")

    # Encode corpus into BPE tokens and count
    bpe_freq = Counter()
    for w in tqdm(words, desc="BPE encoding"):
        pieces = encode_word_bpe(w, model)
        for p in pieces:
            bpe_freq[p] += 1

    bpe_df = (
        pd.DataFrame(bpe_freq.items(), columns=["bpe_token", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    bpe_df.to_csv(out_dir / "bpe_token_freq.csv", index=False, encoding="utf-8")

    # Save a few examples
    examples = []
    for w, _ in Counter(words).most_common(sample_words):
        examples.append({"word": w, "bpe": encode_word_bpe(w, model)})

    summary = {
        "documents": int(len(df)),
        "lowercase": bool(lowercase),
        "num_merges": int(len(model.merges)),
        "min_word_freq": int(min_word_freq),
        "word_tokens_total": int(len(words)),
        "bpe_types": int(len(bpe_freq)),
        "bpe_tokens_total": int(sum(bpe_freq.values())),
        "examples": examples[:20],
    }
    (out_dir / "bpe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Task 3] merges={len(model.merges)} bpe_types={len(bpe_freq)}")
    print(f"Saved: {merges_path}")
    print(f"Saved: {out_dir / 'bpe_token_freq.csv'}")
    print(f"Saved: {out_dir / 'bpe_summary.json'}")


if __name__ == "__main__":
    task3_bpe()