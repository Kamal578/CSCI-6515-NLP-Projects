from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import regex as re
import pandas as pd

from .tokenize import iter_tokens


WORD_CHARS_RE = re.compile(r"^[\p{L}]+$", re.UNICODE)


def count_tokens(corpus_path: str, lowercase: bool = True) -> Counter:
    df = pd.read_csv(corpus_path)
    freqs = Counter(iter_tokens(df["text"].fillna("").astype(str).tolist(), lowercase=lowercase))
    return freqs


def filter_counts(
    freqs: Counter,
    min_freq: int = 2,
    max_freq: int | None = None,
    min_len: int = 2,
    alpha_only: bool = True,
) -> Counter:
    out = Counter()
    for w, c in freqs.items():
        if c < min_freq:
            continue
        if max_freq is not None and c > max_freq:
            continue
        if len(w) < min_len:
            continue
        if alpha_only and not WORD_CHARS_RE.match(w):
            continue
        out[w] = c
    return out


def save_vocab(vocab: Counter, vocab_path: Path) -> None:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        for w, _ in vocab.most_common():
            f.write(f"{w}\n")


def save_summary(summary_path: Path, before: int, after: int, params: dict) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tokens_before_filter": before, "tokens_after_filter": after, **params}
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Build filtered vocabulary from corpus with frequency thresholds.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv", help="CSV with 'text' column.")
    ap.add_argument("--vocab_path", type=str, default="data/processed/vocab.txt", help="Where to write vocab.")
    ap.add_argument("--summary_path", type=str, default="outputs/stats/vocab_summary.json")
    ap.add_argument("--min_freq", type=int, default=2, help="Keep tokens with frequency >= this.")
    ap.add_argument("--max_freq", type=int, default=None, help="Optional cap to drop ultra-common tokens.")
    ap.add_argument("--min_len", type=int, default=2, help="Drop tokens shorter than this.")
    ap.add_argument("--alpha_only", action="store_true", default=True, help="Keep only alphabetic tokens.")
    ap.add_argument("--lowercase", action="store_true", default=True, help="Lowercase tokens before counting.")
    args = ap.parse_args()

    freqs = count_tokens(args.corpus_path, lowercase=args.lowercase)
    filtered = filter_counts(
        freqs,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        min_len=args.min_len,
        alpha_only=args.alpha_only,
    )

    save_vocab(filtered, Path(args.vocab_path))
    save_summary(
        Path(args.summary_path),
        before=len(freqs),
        after=len(filtered),
        params={
            "min_freq": args.min_freq,
            "max_freq": args.max_freq,
            "min_len": args.min_len,
            "alpha_only": args.alpha_only,
            "lowercase": args.lowercase,
        },
    )

    print(f"Vocab size: {len(filtered)} (from {len(freqs)} types).")
    print(f"Wrote vocab to {args.vocab_path}")
    print(f"Wrote summary to {args.summary_path}")


if __name__ == "__main__":
    main()