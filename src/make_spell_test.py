from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple
import argparse
import pandas as pd

from .spell_utils import load_freqs, filter_vocab, AZ_ALPHABET


def corrupt_word(word: str) -> Tuple[str, str]:
    ops = ["delete", "insert", "substitute", "transpose"]
    op = random.choice(ops)

    if len(word) == 1:
        op = random.choice(["insert", "substitute"])  # avoid empty word

    if op == "delete":
        i = random.randrange(len(word))
        return op, word[:i] + word[i + 1 :]

    if op == "insert":
        i = random.randrange(len(word) + 1)
        ch = random.choice(AZ_ALPHABET)
        return op, word[:i] + ch + word[i:]

    if op == "substitute":
        i = random.randrange(len(word))
        ch = random.choice(AZ_ALPHABET)
        return op, word[:i] + ch + word[i + 1 :]

    # transpose
    if len(word) < 2:
        return "insert", random.choice(AZ_ALPHABET) + word
    i = random.randrange(len(word) - 1)
    return op, word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def make_spell_test(
    corpus_path: str,
    out_csv: str = "data/processed/spell_test.csv",
    samples: int = 1000,
    seed: int = 42,
    min_freq: int = 2,
    min_len: int = 3,
) -> None:
    random.seed(seed)
    freqs = load_freqs(corpus_path, lowercase=True)
    freqs = filter_vocab(freqs, min_freq=min_freq, min_len=min_len)

    # weighted sampling by frequency
    words = list(freqs.keys())
    weights = [freqs[w] for w in words]
    total = sum(weights)
    probs = [w / total for w in weights]

    rows: List[Tuple[str, str, str]] = []  # (misspelled, correct, op)
    for _ in range(samples):
        correct = random.choices(words, weights=probs, k=1)[0]
        op, noisy = corrupt_word(correct)
        rows.append((noisy, correct, op))

    df = pd.DataFrame(rows, columns=["misspelled", "correct", "operation"])
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} synthetic misspellings to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Create synthetic spell-check evaluation set.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv", help="Corpus CSV with 'text' column.")
    ap.add_argument("--out_csv", type=str, default="data/processed/spell_test.csv")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--min_len", type=int, default=3)
    args = ap.parse_args()

    make_spell_test(
        corpus_path=args.corpus_path,
        out_csv=args.out_csv,
        samples=args.samples,
        seed=args.seed,
        min_freq=args.min_freq,
        min_len=args.min_len,
    )


if __name__ == "__main__":
    main()
