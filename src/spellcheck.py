from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd

from .tokenize import iter_tokens

from .levenshtein import levenshtein
from .weighted_levenshtein import weighted_levenshtein
from .spell_utils import load_freqs, filter_vocab


def load_vocab(
    corpus_path: str,
    lowercase: bool = True,
    min_freq: int = 2,
    min_len: int = 3,
    max_upper_ratio: float = 0.6,
) -> Counter:
    freqs = load_freqs(corpus_path, lowercase=lowercase)
    freqs = filter_vocab(freqs, min_freq=min_freq, min_len=min_len, max_upper_ratio=max_upper_ratio)
    return freqs


def suggest(
    word: str,
    vocab: Dict[str, int],
    max_dist: int = 2,
    top_k: int = 5,
    weights: Dict | None = None,
) -> List[Tuple[str, int]]:
    candidates: List[Tuple[int, int, str]] = []  # (dist, -freq, token)
    for tok, freq in vocab.items():
        if abs(len(tok) - len(word)) > max_dist:
            continue
        if weights:
            dist = weighted_levenshtein(word, tok, weights, max_cost=max_dist)
        else:
            dist = levenshtein(word, tok, max_dist=max_dist)
        if dist <= max_dist:
            candidates.append((dist, -freq, tok))
    candidates.sort()
    return [(tok, -freq) for dist, freq, tok in candidates[:top_k]]


def load_words_from_file(path: str) -> List[str]:
    return [w.strip() for w in Path(path).read_text(encoding="utf-8").splitlines() if w.strip()]


def main():
    ap = argparse.ArgumentParser(description="Simple spell checker using Levenshtein distance over corpus vocabulary.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv", help="CSV with 'text' column.")
    ap.add_argument("--lowercase", action="store_true", default=True, help="Lowercase tokens for vocab/building.")
    ap.add_argument("--min_freq", type=int, default=2, help="Drop vocab items under this count.")
    ap.add_argument("--min_len", type=int, default=3, help="Drop vocab items shorter than this.")
    ap.add_argument("--max_upper_ratio", type=float, default=0.6, help="Drop tokens with higher uppercase ratio (acronyms).")
    ap.add_argument("--max_dist", type=int, default=2, help="Maximum edit distance for candidates.")
    ap.add_argument("--top_k", type=int, default=5, help="Return up to this many suggestions.")
    ap.add_argument("--confusion", type=str, help="Path to confusion.json (weights) to enable weighted edit distance.")

    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--word", type=str, help="Single word to correct.")
    target.add_argument("--wordlist", type=str, help="File with one word per line to correct.")
    target.add_argument("--text_path", type=str, help="Plain text file; all OOV tokens will be checked.")
    ap.add_argument("--out", type=str, default="outputs/spellcheck/suggestions.txt", help="Where to write suggestions.")
    args = ap.parse_args()

    vocab = load_vocab(
        args.corpus_path,
        lowercase=args.lowercase,
        min_freq=args.min_freq,
        min_len=args.min_len,
        max_upper_ratio=args.max_upper_ratio,
    )
    weights = None
    if args.confusion:
        import json
        weights_data = json.loads(Path(args.confusion).read_text(encoding="utf-8"))
        # weights were stored with stringified keys; eval back the tuple
        w = {}
        for k, v in weights_data.get("weights", {}).items():
            # k format: "('sub', 'a', 'b')" etc.
            w[eval(k)] = float(v)
        weights = w
    known = set(vocab.keys())

    if args.word:
        words = [args.word]
    elif args.wordlist:
        words = load_words_from_file(args.wordlist)
    else:
        text = Path(args.text_path).read_text(encoding="utf-8")
        words = [w for w in iter_tokens([text], lowercase=args.lowercase)]

    suggestions = []
    for w in words:
        if w in known:
            continue
        cands = suggest(w, vocab, max_dist=args.max_dist, top_k=args.top_k, weights=weights)
        if cands:
            suggestion_str = ", ".join([f"{tok} (freq={freq})" for tok, freq in cands])
        else:
            suggestion_str = "(no candidates)"
        suggestions.append(f"{w} -> {suggestion_str}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(suggestions), encoding="utf-8")
    print(f"Wrote {len(suggestions)} suggestions to {out_path}")


if __name__ == "__main__":
    main()
