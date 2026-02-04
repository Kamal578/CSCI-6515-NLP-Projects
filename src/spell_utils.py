from __future__ import annotations

import regex as re
from collections import Counter
from typing import Iterable
import pandas as pd

from .tokenize import iter_tokens

# Azerbaijani + basic Latin alphabet for sampling synthetic typos
AZ_ALPHABET = list("abcçdeəfgğhxıijkqlmnoöpprsşt tuüvyz".replace(" ", "")) + list("abcdefghijklmnopqrstuvwxyz")


def load_freqs(corpus_path: str, lowercase: bool = True) -> Counter:
    df = pd.read_csv(corpus_path)
    freqs = Counter(iter_tokens(df["text"].fillna("").astype(str).tolist(), lowercase=lowercase))
    return freqs


WORD_CHARS_RE = re.compile(r"^[\p{L}]+$", re.UNICODE)


def filter_vocab(
    freqs: Counter,
    min_freq: int = 2,
    min_len: int = 3,
    max_upper_ratio: float = 0.6,
) -> Counter:
    """
    Drop very rare, too short, non-letter, or acronym-like tokens.
    """
    clean = Counter()
    for w, c in freqs.items():
        if len(w) < min_len:
            continue
        if c < min_freq:
            continue
        if not WORD_CHARS_RE.match(w):
            continue
        upper_ratio = sum(1 for ch in w if ch.isupper()) / len(w)
        if upper_ratio > max_upper_ratio:
            continue
        clean[w] = c
    return clean
