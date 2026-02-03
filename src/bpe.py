# src/bpe.py
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

END = "</w>"  # end-of-word marker


def word_to_symbols(word: str) -> Tuple[str, ...]:
    # split into unicode characters + end marker
    return tuple(list(word) + [END])


def get_pair_counts(vocab: Dict[Tuple[str, ...], int]) -> Counter:
    pair_counts = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts


def merge_vocab(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
    a, b = pair
    merged = a + b
    new_vocab: Dict[Tuple[str, ...], int] = {}
    for symbols, freq in vocab.items():
        new_syms: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        new_vocab[tuple(new_syms)] = freq
    return new_vocab


@dataclass
class BPEModel:
    merges: List[Tuple[str, str]]               # merge rules in order
    merge_ranks: Dict[Tuple[str, str], int]     # for fast encoding


def train_bpe(words: Iterable[str], num_merges: int = 5000, min_freq: int = 2) -> BPEModel:
    """
    Train BPE merges on a word stream.
    - words: iterable of word tokens (already normalized/lowercased if desired)
    - min_freq: ignore extremely rare words to reduce noise
    """
    word_freq = Counter(words)
    word_freq = Counter({w: c for w, c in word_freq.items() if c >= min_freq})

    vocab: Dict[Tuple[str, ...], int] = {word_to_symbols(w): c for w, c in word_freq.items()}
    merges: List[Tuple[str, str]] = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break
        (a, b), best = pair_counts.most_common(1)[0]
        if best < 2:
            break
        merges.append((a, b))
        vocab = merge_vocab((a, b), vocab)

    merge_ranks = {m: i for i, m in enumerate(merges)}
    return BPEModel(merges=merges, merge_ranks=merge_ranks)


def encode_word_bpe(word: str, model: BPEModel) -> List[str]:
    """
    Apply BPE merges to one word and return subword tokens (END marker removed).
    """
    symbols = list(word_to_symbols(word))

    while True:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        ranked = [(model.merge_ranks[p], p) for p in pairs if p in model.merge_ranks]
        if not ranked:
            break
        _, best_pair = min(ranked, key=lambda x: x[0])

        a, b = best_pair
        merged = a + b

        new_syms: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        symbols = new_syms

    if symbols and symbols[-1] == END:
        symbols = symbols[:-1]
    return symbols
