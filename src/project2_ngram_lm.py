from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def pad_sentence(tokens: Sequence[str], n: int) -> list[str]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return ([START_TOKEN] * (n - 1)) + list(tokens) + [END_TOKEN]


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


@dataclass
class EvalStats:
    n: int
    num_events: int
    zero_prob_events: int
    log_prob_sum: float
    perplexity: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "num_events": self.num_events,
            "zero_prob_events": self.zero_prob_events,
            "log_prob_sum": self.log_prob_sum,
            "perplexity": self.perplexity,
            "perplexity_is_infinite": math.isinf(self.perplexity),
        }


class MLENgramModel:
    """Maximum-likelihood n-gram model without smoothing."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.total_unigrams = 0
        self.vocab: set[str] = set()
        self.fitted = False

    def fit(self, sentences: Iterable[Sequence[str]]) -> "MLENgramModel":
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.total_unigrams = 0

        for sent in sentences:
            padded = pad_sentence(sent, self.n)
            for ng in iter_ngrams(padded, self.n):
                self.ngram_counts[ng] += 1
                if self.n == 1:
                    self.total_unigrams += 1
                else:
                    self.context_counts[ng[:-1]] += 1

        if self.n == 1:
            self.vocab = {ng[0] for ng in self.ngram_counts.keys()}
        else:
            self.vocab = {tok for ng in self.ngram_counts.keys() for tok in ng}
        self.fitted = True
        return self

    def prob(self, ngram: tuple[str, ...]) -> float:
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        if len(ngram) != self.n:
            raise ValueError(f"Expected {self.n}-gram, got {len(ngram)}")

        count = self.ngram_counts.get(ngram, 0)
        if count == 0:
            return 0.0

        if self.n == 1:
            if self.total_unigrams == 0:
                return 0.0
            return count / self.total_unigrams

        denom = self.context_counts.get(ngram[:-1], 0)
        if denom == 0:
            return 0.0
        return count / denom

    def evaluate(self, sentences: Iterable[Sequence[str]]) -> EvalStats:
        if not self.fitted:
            raise RuntimeError("Model not fitted")

        num_events = 0
        zero_prob_events = 0
        log_prob_sum = 0.0

        for sent in sentences:
            padded = pad_sentence(sent, self.n)
            for ng in iter_ngrams(padded, self.n):
                num_events += 1
                p = self.prob(ng)
                if p <= 0.0:
                    zero_prob_events += 1
                    continue
                log_prob_sum += math.log(p)

        if num_events == 0:
            ppl = float("inf")
        elif zero_prob_events > 0:
            ppl = float("inf")
        else:
            ppl = math.exp(-log_prob_sum / num_events)

        return EvalStats(
            n=self.n,
            num_events=num_events,
            zero_prob_events=zero_prob_events,
            log_prob_sum=log_prob_sum,
            perplexity=ppl,
        )


def build_train_vocab(
    sentences: Iterable[Sequence[str]],
    min_freq: int = 1,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sent in sentences:
        counts.update(sent)
    if min_freq <= 1:
        return counts
    return Counter({tok: c for tok, c in counts.items() if c >= min_freq})


def map_sentence_to_vocab(sentence: Sequence[str], vocab: set[str]) -> list[str]:
    return [tok if tok in vocab else UNK_TOKEN for tok in sentence]

