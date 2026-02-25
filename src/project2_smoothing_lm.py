from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .project2_ngram_lm import EvalStats, iter_ngrams, pad_sentence


@dataclass
class NgramCountStore:
    max_order: int
    ngram_counts: dict[int, Counter[tuple[str, ...]]]
    context_counts: dict[int, Counter[tuple[str, ...]]]
    num_followers: dict[int, dict[tuple[str, ...], int]]
    total_unigrams: int
    pred_vocab: set[str]
    continuation_counts: Counter[str]
    total_distinct_bigrams: int
    fitted: bool = False

    @classmethod
    def build(cls, sentences: Iterable[Sequence[str]], max_order: int = 3) -> "NgramCountStore":
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        ngram_counts = {n: Counter() for n in range(1, max_order + 1)}
        context_counts = {n: Counter() for n in range(2, max_order + 1)}
        total_unigrams = 0

        for sent in sentences:
            for n in range(1, max_order + 1):
                padded = pad_sentence(sent, n)
                for ng in iter_ngrams(padded, n):
                    ngram_counts[n][ng] += 1
                    if n == 1:
                        total_unigrams += 1
                    else:
                        context_counts[n][ng[:-1]] += 1

        pred_vocab = {ng[0] for ng in ngram_counts[1].keys()}

        num_followers: dict[int, dict[tuple[str, ...], int]] = {}
        for n in range(2, max_order + 1):
            followers = defaultdict(int)
            for ng in ngram_counts[n].keys():
                followers[ng[:-1]] += 1
            num_followers[n] = dict(followers)

        continuation_counts: Counter[str] = Counter()
        total_distinct_bigrams = 0
        if max_order >= 2:
            for bg in ngram_counts[2].keys():
                continuation_counts[bg[1]] += 1
            total_distinct_bigrams = len(ngram_counts[2])

        return cls(
            max_order=max_order,
            ngram_counts=ngram_counts,
            context_counts=context_counts,
            num_followers=num_followers,
            total_unigrams=total_unigrams,
            pred_vocab=pred_vocab,
            continuation_counts=continuation_counts,
            total_distinct_bigrams=total_distinct_bigrams,
            fitted=True,
        )

    @property
    def pred_vocab_size(self) -> int:
        return len(self.pred_vocab)

    def mle_prob(self, n: int, ngram: tuple[str, ...]) -> float:
        if not self.fitted:
            raise RuntimeError("Counts not built")
        if len(ngram) != n:
            raise ValueError(f"Expected {n}-gram, got {len(ngram)}")

        count = self.ngram_counts[n].get(ngram, 0)
        if count == 0:
            return 0.0
        if n == 1:
            return count / self.total_unigrams if self.total_unigrams else 0.0
        denom = self.context_counts[n].get(ngram[:-1], 0)
        return count / denom if denom else 0.0

    def count(self, n: int, ngram: tuple[str, ...]) -> int:
        return self.ngram_counts[n].get(ngram, 0)

    def context_count(self, n: int, context: tuple[str, ...]) -> int:
        if n == 1:
            return self.total_unigrams
        return self.context_counts[n].get(context, 0)


class BaseScoredNgramModel:
    def __init__(self, n: int, counts: NgramCountStore):
        if n < 1:
            raise ValueError("n must be >= 1")
        if not counts.fitted:
            raise RuntimeError("Count store is not fitted")
        if counts.max_order < n:
            raise ValueError(f"Count store max_order={counts.max_order} is < model order={n}")
        self.n = n
        self.counts = counts

    def prob(self, ngram: tuple[str, ...]) -> float:
        raise NotImplementedError

    def evaluate(self, sentences: Iterable[Sequence[str]]) -> EvalStats:
        num_events = 0
        zero_prob_events = 0
        log_prob_sum = 0.0

        for sent in sentences:
            padded = pad_sentence(sent, self.n)
            for ng in iter_ngrams(padded, self.n):
                num_events += 1
                p = self.prob(ng)
                if p <= 0.0 or not math.isfinite(p):
                    zero_prob_events += 1
                    continue
                log_prob_sum += math.log(p)

        if num_events == 0 or zero_prob_events > 0:
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


class LaplaceNgramModel(BaseScoredNgramModel):
    def __init__(self, n: int, counts: NgramCountStore, k: float = 1.0):
        super().__init__(n=n, counts=counts)
        if k <= 0:
            raise ValueError("Laplace k must be > 0")
        self.k = float(k)

    def prob(self, ngram: tuple[str, ...]) -> float:
        n = self.n
        c = self.counts.count(n, ngram)
        V = self.counts.pred_vocab_size
        if n == 1:
            denom = self.counts.total_unigrams + self.k * V
            return (c + self.k) / denom if denom else 0.0
        ctx = ngram[:-1]
        ctx_count = self.counts.context_count(n, ctx)
        denom = ctx_count + self.k * V
        return (c + self.k) / denom if denom else 0.0


class InterpolationNgramModel(BaseScoredNgramModel):
    def __init__(self, n: int, counts: NgramCountStore, lambdas: Sequence[float]):
        super().__init__(n=n, counts=counts)
        if len(lambdas) != n:
            raise ValueError(f"Interpolation requires {n} lambdas, got {len(lambdas)}")
        ls = [float(x) for x in lambdas]
        if any(x < 0 for x in ls):
            raise ValueError("Lambdas must be non-negative")
        s = sum(ls)
        if s <= 0:
            raise ValueError("Lambdas must sum to > 0")
        self.lambdas = [x / s for x in ls]

    def prob(self, ngram: tuple[str, ...]) -> float:
        p = 0.0
        for order in range(1, self.n + 1):
            lam = self.lambdas[order - 1]
            if lam == 0:
                continue
            p += lam * self.counts.mle_prob(order, ngram[-order:])
        return min(max(p, 0.0), 1.0)


class DiscountBackoffNgramModel(BaseScoredNgramModel):
    """
    Normalized discount-backoff hybrid using absolute discount interpolation:
      P(w|h) = max(c(hw)-d, 0)/c(h) + lambda(h) * P(w|h')
      lambda(h) = d * T(h) / c(h), T(h)=#distinct followers after h
    """

    def __init__(self, n: int, counts: NgramCountStore, d: float = 0.75):
        super().__init__(n=n, counts=counts)
        if d < 0:
            raise ValueError("discount d must be >= 0")
        self.d = float(d)

    def _prob_order(self, n: int, ngram: tuple[str, ...]) -> float:
        if n == 1:
            return self.counts.mle_prob(1, ngram)

        ctx = ngram[:-1]
        c_ctx = self.counts.context_count(n, ctx)
        lower = self._prob_order(n - 1, ngram[1:])
        if c_ctx <= 0:
            return lower

        c_ng = self.counts.count(n, ngram)
        t_ctx = self.counts.num_followers.get(n, {}).get(ctx, 0)
        first = max(c_ng - self.d, 0.0) / c_ctx
        lam = min(max((self.d * t_ctx) / c_ctx, 0.0), 1.0)
        p = first + lam * lower
        return min(max(p, 0.0), 1.0)

    def prob(self, ngram: tuple[str, ...]) -> float:
        return self._prob_order(self.n, ngram)


class KneserNeyNgramModel(BaseScoredNgramModel):
    """Interpolated Kneser-Ney with a single absolute discount D."""

    def __init__(self, n: int, counts: NgramCountStore, d: float = 0.75):
        super().__init__(n=n, counts=counts)
        if d < 0:
            raise ValueError("discount D must be >= 0")
        self.d = float(d)

    def _unigram_kn(self, w: str) -> float:
        if self.counts.total_distinct_bigrams > 0:
            return self.counts.continuation_counts.get(w, 0) / self.counts.total_distinct_bigrams
        # Fallback if no bigrams exist (degenerate tiny corpus)
        return self.counts.mle_prob(1, (w,))

    def _prob_order(self, n: int, ngram: tuple[str, ...]) -> float:
        if n == 1:
            return self._unigram_kn(ngram[0])

        ctx = ngram[:-1]
        c_ctx = self.counts.context_count(n, ctx)
        lower = self._prob_order(n - 1, ngram[1:])
        if c_ctx <= 0:
            return lower

        c_ng = self.counts.count(n, ngram)
        t_ctx = self.counts.num_followers.get(n, {}).get(ctx, 0)
        first = max(c_ng - self.d, 0.0) / c_ctx
        lam = min(max((self.d * t_ctx) / c_ctx, 0.0), 1.0)
        p = first + lam * lower
        return min(max(p, 0.0), 1.0)

    def prob(self, ngram: tuple[str, ...]) -> float:
        return self._prob_order(self.n, ngram)


def make_laplace_model(n: int, counts: NgramCountStore, params: dict) -> LaplaceNgramModel:
    return LaplaceNgramModel(n=n, counts=counts, k=float(params["k"]))


def make_interpolation_model(n: int, counts: NgramCountStore, params: dict) -> InterpolationNgramModel:
    return InterpolationNgramModel(n=n, counts=counts, lambdas=params["lambdas"])


def make_backoff_model(n: int, counts: NgramCountStore, params: dict) -> DiscountBackoffNgramModel:
    return DiscountBackoffNgramModel(n=n, counts=counts, d=float(params["d"]))


def make_kneser_ney_model(n: int, counts: NgramCountStore, params: dict) -> KneserNeyNgramModel:
    return KneserNeyNgramModel(n=n, counts=counts, d=float(params["d"]))

