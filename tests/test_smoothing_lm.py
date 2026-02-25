from __future__ import annotations

import math

from src.project2_smoothing_lm import (
    DiscountBackoffNgramModel,
    InterpolationNgramModel,
    KneserNeyNgramModel,
    LaplaceNgramModel,
    NgramCountStore,
)


def _toy_train():
    return [
        ["a", "b"],
        ["a", "c"],
        ["b", "a"],
    ]


def test_laplace_bigram_probability_matches_formula():
    counts = NgramCountStore.build(_toy_train(), max_order=2)
    model = LaplaceNgramModel(n=2, counts=counts, k=1.0)

    # With sentence padding, context ('a',) appears before: b, c, and </s> -> c(a)=3.
    # V is the unigram prediction vocab (including <s> and </s> in this implementation).
    V = counts.pred_vocab_size
    p = model.prob(("a", "b"))
    expected = (1 + 1.0) / (3 + 1.0 * V)
    assert math.isclose(p, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_interpolation_bigram_probability_is_convex_combination():
    counts = NgramCountStore.build(_toy_train(), max_order=2)
    lambdas = [0.25, 0.75]  # [unigram, bigram]
    model = InterpolationNgramModel(n=2, counts=counts, lambdas=lambdas)

    ng = ("a", "b")
    expected = 0.25 * counts.mle_prob(1, ("b",)) + 0.75 * counts.mle_prob(2, ng)
    assert math.isclose(model.prob(ng), expected, rel_tol=1e-12, abs_tol=1e-12)


def test_smoothed_models_return_finite_trigram_perplexity_on_unseen_trigrams():
    train = [
        ["a", "b", "c"],
        ["a", "b", "d"],
        ["b", "c", "d"],
    ]
    test = [
        ["a", "c", "d"],  # unseen trigram contexts/events likely present
        ["d", "a", "b"],
    ]
    counts = NgramCountStore.build(train, max_order=3)

    models = [
        LaplaceNgramModel(n=3, counts=counts, k=0.1),
        InterpolationNgramModel(n=3, counts=counts, lambdas=[0.2, 0.3, 0.5]),
        DiscountBackoffNgramModel(n=3, counts=counts, d=0.5),
        KneserNeyNgramModel(n=3, counts=counts, d=0.75),
    ]

    for model in models:
        stats = model.evaluate(test)
        assert stats.zero_prob_events == 0
        assert math.isfinite(stats.perplexity)


def test_backoff_and_kneser_ney_probabilities_approximately_normalize_for_context():
    counts = NgramCountStore.build(_toy_train(), max_order=3)
    pred_vocab = sorted(counts.pred_vocab)

    backoff = DiscountBackoffNgramModel(n=2, counts=counts, d=0.5)
    kn = KneserNeyNgramModel(n=2, counts=counts, d=0.75)

    for model in (backoff, kn):
        s = sum(model.prob(("a", w)) for w in pred_vocab)
        assert 0.95 <= s <= 1.05
