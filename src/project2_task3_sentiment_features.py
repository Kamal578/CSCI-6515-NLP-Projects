from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer

from .project2_task3_sentiment_preprocess import legacy_tokenize


@dataclass
class LexiconData:
    tokens: list[str]
    weights: np.ndarray
    vectorizer: CountVectorizer
    pos_mask: np.ndarray
    neg_mask: np.ndarray
    token_to_weight: dict[str, float]


def build_sentiment_lexicon(
    texts: pd.Series,
    labels: pd.Series,
    top_k_each: int = 500,
    min_count: int = 5,
    alpha: float = 1.0,
) -> LexiconData:
    # Lexicon induction uses only positive/negative rows so "neutral" does not blur polarity weights.
    polarity_mask = labels.isin(["positive", "negative"])
    texts_pol = texts[polarity_mask]
    labels_pol = labels[polarity_mask]

    uni = CountVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        ngram_range=(1, 1),
        min_df=1,
    )
    X = uni.fit_transform(texts_pol)
    vocab = np.array(uni.get_feature_names_out())

    pos_rows = labels_pol.to_numpy() == "positive"
    neg_rows = labels_pol.to_numpy() == "negative"
    if pos_rows.sum() == 0 or neg_rows.sum() == 0:
        raise ValueError("Need both positive and negative samples to build lexicon.")

    pos_counts = np.asarray(X[pos_rows].sum(axis=0)).ravel()
    neg_counts = np.asarray(X[neg_rows].sum(axis=0)).ravel()
    total_counts = pos_counts + neg_counts

    keep = total_counts >= min_count
    vocab = vocab[keep]
    pos_counts = pos_counts[keep]
    neg_counts = neg_counts[keep]

    V = len(vocab)
    pos_total = pos_counts.sum()
    neg_total = neg_counts.sum()
    logprob_pos = np.log((pos_counts + alpha) / (pos_total + alpha * V))
    logprob_neg = np.log((neg_counts + alpha) / (neg_total + alpha * V))
    polarity_score = logprob_pos - logprob_neg

    pos_idx = np.argsort(-polarity_score)[:top_k_each]
    neg_idx = np.argsort(polarity_score)[:top_k_each]
    sel_idx = np.unique(np.concatenate([pos_idx, neg_idx]))

    sel_vocab = vocab[sel_idx]
    sel_scores = polarity_score[sel_idx]

    # Stable ordering makes the preview and saved vocab easier to inspect.
    order = np.argsort(sel_vocab)
    sel_vocab = sel_vocab[order]
    sel_scores = sel_scores[order]

    lex_vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        vocabulary={tok: i for i, tok in enumerate(sel_vocab.tolist())},
    )

    return LexiconData(
        tokens=sel_vocab.tolist(),
        weights=sel_scores.astype(np.float64),
        vectorizer=lex_vectorizer,
        pos_mask=sel_scores > 0,
        neg_mask=sel_scores < 0,
        token_to_weight={tok: float(w) for tok, w in zip(sel_vocab.tolist(), sel_scores.tolist())},
    )


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b != 0)
    return out


def make_lexicon_feature_matrix(texts: pd.Series, lex: LexiconData) -> tuple[csr_matrix, list[str]]:
    X_counts = lex.vectorizer.transform(texts)
    weights = lex.weights

    total_lex = np.asarray(X_counts.sum(axis=1)).ravel().astype(np.float64)
    pos_count = np.asarray(X_counts[:, lex.pos_mask].sum(axis=1)).ravel().astype(np.float64)
    neg_count = np.asarray(X_counts[:, lex.neg_mask].sum(axis=1)).ravel().astype(np.float64)
    pos_unique = np.asarray((X_counts[:, lex.pos_mask] > 0).sum(axis=1)).ravel().astype(np.float64)
    neg_unique = np.asarray((X_counts[:, lex.neg_mask] > 0).sum(axis=1)).ravel().astype(np.float64)

    pos_w = np.clip(weights, 0, None)
    neg_w_abs = np.clip(-weights, 0, None)
    pos_weight_sum = np.asarray(X_counts @ pos_w).ravel().astype(np.float64)
    neg_weight_sum = np.asarray(X_counts @ neg_w_abs).ravel().astype(np.float64)

    agg = np.column_stack(
        [
            total_lex,
            pos_count,
            neg_count,
            pos_unique,
            neg_unique,
            pos_weight_sum,
            neg_weight_sum,
            np.abs(pos_weight_sum - neg_weight_sum),
            _safe_div(pos_count, total_lex),
            _safe_div(neg_count, total_lex),
            (total_lex > 0).astype(np.float64),
        ]
    )
    agg_names = [
        "lex_total_count",
        "lex_pos_count",
        "lex_neg_count",
        "lex_pos_unique",
        "lex_neg_unique",
        "lex_pos_weight_sum",
        "lex_neg_weight_sum",
        "lex_balance_abs",
        "lex_pos_ratio",
        "lex_neg_ratio",
        "lex_has_any",
    ]
    return csr_matrix(agg, dtype=np.float64), agg_names


def build_feature_matrices(
    train_texts: pd.Series,
    test_texts: pd.Series,
    lex: LexiconData,
    min_df: int,
    max_features: int,
    ngram_max: int,
) -> tuple[dict[str, tuple[csr_matrix, csr_matrix]], dict[str, int]]:
    bow_vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
    )
    X_train_bow = bow_vectorizer.fit_transform(train_texts)
    X_test_bow = bow_vectorizer.transform(test_texts)

    X_train_lex, lex_feat_names = make_lexicon_feature_matrix(train_texts, lex)
    X_test_lex, _ = make_lexicon_feature_matrix(test_texts, lex)

    X_train_combined = hstack([X_train_bow, X_train_lex], format="csr")
    X_test_combined = hstack([X_test_bow, X_test_lex], format="csr")

    feature_sets = {
        "bow": (X_train_bow, X_test_bow),
        "lexicon": (X_train_lex, X_test_lex),
        "bow+lexicon": (X_train_combined, X_test_combined),
    }
    metadata = {
        "bow_vocab_size": len(bow_vectorizer.get_feature_names_out()),
        "lexicon_token_count": len(lex.tokens),
        "lexicon_stats_feature_count": len(lex_feat_names),
        "combined_feature_count": int(X_train_combined.shape[1]),
    }
    return feature_sets, metadata
