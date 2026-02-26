from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer

from .project2_task3_sentiment_data import load_sentiment_dataset
from .project2_task3_sentiment_features import (
    LexiconData,
    build_sentiment_lexicon,
    make_lexicon_feature_matrix,
)
from .project2_task3_sentiment_models import get_models
from .project2_task3_sentiment_preprocess import legacy_tokenize


@dataclass
class SentimentInferenceBundle:
    model_name: str
    feature_set: str
    label_mode: str
    labels_order: list[str]
    classifier: Any
    bow_vectorizer: CountVectorizer | None
    lexicon: LexiconData | None
    feature_config: dict[str, Any]
    dataset_config: dict[str, Any]
    training_summary: dict[str, Any]


def _fit_bow_vectorizer(
    train_texts: pd.Series,
    min_df: int,
    max_features: int,
    ngram_max: int,
) -> tuple[CountVectorizer, Any]:
    vec = CountVectorizer(
        lowercase=False,
        tokenizer=legacy_tokenize,
        token_pattern=None,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_features=max_features,
    )
    return vec, vec.fit_transform(train_texts)


def _build_feature_matrix_for_training(
    train_texts: pd.Series,
    labels: pd.Series,
    feature_cfg: dict[str, Any],
    feature_set: str,
) -> tuple[Any, CountVectorizer | None, LexiconData | None]:
    bow_vec = None
    X_bow = None
    if feature_set in {"bow", "bow+lexicon"}:
        bow_vec, X_bow = _fit_bow_vectorizer(
            train_texts=train_texts,
            min_df=int(feature_cfg["bow"]["min_df"]),
            max_features=int(feature_cfg["bow"]["max_features"]),
            ngram_max=int(feature_cfg["bow"]["ngram_max"]),
        )

    lex = None
    X_lex = None
    if feature_set in {"lexicon", "bow+lexicon"}:
        lex = build_sentiment_lexicon(
            train_texts,
            labels,
            top_k_each=int(feature_cfg["lexicon"]["top_k_each"]),
            min_count=int(feature_cfg["lexicon"]["min_count"]),
        )
        X_lex, _ = make_lexicon_feature_matrix(train_texts, lex)

    if feature_set == "bow":
        return X_bow, bow_vec, None
    if feature_set == "lexicon":
        return X_lex, None, lex
    if feature_set == "bow+lexicon":
        return hstack([X_bow, X_lex], format="csr"), bow_vec, lex
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def _transform_texts_for_bundle(texts: list[str], bundle: SentimentInferenceBundle):
    s = pd.Series([str(t) for t in texts], dtype="object")
    X_bow = None
    X_lex = None
    if bundle.feature_set in {"bow", "bow+lexicon"}:
        if bundle.bow_vectorizer is None:
            raise ValueError("Bundle is missing bow_vectorizer for this feature set.")
        X_bow = bundle.bow_vectorizer.transform(s)
    if bundle.feature_set in {"lexicon", "bow+lexicon"}:
        if bundle.lexicon is None:
            raise ValueError("Bundle is missing lexicon for this feature set.")
        X_lex, _ = make_lexicon_feature_matrix(s, bundle.lexicon)

    if bundle.feature_set == "bow":
        return X_bow
    if bundle.feature_set == "lexicon":
        return X_lex
    return hstack([X_bow, X_lex], format="csr")


def _fit_classifier(model_name: str, X_train, y_train):
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {sorted(models)}")
    model = models[model_name]
    if model_name == "bernoulli_nb":
        binarizer = Binarizer(copy=True)
        X_fit = binarizer.fit_transform(X_train)
        model.fit(X_fit, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def train_bundle_from_summary(summary_path: str, out_path: str | None = None) -> SentimentInferenceBundle:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    dataset_cfg = summary["dataset"]
    feature_cfg = summary["features"]
    best = summary["best_overall"]

    model_name = str(best["model"])
    feature_set = str(best["feature_set"])

    ds = load_sentiment_dataset(
        train_path=str(dataset_cfg["train_path"]),
        test_path=str(dataset_cfg["test_path"]),
        text_col=str(dataset_cfg["text_col"]),
        score_col=str(dataset_cfg["score_col"]),
        label_mode=str(dataset_cfg["label_mode"]),
    )

    train_df = ds.train_df
    X_train, bow_vec, lex = _build_feature_matrix_for_training(
        train_texts=train_df[str(dataset_cfg["text_col"])],
        labels=train_df["label"],
        feature_cfg=feature_cfg,
        feature_set=feature_set,
    )
    y_train = train_df["label"].astype(str).to_numpy()
    clf = _fit_classifier(model_name, X_train, y_train)

    bundle = SentimentInferenceBundle(
        model_name=model_name,
        feature_set=feature_set,
        label_mode=str(dataset_cfg["label_mode"]),
        labels_order=ds.labels_order,
        classifier=clf,
        bow_vectorizer=bow_vec,
        lexicon=lex,
        feature_config={
            "bow": {
                "min_df": int(feature_cfg["bow"]["min_df"]),
                "max_features": int(feature_cfg["bow"]["max_features"]),
                "ngram_max": int(feature_cfg["bow"]["ngram_max"]),
            },
            "lexicon": {
                "top_k_each": int(feature_cfg["lexicon"]["top_k_each"]),
                "min_count": int(feature_cfg["lexicon"]["min_count"]),
            },
        },
        dataset_config={
            "train_path": str(dataset_cfg["train_path"]),
            "test_path": str(dataset_cfg["test_path"]),
            "text_col": str(dataset_cfg["text_col"]),
            "score_col": str(dataset_cfg["score_col"]),
        },
        training_summary={
            "task": summary.get("task"),
            "best_overall": best,
            "n_train": int(best.get("n_train", len(train_df))),
        },
    )

    if out_path:
        save_bundle(bundle, out_path)
    return bundle


def save_bundle(bundle: SentimentInferenceBundle, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(bundle, f)


def load_bundle(bundle_path: str) -> SentimentInferenceBundle:
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Bundles created via `python -m ...` may serialize this dataclass under __main__.
            if module == "__main__" and name == "SentimentInferenceBundle":
                return SentimentInferenceBundle
            return super().find_class(module, name)

    with Path(bundle_path).open("rb") as f:
        obj = _CompatUnpickler(f).load()
    if not isinstance(obj, SentimentInferenceBundle):
        raise TypeError("Loaded object is not a SentimentInferenceBundle.")
    return obj


def predict_text(bundle: SentimentInferenceBundle, text: str) -> dict[str, Any]:
    text = "" if text is None else str(text)
    X = _transform_texts_for_bundle([text], bundle)
    X_model = Binarizer(copy=True).fit_transform(X) if bundle.model_name == "bernoulli_nb" else X

    pred = str(bundle.classifier.predict(X_model)[0])
    probs_sorted: list[dict[str, Any]] = []
    if hasattr(bundle.classifier, "predict_proba"):
        proba = bundle.classifier.predict_proba(X_model)[0]
        classes = [str(c) for c in bundle.classifier.classes_.tolist()]
        probs_sorted = sorted(
            [{"label": lab, "prob": float(p)} for lab, p in zip(classes, proba)],
            key=lambda x: x["prob"],
            reverse=True,
        )

    tokens = legacy_tokenize(text)
    token_counts = Counter(tokens)
    lex_hits: list[dict[str, Any]] = []
    pos_sum = 0.0
    neg_sum = 0.0
    if bundle.lexicon is not None:
        for tok, cnt in token_counts.items():
            w = bundle.lexicon.token_to_weight.get(tok)
            if w is None:
                continue
            contrib = float(w) * int(cnt)
            if contrib > 0:
                pos_sum += contrib
            elif contrib < 0:
                neg_sum += -contrib
            lex_hits.append(
                {
                    "token": tok,
                    "count": int(cnt),
                    "weight": float(w),
                    "contribution": contrib,
                }
            )
        lex_hits.sort(key=lambda x: abs(float(x["contribution"])), reverse=True)

    return {
        "input_text": text,
        "predicted_label": pred,
        "probabilities": probs_sorted,
        "bundle_info": {
            "model_name": bundle.model_name,
            "feature_set": bundle.feature_set,
            "label_mode": bundle.label_mode,
            "labels_order": bundle.labels_order,
        },
        "debug": {
            "tokens": tokens[:80],
            "token_count": len(tokens),
            "lexicon_hits": lex_hits[:20],
            "lexicon_positive_contrib_sum": pos_sum,
            "lexicon_negative_contrib_sum": neg_sum,
        },
    }


def bundle_metadata(bundle: SentimentInferenceBundle) -> dict[str, Any]:
    return {
        "model_name": bundle.model_name,
        "feature_set": bundle.feature_set,
        "label_mode": bundle.label_mode,
        "labels_order": bundle.labels_order,
        "dataset_config": bundle.dataset_config,
        "feature_config": bundle.feature_config,
        "training_summary": bundle.training_summary,
        "has_bow_vectorizer": bundle.bow_vectorizer is not None,
        "has_lexicon": bundle.lexicon is not None,
        "lexicon_size": 0 if bundle.lexicon is None else len(bundle.lexicon.tokens),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/load a Task 3 sentiment inference bundle.")
    ap.add_argument(
        "--summary",
        default="outputs/project2/task3_sentiment/summary.json",
        help="Task 3 summary.json produced by training/evaluation script.",
    )
    ap.add_argument(
        "--out",
        default="outputs/project2/task3_sentiment/best_model_bundle.pkl",
        help="Where to write the serialized inference bundle (.pkl).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and train in-memory but do not write a bundle.",
    )
    args = ap.parse_args()

    bundle = train_bundle_from_summary(args.summary, None if args.dry_run else args.out)
    meta = bundle_metadata(bundle)
    print("Built Task 3 sentiment inference bundle")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if not args.dry_run:
        print(f"Saved bundle to {args.out}")


if __name__ == "__main__":
    main()
