from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from .project2_task3_sentiment_data import load_sentiment_dataset
from .project2_task3_sentiment_features import build_feature_matrices, build_sentiment_lexicon
from .project2_task3_sentiment_models import fit_predict, get_models
from .project2_task3_sentiment_stats import compute_metrics, confusion_as_list, holm_bonferroni, mcnemar_exact


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Project 2 Task 3: compare MultinomialNB, BernoulliNB, and Logistic Regression "
            "using Bag-of-Words and sentiment lexicon features on Azerbaijani review sentiment data."
        )
    )
    p.add_argument("--train", default="data/external/train.csv", help="Path to training CSV")
    p.add_argument("--test", default="data/external/test.csv", help="Path to test CSV")
    p.add_argument("--text-col", default="content", help="Text column name in CSV files")
    p.add_argument("--score-col", default="score", help="Score column name")
    p.add_argument(
        "--label-mode",
        choices=["sentiment3", "score5", "binary"],
        default="sentiment3",
        help=(
            "sentiment3: 1-2=negative, 3=neutral, 4-5=positive; "
            "score5: keep 1..5 labels; "
            "binary: drop score=3 and use 1-2 vs 4-5."
        ),
    )
    p.add_argument("--min-df", type=int, default=2, help="BoW min_df")
    p.add_argument("--max-features", type=int, default=60000, help="BoW max_features")
    p.add_argument("--ngram-max", type=int, default=2, choices=[1, 2, 3], help="BoW max n-gram")
    p.add_argument("--lexicon-top-k", type=int, default=500, help="Top-k positive and top-k negative lexicon tokens")
    p.add_argument("--lexicon-min-count", type=int, default=5, help="Minimum token count to enter lexicon")
    p.add_argument(
        "--output-dir",
        default="outputs/project2/task3_sentiment",
        help="Directory to save metrics/reports/predictions",
    )
    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_sentiment_dataset(
        train_path=args.train,
        test_path=args.test,
        text_col=args.text_col,
        score_col=args.score_col,
        label_mode=args.label_mode,
    )
    train_df = ds.train_df
    test_df = ds.test_df
    y_train = train_df["label"].astype(str).to_numpy()
    y_test = test_df["label"].astype(str).to_numpy()
    labels_order = ds.labels_order

    lexicon = build_sentiment_lexicon(
        train_df[args.text_col],
        train_df["label"],
        top_k_each=args.lexicon_top_k,
        min_count=args.lexicon_min_count,
    )
    feature_sets, feat_meta = build_feature_matrices(
        train_df[args.text_col],
        test_df[args.text_col],
        lex=lexicon,
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
    )

    results_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    reports: dict[str, str] = {}
    confusions: dict[str, list[list[int]]] = {}

    for feat_name, (X_train, X_test) in feature_sets.items():
        for model_name, model in get_models().items():
            key = f"{model_name}__{feat_name}"
            y_pred = fit_predict(model_name, model, X_train, y_train, X_test)
            predictions[key] = y_pred

            metrics = compute_metrics(y_test, y_pred, labels_order)
            metrics.update(
                {
                    "model": model_name,
                    "feature_set": feat_name,
                    "experiment_key": key,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                }
            )
            results_rows.append(metrics)
            reports[key] = classification_report(y_test, y_pred, digits=4, zero_division=0)
            confusions[key] = confusion_as_list(y_test, y_pred, labels_order)

    results_df = (
        pd.DataFrame(results_rows)
        .sort_values(by=["f1_macro", "accuracy", "f1_weighted"], ascending=False)
        .reset_index(drop=True)
    )

    # Pairwise McNemar tests within each feature representation.
    sig_rows: list[dict] = []
    for feat_name in feature_sets.keys():
        keys = [k for k in predictions if k.endswith(f"__{feat_name}")]
        local_rows: list[dict] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = keys[i]
                b = keys[j]
                local_rows.append(
                    {
                        "feature_set": feat_name,
                        "model_a": a.split("__")[0],
                        "model_b": b.split("__")[0],
                        **mcnemar_exact(y_test, predictions[a], predictions[b]),
                    }
                )
        if local_rows:
            indexed = list(enumerate([r["mcnemar_exact_pvalue"] for r in local_rows]))
            reject_flags = holm_bonferroni(indexed, alpha=0.05)
            for idx, r in enumerate(local_rows):
                r["holm_reject_0_05"] = bool(reject_flags[idx])
                sig_rows.append(r)
    sig_df = pd.DataFrame(sig_rows)

    # Secondary view: compare top model vs every other experiment.
    top_key = str(results_df.iloc[0]["experiment_key"])
    top_vs_rows: list[dict] = []
    pairs: list[tuple[int, float]] = []
    other_keys = [k for k in predictions.keys() if k != top_key]
    for idx, other in enumerate(other_keys):
        row = {"top_model": top_key, "other_model": other, **mcnemar_exact(y_test, predictions[top_key], predictions[other])}
        top_vs_rows.append(row)
        pairs.append((idx, float(row["mcnemar_exact_pvalue"])))
    if top_vs_rows:
        reject = holm_bonferroni(pairs, alpha=0.05)
        for i, row in enumerate(top_vs_rows):
            row["holm_reject_0_05"] = bool(reject[i])
    top_vs_df = pd.DataFrame(top_vs_rows)

    best_by_classifier = (
        results_df.sort_values(["f1_macro", "accuracy"], ascending=False)
        .groupby("model", as_index=False)
        .first()[["model", "feature_set", "accuracy", "f1_macro", "f1_weighted"]]
        .sort_values(["f1_macro", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )
    best_by_feature = (
        results_df.sort_values(["f1_macro", "accuracy"], ascending=False)
        .groupby("feature_set", as_index=False)
        .first()[["feature_set", "model", "accuracy", "f1_macro", "f1_weighted"]]
        .sort_values(["f1_macro", "accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(out_dir / "metrics.csv", index=False)
    if not sig_df.empty:
        sig_df.to_csv(out_dir / "significance_within_feature_set.csv", index=False)
    if not top_vs_df.empty:
        top_vs_df.to_csv(out_dir / "significance_top_vs_others.csv", index=False)

    (out_dir / "classification_reports.txt").write_text(
        "\n\n".join(f"## {k}\n{reports[k]}" for k in sorted(reports)),
        encoding="utf-8",
    )
    _write_json(
        out_dir / "confusion_matrices.json",
        {"labels_order": labels_order, "matrices": confusions},
    )
    _write_json(
        out_dir / "lexicon_preview.json",
        {
            "size": len(lexicon.tokens),
            "top_positive": sorted(
                [(t, w) for t, w in lexicon.token_to_weight.items() if w > 0],
                key=lambda x: -x[1],
            )[:40],
            "top_negative": sorted(
                [(t, w) for t, w in lexicon.token_to_weight.items() if w < 0],
                key=lambda x: x[1],
            )[:40],
            "feature_metadata": feat_meta,
        },
    )
    pd.DataFrame(
        {
            "text": test_df[args.text_col],
            "true_label": y_test,
            **{k: v for k, v in predictions.items()},
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)

    summary = {
        "task": "Project 2 Task 3 - Sentiment classification comparison",
        "dataset": {
            "train_path": args.train,
            "test_path": args.test,
            "text_col": args.text_col,
            "score_col": args.score_col,
            "label_mode": args.label_mode,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_label_distribution": train_df["label"].value_counts().to_dict(),
            "test_label_distribution": test_df["label"].value_counts().to_dict(),
        },
        "features": {
            "bow": {
                "min_df": args.min_df,
                "max_features": args.max_features,
                "ngram_max": args.ngram_max,
            },
            "lexicon": {
                "top_k_each": args.lexicon_top_k,
                "min_count": args.lexicon_min_count,
                "induced_lexicon_size": len(lexicon.tokens),
            },
            **feat_meta,
        },
        "best_overall": results_df.iloc[0].to_dict(),
        "best_by_classifier": best_by_classifier.to_dict(orient="records"),
        "best_by_feature_set": best_by_feature.to_dict(orient="records"),
        "output_dir": str(out_dir),
    }
    _write_json(out_dir / "summary.json", summary)

    print("Dataset summary")
    print(f"  train={len(train_df)} test={len(test_df)} label_mode={args.label_mode}")
    print("  train label distribution:")
    print(train_df["label"].value_counts())
    print("  test label distribution:")
    print(test_df["label"].value_counts())

    print("\nFeature metadata")
    for k, v in feat_meta.items():
        print(f"  {k}: {v}")

    print("\nTop experiments (sorted by macro-F1)")
    cols = ["experiment_key", "accuracy", "f1_macro", "f1_weighted"]
    print(results_df[cols].head(10).to_string(index=False))

    print("\nBest feature set per classifier")
    print(best_by_classifier.to_string(index=False))

    print("\nBest classifier per feature set")
    print(best_by_feature.to_string(index=False))

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
