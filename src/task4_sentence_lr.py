from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .load_data import load_corpus_csv
from .task4_dot_data import group_rows_by_doc, load_labeled_dot_rows, split_labeled_rows_by_doc
from .task4_dot_features import WINDOW_SIZE, CHAR_WINDOW_RADIUS, extract_features_from_row, rule_guess_eos_for_dot
from .task4_dot_model import (
    compute_dot_metrics,
    fit_dot_model,
    predict_dot_labels,
    save_feature_config,
    save_model_artifact,
    tune_lr_models,
)
from .task4_sentence_utils import split_on_boundaries


def _load_corpus_text_lookup(corpus_path: str, text_column: str = "text") -> dict[int, str]:
    df = load_corpus_csv(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")
    texts = df[text_column].fillna("").astype(str).tolist()
    return {i: t for i, t in enumerate(texts)}


def validate_rows_against_corpus(rows: list[dict[str, Any]], text_lookup: dict[int, str]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        doc_idx = int(row["doc_idx"])
        char_index = int(row["char_index"])
        if doc_idx not in text_lookup:
            raise ValueError(f"doc_idx={doc_idx} from labels CSV not found in corpus")
        text = text_lookup[doc_idx]
        if not (0 <= char_index < len(text)):
            raise ValueError(f"row_id={row['row_id']}: char_index out of range for doc_idx={doc_idx}")
        if text[char_index] != ".":
            raise ValueError(
                f"row_id={row['row_id']}: expected '.' at char_index={char_index}, found {text[char_index]!r}"
            )
        rg = row.get("rule_guess", None)
        if rg is None or str(rg) == "":
            row["rule_guess"] = rule_guess_eos_for_dot(text, char_index)
    return warnings


def build_feature_label_matrices(
    rows: list[dict[str, Any]],
    text_lookup: dict[int, str],
) -> tuple[list[dict[str, Any]], list[int]]:
    X: list[dict[str, Any]] = []
    y: list[int] = []
    for row in rows:
        text = text_lookup[int(row["doc_idx"])]
        X.append(extract_features_from_row(row, text))
        y.append(int(row["gold_label"]))
    return X, y


def _gold_dot_eos_set(rows_for_doc: list[dict[str, Any]]) -> set[int]:
    return {int(r["char_index"]) for r in rows_for_doc if int(r["gold_label"]) == 1}


def _pred_dot_eos_set(rows_for_doc: list[dict[str, Any]], preds_by_row_id: dict[str, int]) -> set[int]:
    return {int(r["char_index"]) for r in rows_for_doc if int(preds_by_row_id[r["row_id"]]) == 1}


def _rule_dot_eos_set(rows_for_doc: list[dict[str, Any]]) -> set[int]:
    return {int(r["char_index"]) for r in rows_for_doc if int(r.get("rule_guess", 0)) == 1}


def _boundary_metrics_from_sets(pred_sets: list[set[int]], gold_sets: list[set[int]]) -> dict[str, Any]:
    tp = fp = fn = 0
    pred_total = gold_total = 0
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        pred_total += len(pred)
        gold_total += len(gold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    bder = (fp + fn) / gold_total if gold_total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bder": bder,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_predicted_boundaries": pred_total,
        "num_gold_boundaries": gold_total,
    }


def sentence_metrics_for_rows(
    rows: list[dict[str, Any]],
    text_lookup: dict[int, str],
    preds_by_row_id: dict[str, int],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    grouped = group_rows_by_doc(rows)
    pred_sets: list[set[int]] = []
    gold_sets: list[set[int]] = []
    per_doc: dict[int, dict[str, Any]] = {}
    for doc_idx, doc_rows in grouped.items():
        text = text_lookup[doc_idx]
        gold_dot = _gold_dot_eos_set(doc_rows)
        pred_dot = _pred_dot_eos_set(doc_rows, preds_by_row_id)
        gold_sents, gold_bounds = split_on_boundaries(text, gold_dot)
        pred_sents, pred_bounds = split_on_boundaries(text, pred_dot)
        gold_set = set(gold_bounds)
        pred_set = set(pred_bounds)
        gold_sets.append(gold_set)
        pred_sets.append(pred_set)
        per_doc[doc_idx] = {
            "gold_sentence_count": len(gold_sents),
            "pred_sentence_count": len(pred_sents),
            "gold_boundaries": gold_bounds,
            "pred_boundaries": pred_bounds,
        }
    return _boundary_metrics_from_sets(pred_sets, gold_sets), per_doc


def rule_baseline_metrics_for_rows(
    rows: list[dict[str, Any]],
    text_lookup: dict[int, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_true = [int(r["gold_label"]) for r in rows]
    y_pred = [int(r.get("rule_guess", 0)) for r in rows]
    dot_metrics = compute_dot_metrics(y_true, y_pred).to_dict()

    grouped = group_rows_by_doc(rows)
    pred_sets: list[set[int]] = []
    gold_sets: list[set[int]] = []
    total_gold_sent = 0
    total_pred_sent = 0
    for doc_idx, doc_rows in grouped.items():
        text = text_lookup[doc_idx]
        gold_dot = _gold_dot_eos_set(doc_rows)
        rule_dot = _rule_dot_eos_set(doc_rows)
        gold_sents, gold_bounds = split_on_boundaries(text, gold_dot)
        pred_sents, pred_bounds = split_on_boundaries(text, rule_dot)
        total_gold_sent += len(gold_sents)
        total_pred_sent += len(pred_sents)
        gold_sets.append(set(gold_bounds))
        pred_sets.append(set(pred_bounds))
    sent_metrics = _boundary_metrics_from_sets(pred_sets, gold_sets)
    sent_metrics["num_gold_sentences"] = total_gold_sent
    sent_metrics["num_pred_sentences"] = total_pred_sent
    return dot_metrics, sent_metrics


def _rows_to_preds_map(rows: list[dict[str, Any]], preds: np.ndarray) -> dict[str, int]:
    return {str(r["row_id"]): int(p) for r, p in zip(rows, preds.tolist())}


def _probs_map(rows: list[dict[str, Any]], probs: np.ndarray) -> dict[str, float]:
    return {str(r["row_id"]): float(p) for r, p in zip(rows, probs.tolist())}


def _count_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    y = [int(r["gold_label"]) for r in rows]
    return {
        "num_rows": len(rows),
        "num_pos_eos": int(sum(y)),
        "num_neg_not_eos": int(len(rows) - sum(y)),
    }


def _unique_docs(rows: list[dict[str, Any]]) -> int:
    return len({int(r["doc_idx"]) for r in rows})


def _validate_split_classes(split_name: str, rows: list[dict[str, Any]]) -> None:
    labels = {int(r["gold_label"]) for r in rows}
    if labels != {0, 1}:
        raise ValueError(
            f"{split_name} split must contain both classes 0 and 1 for logistic regression. Present: {sorted(labels)}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _train_eval_final_model(
    regularization: str,
    selected_c: float,
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    text_lookup: dict[int, str],
) -> dict[str, Any]:
    train_dev_rows = list(train_rows) + list(dev_rows)
    X_train_dev, y_train_dev = build_feature_label_matrices(train_dev_rows, text_lookup)
    X_test, y_test = build_feature_label_matrices(test_rows, text_lookup)

    model = fit_dot_model(
        X_train=X_train_dev,
        y_train=y_train_dev,
        regularization=regularization,
        c_value=selected_c,
        class_weight="balanced",
    )
    test_preds, test_probs = predict_dot_labels(model, X_test)
    dot_metrics = compute_dot_metrics(y_test, test_preds).to_dict()
    preds_map = _rows_to_preds_map(test_rows, test_preds)
    probs_map = _probs_map(test_rows, test_probs)
    sent_metrics, per_doc = sentence_metrics_for_rows(test_rows, text_lookup, preds_map)

    total_gold_sent = total_pred_sent = 0
    for doc_idx, doc_rows in group_rows_by_doc(test_rows).items():
        text = text_lookup[doc_idx]
        gold_sents, _ = split_on_boundaries(text, _gold_dot_eos_set(doc_rows))
        pred_sents, _ = split_on_boundaries(text, _pred_dot_eos_set(doc_rows, preds_map))
        total_gold_sent += len(gold_sents)
        total_pred_sent += len(pred_sents)
    sent_metrics["num_gold_sentences"] = total_gold_sent
    sent_metrics["num_pred_sentences"] = total_pred_sent

    return {
        "model": model,
        "test_preds": test_preds,
        "test_probs": test_probs,
        "dot_metrics": dot_metrics,
        "sentence_metrics": sent_metrics,
        "per_doc_sentence_metrics": per_doc,
        "preds_map": preds_map,
        "probs_map": probs_map,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 4: Logistic regression for dot sentence-boundary detection (L1 vs L2).")
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--corpus_path", required=True)
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--c_grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--primary_metric", choices=["sent_f1", "dot_f1"], default="sent_f1")
    ap.add_argument("--compare_rule_baseline", action="store_true")
    ap.add_argument("--out_dir", default="outputs/project2/task4_lr")
    ap.add_argument("--save_models", action="store_true", default=True)
    ap.add_argument("--no-save-models", dest="save_models", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading labeled dot rows...")
    rows = load_labeled_dot_rows(args.labels_csv, max_rows=args.max_rows)
    text_lookup = _load_corpus_text_lookup(args.corpus_path, text_column=args.text_column)
    validation_warnings = validate_rows_against_corpus(rows, text_lookup)

    split = split_labeled_rows_by_doc(rows, test_ratio=args.test_ratio, dev_ratio=args.dev_ratio, seed=args.seed)
    for name, subset in [("train", split.train), ("dev", split.dev), ("test", split.test)]:
        _validate_split_classes(name, subset)

    print(
        f"Split mode={split.split_mode}: train={len(split.train)} rows ({_unique_docs(split.train)} docs), "
        f"dev={len(split.dev)} rows ({_unique_docs(split.dev)} docs), "
        f"test={len(split.test)} rows ({_unique_docs(split.test)} docs)"
    )

    X_train, y_train = build_feature_label_matrices(split.train, text_lookup)
    X_dev, y_dev = build_feature_label_matrices(split.dev, text_lookup)

    def scorer_fn(reg: str, model, X_dev_local, y_dev_local):
        dev_preds, dev_probs = predict_dot_labels(model, X_dev_local)
        dot_m = compute_dot_metrics(y_dev_local, dev_preds)
        preds_map = _rows_to_preds_map(split.dev, dev_preds)
        sent_m, _ = sentence_metrics_for_rows(split.dev, text_lookup, preds_map)
        score = sent_m["f1"] if args.primary_metric == "sent_f1" else dot_m.f1
        return {
            "score": float(score),
            "dot_f1": float(dot_m.f1),
            "dev_dot_precision": float(dot_m.precision),
            "dev_dot_recall": float(dot_m.recall),
            "dev_dot_f1": float(dot_m.f1),
            "dev_sent_precision": float(sent_m["precision"]),
            "dev_sent_recall": float(sent_m["recall"]),
            "dev_sent_f1": float(sent_m["f1"]),
            "dev_bder": float(sent_m["bder"]),
            "dev_num_pred_boundaries": int(sent_m["num_predicted_boundaries"]),
            "dev_num_gold_boundaries": int(sent_m["num_gold_boundaries"]),
            "dev_num_zero_probs_placeholder": 0,
            "dev_mean_prob": float(np.mean(dev_probs)) if len(dev_probs) else 0.0,
        }

    print("Tuning L1 and L2 logistic regression on dev split...")
    selected_models_train, tuning_rows = tune_lr_models(
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        c_grid=args.c_grid,
        scorer_fn=scorer_fn,
        class_weight="balanced",
    )

    selected_c = {reg: selected_models_train[reg].c_value for reg in ("l1", "l2")}

    print("Retraining selected L1/L2 on train+dev and evaluating on test...")
    final_l1 = _train_eval_final_model("l1", selected_c["l1"], split.train, split.dev, split.test, text_lookup)
    final_l2 = _train_eval_final_model("l2", selected_c["l2"], split.train, split.dev, split.test, text_lookup)

    y_test = [int(r["gold_label"]) for r in split.test]
    rule_dot_metrics = None
    rule_sent_metrics = None
    if args.compare_rule_baseline:
        rule_dot_metrics, rule_sent_metrics = rule_baseline_metrics_for_rows(split.test, text_lookup)

    # Winner selection
    l1_sent_f1 = float(final_l1["sentence_metrics"]["f1"])
    l2_sent_f1 = float(final_l2["sentence_metrics"]["f1"])
    if l1_sent_f1 > l2_sent_f1:
        winner = "lr_l1"
        reason = "Higher test sentence segmentation F1."
    elif l2_sent_f1 > l1_sent_f1:
        winner = "lr_l2"
        reason = "Higher test sentence segmentation F1."
    else:
        l1_dot_f1 = float(final_l1["dot_metrics"]["f1"])
        l2_dot_f1 = float(final_l2["dot_metrics"]["f1"])
        if l1_dot_f1 > l2_dot_f1:
            winner = "lr_l1"
            reason = "Sentence F1 tie; higher dot-boundary F1."
        elif l2_dot_f1 > l1_dot_f1:
            winner = "lr_l2"
            reason = "Sentence F1 tie; higher dot-boundary F1."
        else:
            l1_fp = int(final_l1["dot_metrics"]["fp"])
            l2_fp = int(final_l2["dot_metrics"]["fp"])
            if l1_fp <= l2_fp:
                winner = "lr_l1"
                reason = "Sentence and dot F1 tie; fewer/equal false-positive EOS predictions."
            else:
                winner = "lr_l2"
                reason = "Sentence and dot F1 tie; fewer false-positive EOS predictions."

    # predictions_test.csv
    preds_test_rows: list[dict[str, Any]] = []
    l1_preds_map = final_l1["preds_map"]
    l1_probs_map = final_l1["probs_map"]
    l2_preds_map = final_l2["preds_map"]
    l2_probs_map = final_l2["probs_map"]
    for row in split.test:
        rid = str(row["row_id"])
        preds_test_rows.append(
            {
                "row_id": rid,
                "doc_idx": int(row["doc_idx"]),
                "doc_id": row.get("doc_id", ""),
                "char_index": int(row["char_index"]),
                "gold_label": int(row["gold_label"]),
                "rule_guess": int(row.get("rule_guess", 0)),
                "pred_l1": int(l1_preds_map[rid]),
                "prob_l1": float(l1_probs_map[rid]),
                "pred_l2": int(l2_preds_map[rid]),
                "prob_l2": float(l2_probs_map[rid]),
                "window_text": row.get("window_text", ""),
                "prev_token": row.get("prev_token", ""),
                "next_token": row.get("next_token", ""),
            }
        )
    _write_csv(
        out_dir / "predictions_test.csv",
        preds_test_rows,
        [
            "row_id",
            "doc_idx",
            "doc_id",
            "char_index",
            "gold_label",
            "rule_guess",
            "pred_l1",
            "prob_l1",
            "pred_l2",
            "prob_l2",
            "window_text",
            "prev_token",
            "next_token",
        ],
    )

    comparison_rows = []
    if args.compare_rule_baseline and rule_dot_metrics and rule_sent_metrics:
        comparison_rows.append(
            {
                "model": "rule_based",
                "dot_precision": rule_dot_metrics["precision"],
                "dot_recall": rule_dot_metrics["recall"],
                "dot_f1": rule_dot_metrics["f1"],
                "sent_precision": rule_sent_metrics["precision"],
                "sent_recall": rule_sent_metrics["recall"],
                "sent_f1": rule_sent_metrics["f1"],
                "bder": rule_sent_metrics["bder"],
                "test_docs": _unique_docs(split.test),
                "notes": "Rule-derived dot baseline from existing sentence segmentation heuristics",
            }
        )
    for model_name, final in [("lr_l1", final_l1), ("lr_l2", final_l2)]:
        comparison_rows.append(
            {
                "model": model_name,
                "dot_precision": final["dot_metrics"]["precision"],
                "dot_recall": final["dot_metrics"]["recall"],
                "dot_f1": final["dot_metrics"]["f1"],
                "sent_precision": final["sentence_metrics"]["precision"],
                "sent_recall": final["sentence_metrics"]["recall"],
                "sent_f1": final["sentence_metrics"]["f1"],
                "bder": final["sentence_metrics"]["bder"],
                "test_docs": _unique_docs(split.test),
                "notes": f"Logistic regression ({model_name[-2:].upper()})",
            }
        )
    _write_csv(
        out_dir / "comparison.csv",
        comparison_rows,
        [
            "model",
            "dot_precision",
            "dot_recall",
            "dot_f1",
            "sent_precision",
            "sent_recall",
            "sent_f1",
            "bder",
            "test_docs",
            "notes",
        ],
    )

    tuning_csv_rows = []
    for r in tuning_rows:
        tuning_csv_rows.append(
            {
                "regularization": r["regularization"],
                "C": r["C"],
                "dev_dot_f1": r.get("dev_dot_f1"),
                "dev_sent_f1": r.get("dev_sent_f1"),
                "dev_bder": r.get("dev_bder"),
                "selected": r.get("selected", False),
            }
        )
    _write_csv(
        out_dir / "tuning_results.csv",
        tuning_csv_rows,
        ["regularization", "C", "dev_dot_f1", "dev_sent_f1", "dev_bder", "selected"],
    )

    if args.save_models:
        save_model_artifact(final_l1["model"], out_dir / "model_l1.joblib")
        save_model_artifact(final_l2["model"], out_dir / "model_l2.joblib")
        save_feature_config(
            out_dir / "feature_config.json",
            {
                "char_window_radius": CHAR_WINDOW_RADIUS,
                "preview_window_size": WINDOW_SIZE,
                "primary_metric": args.primary_metric,
                "threshold": 0.5,
                "class_weight": "balanced",
                "solver": "liblinear",
            },
        )

    summary = {
        "task": "Project 2 Task 4 - Logistic regression for dot sentence boundary detection (L1 vs L2)",
        "config": {
            "labels_csv": args.labels_csv,
            "corpus_path": args.corpus_path,
            "text_column": args.text_column,
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "test_ratio": args.test_ratio,
            "c_grid": args.c_grid,
            "max_rows": args.max_rows,
            "primary_metric": args.primary_metric,
            "compare_rule_baseline": args.compare_rule_baseline,
            "out_dir": str(out_dir),
            "save_models": args.save_models,
        },
        "label_dataset": {
            "total_rows": len(rows),
            "total_docs": _unique_docs(rows),
            "class_counts": _count_labels(rows),
        },
        "splits": {
            "split_mode": split.split_mode,
            "warnings": split.warnings + validation_warnings,
            "train": {"docs": _unique_docs(split.train), **_count_labels(split.train)},
            "dev": {"docs": _unique_docs(split.dev), **_count_labels(split.dev)},
            "test": {"docs": _unique_docs(split.test), **_count_labels(split.test)},
        },
        "feature_config": {
            "style": "interpretable_engineered_features",
            "char_window_radius": CHAR_WINDOW_RADIUS,
            "preview_window_size": WINDOW_SIZE,
        },
        "tuning": {
            "primary_metric": args.primary_metric,
            "selected_c": {"l1": selected_c["l1"], "l2": selected_c["l2"]},
            "trials": tuning_csv_rows,
        },
        "test_results": {
            "lr_l1": {
                "dot_metrics": final_l1["dot_metrics"],
                "sentence_metrics": final_l1["sentence_metrics"],
                "c_value": selected_c["l1"],
                "warnings": final_l1["model"].warnings,
            },
            "lr_l2": {
                "dot_metrics": final_l2["dot_metrics"],
                "sentence_metrics": final_l2["sentence_metrics"],
                "c_value": selected_c["l2"],
                "warnings": final_l2["model"].warnings,
            },
            "rule_based": (
                {
                    "dot_metrics": rule_dot_metrics,
                    "sentence_metrics": rule_sent_metrics,
                }
                if args.compare_rule_baseline and rule_dot_metrics and rule_sent_metrics
                else None
            ),
        },
        "winner": {
            "model": winner,
            "primary_metric": args.primary_metric,
            "reason": reason,
        },
        "notes": [
            "Task 4 supervises only '.' (dot) EOS classification; '!' and '?' remain deterministic split cues.",
            "Sentence-level metrics are computed from reconstructed boundaries using predicted dot EOS + deterministic !/? splits.",
            "Rule baseline uses rule-derived dot guesses based on existing sentence segmentation heuristics.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved summary: {out_dir / 'summary.json'}")
    print(f"Saved comparison: {out_dir / 'comparison.csv'}")
    print(f"Saved tuning results: {out_dir / 'tuning_results.csv'}")
    print(f"Saved test predictions: {out_dir / 'predictions_test.csv'}")
    print(f"Winner: {winner} ({reason})")


if __name__ == "__main__":
    main()
