from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


@dataclass
class DotMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


@dataclass
class TrainedDotModel:
    regularization: str
    c_value: float
    pipeline: Pipeline
    warnings: list[str]


def make_lr_pipeline(regularization: str, c_value: float, class_weight: str | None = "balanced") -> Pipeline:
    if regularization not in {"l1", "l2"}:
        raise ValueError("regularization must be 'l1' or 'l2'")
    clf = LogisticRegression(
        penalty=regularization,
        C=float(c_value),
        solver="liblinear",
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("vec", DictVectorizer(sparse=True)),
            ("clf", clf),
        ]
    )


def fit_dot_model(
    X_train: list[dict[str, Any]],
    y_train: list[int],
    regularization: str,
    c_value: float,
    class_weight: str | None = "balanced",
) -> TrainedDotModel:
    pipe = make_lr_pipeline(regularization=regularization, c_value=c_value, class_weight=class_weight)
    warn_msgs: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X_train, y_train)
        for w in caught:
            warn_msgs.append(f"{w.category.__name__}: {w.message}")
    return TrainedDotModel(
        regularization=regularization,
        c_value=float(c_value),
        pipeline=pipe,
        warnings=warn_msgs,
    )


def predict_dot_labels(model: TrainedDotModel, X: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    probs = model.pipeline.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def compute_dot_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> DotMetrics:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    return DotMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
    )


def tune_lr_models(
    X_train: list[dict[str, Any]],
    y_train: list[int],
    X_dev: list[dict[str, Any]],
    y_dev: list[int],
    c_grid: list[float],
    scorer_fn,
    class_weight: str | None = "balanced",
) -> tuple[dict[str, TrainedDotModel], list[dict[str, Any]]]:
    """
    scorer_fn signature:
      scorer_fn(regularization: str, model: TrainedDotModel, X_dev, y_dev) -> dict with keys:
        score (float), dot_metrics (DotMetrics|dict), extra (dict)
    """
    selected: dict[str, TrainedDotModel] = {}
    tuning_rows: list[dict[str, Any]] = []

    for reg in ("l1", "l2"):
        best_model: TrainedDotModel | None = None
        best_score = float("-inf")
        best_dot_f1 = float("-inf")
        for c in c_grid:
            model = fit_dot_model(X_train, y_train, regularization=reg, c_value=c, class_weight=class_weight)
            scored = scorer_fn(reg, model, X_dev, y_dev)
            score = float(scored["score"])
            dot_f1 = float(scored.get("dot_f1", float("-inf")))
            row = {
                "regularization": reg,
                "C": float(c),
                "score": score,
                **{k: v for k, v in scored.items() if k != "score"},
                "selected": False,
            }
            tuning_rows.append(row)
            if (score > best_score) or (score == best_score and dot_f1 > best_dot_f1):
                best_score = score
                best_dot_f1 = dot_f1
                best_model = model
        if best_model is None:
            raise RuntimeError(f"Failed to tune any model for regularization={reg}")
        selected[reg] = best_model

    # mark selected rows
    for row in tuning_rows:
        reg = row["regularization"]
        row["selected"] = bool(reg in selected and abs(float(row["C"]) - selected[reg].c_value) < 1e-12)
    return selected, tuning_rows


def save_model_artifact(model: TrainedDotModel, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "regularization": model.regularization,
            "c_value": model.c_value,
            "pipeline": model.pipeline,
            "warnings": model.warnings,
        },
        p,
    )


def load_model_artifact(path: str | Path) -> TrainedDotModel:
    obj = joblib.load(path)
    return TrainedDotModel(
        regularization=obj["regularization"],
        c_value=float(obj["c_value"]),
        pipeline=obj["pipeline"],
        warnings=list(obj.get("warnings", [])),
    )


def save_feature_config(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

