from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels_order: list[str]) -> dict[str, Any]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
    }
    for label in labels_order:
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=[label], average=None, zero_division=0
        )
        metrics[f"f1_{label}"] = float(f[0])
        metrics[f"support_{label}"] = int(s[0])
    return metrics


def confusion_as_list(y_true: np.ndarray, y_pred: np.ndarray, labels_order: list[str]) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=labels_order).tolist()


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    b = int(np.sum(a_correct & ~b_correct))  # A correct, B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong, B correct
    n = b + c
    pvalue = 1.0 if n == 0 else float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {
        "b_A_correct_B_wrong": b,
        "c_A_wrong_B_correct": c,
        "discordant_total": n,
        "mcnemar_exact_pvalue": pvalue,
    }


def holm_bonferroni(pvals: list[tuple[int, float]], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    ordered = sorted(pvals, key=lambda x: x[1])
    rejected = [False] * m
    for rank, (idx, p) in enumerate(ordered, start=1):
        threshold = alpha / (m - rank + 1)
        if p <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected
