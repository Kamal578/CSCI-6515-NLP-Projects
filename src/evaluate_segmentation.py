from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

from .sentence_segment import sentence_segment


def load_indices(path: str) -> List[int]:
    """
    Load boundary indices from a file.
    Accepted formats:
      - JSON array of integers
      - newline-separated integers
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Boundary file not found: {p}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt.startswith("["):
        arr = json.loads(txt)
        return [int(x) for x in arr]
    # newline-separated
    return [int(line) for line in txt.splitlines() if line.strip()]


def segment_text_to_indices(text: str) -> List[int]:
    """
    Run the project sentence segmenter and return boundary indices
    defined as sentence positions (0-based).
    """
    sentences = sentence_segment(text)
    return list(range(len(sentences)))


def segment_corpus_to_indices(corpus_path: str, limit: int | None = None) -> List[int]:
    df = pd.read_csv(corpus_path)
    texts = df["text"].dropna().astype(str)
    if limit:
        texts = texts.head(limit)
    text = " ".join(texts.tolist())
    return segment_text_to_indices(text)


def to_binary_vector(indices: Sequence[int], length: int) -> np.ndarray:
    vec = np.zeros(length, dtype=int)
    for i in indices:
        if 0 <= i < length:
            vec[i] = 1
    return vec


def compute_metrics(pred: Sequence[int], gold: Sequence[int]) -> Tuple[float, float, float, float]:
    """
    Compute Precision, Recall, F1, and Boundary Detection Error Rate (BDER).
    BDER here is defined as (FP + FN) / |gold|, i.e., rate of boundary mistakes
    relative to gold boundary count.
    """
    if not gold and not pred:
        return 1.0, 1.0, 1.0, 0.0

    max_len = 0
    if gold:
        max_len = max(max_len, max(gold) + 1)
    if pred:
        max_len = max(max_len, max(pred) + 1)
    max_len = max(max_len, 1)

    gold_vec = to_binary_vector(gold, max_len)
    pred_vec = to_binary_vector(pred, max_len)

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_vec, pred_vec, average="binary", zero_division=0
    )

    tp = int(((gold_vec == 1) & (pred_vec == 1)).sum())
    fp = int(((gold_vec == 0) & (pred_vec == 1)).sum())
    fn = int(((gold_vec == 1) & (pred_vec == 0)).sum())
    denom = len(gold) if gold else 1
    bder = (fp + fn) / denom

    return float(precision), float(recall), float(f1), float(bder)


def save_metrics(out_path: str, precision: float, recall: float, f1: float, bder: float) -> None:
    payload = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "bder": bder,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Evaluate sentence segmentation boundaries.")
    ap.add_argument("--gold", required=True, help="Gold boundary indices file (JSON array or newline-separated ints).")
    ap.add_argument("--pred", help="Predicted boundary indices file.")
    ap.add_argument("--pred_text", help="Plain text file to segment with src.sentence_segment.")
    ap.add_argument("--pred_corpus", help="CSV with 'text' column; will be segmented.")
    ap.add_argument("--limit", type=int, default=None, help="Optional row limit when using --pred_corpus.")
    ap.add_argument("--out", required=True, help="Where to write metrics JSON.")
    args = ap.parse_args()

    gold_indices = load_indices(args.gold)
    if args.pred:
        pred_indices = load_indices(args.pred)
    elif args.pred_text:
        text = Path(args.pred_text).read_text(encoding="utf-8")
        pred_indices = segment_text_to_indices(text)
    elif args.pred_corpus:
        pred_indices = segment_corpus_to_indices(args.pred_corpus, limit=args.limit)
    else:
        raise SystemExit("Provide one of --pred, --pred_text, or --pred_corpus for predicted boundaries.")

    precision, recall, f1, bder = compute_metrics(pred_indices, gold_indices)

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"BDER:      {bder:.3f} (FP+FN normalized by |gold|)")

    save_metrics(args.out, precision, recall, f1, bder)
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
