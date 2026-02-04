from __future__ import annotations

import json
from pathlib import Path
import argparse
import pandas as pd

from .spell_utils import load_freqs, filter_vocab
from .spellcheck import suggest


def eval_spellcheck(
    test_csv: str,
    corpus_path: str,
    out_summary: str = "outputs/spellcheck/spell_eval.json",
    out_samples: str = "outputs/spellcheck/sample_predictions.csv",
    max_dist: int = 2,
    top_k: int = 5,
    min_freq: int = 2,
    min_len: int = 3,
    confusion_path: str | None = None,
):
    df = pd.read_csv(test_csv)
    freqs = load_freqs(corpus_path, lowercase=True)
    vocab = filter_vocab(freqs, min_freq=min_freq, min_len=min_len)

    weights = None
    if confusion_path:
        import json
        raw = json.loads(Path(confusion_path).read_text(encoding="utf-8"))
        weights = {eval(k): float(v) for k, v in raw.get("weights", {}).items()}

    total = len(df)
    hits1 = 0
    hits5 = 0
    samples = []

    for _, row in df.iterrows():
        miss = row["misspelled"]
        correct = row["correct"]
        cands = suggest(miss, vocab, max_dist=max_dist, top_k=top_k, weights=weights)
        cand_words = [w for w, _ in cands]
        if cand_words and cand_words[0] == correct:
            hits1 += 1
        if correct in cand_words:
            hits5 += 1
        samples.append(
            {
                "misspelled": miss,
                "correct": correct,
                "candidates": cand_words,
            }
        )

    acc1 = hits1 / total if total else 0.0
    acc5 = hits5 / total if total else 0.0

    summary = {
        "total": total,
        "accuracy@1": acc1,
        "accuracy@5": acc5,
        "max_dist": max_dist,
        "top_k": top_k,
        "vocab_size": len(vocab),
        "test_csv": test_csv,
        "confusion_path": confusion_path,
    }

    out_sum_path = Path(out_summary)
    out_sum_path.parent.mkdir(parents=True, exist_ok=True)
    out_sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_samp_path = Path(out_samples)
    out_samp_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(out_samp_path, index=False, encoding="utf-8")

    print(f"Accuracy@1={acc1:.3f} Accuracy@5={acc5:.3f} (n={total})")
    print(f"Saved summary to {out_sum_path}")
    print(f"Saved sample predictions to {out_samp_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate spell checker on synthetic test set.")
    ap.add_argument("--test_csv", type=str, default="data/processed/spell_test.csv")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv")
    ap.add_argument("--out_summary", type=str, default="outputs/spellcheck/spell_eval.json")
    ap.add_argument("--out_samples", type=str, default="outputs/spellcheck/sample_predictions.csv")
    ap.add_argument("--max_dist", type=int, default=2)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--confusion", type=str, help="Path to confusion.json for weighted edit distance.")
    args = ap.parse_args()

    eval_spellcheck(
        test_csv=args.test_csv,
        corpus_path=args.corpus_path,
        out_summary=args.out_summary,
        out_samples=args.out_samples,
        max_dist=args.max_dist,
        top_k=args.top_k,
        min_freq=args.min_freq,
        min_len=args.min_len,
        confusion_path=args.confusion,
    )


if __name__ == "__main__":
    main()
