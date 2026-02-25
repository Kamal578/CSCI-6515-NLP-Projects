from __future__ import annotations

import argparse
import json

from .project2_task4_dot_data import build_task4_labels_from_gold_sentences


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Task 4 labeled dot CSV + pseudo corpus from sentence-per-line gold file."
    )
    ap.add_argument("--gold_sentences", type=str, required=True, help="Sentence-per-line gold file (manual).")
    ap.add_argument(
        "--out_labels_csv",
        type=str,
        default="data/processed/task4_dot_labels_from_gold_sentences.csv",
        help="Output labeled dot CSV for Task 4.",
    )
    ap.add_argument(
        "--out_corpus_csv",
        type=str,
        default="data/processed/task4_gold_pseudo_corpus.csv",
        help="Output pseudo corpus CSV (doc_id,text).",
    )
    ap.add_argument(
        "--sentences_per_doc",
        type=int,
        default=10,
        help="How many gold sentences to group into each pseudo-document.",
    )
    args = ap.parse_args()

    info = build_task4_labels_from_gold_sentences(
        gold_sentences_path=args.gold_sentences,
        out_labels_csv=args.out_labels_csv,
        out_corpus_csv=args.out_corpus_csv,
        sentences_per_doc=args.sentences_per_doc,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

