from __future__ import annotations

import argparse
import json

from .project2_task4_dot_data import export_dot_label_template


def main() -> None:
    ap = argparse.ArgumentParser(description="Export manual-label template CSV for Task 4 dot EOS classification.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv")
    ap.add_argument("--text_column", type=str, default="text")
    ap.add_argument("--max_docs", type=int, default=200)
    ap.add_argument("--target_dots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="data/processed/task4_dot_labels_template.csv")
    args = ap.parse_args()

    info = export_dot_label_template(
        corpus_path=args.corpus_path,
        text_column=args.text_column,
        max_docs=args.max_docs,
        target_dots=args.target_dots,
        seed=args.seed,
        out_csv=args.out_csv,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

