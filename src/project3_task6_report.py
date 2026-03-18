from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .project3_common import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3 Task 6: build consolidated report artifacts.")
    p.add_argument("--task1_summary", default="outputs/project3/task1_matrices/summary.json")
    p.add_argument("--task2_summary", default="outputs/project3/task2_word2vec/summary.json")
    p.add_argument("--task3_summary", default="outputs/project3/task3_glove/summary.json")
    p.add_argument("--task4_summary", default="outputs/project3/task4_compare/summary.json")
    p.add_argument("--task5_summary", default="outputs/project3/task5_dl/summary.json")
    p.add_argument("--task4_csv", default="outputs/project3/task4_compare/comparison.csv")
    p.add_argument("--task5_csv", default="outputs/project3/task5_dl/leaderboard.csv")
    p.add_argument("--out_dir", default="outputs/project3/task6_report")
    return p.parse_args()


def _safe_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    s1 = read_json(args.task1_summary)
    s2 = read_json(args.task2_summary)
    s3 = read_json(args.task3_summary)
    s4 = read_json(args.task4_summary)
    s5 = read_json(args.task5_summary)

    compare_df = _safe_df(args.task4_csv)
    leaderboard_df = _safe_df(args.task5_csv)

    if not compare_df.empty:
        compare_df.to_csv(out_dir / "embedding_comparison_table.csv", index=False)
    if not leaderboard_df.empty:
        leaderboard_df.to_csv(out_dir / "dl_results_table.csv", index=False)

    lines = [
        "# Project 3 Report Artifacts",
        "",
        "## Task 1",
        f"- Docs: {s1.get('dataset', {}).get('num_docs', 'n/a')}",
        f"- Tokens: {s1.get('dataset', {}).get('num_tokens', 'n/a')}",
        f"- Vocab: {s1.get('dataset', {}).get('vocabulary_size', 'n/a')}",
        "",
        "## Task 2 (Word2Vec)",
        f"- Vocab: {s2.get('model', {}).get('vocab_size', 'n/a')}",
        f"- Dim: {s2.get('model', {}).get('vector_dim', 'n/a')}",
        f"- Analogy hit@k: {s2.get('analogy_summary', {}).get('hit_at_k', 'n/a')}",
        "",
        "## Task 3 (GloVe)",
        f"- Vocab: {s3.get('model', {}).get('vocab_size', 'n/a')}",
        f"- Dim: {s3.get('model', {}).get('vector_dim', 'n/a')}",
        f"- Analogy hit@k: {s3.get('analogy_summary', {}).get('hit_at_k', 'n/a')}",
        "",
        "## Task 4",
        f"- Winner by analogy: {s4.get('winner_by_metric', {}).get('analogy_hit_at_k', 'n/a')}",
        f"- Winner by coverage: {s4.get('winner_by_metric', {}).get('coverage_ratio', 'n/a')}",
        "",
        "## Task 5",
        f"- Best overall: {s5.get('best_overall', {}).get('experiment_key', 'n/a')}",
        f"- Macro-F1: {s5.get('best_overall', {}).get('f1_macro', 'n/a')}",
        "",
        "## Files",
        "- `embedding_comparison_table.csv`",
        "- `dl_results_table.csv`",
        "- `report_artifacts.md`",
    ]
    (out_dir / "report_artifacts.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "task": "Project 3 Task 6 - Report artifacts",
        "input_summaries": {
            "task1": args.task1_summary,
            "task2": args.task2_summary,
            "task3": args.task3_summary,
            "task4": args.task4_summary,
            "task5": args.task5_summary,
        },
        "artifacts": {
            "embedding_comparison_table_csv": str(out_dir / "embedding_comparison_table.csv"),
            "dl_results_table_csv": str(out_dir / "dl_results_table.csv"),
            "report_artifacts_md": str(out_dir / "report_artifacts.md"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("Task 6 completed")
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
