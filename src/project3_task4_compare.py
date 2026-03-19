from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .project2_task3_sentiment_data import load_sentiment_dataset
from .project2_task3_sentiment_preprocess import legacy_tokenize
from .project3_common import ensure_dir, read_json, write_json
from .project3_embeddings import average_neighbor_cosine, load_text_vectors


def _coverage_ratio(vectors_path: str, texts: pd.Series) -> tuple[float, int, int]:
    space = load_text_vectors(vectors_path)
    vocab = set(space.words)
    token_counts = Counter(tok for txt in texts.astype(str).tolist() for tok in legacy_tokenize(txt))
    unique_tokens = set(token_counts.keys())
    covered = sum(1 for t in unique_tokens if t in vocab)
    return (covered / len(unique_tokens) if unique_tokens else 0.0, covered, len(unique_tokens))


def _neighbor_quality(rows_df: pd.DataFrame) -> float:
    if rows_df.empty:
        return 0.0
    rows = rows_df.to_dict(orient="records")
    return average_neighbor_cosine(rows, top_n=3)


def _qualitative_notes(compare_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if compare_df.empty:
        return ["No comparison rows available."]

    best_cov = compare_df.sort_values("coverage_ratio", ascending=False).iloc[0]
    best_ana = compare_df.sort_values("analogy_hit_at_k", ascending=False).iloc[0]
    best_nn = compare_df.sort_values("avg_neighbor_cosine_top3", ascending=False).iloc[0]

    notes.append(
        f"Highest sentiment-vocabulary coverage: {best_cov['model']} ({best_cov['coverage_ratio']:.4f})."
    )
    notes.append(f"Best analogy hit@k: {best_ana['model']} ({best_ana['analogy_hit_at_k']:.4f}).")
    notes.append(
        f"Best average top-3 neighbor cosine: {best_nn['model']} ({best_nn['avg_neighbor_cosine_top3']:.4f})."
    )

    if best_cov["model"] != best_ana["model"]:
        notes.append("Coverage and analogy accuracy favor different models; downstream choice should match objective.")
    else:
        notes.append("One model leads both in vocabulary coverage and analogy quality in this run.")
    return notes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3 Task 4: compare Word2Vec vs GloVe.")
    p.add_argument("--word2vec_summary", default="outputs/project3/task2_word2vec/summary.json")
    p.add_argument("--glove_summary", default="outputs/project3/task3_glove/summary.json")
    p.add_argument("--word2vec_neighbors", default="outputs/project3/task2_word2vec/nearest_neighbors.csv")
    p.add_argument("--glove_neighbors", default="outputs/project3/task3_glove/nearest_neighbors.csv")
    p.add_argument("--out_dir", default="outputs/project3/task4_compare")
    p.add_argument("--train_csv", default="data/external/train.csv")
    p.add_argument("--test_csv", default="data/external/test.csv")
    p.add_argument("--text_col", default="content")
    p.add_argument("--score_col", default="score")
    p.add_argument("--label_mode", default="sentiment3", choices=["sentiment3", "score5", "binary"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    w2v_summary = read_json(args.word2vec_summary)
    glove_summary = read_json(args.glove_summary)

    w2v_vectors = w2v_summary.get("model", {}).get("vectors_path")
    glove_vectors = glove_summary.get("model", {}).get("vectors_path")
    if not w2v_vectors or not glove_vectors:
        raise ValueError("Missing vectors_path in Task 2/3 summary files.")

    ds = load_sentiment_dataset(
        train_path=args.train_csv,
        test_path=args.test_csv,
        text_col=args.text_col,
        score_col=args.score_col,
        label_mode=args.label_mode,
    )
    all_texts = pd.concat([ds.train_df[args.text_col], ds.test_df[args.text_col]], axis=0).reset_index(drop=True)

    w2v_cov_ratio, w2v_cov_n, w2v_cov_total = _coverage_ratio(w2v_vectors, all_texts)
    glove_cov_ratio, glove_cov_n, glove_cov_total = _coverage_ratio(glove_vectors, all_texts)

    w2v_neighbors_df = pd.read_csv(args.word2vec_neighbors) if Path(args.word2vec_neighbors).exists() else pd.DataFrame()
    glove_neighbors_df = pd.read_csv(args.glove_neighbors) if Path(args.glove_neighbors).exists() else pd.DataFrame()

    rows = [
        {
            "model": "word2vec",
            "vector_dim": int(w2v_summary.get("model", {}).get("vector_dim", 0)),
            "vocab_size": int(w2v_summary.get("model", {}).get("vocab_size", 0)),
            "analogy_hit_at_1": float(w2v_summary.get("analogy_summary", {}).get("hit_at_1", 0.0)),
            "analogy_hit_at_k": float(w2v_summary.get("analogy_summary", {}).get("hit_at_k", 0.0)),
            "avg_neighbor_cosine_top3": _neighbor_quality(w2v_neighbors_df),
            "coverage_ratio": w2v_cov_ratio,
            "coverage_unique_tokens": w2v_cov_n,
            "sentiment_unique_tokens": w2v_cov_total,
        },
        {
            "model": "glove",
            "vector_dim": int(glove_summary.get("model", {}).get("vector_dim", 0)),
            "vocab_size": int(glove_summary.get("model", {}).get("vocab_size", 0)),
            "analogy_hit_at_1": float(glove_summary.get("analogy_summary", {}).get("hit_at_1", 0.0)),
            "analogy_hit_at_k": float(glove_summary.get("analogy_summary", {}).get("hit_at_k", 0.0)),
            "avg_neighbor_cosine_top3": _neighbor_quality(glove_neighbors_df),
            "coverage_ratio": glove_cov_ratio,
            "coverage_unique_tokens": glove_cov_n,
            "sentiment_unique_tokens": glove_cov_total,
        },
    ]
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(out_dir / "comparison.csv", index=False)

    notes = _qualitative_notes(compare_df)

    summary = {
        "task": "Project 3 Task 4 - Word2Vec vs GloVe comparison",
        "dataset": {
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "text_col": args.text_col,
            "score_col": args.score_col,
            "label_mode": args.label_mode,
            "n_train": int(len(ds.train_df)),
            "n_test": int(len(ds.test_df)),
            "labels": ds.labels_order,
        },
        "models": rows,
        "qualitative_notes": notes,
        "winner_by_metric": {
            "analogy_hit_at_k": compare_df.sort_values("analogy_hit_at_k", ascending=False).iloc[0]["model"],
            "coverage_ratio": compare_df.sort_values("coverage_ratio", ascending=False).iloc[0]["model"],
            "avg_neighbor_cosine_top3": compare_df.sort_values("avg_neighbor_cosine_top3", ascending=False).iloc[0]["model"],
        },
        "artifacts": {
            "comparison_csv": str(out_dir / "comparison.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("Task 4 completed")
    print(compare_df.to_string(index=False))
    print(f"outputs={out_dir}")


if __name__ == "__main__":
    main()
