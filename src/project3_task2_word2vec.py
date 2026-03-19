from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from .project3_common import ensure_dir, prepare_tokenized_corpus, write_json
from .project3_embeddings import evaluate_analogies, evaluate_targets, load_text_vectors

DEFAULT_TARGETS = [
    "azərbaycan",
    "şəhər",
    "tarix",
    "dövlət",
    "dünya",
    "insan",
    "dil",
    "elm",
    "mədəniyyət",
    "film",
    "qadın",
    "kişi",
    "kitab",
]

DEFAULT_ANALOGIES = [
    ("kişi", "qadın", "oğlan", "qız"),
    ("bakı", "azərbaycan", "ankara", "türkiyə"),
    ("ata", "ana", "oğul", "qız"),
    ("keçmiş", "indiki", "dünən", "bugün"),
    ("şimal", "cənub", "şərq", "qərb"),
]


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _parse_csv_list(arg: str | None) -> list[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3 Task 2: train/evaluate Word2Vec embeddings.")
    p.add_argument("--corpus_path", default="data/raw/corpus.csv")
    p.add_argument("--text_column", default="text")
    p.add_argument("--out_dir", default="outputs/project3/task2_word2vec")
    p.add_argument("--shared_corpus_out", default="outputs/project3/shared/tokenized_corpus.txt")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--word2vec_dir", default="word2vec")
    p.add_argument("--rebuild", action="store_true")

    p.add_argument("--size", type=int, default=200)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--negative", type=int, default=10)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--sample", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--cbow", type=int, default=0, choices=[0, 1], help="0=skip-gram, 1=CBOW")

    p.add_argument("--targets", default=None, help="Comma-separated target words for similarity analysis")
    p.add_argument("--top_k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    tokenized_stats = prepare_tokenized_corpus(
        corpus_path=args.corpus_path,
        text_column=args.text_column,
        out_path=args.shared_corpus_out,
        lowercase=args.lowercase,
        max_docs=args.max_docs,
    )

    w2v_dir = Path(args.word2vec_dir)
    binary = w2v_dir / "word2vec"
    if args.rebuild or not binary.exists():
        _run(["make"], cwd=str(w2v_dir))

    vectors_path = out_dir / "vectors.txt"
    train_cmd = [
        str(binary),
        "-train",
        str(Path(args.shared_corpus_out)),
        "-output",
        str(vectors_path),
        "-cbow",
        str(args.cbow),
        "-size",
        str(args.size),
        "-window",
        str(args.window),
        "-negative",
        str(args.negative),
        "-hs",
        "0",
        "-sample",
        str(args.sample),
        "-threads",
        str(args.threads),
        "-binary",
        "0",
        "-iter",
        str(args.iters),
        "-min-count",
        str(args.min_count),
    ]
    _run(train_cmd)

    space = load_text_vectors(vectors_path)
    user_targets = _parse_csv_list(args.targets)
    targets = user_targets if user_targets else [w for w in DEFAULT_TARGETS if space.has_word(w)]
    if len(targets) < 10:
        targets = (targets + space.words[:10])[:10]
    else:
        targets = targets[:10]

    nn_rows = evaluate_targets(space, targets=targets, top_k=args.top_k)
    analog_rows, analog_summary = evaluate_analogies(space, DEFAULT_ANALOGIES, top_k=args.top_k)

    pd.DataFrame(nn_rows).to_csv(out_dir / "nearest_neighbors.csv", index=False)
    pd.DataFrame(analog_rows).to_csv(out_dir / "analogy_results.csv", index=False)

    summary = {
        "task": "Project 3 Task 2 - Word2Vec",
        "config": {
            "corpus_path": args.corpus_path,
            "text_column": args.text_column,
            "shared_corpus_out": args.shared_corpus_out,
            "max_docs": args.max_docs,
            "lowercase": args.lowercase,
            "size": args.size,
            "window": args.window,
            "negative": args.negative,
            "min_count": args.min_count,
            "iters": args.iters,
            "sample": args.sample,
            "threads": args.threads,
            "cbow": args.cbow,
            "sg_mode": "skip-gram" if args.cbow == 0 else "cbow",
            "top_k": args.top_k,
            "targets": targets,
        },
        "tokenized_corpus": tokenized_stats.__dict__,
        "model": {
            "vocab_size": space.vocab_size,
            "vector_dim": space.dim,
            "vectors_path": str(vectors_path),
        },
        "analogy_summary": analog_summary,
        "artifacts": {
            "nearest_neighbors_csv": str(out_dir / "nearest_neighbors.csv"),
            "analogy_results_csv": str(out_dir / "analogy_results.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("Task 2 completed")
    print(f"  vocab={space.vocab_size} dim={space.dim}")
    print(f"  analogy_hit@{args.top_k}={analog_summary['hit_at_k']:.4f} evaluated={analog_summary['num_evaluated']}")
    print(f"  outputs={out_dir}")


if __name__ == "__main__":
    main()
