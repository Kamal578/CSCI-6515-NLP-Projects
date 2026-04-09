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
    p = argparse.ArgumentParser(description="Project 3 Task 3: train/evaluate GloVe embeddings.")
    p.add_argument("--corpus_path", default="data/raw/corpus.csv")
    p.add_argument("--text_column", default="text")
    p.add_argument("--out_dir", default="outputs/project3/task3_glove")
    p.add_argument("--shared_corpus_out", default="outputs/project3/shared/tokenized_corpus.txt")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--glove_dir", default="GloVe")
    p.add_argument("--rebuild", action="store_true")

    p.add_argument("--vector_size", type=int, default=200)
    p.add_argument("--window_size", type=int, default=10)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--x_max", type=float, default=10.0)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--memory", type=float, default=4.0)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--verbose", type=int, default=2)

    p.add_argument("--targets", default=None)
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

    glove_dir = Path(args.glove_dir)
    build_dir = glove_dir / "build"
    glove_binary = build_dir / "glove"
    if args.rebuild or not glove_binary.exists():
        _run(["make"], cwd=str(glove_dir))

    vocab_file = out_dir / "vocab.txt"
    cooc_file = out_dir / "cooccurrence.bin"
    cooc_shuf_file = out_dir / "cooccurrence.shuf.bin"
    vectors_prefix = out_dir / "vectors"
    vectors_txt = out_dir / "vectors.txt"

    with Path(args.shared_corpus_out).open("r", encoding="utf-8") as fin, vocab_file.open("w", encoding="utf-8") as fout:
        subprocess.run(
            [str(build_dir / "vocab_count"), "-min-count", str(args.min_count), "-verbose", str(args.verbose)],
            stdin=fin,
            stdout=fout,
            check=True,
        )

    with Path(args.shared_corpus_out).open("r", encoding="utf-8") as fin, cooc_file.open("wb") as fout:
        subprocess.run(
            [
                str(build_dir / "cooccur"),
                "-memory",
                str(args.memory),
                "-vocab-file",
                str(vocab_file),
                "-verbose",
                str(args.verbose),
                "-window-size",
                str(args.window_size),
            ],
            stdin=fin,
            stdout=fout,
            check=True,
        )

    with cooc_file.open("rb") as fin, cooc_shuf_file.open("wb") as fout:
        subprocess.run(
            [str(build_dir / "shuffle"), "-memory", str(args.memory), "-verbose", str(args.verbose)],
            stdin=fin,
            stdout=fout,
            check=True,
        )

    _run(
        [
            str(glove_binary),
            "-save-file",
            str(vectors_prefix),
            "-threads",
            str(args.threads),
            "-input-file",
            str(cooc_shuf_file),
            "-x-max",
            str(args.x_max),
            "-iter",
            str(args.max_iter),
            "-vector-size",
            str(args.vector_size),
            "-binary",
            "0",
            "-vocab-file",
            str(vocab_file),
            "-verbose",
            str(args.verbose),
            "-eta",
            str(args.eta),
            "-alpha",
            str(args.alpha),
        ]
    )

    if not vectors_txt.exists():
        # Some builds may emit vectors directly as `<prefix>` without extension.
        fallback = Path(str(vectors_prefix))
        if fallback.exists():
            fallback.rename(vectors_txt)
        else:
            raise FileNotFoundError(f"Expected GloVe vectors not found: {vectors_txt}")

    space = load_text_vectors(vectors_txt)
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
        "task": "Project 3 Task 3 - GloVe",
        "config": {
            "corpus_path": args.corpus_path,
            "text_column": args.text_column,
            "shared_corpus_out": args.shared_corpus_out,
            "max_docs": args.max_docs,
            "lowercase": args.lowercase,
            "vector_size": args.vector_size,
            "window_size": args.window_size,
            "min_count": args.min_count,
            "max_iter": args.max_iter,
            "x_max": args.x_max,
            "eta": args.eta,
            "alpha": args.alpha,
            "memory": args.memory,
            "threads": args.threads,
            "verbose": args.verbose,
            "top_k": args.top_k,
            "targets": targets,
        },
        "tokenized_corpus": tokenized_stats.__dict__,
        "model": {
            "vocab_size": space.vocab_size,
            "vector_dim": space.dim,
            "vectors_path": str(vectors_txt),
            "vocab_file": str(vocab_file),
        },
        "analogy_summary": analog_summary,
        "artifacts": {
            "nearest_neighbors_csv": str(out_dir / "nearest_neighbors.csv"),
            "analogy_results_csv": str(out_dir / "analogy_results.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("Task 3 completed")
    print(f"  vocab={space.vocab_size} dim={space.dim}")
    print(f"  analogy_hit@{args.top_k}={analog_summary['hit_at_k']:.4f} evaluated={analog_summary['num_evaluated']}")
    print(f"  outputs={out_dir}")


if __name__ == "__main__":
    main()
