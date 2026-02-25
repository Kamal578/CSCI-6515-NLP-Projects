from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

from .load_data import load_corpus_csv
from .ngram_lm import END_TOKEN, UNK_TOKEN, MLENgramModel, build_train_vocab, map_sentence_to_vocab
from .sentence_segment import sentence_segment
from .tokenize import tokenize


def split_docs(
    texts: list[str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    idxs = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    test_size = max(1, int(round(len(idxs) * test_ratio))) if len(idxs) > 1 else 0
    test_ids = set(idxs[:test_size])
    train_texts = [t for i, t in enumerate(texts) if i not in test_ids]
    test_texts = [t for i, t in enumerate(texts) if i in test_ids]
    return train_texts, test_texts


def docs_to_tokenized_sentences(texts: Iterable[str], lowercase: bool = True) -> list[list[str]]:
    sentences: list[list[str]] = []
    for text in texts:
        for sent in sentence_segment(text):
            toks = tokenize(sent, lowercase=lowercase)
            if toks:
                sentences.append(toks)
    return sentences


def write_top_ngrams_csv(
    counts: Counter[tuple[str, ...]],
    out_path: Path,
    top_n: int = 200,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = counts.most_common(top_n) if top_n and top_n > 0 else counts.most_common()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ngram", "count"])
        for ng, count in rows:
            writer.writerow([" ".join(ng), count])


def count_predicted_tokens(sentences: list[list[str]]) -> int:
    # Each sentence contributes its tokens plus one explicit end-of-sentence token.
    return sum(len(s) + 1 for s in sentences)


def main() -> None:
    ap = argparse.ArgumentParser(description="Project 2 Task 1: n-gram LM (MLE, no smoothing) + perplexity.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv")
    ap.add_argument("--text_column", type=str, default="text", help="CSV column to use (e.g., text or text_clean).")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_docs", type=int, default=None, help="Optional cap on number of documents (debug/smoke runs).")
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.add_argument(
        "--unk_min_freq",
        type=int,
        default=2,
        help="Map train tokens with freq < this threshold to <unk> before fitting (recommended >=2).",
    )
    ap.add_argument("--top_n", type=int, default=200, help="How many top n-grams to export per model (0=all).")
    ap.add_argument("--out_dir", type=str, default="outputs/project2/task1_lm")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_corpus_csv(args.corpus_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found. Available: {df.columns.tolist()}")
    texts = df[args.text_column].fillna("").astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if args.max_docs is not None:
        texts = texts[: args.max_docs]
    if len(texts) < 2:
        raise ValueError("Need at least 2 non-empty documents for a train/test split.")

    train_docs, test_docs = split_docs(texts, test_ratio=args.test_ratio, seed=args.seed)

    print(f"Loaded {len(texts)} docs (train={len(train_docs)}, test={len(test_docs)}).")
    print("Segmenting + tokenizing train docs...")
    train_sents_raw = docs_to_tokenized_sentences(train_docs, lowercase=args.lowercase)
    print("Segmenting + tokenizing test docs...")
    test_sents_raw = docs_to_tokenized_sentences(test_docs, lowercase=args.lowercase)
    if not train_sents_raw:
        raise ValueError("No tokenized training sentences were produced.")
    if not test_sents_raw:
        raise ValueError("No tokenized test sentences were produced.")

    raw_train_token_counts = Counter(tok for s in train_sents_raw for tok in s)
    raw_test_token_count = sum(len(s) for s in test_sents_raw)
    raw_test_oov_token_count = sum(
        1 for s in test_sents_raw for tok in s if tok not in raw_train_token_counts
    )

    train_vocab_counts = build_train_vocab(train_sents_raw, min_freq=args.unk_min_freq)
    train_vocab = set(train_vocab_counts.keys())
    train_vocab.add(UNK_TOKEN)

    train_sents = [map_sentence_to_vocab(s, train_vocab) for s in train_sents_raw]
    test_sents = [map_sentence_to_vocab(s, train_vocab) for s in test_sents_raw]

    print("Training/evaluating unigram, bigram, trigram MLE models...")
    models = {}
    metrics = {}
    for n in (1, 2, 3):
        model = MLENgramModel(n).fit(train_sents)
        stats = model.evaluate(test_sents)
        models[n] = model
        metrics[n] = stats

    write_top_ngrams_csv(models[1].ngram_counts, out_dir / "unigram_top.csv", top_n=args.top_n)
    write_top_ngrams_csv(models[2].ngram_counts, out_dir / "bigram_top.csv", top_n=args.top_n)
    write_top_ngrams_csv(models[3].ngram_counts, out_dir / "trigram_top.csv", top_n=args.top_n)

    summary = {
        "task": "Project 2 Task 1 - N-gram LM baseline (MLE, no smoothing)",
        "config": {
            "corpus_path": args.corpus_path,
            "text_column": args.text_column,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "max_docs": args.max_docs,
            "lowercase": args.lowercase,
            "unk_min_freq": args.unk_min_freq,
            "top_n": args.top_n,
            "out_dir": str(out_dir),
            "smoothing": None,
        },
        "data_split": {
            "num_docs_total": len(texts),
            "num_docs_train": len(train_docs),
            "num_docs_test": len(test_docs),
            "num_sentences_train": len(train_sents),
            "num_sentences_test": len(test_sents),
            "num_tokens_train_excluding_eos": int(sum(len(s) for s in train_sents)),
            "num_tokens_test_excluding_eos": int(sum(len(s) for s in test_sents)),
            "num_predicted_events_per_model_test": {
                "unigram": int(count_predicted_tokens(test_sents)),
                "bigram": int(count_predicted_tokens(test_sents)),
                "trigram": int(count_predicted_tokens(test_sents)),
            },
        },
        "vocab": {
            "train_vocab_size_after_unk_threshold": len(train_vocab),
            "unk_token": UNK_TOKEN,
            "eos_token": END_TOKEN,
            "raw_train_unique_tokens_before_threshold": len(raw_train_token_counts),
            "raw_test_oov_token_count_vs_train_vocab": int(raw_test_oov_token_count),
            "raw_test_token_count": int(raw_test_token_count),
            "raw_test_oov_rate_vs_train_vocab": (
                raw_test_oov_token_count / raw_test_token_count if raw_test_token_count else 0.0
            ),
            "unk_count_in_train_after_mapping": int(
                sum(tok == UNK_TOKEN for s in train_sents for tok in s)
            ),
        },
        "models": {
            "unigram": {
                "num_ngrams": len(models[1].ngram_counts),
                **metrics[1].to_dict(),
            },
            "bigram": {
                "num_ngrams": len(models[2].ngram_counts),
                "num_contexts": len(models[2].context_counts),
                **metrics[2].to_dict(),
            },
            "trigram": {
                "num_ngrams": len(models[3].ngram_counts),
                "num_contexts": len(models[3].context_counts),
                **metrics[3].to_dict(),
            },
        },
        "notes": [
            "No smoothing is applied in Task 1 (MLE baseline).",
            "Perplexity can be infinite when unseen n-grams occur in the test set.",
            "Train/test split, tokenization, and <unk> policy should be kept fixed for Task 2 smoothing comparisons.",
        ],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print(f"Saved summary: {out_dir / 'summary.json'}")
    for name in ("unigram", "bigram", "trigram"):
        ppl = summary["models"][name]["perplexity"]
        ppl_str = "inf" if (isinstance(ppl, float) and math.isinf(ppl)) else f"{ppl:.6f}"
        z = summary["models"][name]["zero_prob_events"]
        events = summary["models"][name]["num_events"]
        print(f"{name:7s} perplexity={ppl_str} zero_prob_events={z}/{events}")


if __name__ == "__main__":
    main()
