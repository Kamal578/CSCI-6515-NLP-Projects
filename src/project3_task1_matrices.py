from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .project3_common import ensure_dir, load_texts, write_json
from .sentence_segment import sentence_segment
from .tokenize import tokenize


def _tokenize_docs(texts: list[str], lowercase: bool = True) -> tuple[list[list[str]], list[list[str]]]:
    doc_tokens: list[list[str]] = []
    sentence_tokens: list[list[str]] = []
    for text in texts:
        toks_doc = tokenize(text, lowercase=lowercase)
        if toks_doc:
            doc_tokens.append(toks_doc)
        for sent in sentence_segment(text):
            toks = tokenize(sent, lowercase=lowercase)
            if toks:
                sentence_tokens.append(toks)
    return doc_tokens, sentence_tokens


def _build_cooccurrence(sentences: list[list[str]], vocab_to_idx: dict[str, int], window: int = 2) -> np.ndarray:
    n = len(vocab_to_idx)
    mat = np.zeros((n, n), dtype=np.int64)
    for sent in sentences:
        idxs = [vocab_to_idx[t] for t in sent if t in vocab_to_idx]
        for i, c in enumerate(idxs):
            lo = max(0, i - window)
            hi = min(len(idxs), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                mat[c, idxs[j]] += 1
    return mat


def _plot_heatmap(mat: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    log_mat = np.log1p(mat)
    im = ax.imshow(log_mat, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 3 Task 1: dataset stats + term-document/word-word matrices.")
    p.add_argument("--corpus_path", default="data/raw/corpus.csv")
    p.add_argument("--text_column", default="text")
    p.add_argument("--out_dir", default="outputs/project3/task1_matrices")
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--top_terms", type=int, default=100)
    p.add_argument("--top_docs", type=int, default=100)
    p.add_argument("--frequent_threshold", type=int, default=100)
    p.add_argument("--rare_threshold", type=int, default=2)
    p.add_argument("--cooc_window", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    texts = load_texts(args.corpus_path, args.text_column, args.max_docs)
    doc_tokens, sent_tokens = _tokenize_docs(texts=texts, lowercase=args.lowercase)
    if not doc_tokens:
        raise ValueError("No tokenized documents were produced.")

    token_counts = Counter(tok for doc in doc_tokens for tok in doc)
    vocab_size = len(token_counts)
    total_tokens = int(sum(token_counts.values()))
    num_docs = len(doc_tokens)

    frequent_words = {w: c for w, c in token_counts.items() if c >= args.frequent_threshold}
    rare_words = {w: c for w, c in token_counts.items() if c <= args.rare_threshold}

    docs_as_text = [" ".join(toks) for toks in doc_tokens]
    vec = CountVectorizer(lowercase=False, tokenizer=str.split, token_pattern=None)
    X_td = vec.fit_transform(docs_as_text)
    freq_series = pd.Series(token_counts, name="count").sort_values(ascending=False)
    top_terms = freq_series.head(args.top_terms).index.tolist()
    top_term_to_idx = {t: i for i, t in enumerate(top_terms)}

    term_doc_top = np.asarray(X_td[: args.top_docs, [vec.vocabulary_[t] for t in top_terms if t in vec.vocabulary_]].todense())
    cooc_top = _build_cooccurrence(sentences=sent_tokens, vocab_to_idx=top_term_to_idx, window=args.cooc_window)

    sparse.save_npz(out_dir / "term_document_sparse.npz", X_td)
    pd.DataFrame({"term": freq_series.index, "count": freq_series.values}).to_csv(out_dir / "word_frequency_distribution.csv", index=False)
    pd.DataFrame(term_doc_top, columns=top_terms[: term_doc_top.shape[1]]).to_csv(out_dir / "term_document_top_matrix.csv", index=False)
    pd.DataFrame(cooc_top, index=top_terms, columns=top_terms).to_csv(out_dir / "word_word_top_matrix.csv", index=True)

    _plot_heatmap(term_doc_top.T, labels=[f"doc{i+1}" for i in range(term_doc_top.shape[0])], title="Term-Document Matrix (Top Terms x Top Docs)", out_path=out_dir / "term_document_heatmap.png")
    _plot_heatmap(cooc_top, labels=top_terms, title="Word-Word Co-occurrence Matrix (Top Terms)", out_path=out_dir / "word_word_heatmap.png")

    summary = {
        "task": "Project 3 Task 1 - Dataset and matrix analysis",
        "config": {
            "corpus_path": args.corpus_path,
            "text_column": args.text_column,
            "max_docs": args.max_docs,
            "lowercase": args.lowercase,
            "top_terms": args.top_terms,
            "top_docs": args.top_docs,
            "frequent_threshold": args.frequent_threshold,
            "rare_threshold": args.rare_threshold,
            "cooc_window": args.cooc_window,
            "out_dir": str(out_dir),
        },
        "dataset": {
            "num_docs": num_docs,
            "num_sentences": len(sent_tokens),
            "num_tokens": total_tokens,
            "vocabulary_size": vocab_size,
            "num_frequent_words": len(frequent_words),
            "num_rare_words": len(rare_words),
            "top_30_words": freq_series.head(30).to_dict(),
        },
        "matrices": {
            "term_document_sparse_shape": [int(X_td.shape[0]), int(X_td.shape[1])],
            "term_document_sparse_nnz": int(X_td.nnz),
            "term_document_top_shape": [int(term_doc_top.shape[0]), int(term_doc_top.shape[1])],
            "word_word_top_shape": [int(cooc_top.shape[0]), int(cooc_top.shape[1])],
            "note": "Word-word matrix is built on top-N terms for tractability.",
        },
        "artifacts": {
            "term_document_sparse": str(out_dir / "term_document_sparse.npz"),
            "frequency_csv": str(out_dir / "word_frequency_distribution.csv"),
            "term_document_top_csv": str(out_dir / "term_document_top_matrix.csv"),
            "word_word_top_csv": str(out_dir / "word_word_top_matrix.csv"),
            "term_document_heatmap": str(out_dir / "term_document_heatmap.png"),
            "word_word_heatmap": str(out_dir / "word_word_heatmap.png"),
        },
    }
    write_json(out_dir / "summary.json", summary)

    print("Task 1 completed")
    print(f"  docs={num_docs} tokens={total_tokens} vocab={vocab_size}")
    print(f"  frequent(>={args.frequent_threshold})={len(frequent_words)} rare(<={args.rare_threshold})={len(rare_words)}")
    print(f"  outputs={out_dir}")


if __name__ == "__main__":
    main()
