from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .load_data import load_corpus_csv
from .sentence_segment import sentence_segment
from .tokenize import tokenize


@dataclass
class TokenizedCorpusStats:
    corpus_path: str
    text_column: str
    max_docs: int | None
    num_docs: int
    num_sentences: int
    num_tokens: int
    output_path: str


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_texts(corpus_path: str, text_column: str = "text", max_docs: int | None = None) -> list[str]:
    df = load_corpus_csv(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")
    texts = [t for t in df[text_column].fillna("").astype(str).tolist() if t.strip()]
    if max_docs is not None:
        texts = texts[:max_docs]
    return texts


def iter_tokenized_sentences(texts: Iterable[str], lowercase: bool = True) -> Iterable[list[str]]:
    for text in texts:
        for sent in sentence_segment(text):
            toks = tokenize(sent, lowercase=lowercase)
            if toks:
                yield toks


def prepare_tokenized_corpus(
    corpus_path: str,
    text_column: str,
    out_path: str | Path,
    lowercase: bool = True,
    max_docs: int | None = None,
) -> TokenizedCorpusStats:
    texts = load_texts(corpus_path=corpus_path, text_column=text_column, max_docs=max_docs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sent_count = 0
    token_count = 0
    with out.open("w", encoding="utf-8") as f:
        for toks in iter_tokenized_sentences(texts, lowercase=lowercase):
            f.write(" ".join(toks) + "\n")
            sent_count += 1
            token_count += len(toks)

    return TokenizedCorpusStats(
        corpus_path=corpus_path,
        text_column=text_column,
        max_docs=max_docs,
        num_docs=len(texts),
        num_sentences=sent_count,
        num_tokens=token_count,
        output_path=str(out),
    )
