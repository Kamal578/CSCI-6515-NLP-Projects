from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .load_data import load_corpus_csv
from .ngram_lm import END_TOKEN, UNK_TOKEN, build_train_vocab, map_sentence_to_vocab
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


def split_sentences(
    sentences: list[list[str]],
    dev_ratio: float = 0.1,
    seed: int = 43,
) -> tuple[list[list[str]], list[list[str]]]:
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be in (0, 1)")
    if len(sentences) < 2:
        raise ValueError("Need at least 2 training sentences to create train/dev split.")
    idxs = list(range(len(sentences)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    dev_size = max(1, int(round(len(idxs) * dev_ratio)))
    if dev_size >= len(idxs):
        dev_size = len(idxs) - 1
    dev_ids = set(idxs[:dev_size])
    train_core = [s for i, s in enumerate(sentences) if i not in dev_ids]
    dev = [s for i, s in enumerate(sentences) if i in dev_ids]
    return train_core, dev


def docs_to_tokenized_sentences(texts: Iterable[str], lowercase: bool = True) -> list[list[str]]:
    sentences: list[list[str]] = []
    for text in texts:
        for sent in sentence_segment(text):
            toks = tokenize(sent, lowercase=lowercase)
            if toks:
                sentences.append(toks)
    return sentences


def count_predicted_events(sentences: list[list[str]]) -> int:
    # Predicted events for unigram/bigram/trigram eval all equal to tokens + explicit </s> per sentence.
    return sum(len(s) + 1 for s in sentences)


@dataclass
class PreparedLMSplits:
    texts_total: int
    train_docs: list[str]
    test_docs: list[str]
    train_sentences_raw: list[list[str]]
    test_sentences_raw: list[list[str]]
    train_sentences: list[list[str]]
    test_sentences: list[list[str]]
    train_core_sentences: list[list[str]]
    dev_sentences: list[list[str]]
    train_vocab: set[str]
    raw_train_token_counts: Counter[str]
    raw_test_token_count: int
    raw_test_oov_token_count: int
    unk_min_freq: int
    lowercase: bool
    corpus_path: str
    text_column: str
    seed: int
    dev_seed: int
    test_ratio: float
    dev_ratio: float
    max_docs: int | None

    def stats_dict(self) -> dict:
        return {
            "num_docs_total": self.texts_total,
            "num_docs_train": len(self.train_docs),
            "num_docs_test": len(self.test_docs),
            "num_sentences_train_total": len(self.train_sentences),
            "num_sentences_train_core": len(self.train_core_sentences),
            "num_sentences_dev": len(self.dev_sentences),
            "num_sentences_test": len(self.test_sentences),
            "num_tokens_train_excluding_eos": int(sum(len(s) for s in self.train_sentences)),
            "num_tokens_train_core_excluding_eos": int(sum(len(s) for s in self.train_core_sentences)),
            "num_tokens_dev_excluding_eos": int(sum(len(s) for s in self.dev_sentences)),
            "num_tokens_test_excluding_eos": int(sum(len(s) for s in self.test_sentences)),
            "num_predicted_events_test": int(count_predicted_events(self.test_sentences)),
            "num_predicted_events_dev": int(count_predicted_events(self.dev_sentences)),
        }

    def vocab_dict(self) -> dict:
        raw_train_unique = len(self.raw_train_token_counts)
        raw_test_oov_rate = (
            self.raw_test_oov_token_count / self.raw_test_token_count if self.raw_test_token_count else 0.0
        )
        return {
            "train_vocab_size_after_unk_threshold": len(self.train_vocab),
            "unk_token": UNK_TOKEN,
            "eos_token": END_TOKEN,
            "raw_train_unique_tokens_before_threshold": raw_train_unique,
            "raw_test_oov_token_count_vs_train_vocab": int(self.raw_test_oov_token_count),
            "raw_test_token_count": int(self.raw_test_token_count),
            "raw_test_oov_rate_vs_train_vocab": raw_test_oov_rate,
            "unk_count_in_train_after_mapping": int(sum(tok == UNK_TOKEN for s in self.train_sentences for tok in s)),
        }


def prepare_lm_splits(
    corpus_path: str = "data/raw/corpus.csv",
    text_column: str = "text",
    test_ratio: float = 0.2,
    dev_ratio: float = 0.1,
    seed: int = 42,
    dev_seed: int = 43,
    lowercase: bool = True,
    unk_min_freq: int = 2,
    max_docs: int | None = None,
) -> PreparedLMSplits:
    df = load_corpus_csv(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")

    texts = df[text_column].fillna("").astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if max_docs is not None:
        texts = texts[:max_docs]
    if len(texts) < 2:
        raise ValueError("Need at least 2 non-empty documents for a train/test split.")

    train_docs, test_docs = split_docs(texts, test_ratio=test_ratio, seed=seed)

    train_sents_raw = docs_to_tokenized_sentences(train_docs, lowercase=lowercase)
    test_sents_raw = docs_to_tokenized_sentences(test_docs, lowercase=lowercase)
    if not train_sents_raw:
        raise ValueError("No tokenized training sentences were produced.")
    if not test_sents_raw:
        raise ValueError("No tokenized test sentences were produced.")

    raw_train_token_counts = Counter(tok for s in train_sents_raw for tok in s)
    raw_test_token_count = sum(len(s) for s in test_sents_raw)
    raw_test_oov_token_count = sum(
        1 for s in test_sents_raw for tok in s if tok not in raw_train_token_counts
    )

    train_vocab_counts = build_train_vocab(train_sents_raw, min_freq=unk_min_freq)
    train_vocab = set(train_vocab_counts.keys())
    train_vocab.add(UNK_TOKEN)

    train_sents = [map_sentence_to_vocab(s, train_vocab) for s in train_sents_raw]
    test_sents = [map_sentence_to_vocab(s, train_vocab) for s in test_sents_raw]

    train_core_sents, dev_sents = split_sentences(train_sents, dev_ratio=dev_ratio, seed=dev_seed)

    return PreparedLMSplits(
        texts_total=len(texts),
        train_docs=train_docs,
        test_docs=test_docs,
        train_sentences_raw=train_sents_raw,
        test_sentences_raw=test_sents_raw,
        train_sentences=train_sents,
        test_sentences=test_sents,
        train_core_sentences=train_core_sents,
        dev_sentences=dev_sents,
        train_vocab=train_vocab,
        raw_train_token_counts=raw_train_token_counts,
        raw_test_token_count=raw_test_token_count,
        raw_test_oov_token_count=raw_test_oov_token_count,
        unk_min_freq=unk_min_freq,
        lowercase=lowercase,
        corpus_path=corpus_path,
        text_column=text_column,
        seed=seed,
        dev_seed=dev_seed,
        test_ratio=test_ratio,
        dev_ratio=dev_ratio,
        max_docs=max_docs,
    )

