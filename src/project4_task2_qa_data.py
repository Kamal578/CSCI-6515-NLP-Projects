from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
GLOVE_6B_100D_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(slots=True)
class SquadExample:
    question_id: str
    title: str
    context: str
    question: str
    answers: list[str]
    answer_starts: list[int]


@dataclass(slots=True)
class TokenizedQaExample:
    question_id: str
    title: str
    context: str
    question: str
    context_words: list[str]
    context_offsets: list[tuple[int, int]]
    question_words: list[str]
    answer_texts: list[str]
    answer_word_spans: list[tuple[int, int]]


@dataclass(slots=True)
class WindowedQaExample:
    question_id: str
    title: str
    context_words: list[str]
    question_words: list[str]
    context_start_word: int
    context_end_word: int
    answer_texts: list[str]
    start_position: int | None
    end_position: int | None

    @property
    def window_id(self) -> str:
        return f"{self.question_id}::{self.context_start_word}:{self.context_end_word}"


@dataclass
class QaBatch:
    question_ids: list[str]
    window_ids: list[str]
    context_start_words: list[int]
    question_words: list[list[str]]
    context_words: list[list[str]]
    question_mask: torch.Tensor
    context_mask: torch.Tensor
    question_token_ids: torch.Tensor | None = None
    context_token_ids: torch.Tensor | None = None
    start_positions: torch.Tensor | None = None
    end_positions: torch.Tensor | None = None

    def to(self, device: torch.device) -> QaBatch:
        if self.question_mask is not None:
            self.question_mask = self.question_mask.to(device)
        if self.context_mask is not None:
            self.context_mask = self.context_mask.to(device)
        if self.question_token_ids is not None:
            self.question_token_ids = self.question_token_ids.to(device)
        if self.context_token_ids is not None:
            self.context_token_ids = self.context_token_ids.to(device)
        if self.start_positions is not None:
            self.start_positions = self.start_positions.to(device)
        if self.end_positions is not None:
            self.end_positions = self.end_positions.to(device)
        return self


class QaWindowDataset(Dataset):
    def __init__(self, windows: list[WindowedQaExample]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> WindowedQaExample:
        return self.windows[idx]


def word_tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in WORD_RE.finditer(text):
        words.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return words, offsets


def map_answer_chars_to_word_span(
    offsets: list[tuple[int, int]],
    answer_start: int,
    answer_text: str,
) -> tuple[int, int]:
    answer_end = answer_start + len(answer_text)
    overlapping = [
        idx
        for idx, (token_start, token_end) in enumerate(offsets)
        if token_start < answer_end and token_end > answer_start
    ]
    if not overlapping:
        raise ValueError(f"Answer span ({answer_start}, {answer_end}) does not overlap any token offsets.")
    return overlapping[0], overlapping[-1]


def extract_answer_text(example: TokenizedQaExample, start_word: int, end_word: int) -> str:
    if not example.context_offsets:
        return ""
    start_word = max(0, min(start_word, len(example.context_offsets) - 1))
    end_word = max(start_word, min(end_word, len(example.context_offsets) - 1))
    start_char = example.context_offsets[start_word][0]
    end_char = example.context_offsets[end_word][1]
    return example.context[start_char:end_char].strip()


def _normalize_text(text: str) -> str:
    return text.lower()


def _load_squad_json(path: str | Path) -> list[SquadExample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[SquadExample] = []
    for article in payload.get("data", []):
        title = _normalize_text(article.get("title", ""))
        for paragraph in article.get("paragraphs", []):
            context = _normalize_text(paragraph["context"])
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                examples.append(
                    SquadExample(
                        question_id=str(qa["id"]),
                        title=title,
                        context=context,
                        question=_normalize_text(qa["question"]),
                        answers=[_normalize_text(a["text"]) for a in answers],
                        answer_starts=[int(a["answer_start"]) for a in answers],
                    )
                )
    return examples


def load_squad_splits(
    train_json: str | None,
    val_json: str | None,
    cache_dir: str | None = None,
) -> tuple[list[SquadExample], list[SquadExample]]:
    if train_json and val_json:
        return _load_squad_json(train_json), _load_squad_json(val_json)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for loading SQuAD automatically. "
            "Install it or pass --train_json and --val_json."
        ) from exc

    dataset = load_dataset("squad", cache_dir=cache_dir)
    train_examples = [
        SquadExample(
            question_id=str(row["id"]),
            title=_normalize_text(row.get("title", "")),
            context=_normalize_text(row["context"]),
            question=_normalize_text(row["question"]),
            answers=[_normalize_text(text) for text in row["answers"]["text"]],
            answer_starts=[int(v) for v in row["answers"]["answer_start"]],
        )
        for row in dataset["train"]
    ]
    val_examples = [
        SquadExample(
            question_id=str(row["id"]),
            title=_normalize_text(row.get("title", "")),
            context=_normalize_text(row["context"]),
            question=_normalize_text(row["question"]),
            answers=[_normalize_text(text) for text in row["answers"]["text"]],
            answer_starts=[int(v) for v in row["answers"]["answer_start"]],
        )
        for row in dataset["validation"]
    ]
    return train_examples, val_examples


def maybe_limit_examples(
    examples: list[SquadExample],
    max_examples: int | None,
    seed: int,
) -> list[SquadExample]:
    if max_examples is None or max_examples >= len(examples):
        return examples
    rng = random.Random(seed)
    idxs = list(range(len(examples)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:max_examples])
    return [examples[idx] for idx in chosen]


def tokenize_qa_example(example: SquadExample, max_question_words: int) -> TokenizedQaExample:
    context_words, context_offsets = word_tokenize_with_offsets(example.context)
    question_words, _ = word_tokenize_with_offsets(example.question)
    question_words = question_words[:max_question_words]

    answer_word_spans: list[tuple[int, int]] = []
    for answer_start, answer_text in zip(example.answer_starts, example.answers):
        try:
            answer_word_spans.append(map_answer_chars_to_word_span(context_offsets, answer_start, answer_text))
        except ValueError:
            continue

    return TokenizedQaExample(
        question_id=example.question_id,
        title=example.title,
        context=example.context,
        question=example.question,
        context_words=context_words,
        context_offsets=context_offsets,
        question_words=question_words,
        answer_texts=example.answers,
        answer_word_spans=answer_word_spans,
    )


def make_bert_context_trimmer(
    bert_model_name: str,
    cache_dir: str | None,
    bert_max_length: int,
) -> Callable[[list[str]], list[str]]:
    try:
        from transformers import BertTokenizerFast
    except ImportError as exc:
        raise ImportError("transformers is required for the bert variant.") from exc

    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name, cache_dir=cache_dir)
    def trim(words: list[str]) -> list[str]:
        current = list(words)
        while current:
            encoded = tokenizer(
                current,
                is_split_into_words=True,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if len(encoded["input_ids"]) <= bert_max_length:
                return current
            current = current[:-1]
        return []

    return trim


def build_windowed_examples(
    tokenized_examples: list[TokenizedQaExample],
    context_window_words: int,
    doc_stride_words: int,
    is_train: bool,
    context_word_trimmer: Callable[[list[str]], list[str]] | None = None,
) -> list[WindowedQaExample]:
    windows: list[WindowedQaExample] = []
    for example in tokenized_examples:
        if not example.context_words or not example.question_words:
            continue

        total_words = len(example.context_words)
        window_start = 0
        while window_start < total_words:
            window_end = min(total_words, window_start + context_window_words)
            window_words = example.context_words[window_start:window_end]
            if context_word_trimmer is not None:
                trimmed_words = context_word_trimmer(window_words)
            else:
                trimmed_words = window_words
            if not trimmed_words:
                if window_end >= total_words:
                    break
                window_start += doc_stride_words
                continue

            effective_end = window_start + len(trimmed_words)
            if is_train:
                kept = False
                for gold_start, gold_end in example.answer_word_spans:
                    if window_start <= gold_start and gold_end < effective_end:
                        windows.append(
                            WindowedQaExample(
                                question_id=example.question_id,
                                title=example.title,
                                context_words=trimmed_words,
                                question_words=example.question_words,
                                context_start_word=window_start,
                                context_end_word=effective_end,
                                answer_texts=example.answer_texts,
                                start_position=gold_start - window_start,
                                end_position=gold_end - window_start,
                            )
                        )
                        kept = True
                if not kept:
                    pass
            else:
                windows.append(
                    WindowedQaExample(
                        question_id=example.question_id,
                        title=example.title,
                        context_words=trimmed_words,
                        question_words=example.question_words,
                        context_start_word=window_start,
                        context_end_word=effective_end,
                        answer_texts=example.answer_texts,
                        start_position=None,
                        end_position=None,
                    )
                )

            if window_end >= total_words:
                break
            window_start += doc_stride_words
    return windows


def build_word_vocab(
    windows: list[WindowedQaExample],
    min_freq: int = 1,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for example in windows:
        counter.update(example.context_words)
        counter.update(example.question_words)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counter.items()):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_words(words: list[str], vocab: dict[str, int]) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(word, unk_id) for word in words]


def make_collate_fn(
    variant: str,
    vocab: dict[str, int] | None = None,
) -> Callable[[list[WindowedQaExample]], QaBatch]:
    if variant == "glove" and vocab is None:
        raise ValueError("vocab is required for the glove variant.")

    def collate(batch: list[WindowedQaExample]) -> QaBatch:
        question_lengths = [len(item.question_words) for item in batch]
        context_lengths = [len(item.context_words) for item in batch]
        max_q = max(question_lengths)
        max_c = max(context_lengths)

        question_mask = torch.zeros(len(batch), max_q, dtype=torch.bool)
        context_mask = torch.zeros(len(batch), max_c, dtype=torch.bool)
        question_token_ids = None
        context_token_ids = None
        if variant == "glove":
            assert vocab is not None
            question_token_ids = torch.zeros(len(batch), max_q, dtype=torch.long)
            context_token_ids = torch.zeros(len(batch), max_c, dtype=torch.long)

        start_positions: list[int] = []
        end_positions: list[int] = []
        has_labels = batch[0].start_position is not None and batch[0].end_position is not None

        for idx, item in enumerate(batch):
            q_len = len(item.question_words)
            c_len = len(item.context_words)
            question_mask[idx, :q_len] = True
            context_mask[idx, :c_len] = True

            if variant == "glove":
                question_token_ids[idx, :q_len] = torch.tensor(encode_words(item.question_words, vocab), dtype=torch.long)
                context_token_ids[idx, :c_len] = torch.tensor(encode_words(item.context_words, vocab), dtype=torch.long)

            if has_labels:
                start_positions.append(int(item.start_position))
                end_positions.append(int(item.end_position))

        return QaBatch(
            question_ids=[item.question_id for item in batch],
            window_ids=[item.window_id for item in batch],
            context_start_words=[item.context_start_word for item in batch],
            question_words=[item.question_words for item in batch],
            context_words=[item.context_words for item in batch],
            question_mask=question_mask,
            context_mask=context_mask,
            question_token_ids=question_token_ids,
            context_token_ids=context_token_ids,
            start_positions=torch.tensor(start_positions, dtype=torch.long) if has_labels else None,
            end_positions=torch.tensor(end_positions, dtype=torch.long) if has_labels else None,
        )

    return collate


def ensure_glove_path(glove_path: str | None, cache_dir: str | Path) -> Path:
    if glove_path:
        resolved = Path(glove_path)
        if not resolved.exists():
            raise FileNotFoundError(f"GloVe file not found: {resolved}")
        return resolved

    cache_dir = Path(cache_dir)
    glove_dir = cache_dir / "glove"
    glove_dir.mkdir(parents=True, exist_ok=True)
    target = glove_dir / "glove.6B.100d.txt"
    if target.exists():
        return target

    zip_path = glove_dir / "glove.6B.zip"
    response = requests.get(GLOVE_6B_100D_URL, timeout=60, stream=True)
    response.raise_for_status()
    with zip_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("glove.6B.100d.txt", glove_dir)
    return target


def load_glove_embedding_matrix(
    vocab: dict[str, int],
    glove_path: str | Path,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)

    hits = 0
    glove_file = Path(glove_path)
    with glove_file.open("r", encoding="utf-8") as f:
        for line in f:
            pieces = line.rstrip("\n").split(" ")
            if len(pieces) != embedding_dim + 1:
                continue
            token = pieces[0]
            if token not in vocab:
                continue
            vector = np.asarray(pieces[1:], dtype=np.float32)
            matrix[vocab[token]] = vector
            hits += 1

    stats = {
        "path": str(glove_file),
        "embedding_dim": embedding_dim,
        "vocab_size": len(vocab),
        "matched_tokens": hits,
        "coverage": hits / max(1, len(vocab) - 2),
    }
    return matrix, stats
