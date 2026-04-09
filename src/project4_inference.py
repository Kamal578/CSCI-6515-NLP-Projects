from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .project3_common import read_json
from .project4_task1_sentiment import normalize_azerbaijani_text, resolve_device
from .project4_task2_qa_data import (
    SquadExample,
    extract_answer_text,
    make_bert_context_trimmer,
    make_collate_fn,
    tokenize_qa_example,
    build_windowed_examples,
)
from .project4_task2_qa_metrics import select_best_span
from .project4_task2_qa_model import FrozenBertBidafQaModel, GloveBidafQaModel


@dataclass(slots=True)
class SentimentBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    label_names: list[str]
    summary: dict[str, Any]


@dataclass(slots=True)
class QaBundle:
    model: Any
    device: torch.device
    summary: dict[str, Any]
    vocab: dict[str, int] | None
    variant: str


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_sentiment_bundle(model_root: str | Path, device: str = "auto") -> SentimentBundle:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for sentiment inference.") from exc

    model_path = Path(model_root)
    summary_path = model_path.parent / "summary.json" if model_path.name == "model" else model_path / "summary.json"
    if model_path.name != "model":
        model_path = model_path / "model"
    summary = read_json(summary_path)
    runtime_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(runtime_device)
    model.eval()
    labels = summary.get("model", {}).get("labels")
    if not labels:
        id2label = getattr(model.config, "id2label", {}) or {}
        labels = [id2label[str(idx)] if str(idx) in id2label else id2label.get(idx, str(idx)) for idx in range(model.config.num_labels)]
    return SentimentBundle(
        model=model,
        tokenizer=tokenizer,
        device=runtime_device,
        label_names=[str(label) for label in labels],
        summary=summary,
    )


def predict_sentiment(bundle: SentimentBundle, text: str, max_length: int | None = None) -> dict[str, Any]:
    cleaned = normalize_azerbaijani_text(text)
    if not cleaned:
        raise ValueError("Input text is empty after normalization.")
    max_len = max_length or int(bundle.summary.get("model", {}).get("max_length", 192))
    inputs = bundle.tokenizer(
        cleaned,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = bundle.model(**inputs).logits.detach().cpu()
    probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    predicted_id = int(logits.argmax(dim=-1).item())
    return {
        "normalized_text": cleaned,
        "predicted_label_id": predicted_id,
        "predicted_label": bundle.label_names[predicted_id],
        "confidence": float(max(probabilities)),
        "probabilities": {
            bundle.label_names[idx]: float(probability)
            for idx, probability in enumerate(probabilities)
        },
    }


def load_qa_bundle(variant_root: str | Path, device: str = "auto") -> QaBundle:
    variant_path = Path(variant_root)
    summary_path = variant_path / "summary.json"
    summary = _load_json(summary_path)
    if not summary:
        raise FileNotFoundError(f"Missing QA summary at {summary_path}")
    variant = str(summary.get("variant", variant_path.name))
    config = summary.get("config", {})
    checkpoint_path = Path(summary.get("artifacts", {}).get("checkpoint", variant_path / "model.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing QA checkpoint at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if variant == "glove":
        vocab_path = variant_path / "vocab.json"
        vocab = _load_json(vocab_path)
        if not vocab:
            raise FileNotFoundError(f"Missing GloVe vocab at {vocab_path}")
        embedding_matrix = state_dict["embedding.weight"]
        model = GloveBidafQaModel(
            embedding_matrix=embedding_matrix,
            hidden_size=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
        )
    else:
        vocab = None
        model = FrozenBertBidafQaModel(
            bert_model_name=str(config["bert_model_name"]),
            cache_dir=config.get("cache_dir"),
            projection_dim=int(config["embedding_dim"]),
            hidden_size=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
            bert_max_length=int(config["bert_max_length"]),
        )
    model.load_state_dict(state_dict)
    runtime_device = resolve_device(device)
    model = model.to(runtime_device)
    model.eval()
    return QaBundle(
        model=model,
        device=runtime_device,
        summary=summary,
        vocab=vocab,
        variant=variant,
    )


def predict_qa_answer(bundle: QaBundle, context: str, question: str) -> dict[str, Any]:
    config = bundle.summary.get("config", {})
    normalized_context = context.lower().strip()
    normalized_question = question.lower().strip()
    if not normalized_context or not normalized_question:
        raise ValueError("Context and question must both be non-empty.")

    example = SquadExample(
        question_id="interactive",
        title="interactive",
        context=normalized_context,
        question=normalized_question,
        answers=[],
        answer_starts=[],
    )
    tokenized = tokenize_qa_example(example, max_question_words=int(config["max_question_words"]))
    context_trimmer = None
    if bundle.variant == "bert":
        context_trimmer = make_bert_context_trimmer(
            bert_model_name=str(config["bert_model_name"]),
            cache_dir=config.get("cache_dir"),
            bert_max_length=int(config["bert_max_length"]),
        )
    windows = build_windowed_examples(
        [tokenized],
        context_window_words=int(config["context_window_words"]),
        doc_stride_words=int(config["doc_stride_words"]),
        is_train=False,
        context_word_trimmer=context_trimmer,
    )
    if not windows:
        return {
            "answer": "",
            "score": float("-inf"),
            "num_windows": 0,
            "message": "No valid windows were created for the supplied context/question pair.",
        }

    collate = make_collate_fn(bundle.variant, vocab=bundle.vocab)
    batch = collate(windows).to(bundle.device)
    with torch.no_grad():
        start_logits, end_logits = bundle.model(batch)

    best_answer = ""
    best_score = float("-inf")
    best_start = 0
    best_end = 0
    for row_idx in range(len(windows)):
        local_start, local_end, score = select_best_span(
            start_logits[row_idx].detach().cpu(),
            end_logits[row_idx].detach().cpu(),
            batch.context_mask[row_idx].detach().cpu(),
            max_answer_words=int(config["max_answer_words"]),
        )
        global_start = batch.context_start_words[row_idx] + local_start
        global_end = batch.context_start_words[row_idx] + local_end
        answer_text = extract_answer_text(tokenized, global_start, global_end)
        if score > best_score:
            best_score = score
            best_answer = answer_text
            best_start = global_start
            best_end = global_end

    return {
        "answer": best_answer,
        "score": float(best_score),
        "start_word": best_start,
        "end_word": best_end,
        "num_windows": len(windows),
        "variant": bundle.variant,
    }
