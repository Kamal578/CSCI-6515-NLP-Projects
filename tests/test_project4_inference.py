from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from src.project4_inference import load_qa_bundle, predict_qa_answer
from src.project4_task2_qa_model import FrozenBertBidafQaModel


def _make_tiny_bert_dir(tmp_path: Path) -> Path:
    import transformers

    model_dir = tmp_path / "tiny_qa_bert"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "azerbaijan",
        "capital",
        "is",
        "baku",
        "what",
        "?",
    ]
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(model_dir)

    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = transformers.BertModel(config)
    model.save_pretrained(model_dir)
    return model_dir


def test_load_qa_bundle_and_predict_answer_with_local_checkpoint(tmp_path: Path):
    model_dir = _make_tiny_bert_dir(tmp_path)
    variant_dir = tmp_path / "bert"
    variant_dir.mkdir(parents=True, exist_ok=True)

    model = FrozenBertBidafQaModel(
        bert_model_name=str(model_dir),
        cache_dir=None,
        projection_dim=16,
        hidden_size=8,
        dropout=0.1,
        bert_max_length=64,
    )
    checkpoint_path = variant_dir / "model.pt"
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint_path)

    summary = {
        "variant": "bert",
        "config": {
            "bert_model_name": str(model_dir),
            "cache_dir": None,
            "embedding_dim": 16,
            "hidden_size": 8,
            "dropout": 0.1,
            "bert_max_length": 64,
            "max_question_words": 8,
            "context_window_words": 16,
            "doc_stride_words": 8,
            "max_answer_words": 4,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
        },
    }
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bundle = load_qa_bundle(variant_dir, device="cpu")
    result = predict_qa_answer(
        bundle,
        context="Azerbaijan capital is Baku.",
        question="What is capital?",
    )

    assert result["variant"] == "bert"
    assert isinstance(result["answer"], str)
    assert result["num_windows"] >= 1
