from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("torch")

from src.project4_task2_qa_data import QaBatch
from src.project4_task2_qa_model import FrozenBertBidafQaModel, GloveBidafQaModel


def _make_tiny_bert_dir(tmp_path: Path) -> Path:
    transformers = pytest.importorskip("transformers")

    model_dir = tmp_path / "tiny_bert"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "what",
        "is",
        "alpha",
        "beta",
        "gamma",
        "?",
        "the",
        "answer",
        "token",
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


def test_glove_bidaf_forward_shapes():
    embedding_matrix = torch.randn(20, 100)
    model = GloveBidafQaModel(embedding_matrix=embedding_matrix, hidden_size=8, dropout=0.1)
    batch = QaBatch(
        question_ids=["q1", "q2"],
        window_ids=["w1", "w2"],
        context_start_words=[0, 0],
        question_words=[["what", "is", "alpha"], ["what", "is"]],
        context_words=[["alpha", "beta", "gamma", "delta"], ["beta", "gamma"]],
        question_mask=torch.tensor([[True, True, True], [True, True, False]]),
        context_mask=torch.tensor([[True, True, True, True], [True, True, False, False]]),
        question_token_ids=torch.tensor([[1, 2, 3], [1, 2, 0]]),
        context_token_ids=torch.tensor([[4, 5, 6, 7], [5, 6, 0, 0]]),
        start_positions=torch.tensor([1, 0]),
        end_positions=torch.tensor([2, 1]),
    )

    start_logits, end_logits = model(batch)
    assert start_logits.shape == (2, 4)
    assert end_logits.shape == (2, 4)
    assert start_logits[1, 2].item() < -1e10


def test_frozen_bert_bidaf_forward_shapes(tmp_path: Path):
    model_dir = _make_tiny_bert_dir(tmp_path)
    model = FrozenBertBidafQaModel(
        bert_model_name=str(model_dir),
        cache_dir=None,
        projection_dim=100,
        hidden_size=8,
        dropout=0.1,
        bert_max_length=64,
    )
    batch = QaBatch(
        question_ids=["q1", "q2"],
        window_ids=["w1", "w2"],
        context_start_words=[0, 0],
        question_words=[["what", "is", "alpha", "?"], ["what", "is", "beta", "?"]],
        context_words=[["alpha", "beta", "gamma"], ["beta", "gamma"]],
        question_mask=torch.tensor([[True, True, True, True], [True, True, True, True]]),
        context_mask=torch.tensor([[True, True, True], [True, True, False]]),
        start_positions=torch.tensor([1, 0]),
        end_positions=torch.tensor([1, 0]),
    )

    start_logits, end_logits = model(batch)
    assert start_logits.shape == (2, 3)
    assert end_logits.shape == (2, 3)
