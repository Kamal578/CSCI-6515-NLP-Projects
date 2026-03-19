from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from src.project3_task5_dl import build_sequence_model


@pytest.mark.parametrize("arch", ["rnn", "birnn", "lstm"])
def test_sequence_model_forward_smoke(arch: str):
    emb = np.random.randn(12, 16).astype(np.float32)
    model = build_sequence_model(
        architecture=arch,
        embedding_matrix=emb,
        num_classes=3,
        hidden_size=8,
        dropout=0.1,
        trainable_embeddings=False,
    )
    x = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([3, 2], dtype=torch.long)
    logits = model(x, lengths)
    assert logits.shape == (2, 3)
