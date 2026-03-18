from __future__ import annotations

import numpy as np

from src.project3_task1_matrices import _build_cooccurrence


def test_cooccurrence_shape_and_counts():
    vocab = {"a": 0, "b": 1, "c": 2}
    sents = [["a", "b", "c"], ["b", "a"]]
    mat = _build_cooccurrence(sents, vocab_to_idx=vocab, window=1)
    assert mat.shape == (3, 3)
    assert int(mat[0, 1]) > 0  # a near b
    assert int(mat[1, 0]) > 0  # b near a
    assert np.all(np.diag(mat) == 0)
