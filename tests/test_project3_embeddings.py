from __future__ import annotations

import numpy as np

from src.project3_embeddings import EmbeddingSpace, evaluate_analogies, iter_text_vectors


def test_neighbors_and_analogy_smoke():
    words = ["man", "woman", "king", "queen", "apple"]
    vecs = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    sp = EmbeddingSpace(words=words, vectors=vecs)
    nns = sp.nearest_neighbors("king", top_k=2)
    assert len(nns) == 2

    rows, summary = evaluate_analogies(sp, [("man", "king", "woman", "queen")], top_k=3)
    assert len(rows) == 1
    assert summary["num_total"] == 1


def test_iter_text_vectors_skips_word2vec_style_header(tmp_path):
    path = tmp_path / "vectors.txt"
    path.write_text("2 3\nalpha 1 2 3\nbeta 4 5 6\n", encoding="utf-8")

    rows = list(iter_text_vectors(path))
    assert len(rows) == 2
    assert rows[0][0] == "alpha"
    assert rows[0][1].shape == (3,)
