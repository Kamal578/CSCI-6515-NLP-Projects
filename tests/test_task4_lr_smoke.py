from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from src.task4_dot_model import fit_dot_model, predict_dot_labels, save_model_artifact, load_model_artifact


def test_lr_smoke_train_predict_and_save(tmp_path: Path):
    X = [
        {"is_decimal_pattern": True, "next_nonspace_is_upper": False, "joined_no_space_around_dot": True},
        {"is_decimal_pattern": False, "next_nonspace_is_upper": True, "joined_no_space_around_dot": False},
        {"is_decimal_pattern": True, "next_nonspace_is_upper": False, "joined_no_space_around_dot": True},
        {"is_decimal_pattern": False, "next_nonspace_is_upper": True, "joined_no_space_around_dot": False},
    ]
    y = [0, 1, 0, 1]

    for reg in ("l1", "l2"):
        model = fit_dot_model(X, y, regularization=reg, c_value=1.0)
        preds, probs = predict_dot_labels(model, X)
        assert len(preds) == len(y)
        assert len(probs) == len(y)
        assert set(int(p) for p in preds.tolist()).issubset({0, 1})

        out = tmp_path / f"{reg}.joblib"
        save_model_artifact(model, out)
        loaded = load_model_artifact(out)
        preds2, probs2 = predict_dot_labels(loaded, X)
        assert preds.tolist() == preds2.tolist()
        assert len(probs2) == len(y)
