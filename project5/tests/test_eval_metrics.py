from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import load_src_module


def test_compute_metrics_basic() -> None:
    module = load_src_module("05_eval_metrics.py", "eval_metrics_mod_basic")

    records = [
        {
            "question": "Q1",
            "baseline_answer": "short answer",
            "rag_answer": "longer grounded answer text",
            "retrieved_files": ["AAPL_2022.txt", "MSFT_2021.txt"],
        },
        {
            "question": "Q2",
            "baseline_answer": "",
            "rag_answer": "another answer",
            "retrieved_files": [],
        },
    ]

    metrics = module.compute_metrics(records)
    assert metrics["question_count"] == 2
    assert metrics["baseline_non_empty_rate"] == 0.5
    assert metrics["rag_non_empty_rate"] == 1.0
    assert metrics["avg_retrieved_sources_per_question"] == 1.0
    assert metrics["questions_with_sources_rate"] == 0.5
    assert metrics["unique_retrieved_sources_count"] == 2


def test_latest_eval_json_selects_latest(tmp_path: Path) -> None:
    module = load_src_module("05_eval_metrics.py", "eval_metrics_mod_latest")

    out = tmp_path / "eval"
    out.mkdir()
    older = out / "rag_eval_20240101T000000Z.json"
    newer = out / "rag_eval_20240102T000000Z.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    selected = module.latest_eval_json(out)
    assert selected == newer


def test_load_eval_payload_rejects_bad_format(tmp_path: Path) -> None:
    module = load_src_module("05_eval_metrics.py", "eval_metrics_mod_payload")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected evaluation file format"):
        module.load_eval_payload(bad)
