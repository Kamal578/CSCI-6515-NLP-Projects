from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _write_squad_json(path: Path, examples: list[dict[str, str]]) -> None:
    payload = {
        "data": [
            {
                "title": "demo",
                "paragraphs": [
                    {
                        "context": example["context"],
                        "qas": [
                            {
                                "id": example["id"],
                                "question": example["question"],
                                "answers": [
                                    {
                                        "text": example["answer_text"],
                                        "answer_start": example["answer_start"],
                                    }
                                ],
                            }
                        ],
                    }
                    for example in examples
                ],
            }
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_glove_file(path: Path) -> None:
    tokens = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "what",
        "comes",
        "after",
        "which",
        "token",
        "is",
        "first",
        "?",
    ]
    rows = []
    for row_idx, token in enumerate(tokens, start=1):
        vector = " ".join(f"{(row_idx + col_idx) / 1000:.6f}" for col_idx in range(100))
        rows.append(f"{token} {vector}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


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
        "comes",
        "after",
        "which",
        "token",
        "is",
        "first",
        "alpha",
        "beta",
        "gamma",
        "delta",
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


@pytest.mark.parametrize("variant", ["glove", "bert"])
def test_project4_task2_cli_smoke_runs_with_local_resources(tmp_path: Path, variant: str):
    if variant == "bert":
        bert_dir = _make_tiny_bert_dir(tmp_path)
    else:
        bert_dir = None

    train_examples = [
        {
            "id": "train-1",
            "context": "alpha beta gamma delta",
            "question": "what comes after alpha?",
            "answer_text": "beta",
            "answer_start": 6,
        },
        {
            "id": "train-2",
            "context": "beta gamma delta alpha",
            "question": "which token is first?",
            "answer_text": "beta",
            "answer_start": 0,
        },
    ]
    val_examples = [
        {
            "id": "val-1",
            "context": "alpha beta gamma delta",
            "question": "what comes after alpha?",
            "answer_text": "beta",
            "answer_start": 6,
        }
    ]
    train_json = tmp_path / "train.json"
    val_json = tmp_path / "val.json"
    _write_squad_json(train_json, train_examples)
    _write_squad_json(val_json, val_examples)

    glove_path = tmp_path / "glove.6B.100d.txt"
    _write_glove_file(glove_path)

    out_dir = tmp_path / "outputs"
    cache_dir = tmp_path / "cache"
    cmd = [
        sys.executable,
        "-m",
        "src.project4_task2_reading_comprehension",
        "--variant",
        variant,
        "--train_json",
        str(train_json),
        "--val_json",
        str(val_json),
        "--cache_dir",
        str(cache_dir),
        "--out_dir",
        str(out_dir),
        "--glove_path",
        str(glove_path),
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--eval_batch_size",
        "2",
        "--hidden_size",
        "8",
        "--context_window_words",
        "8",
        "--doc_stride_words",
        "4",
        "--max_question_words",
        "8",
        "--device",
        "cpu",
        "--smoke",
    ]
    if bert_dir is not None:
        cmd.extend(["--bert_model_name", str(bert_dir)])

    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    assert (out_dir / variant / "summary.json").exists()
    assert (out_dir / variant / "history.csv").exists()
    assert (out_dir / variant / "predictions.json").exists()
    assert (out_dir / "comparison.csv").exists()
    assert (out_dir / "report_notes.md").exists()
