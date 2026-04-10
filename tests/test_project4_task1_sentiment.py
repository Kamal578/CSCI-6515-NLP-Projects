from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.project4_inference import load_sentiment_bundle, predict_sentiment
from src.project4_task1_sentiment import normalize_azerbaijani_text, score_to_label


def _make_tiny_bert_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny_sentiment_bert"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "bu",
        "tetbiq",
        "çox",
        "ela",
        "pis",
        "amma",
        "gec",
        "aci",
        "acilir",
        "rahatdir",
        ".",
    ]
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(model_dir)

    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=5,
    )
    model = transformers.BertForSequenceClassification(config)
    model.save_pretrained(model_dir)
    return model_dir


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["content", "score", "upvotes"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_normalize_azerbaijani_text_cleans_noise_without_losing_letters():
    text = "  Çooox   gözəl\u200b tətbiqdir!!!  "
    normalized = normalize_azerbaijani_text(text)
    assert normalized == "Çooox gözəl tətbiqdir!!"
    assert "ə" in normalized


def test_score_to_label_supports_multiple_label_modes():
    assert score_to_label(5, "score5") == (4, "5 stars")
    assert score_to_label(3, "sentiment3") == (1, "neutral")
    assert score_to_label(1, "binary") == (0, "negative")
    assert score_to_label(3, "binary") == (None, None)


def test_project4_task1_cli_smoke_runs_with_local_resources(tmp_path: Path):
    model_dir = _make_tiny_bert_dir(tmp_path)
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    out_dir = tmp_path / "outputs"

    train_rows = [
        {"content": "Bu tətbiq çox ela", "score": 5, "upvotes": 2},
        {"content": "Bu tətbiq pis", "score": 1, "upvotes": 0},
        {"content": "Tetbiq rahatdir", "score": 4, "upvotes": 1},
        {"content": "Tetbiq gec acilir", "score": 2, "upvotes": 0},
        {"content": "Ela amma gec acilir", "score": 3, "upvotes": 0},
        {"content": "Çox ela tetbiq", "score": 5, "upvotes": 3},
        {"content": "Pis amma rahatdir", "score": 2, "upvotes": 0},
        {"content": "Bu tətbiq ela", "score": 4, "upvotes": 1},
        {"content": "Tetbiq çox pis", "score": 1, "upvotes": 0},
        {"content": "Rahatdir amma gec", "score": 3, "upvotes": 0},
    ]
    test_rows = [
        {"content": "Bu tətbiq ela", "score": 5, "upvotes": 1},
        {"content": "Bu tətbiq pis", "score": 1, "upvotes": 0},
        {"content": "Gec acilir amma rahatdir", "score": 3, "upvotes": 0},
    ]
    _write_csv(train_csv, train_rows)
    _write_csv(test_csv, test_rows)

    cmd = [
        sys.executable,
        "-m",
        "src.project4_task1_sentiment",
        "--train_file",
        str(train_csv),
        "--test_file",
        str(test_csv),
        "--model_name",
        str(model_dir),
        "--output_dir",
        str(out_dir),
        "--cache_dir",
        str(tmp_path / "cache"),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--eval_batch_size",
        "2",
        "--max_length",
        "48",
        "--validation_ratio",
        "0.2",
        "--log_every_steps",
        "1",
        "--smoke",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "model").exists()
    assert (out_dir / "test_predictions.json").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"]["train_examples"] > 0
    assert summary["model"]["num_labels"] == 5

    bundle = load_sentiment_bundle(out_dir, device="cpu")
    prediction = predict_sentiment(bundle, "Bu tətbiq çox ela")
    assert prediction["predicted_label"] in bundle.label_names
    assert 0.0 <= prediction["confidence"] <= 1.0
