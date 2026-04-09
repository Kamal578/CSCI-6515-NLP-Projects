from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .project3_common import ensure_dir, write_json


AZ_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
TEXT_REPLACEMENTS = {
    "\u200b": " ",
    "\u200c": " ",
    "\u200d": " ",
    "\ufeff": " ",
    "’": "'",
    "`": "'",
    "´": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "…": "...",
}
COMMON_SUFFIXES = [
    "lar",
    "lər",
    "lıq",
    "lik",
    "luq",
    "lük",
    "çı",
    "çi",
    "çu",
    "çü",
    "sız",
    "siz",
    "suz",
    "süz",
    "lıqdır",
    "likdir",
    "dur",
    "dür",
    "dır",
    "dir",
    "mış",
    "miş",
    "muş",
    "müş",
    "acaq",
    "əcək",
    "anda",
    "əndə",
    "dan",
    "dən",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project 4 Task 1: fine-tune a BERT classifier for Azerbaijani review sentiment "
            "with Apple Silicon MPS support."
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hajili/azerbaijani_review_sentiment_classification",
        help="Hugging Face dataset name to load when local CSV files are not provided.",
    )
    parser.add_argument("--train_file", type=str, default=None, help="Optional local training CSV.")
    parser.add_argument("--test_file", type=str, default=None, help="Optional local test CSV.")
    parser.add_argument("--text_col", type=str, default="content")
    parser.add_argument("--score_col", type=str, default="score")
    parser.add_argument("--upvotes_col", type=str, default="upvotes")
    parser.add_argument(
        "--label_mode",
        type=str,
        choices=["score5", "sentiment3", "binary"],
        default="score5",
        help=(
            "score5 keeps 1..5 labels; sentiment3 maps 1-2 negative, 3 neutral, 4-5 positive; "
            "binary removes score=3 and maps 1-2 negative, 4-5 positive."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-multilingual-cased",
        help="Base checkpoint or local model directory.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="data/external/project4_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/project4/task1_sentiment")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--morphology_samples", type=int, default=4000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.max_train_examples = 96 if args.max_train_examples is None else args.max_train_examples
    args.max_val_examples = 32 if args.max_val_examples is None else args.max_val_examples
    args.max_test_examples = 32 if args.max_test_examples is None else args.max_test_examples
    args.batch_size = min(args.batch_size, 8)
    args.eval_batch_size = min(args.eval_batch_size, 8)
    args.epochs = min(args.epochs, 1)
    args.max_length = min(args.max_length, 96)
    args.log_every_steps = min(args.log_every_steps, 10)


def normalize_azerbaijani_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    for source, target in TEXT_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    normalized = re.sub(r"([!?.,])\1{2,}", r"\1\1", normalized)
    normalized = re.sub(r"(\w)\1{3,}", lambda match: match.group(1) * 3, normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def score_to_label(score: int, mode: str) -> tuple[int | None, str | None]:
    value = int(score)
    if mode == "score5":
        if value < 1 or value > 5:
            return None, None
        label_id = value - 1
        return label_id, ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"][label_id]
    if mode == "sentiment3":
        if value <= 2:
            return 0, "negative"
        if value == 3:
            return 1, "neutral"
        return 2, "positive"
    if mode == "binary":
        if value == 3:
            return None, None
        if value <= 2:
            return 0, "negative"
        return 1, "positive"
    raise ValueError(f"Unsupported label mode: {mode}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and getattr(torch, "mps", None) is not None:
        torch.mps.empty_cache()


def _load_raw_dataset(args: argparse.Namespace):
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for Task 1.") from exc

    if args.train_file and args.test_file:
        dataset = load_dataset("csv", data_files={"train": args.train_file, "test": args.test_file}, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict with train/test splits.")
    return dataset


def _prepare_split(split, args: argparse.Namespace):
    required_columns = {args.text_col, args.score_col}
    missing = required_columns.difference(split.column_names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Available columns: {split.column_names}")

    def convert_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized_text = normalize_azerbaijani_text(row[args.text_col])
        label_id, label_name = score_to_label(int(row[args.score_col]), args.label_mode)
        return {
            "text": normalized_text,
            "label": label_id,
            "label_name": label_name,
            "score": int(row[args.score_col]),
            "upvotes": int(row.get(args.upvotes_col, 0) or 0),
        }

    converted = split.map(convert_row)
    converted = converted.filter(lambda row: row["label"] is not None and bool(row["text"].strip()))
    converted = converted.remove_columns(
        [column for column in converted.column_names if column not in {"text", "label", "label_name", "score", "upvotes"}]
    )
    converted = converted.class_encode_column("label")
    return converted


def _limit_dataset(dataset, max_examples: int | None, seed: int):
    if max_examples is None or max_examples >= len(dataset):
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(max_examples))


def _split_train_validation(train_split, args: argparse.Namespace):
    if not 0.0 < args.validation_ratio < 0.5:
        raise ValueError("--validation_ratio must be between 0 and 0.5")
    split = train_split.train_test_split(
        test_size=args.validation_ratio,
        seed=args.seed,
        stratify_by_column="label",
    )
    train_dataset = _limit_dataset(split["train"], args.max_train_examples, args.seed)
    val_dataset = _limit_dataset(split["test"], args.max_val_examples, args.seed + 1)
    return train_dataset, val_dataset


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    encoded = dataset.map(tokenize_batch, batched=True)
    keep_columns = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    remove_columns = [column for column in encoded.column_names if column not in keep_columns]
    if remove_columns:
        encoded = encoded.remove_columns(remove_columns)
    encoded.set_format("torch")
    return encoded


def _iter_words(texts: list[str]) -> list[str]:
    words: list[str] = []
    for text in texts:
        words.extend(AZ_WORD_RE.findall(text.lower()))
    return words


def _suffix_profile(words: list[str]) -> dict[str, list[dict[str, Any]]]:
    profile: dict[str, list[dict[str, Any]]] = {}
    for suffix_len in (2, 3, 4):
        counter = Counter()
        for word in words:
            if len(word) >= suffix_len + 2:
                suffix = word[-suffix_len:]
                if suffix.isalpha():
                    counter[suffix] += 1
        profile[f"top_suffixes_{suffix_len}"] = [
            {"suffix": suffix, "count": count}
            for suffix, count in counter.most_common(15)
        ]
    domain_suffix_hits = Counter()
    for word in words:
        for suffix in COMMON_SUFFIXES:
            if word.endswith(suffix):
                domain_suffix_hits[suffix] += 1
    profile["common_azerbaijani_suffix_hits"] = [
        {"suffix": suffix, "count": count}
        for suffix, count in domain_suffix_hits.most_common(15)
    ]
    return profile


def build_morphology_summary(dataset, tokenizer, max_samples: int) -> dict[str, Any]:
    if len(dataset) == 0:
        return {}
    sample_size = min(max_samples, len(dataset))
    sampled = dataset.shuffle(seed=17).select(range(sample_size))
    texts = [str(text) for text in sampled["text"]]
    words = _iter_words(texts)
    if not words:
        return {}

    piece_counts: list[int] = []
    for word in words[: min(25000, len(words))]:
        pieces = tokenizer.tokenize(word)
        if pieces:
            piece_counts.append(len(pieces))
    suffix_profile = _suffix_profile(words)
    avg_word_pieces = float(sum(piece_counts) / len(piece_counts)) if piece_counts else 0.0
    fragmentation_rate = float(sum(1 for count in piece_counts if count > 1) / len(piece_counts)) if piece_counts else 0.0
    return {
        "sample_reviews": sample_size,
        "sample_words": len(words),
        "avg_words_per_review": float(sum(len(AZ_WORD_RE.findall(text)) for text in texts) / max(1, len(texts))),
        "avg_characters_per_word": float(sum(len(word) for word in words) / len(words)),
        "avg_wordpieces_per_word": avg_word_pieces,
        "multi_piece_word_rate": fragmentation_rate,
        "strategy": {
            "text_normalization": [
                "Unicode NFC normalization",
                "Whitespace cleanup while preserving Azerbaijani diacritics",
                "Punctuation and repeated-character compression for noisy app reviews",
            ],
            "agglutinative_adaptation": [
                "No stemming or suffix stripping is applied because it would destroy sentiment-bearing endings.",
                "Subword BERT tokenization is kept intact so suffix chains remain visible to the encoder.",
                "Suffix-frequency diagnostics are saved to document how often agglutinative endings appear.",
            ],
        },
        **suffix_profile,
    }


def _label_names(mode: str) -> list[str]:
    if mode == "score5":
        return ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    if mode == "sentiment3":
        return ["negative", "neutral", "positive"]
    if mode == "binary":
        return ["negative", "positive"]
    raise ValueError(f"Unsupported label mode: {mode}")


def _metrics_from_predictions(gold: list[int], pred: list[int], label_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(f1_score(gold, pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(gold, pred, average="weighted", labels=labels, zero_division=0)),
        "classification_report": classification_report(
            gold,
            pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _evaluate(model, dataloader: DataLoader, device: torch.device, label_names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    gold: list[int] = []
    pred: list[int] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1).tolist()
            labels = batch["labels"].detach().cpu().tolist()
            gold.extend(labels)
            pred.extend(predictions)
            for label_id, pred_id, prob_vector in zip(labels, predictions, probs.tolist()):
                rows.append(
                    {
                        "gold_label_id": label_id,
                        "gold_label": label_names[label_id],
                        "predicted_label_id": pred_id,
                        "predicted_label": label_names[pred_id],
                        "confidence": float(max(prob_vector)),
                        "probabilities": {label_names[idx]: float(value) for idx, value in enumerate(prob_vector)},
                    }
                )
    return _metrics_from_predictions(gold, pred, label_names), rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_predictions(
    path: Path,
    original_dataset,
    prediction_rows: list[dict[str, Any]],
) -> None:
    payload = []
    for original, pred_row in zip(original_dataset, prediction_rows):
        payload.append(
            {
                "text": original["text"],
                "score": int(original["score"]),
                "upvotes": int(original["upvotes"]),
                **pred_row,
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _apply_smoke_defaults(args)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be > 0")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be > 0")

    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(out_dir / "model")

    raw_dataset = _load_raw_dataset(args)
    train_split = _prepare_split(raw_dataset["train"], args)
    test_split = _prepare_split(raw_dataset["test"], args)
    train_dataset, val_dataset = _split_train_validation(train_split, args)
    test_dataset = _limit_dataset(test_split, args.max_test_examples, args.seed + 2)
    label_names = _label_names(args.label_mode)

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise ImportError("transformers is required for Task 1.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        num_labels=len(label_names),
        ignore_mismatched_sizes=True,
        id2label={idx: label for idx, label in enumerate(label_names)},
        label2id={label: idx for idx, label in enumerate(label_names)},
    ).to(device)

    encoded_train = _tokenize_dataset(train_dataset, tokenizer, args.max_length)
    encoded_val = _tokenize_dataset(val_dataset, tokenizer, args.max_length)
    encoded_test = _tokenize_dataset(test_dataset, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if device.type != "cpu" else None)

    train_loader = DataLoader(
        encoded_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        encoded_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        encoded_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    total_steps = max(1, update_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    morphology_summary = build_morphology_summary(train_dataset, tokenizer, args.morphology_samples)
    history_rows: list[dict[str, Any]] = []
    best_val_f1 = float("-inf")
    best_epoch = 0
    patience_counter = 0
    history_path = out_dir / "history.csv"
    val_predictions_path = out_dir / "validation_predictions.json"
    test_predictions_path = out_dir / "test_predictions.json"

    print(
        "Prepared Azerbaijani sentiment dataset:",
        f"train={len(train_dataset)}",
        f"val={len(val_dataset)}",
        f"test={len(test_dataset)}",
        f"label_mode={args.label_mode}",
        f"device={device}",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.item()) * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step == 1 or step % max(1, args.log_every_steps) == 0 or step == len(train_loader):
                print(
                    f"[task1] epoch {epoch}/{args.epochs} batch {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}"
                )

        train_loss = running_loss / max(1, len(train_loader))
        maybe_empty_cache(device)
        val_metrics, val_prediction_rows = _evaluate(model, val_loader, device, label_names)
        maybe_empty_cache(device)
        test_metrics, test_prediction_rows = _evaluate(model, test_loader, device, label_names)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
            }
        )
        _write_csv(
            history_path,
            history_rows,
            ["epoch", "train_loss", "val_accuracy", "val_f1_macro", "test_accuracy", "test_f1_macro"],
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1_macro"])
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            _save_predictions(val_predictions_path, val_dataset, val_prediction_rows)
            _save_predictions(test_predictions_path, test_dataset, test_prediction_rows)
            print(
                f"[task1] new best epoch={epoch} "
                f"val_acc={val_metrics['accuracy']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f}"
            )
        else:
            patience_counter += 1
            print(
                f"[task1] no improvement at epoch={epoch} "
                f"(patience {patience_counter}/{args.patience})"
            )

        if patience_counter >= args.patience:
            break

    if best_epoch == 0:
        raise RuntimeError("Task 1 training never produced a best checkpoint.")

    summary = {
        "task": "Project 4 Task 1 - Azerbaijani sentiment analysis with BERT",
        "dataset": {
            "source": "local_csv" if args.train_file and args.test_file else args.dataset_name,
            "text_column": args.text_col,
            "score_column": args.score_col,
            "upvotes_column": args.upvotes_col,
            "train_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "test_examples": len(test_dataset),
            "label_mode": args.label_mode,
            "train_label_distribution": dict(sorted(Counter(train_dataset["label_name"]).items())),
            "validation_label_distribution": dict(sorted(Counter(val_dataset["label_name"]).items())),
            "test_label_distribution": dict(sorted(Counter(test_dataset["label_name"]).items())),
        },
        "model": {
            "model_name": args.model_name,
            "num_labels": len(label_names),
            "labels": label_names,
            "max_length": args.max_length,
            "tokenizer_class": tokenizer.__class__.__name__,
            "case_sensitive": not bool(getattr(tokenizer, "do_lower_case", False)),
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "io_contract": {
            "input": "Single Azerbaijani UTF-8 review string",
            "output": label_names,
            "tensor_inputs": ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in encoded_train.features else []),
            "prediction_rule": "argmax over classifier logits",
            "agglutinative_adaptation": morphology_summary.get("strategy", {}),
        },
        "training": {
            "best_epoch": best_epoch,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history_rows),
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "device": str(device),
            "best_validation_accuracy": max(float(row["val_accuracy"]) for row in history_rows),
            "best_validation_f1_macro": best_val_f1,
            "latest_test_accuracy": float(history_rows[-1]["test_accuracy"]),
            "latest_test_f1_macro": float(history_rows[-1]["test_f1_macro"]),
        },
        "morphology": morphology_summary,
        "artifacts": {
            "model_dir": str(model_dir),
            "history_csv": str(history_path),
            "validation_predictions_json": str(val_predictions_path),
            "test_predictions_json": str(test_predictions_path),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "classification_report.json", _metrics_from_predictions(
        [row["gold_label_id"] for row in json.loads(test_predictions_path.read_text(encoding="utf-8"))],
        [row["predicted_label_id"] for row in json.loads(test_predictions_path.read_text(encoding="utf-8"))],
        label_names,
    ))
    print(f"Saved Task 1 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
