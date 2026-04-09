from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .project4_task2_qa_data import (
    QaWindowDataset,
    TokenizedQaExample,
    build_windowed_examples,
    build_word_vocab,
    ensure_glove_path,
    extract_answer_text,
    load_glove_embedding_matrix,
    load_squad_splits,
    make_bert_context_trimmer,
    make_collate_fn,
    maybe_limit_examples,
    tokenize_qa_example,
)
from .project4_task2_qa_metrics import (
    SquadMetricRecord,
    aggregate_squad_metrics,
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    select_best_span,
)
from .project4_task2_qa_model import FrozenBertBidafQaModel, GloveBidafQaModel
from .project3_common import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project 4 Task 2: Reading comprehension with BiDAF using "
            "either pretrained GloVe or frozen BERT embeddings on SQuAD 1.1."
        )
    )
    parser.add_argument("--variant", choices=["glove", "bert", "compare"], default="compare")
    parser.add_argument("--train_json", type=str, default=None, help="Optional local SQuAD-format train JSON.")
    parser.add_argument("--val_json", type=str, default=None, help="Optional local SQuAD-format validation JSON.")
    parser.add_argument("--cache_dir", type=str, default="data/external/project4_cache")
    parser.add_argument("--out_dir", type=str, default="outputs/project4/task2_reading_comprehension")
    parser.add_argument("--glove_path", type=str, default=None, help="Path to glove.6B.100d.txt")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--max_question_words", type=int, default=32)
    parser.add_argument("--context_window_words", type=int, default=160)
    parser.add_argument("--doc_stride_words", type=int, default=64)
    parser.add_argument("--max_answer_words", type=int, default=20)
    parser.add_argument("--bert_max_length", type=int, default=512)
    parser.add_argument("--min_word_freq", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--medium", action="store_true", help="Run a medium-size CPU-friendly configuration.")
    parser.add_argument("--log_every_steps", type=int, default=25, help="Log every N train/eval batches.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny CPU-friendly smoke configuration.")
    return parser.parse_args()


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.max_train_examples = 16 if args.max_train_examples is None else args.max_train_examples
    args.max_val_examples = 8 if args.max_val_examples is None else args.max_val_examples
    args.epochs = 1
    args.batch_size = min(args.batch_size, 4)
    args.eval_batch_size = min(args.eval_batch_size, 4)
    args.hidden_size = min(args.hidden_size, 16)
    args.context_window_words = min(args.context_window_words, 48)
    args.doc_stride_words = min(args.doc_stride_words, 24)
    args.max_question_words = min(args.max_question_words, 24)


def _apply_medium_defaults(args: argparse.Namespace) -> None:
    if not args.medium or args.smoke:
        return
    args.max_train_examples = 512 if args.max_train_examples is None else args.max_train_examples
    args.max_val_examples = 128 if args.max_val_examples is None else args.max_val_examples
    args.epochs = min(args.epochs, 2)
    args.batch_size = min(args.batch_size, 8)
    args.eval_batch_size = min(args.eval_batch_size, 8)
    args.hidden_size = min(args.hidden_size, 32)
    args.context_window_words = min(args.context_window_words, 96)
    args.doc_stride_words = min(args.doc_stride_words, 48)
    args.max_question_words = min(args.max_question_words, 24)


def _resolve_device(requested: str) -> torch.device:
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _serialize_predictions(path: Path, records: list[SquadMetricRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in records]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_variant_summary(
    variant: str,
    train_examples: list[TokenizedQaExample],
    val_examples: list[TokenizedQaExample],
    train_windows: list,
    val_windows: list,
    args: argparse.Namespace,
    device: torch.device,
    history: list[dict[str, float | int]],
    best_epoch: int,
    best_metrics: dict[str, float],
    variant_out_dir: Path,
    checkpoint_path: Path,
    history_path: Path,
    predictions_path: Path,
    vocab: dict[str, int] | None,
    loader_metadata: dict[str, object],
    status: str,
) -> dict[str, object]:
    summary = {
        "task": "Project 4 Task 2 - Reading comprehension",
        "variant": variant,
        "status": status,
        "dataset": {
            "source": "local_json" if args.train_json and args.val_json else "squad",
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "train_windows": len(train_windows),
            "val_windows": len(val_windows),
            "train_batches": loader_metadata["num_train_batches"],
            "val_batches": loader_metadata["num_val_batches"],
        },
        "config": {
            "variant": variant,
            "train_json": args.train_json,
            "val_json": args.val_json,
            "cache_dir": args.cache_dir,
            "glove_path": args.glove_path,
            "bert_model_name": args.bert_model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "embedding_dim": args.embedding_dim,
            "max_question_words": args.max_question_words,
            "context_window_words": args.context_window_words,
            "doc_stride_words": args.doc_stride_words,
            "max_answer_words": args.max_answer_words,
            "bert_max_length": args.bert_max_length,
            "seed": args.seed,
            "device": str(device),
            "medium": bool(args.medium),
            "smoke": bool(args.smoke),
            "log_every_steps": args.log_every_steps,
        },
        "training": {
            "best_epoch": best_epoch,
            "history_rows": len(history),
            "best_exact_match": best_metrics["exact_match"],
            "best_f1": best_metrics["f1"],
            "latest_epoch": int(history[-1]["epoch"]) if history else 0,
            "latest_train_loss": float(history[-1]["train_loss"]) if history else None,
            "latest_val_exact_match": float(history[-1]["val_exact_match"]) if history else None,
            "latest_val_f1": float(history[-1]["val_f1"]) if history else None,
        },
        "artifacts": {
            "summary_json": str(variant_out_dir / "summary.json"),
            "history_csv": str(history_path),
            "predictions_json": str(predictions_path),
            "checkpoint": str(checkpoint_path),
        },
    }
    if vocab is not None:
        summary["vocab"] = {"size": len(vocab), "min_word_freq": args.min_word_freq}
    if "glove_stats" in loader_metadata:
        summary["glove"] = loader_metadata["glove_stats"]
    return summary


def _prepare_tokenized_splits(args: argparse.Namespace) -> tuple[list[TokenizedQaExample], list[TokenizedQaExample]]:
    train_raw, val_raw = load_squad_splits(
        train_json=args.train_json,
        val_json=args.val_json,
        cache_dir=args.cache_dir,
    )
    train_raw = maybe_limit_examples(train_raw, args.max_train_examples, args.seed)
    val_raw = maybe_limit_examples(val_raw, args.max_val_examples, args.seed + 1)
    train_tokenized = [tokenize_qa_example(example, args.max_question_words) for example in train_raw]
    val_tokenized = [tokenize_qa_example(example, args.max_question_words) for example in val_raw]
    train_tokenized = [example for example in train_tokenized if example.answer_word_spans]
    val_tokenized = [example for example in val_tokenized if example.answer_word_spans]
    print(
        "Prepared tokenized QA splits:",
        f"train_examples={len(train_tokenized)}",
        f"val_examples={len(val_tokenized)}",
        f"source={'local_json' if args.train_json and args.val_json else 'squad'}",
    )
    return train_tokenized, val_tokenized


def _build_windows_for_variant(
    variant: str,
    train_examples: list[TokenizedQaExample],
    val_examples: list[TokenizedQaExample],
    args: argparse.Namespace,
) -> tuple[list, list]:
    context_trimmer = None
    if variant == "bert":
        context_trimmer = make_bert_context_trimmer(
            bert_model_name=args.bert_model_name,
            cache_dir=args.cache_dir,
            bert_max_length=args.bert_max_length,
        )
    train_windows = build_windowed_examples(
        train_examples,
        context_window_words=args.context_window_words,
        doc_stride_words=args.doc_stride_words,
        is_train=True,
        context_word_trimmer=context_trimmer,
    )
    val_windows = build_windowed_examples(
        val_examples,
        context_window_words=args.context_window_words,
        doc_stride_words=args.doc_stride_words,
        is_train=False,
        context_word_trimmer=context_trimmer,
    )
    return train_windows, val_windows


def _build_dataloaders(
    variant: str,
    train_windows: list,
    val_windows: list,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, dict[str, int] | None, dict[str, object]]:
    metadata: dict[str, object] = {
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
    }
    vocab = None
    collate = None
    if variant == "glove":
        vocab = build_word_vocab(train_windows, min_freq=args.min_word_freq)
        glove_file = ensure_glove_path(args.glove_path, args.cache_dir)
        embedding_matrix, glove_stats = load_glove_embedding_matrix(
            vocab=vocab,
            glove_path=glove_file,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
        )
        collate = make_collate_fn("glove", vocab=vocab)
        metadata["glove_stats"] = glove_stats
        metadata["embedding_matrix"] = embedding_matrix
    else:
        collate = make_collate_fn("bert")

    train_loader = DataLoader(
        QaWindowDataset(train_windows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        QaWindowDataset(val_windows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    metadata["num_train_batches"] = len(train_loader)
    metadata["num_val_batches"] = len(val_loader)
    return train_loader, val_loader, vocab, metadata


def _build_model(
    variant: str,
    loader_metadata: dict[str, object],
    args: argparse.Namespace,
) -> nn.Module:
    if variant == "glove":
        embedding_matrix = torch.tensor(loader_metadata["embedding_matrix"], dtype=torch.float32)
        return GloveBidafQaModel(
            embedding_matrix=embedding_matrix,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    return FrozenBertBidafQaModel(
        bert_model_name=args.bert_model_name,
        cache_dir=args.cache_dir,
        projection_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        bert_max_length=args.bert_max_length,
    )


def _training_step(
    model: nn.Module,
    batch,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    optimizer.zero_grad()
    start_logits, end_logits = model(batch)
    if batch.start_positions is None or batch.end_positions is None:
        raise ValueError("Training batch is missing labels.")
    loss = criterion(start_logits, batch.start_positions) + criterion(end_logits, batch.end_positions)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    example_lookup: dict[str, TokenizedQaExample],
    device: torch.device,
    max_answer_words: int,
    variant: str,
    log_every_steps: int,
) -> tuple[dict[str, float], list[SquadMetricRecord]]:
    model.eval()
    best_predictions: dict[str, tuple[float, str]] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = batch.to(device)
            start_logits, end_logits = model(batch)
            if batch_idx == 1 or batch_idx % max(1, log_every_steps) == 0 or batch_idx == len(dataloader):
                print(f"[{variant}] eval batch {batch_idx}/{len(dataloader)}")
            for row_idx, question_id in enumerate(batch.question_ids):
                local_start, local_end, score = select_best_span(
                    start_logits[row_idx].detach().cpu(),
                    end_logits[row_idx].detach().cpu(),
                    batch.context_mask[row_idx].detach().cpu(),
                    max_answer_words=max_answer_words,
                )
                global_start = batch.context_start_words[row_idx] + local_start
                global_end = batch.context_start_words[row_idx] + local_end
                prediction_text = extract_answer_text(example_lookup[question_id], global_start, global_end)
                previous = best_predictions.get(question_id)
                if previous is None or score > previous[0]:
                    best_predictions[question_id] = (score, prediction_text)

    records: list[SquadMetricRecord] = []
    for question_id, example in example_lookup.items():
        score, prediction_text = best_predictions.get(question_id, (float("-inf"), ""))
        exact_match = metric_max_over_ground_truths(exact_match_score, prediction_text, example.answer_texts)
        f1 = metric_max_over_ground_truths(f1_score, prediction_text, example.answer_texts)
        records.append(
            SquadMetricRecord(
                question_id=question_id,
                prediction_text=prediction_text,
                gold_answers=example.answer_texts,
                exact_match=exact_match,
                f1=f1,
                score=score,
            )
        )

    metrics = aggregate_squad_metrics(records)
    return metrics, records


def _train_variant(
    variant: str,
    train_examples: list[TokenizedQaExample],
    val_examples: list[TokenizedQaExample],
    args: argparse.Namespace,
    device: torch.device,
    variant_out_dir: Path,
) -> dict[str, object]:
    train_windows, val_windows = _build_windows_for_variant(variant, train_examples, val_examples, args)
    if not train_windows:
        raise RuntimeError(f"No training windows were created for variant={variant}.")
    if not val_windows:
        raise RuntimeError(f"No validation windows were created for variant={variant}.")
    print(
        f"[{variant}] built windows:",
        f"train_windows={len(train_windows)}",
        f"val_windows={len(val_windows)}",
    )

    train_loader, val_loader, vocab, loader_metadata = _build_dataloaders(variant, train_windows, val_windows, args)
    print(
        f"[{variant}] dataloaders ready:",
        f"train_batches={len(train_loader)}",
        f"val_batches={len(val_loader)}",
    )
    model = _build_model(variant, loader_metadata, args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    example_lookup = {example.question_id: example for example in val_examples}
    history: list[dict[str, float | int]] = []
    best_metrics: dict[str, float] = {"exact_match": 0.0, "f1": float("-inf"), "num_examples": 0}
    best_records: list[SquadMetricRecord] = []
    best_epoch = 0
    epochs_without_improvement = 0
    variant_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = variant_out_dir / "model.pt"
    history_path = variant_out_dir / "history.csv"
    predictions_path = variant_out_dir / "predictions.json"

    for epoch in range(1, args.epochs + 1):
        print(f"[{variant}] epoch {epoch}/{args.epochs} started")
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            batch_loss = _training_step(model, batch, criterion, optimizer, args.grad_clip)
            running_loss += batch_loss
            num_batches += 1
            if batch_idx == 1 or batch_idx % max(1, args.log_every_steps) == 0 or batch_idx == len(train_loader):
                print(
                    f"[{variant}] epoch {epoch}/{args.epochs} "
                    f"train batch {batch_idx}/{len(train_loader)} loss={batch_loss:.4f}"
                )

        train_loss = running_loss / max(1, num_batches)
        print(f"[{variant}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}")
        val_metrics, val_records = _evaluate(
            model=model,
            dataloader=val_loader,
            example_lookup=example_lookup,
            device=device,
            max_answer_words=args.max_answer_words,
            variant=variant,
            log_every_steps=args.log_every_steps,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_exact_match": val_metrics["exact_match"],
                "val_f1": val_metrics["f1"],
            }
        )
        if val_metrics["f1"] > best_metrics["f1"]:
            best_metrics = val_metrics
            best_records = val_records
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint_path)
            _serialize_predictions(predictions_path, best_records)
            print(
                f"[{variant}] new best at epoch {epoch}: "
                f"val_em={val_metrics['exact_match']:.4f} val_f1={val_metrics['f1']:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[{variant}] no improvement at epoch {epoch}: "
                f"val_em={val_metrics['exact_match']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"(patience {epochs_without_improvement}/{args.patience})"
            )
        _write_csv(history_path, history, ["epoch", "train_loss", "val_exact_match", "val_f1"])
        write_json(
            variant_out_dir / "summary.json",
            _build_variant_summary(
                variant=variant,
                train_examples=train_examples,
                val_examples=val_examples,
                train_windows=train_windows,
                val_windows=val_windows,
                args=args,
                device=device,
                history=history,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                variant_out_dir=variant_out_dir,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                predictions_path=predictions_path,
                vocab=vocab,
                loader_metadata=loader_metadata,
                status="running",
            ),
        )
        if epochs_without_improvement >= args.patience:
            break

    if best_epoch == 0:
        raise RuntimeError(f"Training did not produce a best state for variant={variant}.")

    summary = _build_variant_summary(
        variant=variant,
        train_examples=train_examples,
        val_examples=val_examples,
        train_windows=train_windows,
        val_windows=val_windows,
        args=args,
        device=device,
        history=history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        variant_out_dir=variant_out_dir,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        predictions_path=predictions_path,
        vocab=vocab,
        loader_metadata=loader_metadata,
        status="completed",
    )
    write_json(variant_out_dir / "summary.json", summary)
    print(
        f"[{variant}] best_epoch={best_epoch} "
        f"val_em={best_metrics['exact_match']:.4f} val_f1={best_metrics['f1']:.4f} "
        f"outputs={variant_out_dir}"
    )
    return summary


def _write_comparison(
    out_dir: Path,
    summaries: list[dict[str, object]],
) -> None:
    rows = []
    by_variant = {}
    for summary in summaries:
        variant = str(summary["variant"])
        by_variant[variant] = summary
        rows.append(
            {
                "variant": variant,
                "best_epoch": summary["training"]["best_epoch"],
                "exact_match": summary["training"]["best_exact_match"],
                "f1": summary["training"]["best_f1"],
                "train_examples": summary["dataset"]["train_examples"],
                "val_examples": summary["dataset"]["val_examples"],
            }
        )

    comparison_path = out_dir / "comparison.csv"
    _write_csv(
        comparison_path,
        rows,
        ["variant", "best_epoch", "exact_match", "f1", "train_examples", "val_examples"],
    )

    notes_lines = ["# Project 4 Task 2 Comparison", ""]
    for row in rows:
        notes_lines.append(
            f"- `{row['variant']}`: EM={row['exact_match']:.4f}, F1={row['f1']:.4f}, best_epoch={row['best_epoch']}"
        )
    if "glove" in by_variant and "bert" in by_variant:
        glove_f1 = float(by_variant["glove"]["training"]["best_f1"])
        bert_f1 = float(by_variant["bert"]["training"]["best_f1"])
        delta_f1 = bert_f1 - glove_f1
        glove_em = float(by_variant["glove"]["training"]["best_exact_match"])
        bert_em = float(by_variant["bert"]["training"]["best_exact_match"])
        delta_em = bert_em - glove_em
        notes_lines.extend(
            [
                "",
                "## BERT vs GloVe",
                f"- Frozen BERT delta EM: {delta_em:+.4f}",
                f"- Frozen BERT delta F1: {delta_f1:+.4f}",
            ]
        )

    notes_path = out_dir / "report_notes.md"
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _apply_smoke_defaults(args)
    _apply_medium_defaults(args)
    if args.context_window_words <= 0:
        raise ValueError("--context_window_words must be > 0")
    if args.doc_stride_words <= 0:
        raise ValueError("--doc_stride_words must be > 0")
    if args.max_question_words <= 0:
        raise ValueError("--max_question_words must be > 0")
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    out_dir = ensure_dir(args.out_dir)
    train_examples, val_examples = _prepare_tokenized_splits(args)
    if not train_examples or not val_examples:
        raise RuntimeError("No tokenized train/validation examples are available after preprocessing.")

    variants = ["glove", "bert"] if args.variant == "compare" else [args.variant]
    summaries: list[dict[str, object]] = []
    for variant in variants:
        variant_out_dir = out_dir / variant
        summary = _train_variant(
            variant=variant,
            train_examples=train_examples,
            val_examples=val_examples,
            args=args,
            device=device,
            variant_out_dir=variant_out_dir,
        )
        summaries.append(summary)

    _write_comparison(out_dir, summaries)
    print(f"Saved comparison artifacts to {out_dir}")


if __name__ == "__main__":
    main()
