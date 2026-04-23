from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import resolve_from_root


WORD_RE = re.compile(r"\b\w+\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute summary metrics for RAG evaluation outputs.")
    parser.add_argument("--input-json", help="Path to rag_eval_*.json. Defaults to latest in outputs/eval.")
    parser.add_argument("--output-dir", default="outputs/eval")
    return parser.parse_args()


def latest_eval_json(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("rag_eval_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No evaluation JSON files found in {output_dir}. Run `python src/03_rag_pipeline.py` first."
        )
    return files[-1]


def load_eval_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"Unexpected evaluation file format: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Unexpected records type in: {path}")
    return payload


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    q_count = len(records)
    if q_count == 0:
        raise ValueError("Evaluation records are empty.")

    baseline_word_counts = [word_count(str(r.get("baseline_answer", ""))) for r in records]
    rag_word_counts = [word_count(str(r.get("rag_answer", ""))) for r in records]
    retrieved_counts = [len(r.get("retrieved_files", []) or []) for r in records]

    baseline_non_empty = sum(1 for x in baseline_word_counts if x > 0)
    rag_non_empty = sum(1 for x in rag_word_counts if x > 0)
    with_sources = sum(1 for x in retrieved_counts if x > 0)

    unique_sources: set[str] = set()
    for record in records:
        for source in record.get("retrieved_files", []) or []:
            if source:
                unique_sources.add(str(source))

    baseline_avg_words = sum(baseline_word_counts) / q_count
    rag_avg_words = sum(rag_word_counts) / q_count

    return {
        "question_count": q_count,
        "baseline_non_empty_rate": baseline_non_empty / q_count,
        "rag_non_empty_rate": rag_non_empty / q_count,
        "baseline_avg_words": baseline_avg_words,
        "rag_avg_words": rag_avg_words,
        "rag_to_baseline_avg_length_ratio": (
            rag_avg_words / baseline_avg_words if baseline_avg_words > 0 else None
        ),
        "avg_retrieved_sources_per_question": sum(retrieved_counts) / q_count,
        "questions_with_sources_rate": with_sources / q_count,
        "unique_retrieved_sources_count": len(unique_sources),
        "unique_retrieved_sources": sorted(unique_sources),
    }


def write_metrics(metrics: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = output_dir / f"rag_metrics_{stamp}.json"
    csv_path = output_dir / f"rag_metrics_{stamp}.csv"

    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")

    csv_fields = [
        "question_count",
        "baseline_non_empty_rate",
        "rag_non_empty_rate",
        "baseline_avg_words",
        "rag_avg_words",
        "rag_to_baseline_avg_length_ratio",
        "avg_retrieved_sources_per_question",
        "questions_with_sources_rate",
        "unique_retrieved_sources_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({key: metrics.get(key) for key in csv_fields})

    return json_path, csv_path


def main() -> int:
    args = parse_args()
    output_dir = resolve_from_root(args.output_dir)

    input_path = resolve_from_root(args.input_json) if args.input_json else latest_eval_json(output_dir)
    payload = load_eval_payload(input_path)
    records = payload["records"]
    metrics = compute_metrics(records)

    json_path, csv_path = write_metrics(metrics, output_dir)

    print(f"Input evaluation file: {input_path}")
    print("=" * 80)
    print(f"Questions: {metrics['question_count']}")
    print(f"Baseline non-empty rate: {metrics['baseline_non_empty_rate']:.2%}")
    print(f"RAG non-empty rate: {metrics['rag_non_empty_rate']:.2%}")
    print(f"Baseline avg words: {metrics['baseline_avg_words']:.2f}")
    print(f"RAG avg words: {metrics['rag_avg_words']:.2f}")
    ratio = metrics["rag_to_baseline_avg_length_ratio"]
    print(f"RAG/Baseline length ratio: {'N/A' if ratio is None else f'{ratio:.2f}'}")
    print(
        "Avg retrieved sources/question: "
        f"{metrics['avg_retrieved_sources_per_question']:.2f}"
    )
    print(f"Questions with >=1 source: {metrics['questions_with_sources_rate']:.2%}")
    print(f"Unique retrieved sources: {metrics['unique_retrieved_sources_count']}")
    print("=" * 80)
    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved metrics CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
