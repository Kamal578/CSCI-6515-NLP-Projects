from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .load_data import load_corpus_csv
from .project2_task4_dot_features import dot_candidate_preview_fields


@dataclass
class CorpusDoc:
    doc_idx: int
    text: str
    doc_id: str


@dataclass
class SplitRows:
    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    test: list[dict[str, Any]]
    split_mode: str
    warnings: list[str]


def load_corpus_docs(
    corpus_path: str,
    text_column: str = "text",
    max_docs: int | None = None,
) -> list[CorpusDoc]:
    df = load_corpus_csv(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")
    texts = df[text_column].fillna("").astype(str).tolist()
    docs: list[CorpusDoc] = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        doc_id = ""
        if "doc_id" in df.columns:
            try:
                doc_id = str(df.iloc[i]["doc_id"])
            except Exception:
                doc_id = ""
        docs.append(CorpusDoc(doc_idx=i, text=text, doc_id=doc_id))
        if max_docs is not None and len(docs) >= max_docs:
            break
    return docs


def iter_dot_candidate_rows(doc: CorpusDoc) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = doc.text
    dot_positions = [i for i, ch in enumerate(text) if ch == "."]
    for local_idx, char_index in enumerate(dot_positions):
        preview = dot_candidate_preview_fields(text, char_index)
        row_id = f"d{doc.doc_idx}_p{char_index}"
        rows.append(
            {
                "row_id": row_id,
                "doc_idx": doc.doc_idx,
                "doc_id": doc.doc_id,
                "char_index": char_index,
                "char": ".",
                "left_context": preview["left_context"],
                "right_context": preview["right_context"],
                "window_text": preview["window_text"],
                "prev_char": preview["prev_char"],
                "next_char": preview["next_char"],
                "prev_token": preview["prev_token"],
                "next_token": preview["next_token"],
                "rule_guess": int(preview["rule_guess"]),
                "gold_label": "",
                "annotator_note": "",
                "dot_idx_in_doc": local_idx,
            }
        )
    return rows


def export_dot_label_template(
    corpus_path: str = "data/raw/corpus.csv",
    text_column: str = "text",
    max_docs: int | None = 200,
    target_dots: int = 2000,
    seed: int = 42,
    out_csv: str = "data/processed/task4_dot_labels_template.csv",
) -> dict[str, Any]:
    docs = load_corpus_docs(corpus_path=corpus_path, text_column=text_column, max_docs=max_docs)
    if not docs:
        raise ValueError("No documents available to export dot candidates.")

    rng = random.Random(seed)
    order = list(range(len(docs)))
    rng.shuffle(order)

    selected_rows: list[dict[str, Any]] = []
    selected_docs = 0
    for idx in order:
        doc = docs[idx]
        doc_rows = iter_dot_candidate_rows(doc)
        if not doc_rows:
            continue
        # Add whole docs to keep sentence reconstruction possible later.
        selected_rows.extend(doc_rows)
        selected_docs += 1
        if len(selected_rows) >= target_dots:
            break

    if not selected_rows:
        raise ValueError("No dot candidates found in sampled documents.")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "doc_idx",
        "doc_id",
        "char_index",
        "char",
        "left_context",
        "right_context",
        "window_text",
        "prev_char",
        "next_char",
        "prev_token",
        "next_token",
        "rule_guess",
        "gold_label",
        "annotator_note",
        "dot_idx_in_doc",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    return {
        "out_csv": str(out_path),
        "selected_docs": selected_docs,
        "selected_dots": len(selected_rows),
        "target_dots": target_dots,
        "seed": seed,
        "max_docs": max_docs,
    }


def load_labeled_dot_rows(labels_csv: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    df = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    required = {"row_id", "doc_idx", "char_index", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {sorted(missing)}")

    if max_rows is not None:
        df = df.head(max_rows)

    if df["row_id"].duplicated().any():
        dupes = df.loc[df["row_id"].duplicated(), "row_id"].tolist()[:10]
        raise ValueError(f"Duplicate row_id values found (sample): {dupes}")

    unlabeled = df[df["gold_label"].astype(str).str.strip() == ""]
    if not unlabeled.empty:
        sample_ids = unlabeled["row_id"].tolist()[:10]
        raise ValueError(f"Missing gold_label for {len(unlabeled)} rows. Sample row_ids: {sample_ids}")

    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        lbl_raw = str(rec["gold_label"]).strip()
        if lbl_raw not in {"0", "1"}:
            raise ValueError(f"Invalid gold_label={lbl_raw!r} for row_id={rec.get('row_id')}")
        rec["gold_label"] = int(lbl_raw)
        rec["doc_idx"] = int(str(rec["doc_idx"]).strip())
        rec["char_index"] = int(str(rec["char_index"]).strip())
        rec["rule_guess"] = int(str(rec.get("rule_guess", "0") or "0"))
        rec["dot_idx_in_doc"] = int(str(rec.get("dot_idx_in_doc", "0") or "0"))
        rows.append(rec)
    return rows


def split_labeled_rows_by_doc(
    rows: list[dict[str, Any]],
    test_ratio: float = 0.15,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> SplitRows:
    if not rows:
        raise ValueError("No labeled rows provided.")
    if not (0 < test_ratio < 1 and 0 < dev_ratio < 1 and test_ratio + dev_ratio < 1):
        raise ValueError("Require 0<test_ratio<1, 0<dev_ratio<1 and test_ratio+dev_ratio<1")

    warnings: list[str] = []
    by_doc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_doc[int(row["doc_idx"])].append(row)

    doc_ids = list(by_doc.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_ids)

    def _row_fallback(extra_warning: str | None = None) -> SplitRows:
        local_warnings = list(warnings)
        if extra_warning:
            local_warnings.append(extra_warning)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(rows) * test_ratio)))
        n_dev = max(1, int(round(len(rows) * dev_ratio)))
        if n_test + n_dev >= len(rows):
            n_test = max(1, len(rows) // 5)
            n_dev = max(1, len(rows) // 5)
        if n_test + n_dev >= len(rows):
            n_test = 1
            n_dev = 1 if len(rows) > 2 else 0
        test_set = set(idxs[:n_test])
        dev_set = set(idxs[n_test : n_test + n_dev])
        train = [r for i, r in enumerate(rows) if i not in test_set and i not in dev_set]
        dev = [r for i, r in enumerate(rows) if i in dev_set]
        test = [r for i, r in enumerate(rows) if i in test_set]
        if not train or not dev or not test:
            raise ValueError("Unable to create non-empty train/dev/test splits from labeled rows.")
        return SplitRows(train=train, dev=dev, test=test, split_mode="row_fallback", warnings=local_warnings)

    if len(doc_ids) < 6:
        return _row_fallback("Too few labeled documents for reliable doc-level split; falling back to row-level random split.")

    n_docs = len(doc_ids)
    n_test_docs = max(1, int(round(n_docs * test_ratio)))
    n_dev_docs = max(1, int(round(n_docs * dev_ratio)))
    if n_test_docs + n_dev_docs >= n_docs:
        n_test_docs = max(1, n_docs // 6)
        n_dev_docs = max(1, n_docs // 6)
        if n_test_docs + n_dev_docs >= n_docs:
            n_dev_docs = 1
            n_test_docs = 1

    test_docs = set(doc_ids[:n_test_docs])
    dev_docs = set(doc_ids[n_test_docs : n_test_docs + n_dev_docs])
    train_docs = set(doc_ids[n_test_docs + n_dev_docs :])

    train = [r for r in rows if r["doc_idx"] in train_docs]
    dev = [r for r in rows if r["doc_idx"] in dev_docs]
    test = [r for r in rows if r["doc_idx"] in test_docs]

    if not train or not dev or not test:
        return _row_fallback("Empty split produced at doc-level; falling back to row-level split.")

    return SplitRows(train=train, dev=dev, test=test, split_mode="doc", warnings=warnings)


def group_rows_by_doc(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[int(row["doc_idx"])].append(row)
    for doc_idx in out:
        out[doc_idx] = sorted(out[doc_idx], key=lambda r: int(r["char_index"]))
    return dict(out)


def build_task4_labels_from_gold_sentences(
    gold_sentences_path: str,
    out_labels_csv: str,
    out_corpus_csv: str,
    sentences_per_doc: int = 10,
) -> dict[str, Any]:
    """
    Convert a sentence-per-line gold file into:
      1) a pseudo corpus CSV (grouped into fixed-size pseudo-docs)
      2) a fully labeled Task 4 dot CSV (one row per dot, gold_label filled)

    This is useful when original source-doc boundaries are unavailable.
    """
    if sentences_per_doc < 1:
        raise ValueError("sentences_per_doc must be >= 1")

    lines = [l.strip() for l in Path(gold_sentences_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"No non-empty sentences found in {gold_sentences_path}")

    pseudo_docs: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    chunks = [lines[i : i + sentences_per_doc] for i in range(0, len(lines), sentences_per_doc)]
    for doc_idx, chunk in enumerate(chunks):
        text_parts: list[str] = []
        gold_eos_positions: set[int] = set()
        cursor = 0
        for s_i, sent in enumerate(chunk):
            if text_parts:
                cursor += 1  # account for join space
            text_parts.append(sent)
            dot_pos = cursor + sent.rfind(".")
            if dot_pos >= cursor:
                gold_eos_positions.add(dot_pos)
            cursor += len(sent)

        text = " ".join(chunk)
        pseudo_docs.append({"doc_id": doc_idx, "text": text})

        dot_positions = [i for i, ch in enumerate(text) if ch == "."]
        for local_idx, char_index in enumerate(dot_positions):
            preview = dot_candidate_preview_fields(text, char_index)
            row_id = f"d{doc_idx}_p{char_index}"
            label_rows.append(
                {
                    "row_id": row_id,
                    "doc_idx": doc_idx,
                    "doc_id": str(doc_idx),
                    "char_index": char_index,
                    "char": ".",
                    "left_context": preview["left_context"],
                    "right_context": preview["right_context"],
                    "window_text": preview["window_text"],
                    "prev_char": preview["prev_char"],
                    "next_char": preview["next_char"],
                    "prev_token": preview["prev_token"],
                    "next_token": preview["next_token"],
                    "rule_guess": int(preview["rule_guess"]),
                    "gold_label": int(char_index in gold_eos_positions),
                    "annotator_note": "",
                    "dot_idx_in_doc": local_idx,
                }
            )

    # write pseudo corpus CSV
    out_corpus = Path(out_corpus_csv)
    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    with out_corpus.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "text"])
        writer.writeheader()
        for row in pseudo_docs:
            writer.writerow(row)

    # write labels CSV (same schema as template but gold_label filled)
    out_labels = Path(out_labels_csv)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "doc_idx",
        "doc_id",
        "char_index",
        "char",
        "left_context",
        "right_context",
        "window_text",
        "prev_char",
        "next_char",
        "prev_token",
        "next_token",
        "rule_guess",
        "gold_label",
        "annotator_note",
        "dot_idx_in_doc",
    ]
    with out_labels.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in label_rows:
            writer.writerow(row)

    return {
        "gold_sentences_path": gold_sentences_path,
        "out_labels_csv": str(out_labels),
        "out_corpus_csv": str(out_corpus),
        "num_gold_sentences": len(lines),
        "num_pseudo_docs": len(pseudo_docs),
        "sentences_per_doc": sentences_per_doc,
        "num_dot_candidates": len(label_rows),
        "num_eos_dots": int(sum(int(r["gold_label"]) for r in label_rows)),
        "num_not_eos_dots": int(len(label_rows) - sum(int(r["gold_label"]) for r in label_rows)),
    }
