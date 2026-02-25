from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .lm_data import prepare_lm_splits
from .ngram_lm import UNK_TOKEN, build_train_vocab, map_sentence_to_vocab
from .smoothing_lm import (
    NgramCountStore,
    make_backoff_model,
    make_interpolation_model,
    make_kneser_ney_model,
    make_laplace_model,
)


MethodFactory = Callable[[int, NgramCountStore, dict], object]


METHOD_FACTORIES: dict[str, MethodFactory] = {
    "laplace": make_laplace_model,
    "interpolation": make_interpolation_model,
    "backoff": make_backoff_model,
    "kneser_ney": make_kneser_ney_model,
}

METHOD_COMPLEXITY_RANK = {
    "laplace": 1,
    "interpolation": 2,
    "backoff": 3,
    "kneser_ney": 4,
}


@dataclass
class TrialResult:
    order: int
    method: str
    params: dict
    dev_perplexity: float
    dev_zero_prob_events: int
    dev_num_events: int

    def row(self) -> dict[str, object]:
        return {
            "order": self.order,
            "method": self.method,
            "params_json": json.dumps(self.params, ensure_ascii=False, sort_keys=True),
            "dev_perplexity": self.dev_perplexity,
            "dev_zero_prob_events": self.dev_zero_prob_events,
            "dev_num_events": self.dev_num_events,
        }


def _float_list(arg_values: Iterable[str]) -> list[float]:
    return [float(x) for x in arg_values]


def _normalize_weights(vals: list[float]) -> list[float]:
    vals = [0.0 if abs(v) < 1e-12 else v for v in vals]
    s = sum(vals)
    if s <= 0:
        raise ValueError("weights must sum to > 0")
    return [v / s for v in vals]


def interpolation_grid(order: int, step: float = 0.1) -> list[dict]:
    if order == 2:
        out = []
        points = int(round(1.0 / step))
        for i in range(points + 1):
            l2 = round(i * step, 10)
            l1 = round(1.0 - l2, 10)
            out.append({"lambdas": _normalize_weights([l1, l2])})
        return out
    if order == 3:
        out = []
        points = int(round(1.0 / step))
        for i in range(points + 1):  # lambda3
            l3 = i * step
            for j in range(points - i + 1):  # lambda2
                l2 = j * step
                l1 = 1.0 - l3 - l2
                if l1 < -1e-9:
                    continue
                out.append({"lambdas": _normalize_weights([max(l1, 0.0), max(l2, 0.0), max(l3, 0.0)])})
        return out
    raise ValueError(f"Interpolation grid only implemented for order 2/3, got {order}")


def default_param_grid(method: str, order: int, disable_grid_search: bool) -> list[dict]:
    if method == "laplace":
        values = [0.1] if disable_grid_search else [0.01, 0.05, 0.1, 0.5, 1.0]
        return [{"k": v} for v in values]
    if method == "interpolation":
        if disable_grid_search:
            if order == 2:
                return [{"lambdas": [0.2, 0.8]}]
            if order == 3:
                return [{"lambdas": [0.1, 0.2, 0.7]}]
        return interpolation_grid(order=order, step=0.1)
    if method == "backoff":
        values = [0.75] if disable_grid_search else [0.1, 0.3, 0.5, 0.75]
        return [{"d": v} for v in values]
    if method == "kneser_ney":
        values = [0.75] if disable_grid_search else [0.5, 0.75, 1.0]
        return [{"d": v} for v in values]
    raise ValueError(f"Unknown method: {method}")


def fit_eval_model(order: int, method: str, counts: NgramCountStore, sentences: list[list[str]], params: dict):
    model = METHOD_FACTORIES[method](order, counts, params)
    return model.evaluate(sentences)


def choose_best_trial(trials: list[TrialResult]) -> TrialResult:
    if not trials:
        raise ValueError("No trials to choose from")
    return min(
        trials,
        key=lambda t: (
            math.isinf(t.dev_perplexity),
            t.dev_perplexity,
            t.dev_zero_prob_events,
            json.dumps(t.params, sort_keys=True),
        ),
    )


def rank_methods(rows: list[dict]) -> list[dict]:
    def key_fn(r: dict):
        return (
            math.isinf(float(r["test_perplexity"])),
            float(r["test_perplexity"]),
            math.isinf(float(r["dev_perplexity"])),
            float(r["dev_perplexity"]),
            METHOD_COMPLEXITY_RANK.get(str(r["method"]), 99),
        )

    ranked = sorted(rows, key=key_fn)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked


def choose_overall_best(ranked_by_order: dict[int, list[dict]]) -> dict:
    trigram_rows = ranked_by_order.get(3, [])
    if not trigram_rows:
        return {"primary_order": 3, "selected_method": None, "reason": "No trigram results available."}

    ranked = sorted(trigram_rows, key=lambda r: r["rank"])
    top = ranked[0]
    reason = "Selected by lowest trigram test perplexity."

    if len(ranked) > 1:
        second = ranked[1]
        p1 = float(top["test_perplexity"])
        p2 = float(second["test_perplexity"])
        if math.isfinite(p1) and math.isfinite(p2):
            rel = abs(p2 - p1) / max(p1, 1e-12)
            if rel < 0.005:
                if float(second["dev_perplexity"]) < float(top["dev_perplexity"]):
                    top = second
                    reason = "Trigram test perplexities within 0.5%; tie broken by lower dev perplexity."
                elif float(second["dev_perplexity"]) == float(top["dev_perplexity"]):
                    if METHOD_COMPLEXITY_RANK.get(second["method"], 99) < METHOD_COMPLEXITY_RANK.get(top["method"], 99):
                        top = second
                        reason = "Trigram test/dev perplexities tied; tie broken by simpler method."
                    else:
                        reason = "Trigram top methods effectively tied (<0.5%); selected by simplicity tie-breaker."
                else:
                    reason = "Trigram top methods effectively tied (<0.5%); selected by lower dev perplexity."

    return {
        "primary_order": 3,
        "selected_method": top["method"],
        "selected_order": int(top["order"]),
        "test_perplexity": top["test_perplexity"],
        "dev_perplexity": top["dev_perplexity"],
        "hyperparams": top["params"],
        "reason": reason,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Project 2 Task 2: compare smoothing methods on n-gram LMs.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv")
    ap.add_argument("--text_column", type=str, default="text")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_seed", type=int, default=43)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--lowercase", action="store_true", default=True)
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false")
    ap.add_argument("--unk_min_freq", type=int, default=2)
    ap.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["laplace", "interpolation", "backoff", "kneser_ney"],
        choices=["laplace", "interpolation", "backoff", "kneser_ney"],
    )
    ap.add_argument("--top_n", type=int, default=0, help="Reserved for future diagnostics; currently unused.")
    ap.add_argument("--disable_grid_search", action="store_true")
    ap.add_argument("--out_dir", type=str, default="outputs/project2/task2_smoothing")
    args = ap.parse_args()

    orders = sorted(set(args.orders))
    if any(n not in (2, 3) for n in orders):
        raise ValueError(f"Task 2 currently supports only orders 2 and 3. Got: {orders}")
    if args.dev_ratio <= 0 or args.dev_ratio >= 1:
        raise ValueError("dev_ratio must be in (0,1)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing train/dev/test splits using Task 1-compatible preprocessing...")
    prepared = prepare_lm_splits(
        corpus_path=args.corpus_path,
        text_column=args.text_column,
        test_ratio=args.test_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        dev_seed=args.dev_seed,
        lowercase=args.lowercase,
        unk_min_freq=args.unk_min_freq,
        max_docs=args.max_docs,
    )

    print(
        f"Docs total={prepared.texts_total} train={len(prepared.train_docs)} test={len(prepared.test_docs)} | "
        f"sent train_core={len(prepared.train_core_sentences)} dev={len(prepared.dev_sentences)} test={len(prepared.test_sentences)}"
    )

    max_order = max(orders)
    # For fair dev tuning, remap train_core/dev using train_core-derived vocab so dev tokens unseen in train_core
    # become <unk> instead of causing artificial zero-probability failures during tuning.
    train_core_vocab_counts = build_train_vocab(prepared.train_core_sentences, min_freq=1)
    train_core_vocab = set(train_core_vocab_counts.keys())
    train_core_vocab.add(UNK_TOKEN)
    train_core_tuning_sents = [map_sentence_to_vocab(s, train_core_vocab) for s in prepared.train_core_sentences]
    dev_tuning_sents = [map_sentence_to_vocab(s, train_core_vocab) for s in prepared.dev_sentences]

    print("Building count stores (train_core for tuning and full_train for final test evaluation)...")
    counts_train_core = NgramCountStore.build(train_core_tuning_sents, max_order=max_order)
    counts_full_train = NgramCountStore.build(prepared.train_sentences, max_order=max_order)

    tuning_rows: list[dict] = []
    comparison_rows: list[dict] = []
    per_order_rank_rows: dict[int, list[dict]] = {n: [] for n in orders}
    detailed_results: dict[str, dict[str, dict]] = {}
    warnings: list[str] = []

    for order in orders:
        detailed_results[str(order)] = {}
        for method in args.methods:
            print(f"Tuning method={method} order={order}...")
            params_grid = default_param_grid(method=method, order=order, disable_grid_search=args.disable_grid_search)
            trials: list[TrialResult] = []

            for params in params_grid:
                try:
                    dev_stats = fit_eval_model(
                        order=order,
                        method=method,
                        counts=counts_train_core,
                        sentences=dev_tuning_sents,
                        params=params,
                    )
                except Exception as e:
                    warnings.append(
                        f"Skipped invalid trial method={method} order={order} params={params}: {type(e).__name__}: {e}"
                    )
                    continue
                trial = TrialResult(
                    order=order,
                    method=method,
                    params=params,
                    dev_perplexity=dev_stats.perplexity,
                    dev_zero_prob_events=dev_stats.zero_prob_events,
                    dev_num_events=dev_stats.num_events,
                )
                trials.append(trial)
                tuning_rows.append(trial.row())

            best_trial = choose_best_trial(trials)

            print(f"Retraining selected {method} (order={order}) on full train and evaluating on test...")
            test_stats = fit_eval_model(
                order=order,
                method=method,
                counts=counts_full_train,
                sentences=prepared.test_sentences,
                params=best_trial.params,
            )

            row = {
                "order": order,
                "method": method,
                "dev_perplexity": best_trial.dev_perplexity,
                "dev_zero_prob_events": best_trial.dev_zero_prob_events,
                "dev_num_events": best_trial.dev_num_events,
                "test_perplexity": test_stats.perplexity,
                "test_zero_prob_events": test_stats.zero_prob_events,
                "test_num_events": test_stats.num_events,
                "params": best_trial.params,
                "params_json": json.dumps(best_trial.params, ensure_ascii=False, sort_keys=True),
            }
            comparison_rows.append(row)
            per_order_rank_rows[order].append(row)

            detailed_results[str(order)][method] = {
                "selected_hyperparams": best_trial.params,
                "dev": {
                    "perplexity": best_trial.dev_perplexity,
                    "zero_prob_events": best_trial.dev_zero_prob_events,
                    "num_events": best_trial.dev_num_events,
                    "perplexity_is_infinite": math.isinf(best_trial.dev_perplexity),
                },
                "test": test_stats.to_dict(),
                "grid_size": len(params_grid),
            }

    ranked_rows_all: list[dict] = []
    best_by_order_summary: dict[str, dict] = {}
    for order in orders:
        ranked = rank_methods(per_order_rank_rows[order])
        ranked_rows_all.extend(ranked)
        if ranked:
            best_by_order_summary[str(order)] = {
                "method": ranked[0]["method"],
                "rankings": [
                    {
                        "rank": r["rank"],
                        "method": r["method"],
                        "test_perplexity": r["test_perplexity"],
                        "dev_perplexity": r["dev_perplexity"],
                        "params": r["params"],
                    }
                    for r in ranked
                ],
            }

    ranked_by_order_map: dict[int, list[dict]] = {}
    for order in orders:
        ranked_by_order_map[order] = sorted(per_order_rank_rows[order], key=lambda r: r.get("rank", 999))

    overall_best = choose_overall_best(ranked_by_order_map)

    comparison_csv_rows = [
        {
            "order": r["order"],
            "method": r["method"],
            "dev_perplexity": r["dev_perplexity"],
            "test_perplexity": r["test_perplexity"],
            "dev_zero_prob_events": r["dev_zero_prob_events"],
            "test_zero_prob_events": r["test_zero_prob_events"],
            "rank": r.get("rank"),
            "hyperparams_json": r["params_json"],
        }
        for r in ranked_rows_all
    ]
    write_csv(
        out_dir / "comparison.csv",
        comparison_csv_rows,
        [
            "order",
            "method",
            "dev_perplexity",
            "test_perplexity",
            "dev_zero_prob_events",
            "test_zero_prob_events",
            "rank",
            "hyperparams_json",
        ],
    )
    write_csv(
        out_dir / "tuning_results.csv",
        tuning_rows,
        ["order", "method", "params_json", "dev_perplexity", "dev_zero_prob_events", "dev_num_events"],
    )

    summary = {
        "task": "Project 2 Task 2 - Smoothing comparison for n-gram LM",
        "config": {
            "corpus_path": args.corpus_path,
            "text_column": args.text_column,
            "test_ratio": args.test_ratio,
            "dev_ratio": args.dev_ratio,
            "seed": args.seed,
            "dev_seed": args.dev_seed,
            "max_docs": args.max_docs,
            "lowercase": args.lowercase,
            "unk_min_freq": args.unk_min_freq,
            "orders": orders,
            "methods": args.methods,
            "disable_grid_search": args.disable_grid_search,
            "out_dir": str(out_dir),
        },
        "data_split": prepared.stats_dict(),
        "vocab": prepared.vocab_dict(),
        "results_by_order": detailed_results,
        "best_method_by_order": best_by_order_summary,
        "overall_recommendation": overall_best,
        "selection_rule": {
            "primary": "lowest trigram test perplexity",
            "secondary": "report bigram rankings separately",
            "tiebreaker": "if trigram test perplexities differ by <0.5%, use lower dev perplexity; then simplicity",
        },
        "notes": [
            "Task 1 fairness settings are preserved (seed/test_ratio/tokenization/sentence segmentation/unk policy).",
            "Dev tuning uses a train_core-derived <unk> remap to avoid artificial zero-probability failures on dev-only tokens.",
            "Interpolation uses MLE component probabilities with lambda grid search on dev.",
            "Backoff uses a normalized absolute-discount backoff/interpolation hybrid (proper probabilities for perplexity).",
            "Kneser-Ney is interpolated Kneser-Ney with single discount D and continuation unigram probabilities.",
        ] + warnings,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    notes_text = [
        "Task 2 smoothing run completed.",
        f"Orders evaluated: {orders}",
        f"Methods evaluated: {args.methods}",
        f"Best trigram method: {overall_best.get('selected_method')}",
        f"Reason: {overall_best.get('reason')}",
    ]
    (out_dir / "notes.txt").write_text("\n".join(notes_text) + "\n", encoding="utf-8")

    print(f"Saved summary: {out_dir / 'summary.json'}")
    print(f"Saved comparison: {out_dir / 'comparison.csv'}")
    print(f"Saved tuning grid results: {out_dir / 'tuning_results.csv'}")
    if overall_best.get("selected_method"):
        print(
            f"Overall recommendation (trigram criterion): {overall_best['selected_method']} "
            f"(test perplexity={overall_best['test_perplexity']})"
        )


if __name__ == "__main__":
    main()
