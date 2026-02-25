from __future__ import annotations

import ast
import json
from pathlib import Path
from functools import lru_cache

from flask import Flask, request, jsonify, send_from_directory

from .spellcheck import load_vocab, suggest
from .spell_utils import generate_az_variants_with_edits
from .levenshtein import levenshtein

app = Flask(__name__, static_folder=None)


@lru_cache(maxsize=1)
def get_vocab(corpus_path: str, min_freq: int, min_len: int, max_upper_ratio: float):
    return load_vocab(
        corpus_path,
        lowercase=True,
        min_freq=min_freq,
        min_len=min_len,
        max_upper_ratio=max_upper_ratio,
    )


@lru_cache(maxsize=8)
def load_weights(confusion_path: str | None):
    if not confusion_path:
        return None
    p = Path(confusion_path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {ast.literal_eval(k): float(v) for k, v in raw.get("weights", {}).items()}


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "spell_ui.html")


@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    data = request.get_json(silent=True) or {}
    word = str(data.get("word", "")).strip()
    if not word:
        return jsonify({"error": "word is required"}), 400

    corpus_path = data.get("corpus_path", "data/raw/corpus.csv")
    min_freq = int(data.get("min_freq", 2))
    min_len = int(data.get("min_len", 3))
    max_upper_ratio = float(data.get("max_upper_ratio", 0.6))
    max_dist = int(data.get("max_dist", 2))
    top_k = int(data.get("top_k", 5))
    confusion_path = data.get("confusion")

    vocab = get_vocab(corpus_path, min_freq, min_len, max_upper_ratio)
    weights = load_weights(confusion_path)

    debug = str(data.get("debug", "0")).strip() in {"1", "true", "yes", "on"}
    cands, debug_info = expand_suggest(
        word,
        vocab,
        max_dist=max_dist,
        top_k=top_k,
        weights=weights,
        max_variant_edits=int(data.get("variant_max_edits", 3)),
        max_variant_candidates=int(data.get("variant_max_candidates", 80)),
        debug=debug,
    )
    payload = {
        "word": word,
        "candidates": [{"token": w, "freq": freq} for w, freq in cands],
        "used_confusion": bool(weights),
        "vocab_size": len(vocab),
    }
    if debug and debug_info is not None:
        payload["debug"] = debug_info
    return jsonify(payload)


def expand_suggest(
    word: str,
    vocab: dict,
    max_dist: int,
    top_k: int,
    weights: dict | None,
    max_variant_edits: int = 3,
    max_variant_candidates: int = 40,
    debug: bool = False,
) -> tuple[list[tuple[str, int]], dict | None]:
    """
    Always try the raw word plus all AZ variants, then merge and rank results.
    """
    query = word.lower()
    merged: dict[str, tuple[int, int, int]] = {}  # token -> (min_dist, freq, variant_edits)
    checked: list[dict[str, int | str]] = []

    def consider(token: str, freq: int, variant_edits: int, variant_dist: int) -> None:
        existing = merged.get(token)
        if existing is None:
            merged[token] = (variant_dist, freq, variant_edits)
            return
        ex_dist, ex_freq, ex_edits = existing
        if variant_dist < ex_dist:
            merged[token] = (variant_dist, freq, variant_edits)
            return
        if variant_dist == ex_dist:
            if freq > ex_freq:
                merged[token] = (variant_dist, freq, variant_edits)
                return
            if freq == ex_freq and variant_edits < ex_edits:
                merged[token] = (variant_dist, freq, variant_edits)

    variants = generate_az_variants_with_edits(
        query,
        max_edits=max_variant_edits,
        max_candidates=max_variant_candidates,
    )
    # Ensure the raw query is tried first, then all other variants in generated order.
    ordered_variants = [(query, 0)] + [(v, e) for v, e in variants if v != query]
    for variant, edits in ordered_variants:
        variant_results = suggest(variant, vocab, max_dist=max_dist, top_k=top_k, weights=weights)
        if debug:
            checked.append({"candidate": variant, "count_returned": len(variant_results)})
        for tok, freq in variant_results:
            dist = levenshtein(variant, tok, max_dist=max_dist + 2)
            consider(tok, freq, edits, dist)

    scored: list[tuple[str, int, int, int]] = []
    for tok, (dist, freq, edits) in merged.items():
        scored.append((tok, freq, edits, dist))

    # Order: smaller distance, higher freq, fewer variant edits, token
    scored.sort(key=lambda x: (x[3], -x[1], x[2], x[0]))
    result = [(tok, freq) for tok, freq, _, _ in scored[:top_k]]
    debug_info = None
    if debug:
        debug_info = {
            "checked_candidates": checked,
            "limits": {"max_edits": max_variant_edits, "max_candidates": max_variant_candidates},
        }
    return result, debug_info


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Serve a simple spellcheck web UI.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
