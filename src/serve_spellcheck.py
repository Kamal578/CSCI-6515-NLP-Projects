from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

from flask import Flask, request, jsonify, send_from_directory

from .spellcheck import load_vocab, suggest

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


def load_weights(confusion_path: str | None):
    if not confusion_path:
        return None
    p = Path(confusion_path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {eval(k): float(v) for k, v in raw.get("weights", {}).items()}


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

    cands = suggest(word, vocab, max_dist=max_dist, top_k=top_k, weights=weights)
    return jsonify({
        "word": word,
        "candidates": [{"token": w, "freq": freq} for w, freq in cands],
        "used_confusion": bool(weights),
        "vocab_size": len(vocab),
    })


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
