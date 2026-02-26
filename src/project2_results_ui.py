from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as re
from flask import Flask, jsonify, request, send_from_directory

from .load_data import load_corpus_csv
from .project2_task3_sentiment_infer import bundle_metadata, load_bundle, predict_text
from .serve_spellcheck import expand_suggest, get_vocab, load_weights
from .sentence_segment import sentence_segment
from .tokenize import tokenize


app = Flask(__name__, static_folder=None)
WORD_TAIL_RE = re.compile(r"[\p{L}\p{N}_'’\-]+$", re.UNICODE)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFUSION_PATH = "outputs/spellcheck/confusion.json"
DEFAULT_SENTIMENT_BUNDLE_PATH = "outputs/project2/task3_sentiment/best_model_bundle.pkl"


@dataclass
class NgramSuggestionIndex:
    unigram_counts: Counter[str]
    bigram_next: dict[str, Counter[str]]
    trigram_next: dict[tuple[str, str], Counter[str]]
    num_docs: int
    num_sentences: int
    num_tokens: int

    @property
    def vocab_size(self) -> int:
        return len(self.unigram_counts)


def _top_items(counter: Counter[str] | None, k: int = 3) -> list[dict[str, int]]:
    if not counter:
        return []
    return [{"token": tok, "count": int(cnt)} for tok, cnt in counter.most_common(k)]


def _context_tokens_for_text(text: str) -> list[str]:
    # Reuse the project tokenizer for consistency with LM tasks.
    return tokenize(text or "", lowercase=True)


@lru_cache(maxsize=2)
def get_ngram_index(corpus_path: str, text_column: str = "text", max_docs: int | None = None) -> NgramSuggestionIndex:
    df = load_corpus_csv(corpus_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")

    texts = df[text_column].fillna("").astype(str).tolist()
    if max_docs is not None:
        texts = texts[:max_docs]

    unigram_counts: Counter[str] = Counter()
    bigram_next: dict[str, Counter[str]] = defaultdict(Counter)
    trigram_next: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    sent_count = 0
    tok_count = 0

    for text in texts:
        if not text.strip():
            continue
        for sent in sentence_segment(text):
            toks = tokenize(sent, lowercase=True)
            if not toks:
                continue
            sent_count += 1
            tok_count += len(toks)
            unigram_counts.update(toks)
            for i in range(1, len(toks)):
                bigram_next[toks[i - 1]][toks[i]] += 1
            for i in range(2, len(toks)):
                trigram_next[(toks[i - 2], toks[i - 1])][toks[i]] += 1

    return NgramSuggestionIndex(
        unigram_counts=unigram_counts,
        bigram_next=dict(bigram_next),
        trigram_next=dict(trigram_next),
        num_docs=len(texts),
        num_sentences=sent_count,
        num_tokens=tok_count,
    )


def _extract_current_partial_word(text: str) -> str:
    m = WORD_TAIL_RE.search(text or "")
    return m.group(0) if m else ""


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_project_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@lru_cache(maxsize=2)
def _get_sentiment_bundle(bundle_path: str):
    return load_bundle(bundle_path)


def _ngram_suggestions_payload(text: str, corpus_path: str, text_column: str, max_docs_ngram: int | None = None) -> dict:
    # Use a consistent positional call style so lru_cache keys are reused.
    idx = get_ngram_index(corpus_path, text_column, max_docs_ngram)
    toks = _context_tokens_for_text(text)

    last1 = toks[-1] if len(toks) >= 1 else None
    last2 = tuple(toks[-2:]) if len(toks) >= 2 else None

    unigram_top = _top_items(idx.unigram_counts, k=3)
    bigram_top = _top_items(idx.bigram_next.get(last1) if last1 else None, k=3)
    trigram_top = _top_items(idx.trigram_next.get(last2) if last2 else None, k=3)

    return {
        "input_text": text,
        "context_tokens": toks[-6:],
        "suggestions": {
            "1gram": {
                "context": [],
                "items": unigram_top,
                "fallback_used": False,
            },
            "2gram": {
                "context": [last1] if last1 else [],
                "items": bigram_top if bigram_top else unigram_top,
                "fallback_used": bool(last1) and not bigram_top,
            },
            "3gram": {
                "context": list(last2) if last2 else [],
                "items": trigram_top if trigram_top else (bigram_top if bigram_top else unigram_top),
                "fallback_used": bool(last2) and not trigram_top,
            },
        },
        "index_stats": {
            "num_docs_indexed": idx.num_docs,
            "num_sentences_indexed": idx.num_sentences,
            "num_tokens_indexed": idx.num_tokens,
            "vocab_size": idx.vocab_size,
        },
    }


@lru_cache(maxsize=512)
def _cached_spell_candidates(
    word: str,
    corpus_path: str,
    confusion_path: str | None,
    top_k: int,
    max_dist: int,
    max_variant_edits: int,
    max_variant_candidates: int,
) -> tuple[tuple[str, int], ...]:
    vocab = get_vocab(corpus_path, 2, 3, 0.6)
    weights = load_weights(confusion_path)
    cands, _ = expand_suggest(
        word=word,
        vocab=vocab,
        max_dist=max_dist,
        top_k=top_k,
        weights=weights,
        max_variant_edits=max_variant_edits,
        max_variant_candidates=max_variant_candidates,
        debug=False,
    )
    return tuple((str(w), int(freq)) for w, freq in cands)


def _spell_suggestions_payload(
    word: str,
    corpus_path: str,
    use_confusion: bool = False,
    confusion_path: str = DEFAULT_CONFUSION_PATH,
    top_k: int = 3,
    max_dist: int = 2,
    max_variant_edits: int = 3,
    max_variant_candidates: int = 80,
) -> dict:
    resolved_confusion_path = confusion_path if use_confusion else None
    cands = _cached_spell_candidates(
        word=word.lower(),
        corpus_path=corpus_path,
        confusion_path=resolved_confusion_path,
        top_k=top_k,
        max_dist=max_dist,
        max_variant_edits=max_variant_edits,
        max_variant_candidates=max_variant_candidates,
    )
    vocab = get_vocab(corpus_path, 2, 3, 0.6)
    weights = load_weights(resolved_confusion_path)
    return {
        "word": word,
        "candidates": [{"token": w, "freq": int(freq)} for w, freq in cands],
        "vocab_size": len(vocab),
        "used_confusion": bool(weights),
    }


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "project2_results_ui.html")


@app.route("/api/spellchecker", methods=["POST"])
def api_spellchecker():
    data = request.get_json(silent=True) or {}
    word = str(data.get("word", "")).strip()
    corpus_path = str(data.get("corpus_path", "data/raw/corpus.csv"))
    use_confusion = _as_bool(data.get("use_confusion"), default=False)
    confusion_path = str(data.get("confusion_path", DEFAULT_CONFUSION_PATH))
    if not word:
        return jsonify({"error": "word is required"}), 400
    return jsonify(
        _spell_suggestions_payload(
            word=word,
            corpus_path=corpus_path,
            use_confusion=use_confusion,
            confusion_path=confusion_path,
            top_k=5,
            max_dist=2,
            max_variant_edits=3,
            max_variant_candidates=80,
        )
    )


@app.route("/api/ngram_suggestions", methods=["POST"])
def api_ngram_suggestions():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))
    corpus_path = str(data.get("corpus_path", "data/raw/corpus.csv"))
    text_column = str(data.get("text_column", "text"))
    max_docs_ngram = data.get("max_docs_ngram", None)
    max_docs_ngram = int(max_docs_ngram) if max_docs_ngram not in (None, "", "null") else None
    return jsonify(_ngram_suggestions_payload(text, corpus_path, text_column, max_docs_ngram=max_docs_ngram))


@app.route("/api/sentence_delimiter", methods=["POST"])
def api_sentence_delimiter():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    sentences = sentence_segment(text)
    return jsonify(
        {
            "sentences": sentences,
            "sentence_count": len(sentences),
        }
    )


@app.route("/api/typing_assist", methods=["POST"])
def api_typing_assist():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))
    corpus_path = str(data.get("corpus_path", "data/raw/corpus.csv"))
    text_column = str(data.get("text_column", "text"))
    use_confusion = _as_bool(data.get("use_confusion"), default=False)
    confusion_path = str(data.get("confusion_path", DEFAULT_CONFUSION_PATH))
    max_docs_ngram = data.get("max_docs_ngram", None)
    max_docs_ngram = int(max_docs_ngram) if max_docs_ngram not in (None, "", "null") else None

    if not text:
        return jsonify({"mode": "empty", "message": "Start typing to see suggestions."})

    if text[-1].isspace():
        payload = _ngram_suggestions_payload(text, corpus_path, text_column, max_docs_ngram=max_docs_ngram)
        return jsonify(
            {
                "mode": "next_word",
                "input_text": text,
                "ngram_suggestions": payload["suggestions"],
                "index_stats": payload["index_stats"],
            }
        )

    partial = _extract_current_partial_word(text)
    if not partial:
        return jsonify({"mode": "empty", "message": "Type letters to get spell suggestions."})
    if len(partial) < 3:
        return jsonify(
            {
                "mode": "spellcheck",
                "input_text": text,
                "current_token": partial,
                "spell_suggestions": [],
                "used_confusion": False,
                "message": "Type at least 3 characters for spell suggestions.",
            }
        )

    # Typing mode uses stricter limits than the dedicated spellchecker tab to stay responsive.
    spell = _spell_suggestions_payload(
        word=partial,
        corpus_path=corpus_path,
        use_confusion=use_confusion,
        confusion_path=confusion_path,
        top_k=3,
        max_dist=1 if len(partial) <= 4 else 2,
        max_variant_edits=1,
        max_variant_candidates=12,
    )
    return jsonify(
        {
            "mode": "spellcheck",
            "input_text": text,
            "current_token": partial,
            "spell_suggestions": spell["candidates"],
            "used_confusion": spell["used_confusion"],
        }
    )


@app.route("/api/ui_status", methods=["GET"])
def api_ui_status():
    corpus_path = request.args.get("corpus_path", "data/raw/corpus.csv")
    corpus_exists = _resolve_project_path(str(corpus_path)).exists()
    confusion_exists = _resolve_project_path(DEFAULT_CONFUSION_PATH).exists()
    sentiment_bundle_exists = _resolve_project_path(DEFAULT_SENTIMENT_BUNDLE_PATH).exists()
    spell_cache_info = get_vocab.cache_info()
    spell_vocab_cached = bool(spell_cache_info.currsize)

    return jsonify(
        {
            "corpus_exists": corpus_exists,
            "confusion_exists": confusion_exists,
            "sentiment_bundle_exists": sentiment_bundle_exists,
            "spell_vocab_cached": spell_vocab_cached,
            "spell_cache_info": {
                "hits": spell_cache_info.hits,
                "misses": spell_cache_info.misses,
                "currsize": spell_cache_info.currsize,
            },
            "ngram_index_cached": bool(get_ngram_index.cache_info().currsize),
            "ngram_cache_info": {
                "hits": get_ngram_index.cache_info().hits,
                "misses": get_ngram_index.cache_info().misses,
                "currsize": get_ngram_index.cache_info().currsize,
            },
        }
    )


@app.route("/api/sentiment_bundle_info", methods=["GET"])
def api_sentiment_bundle_info():
    bundle_path = request.args.get("bundle_path", DEFAULT_SENTIMENT_BUNDLE_PATH)
    p = _resolve_project_path(str(bundle_path))
    if not p.exists():
        return jsonify(
            {
                "bundle_exists": False,
                "bundle_path": str(p),
                "error": (
                    "Sentiment bundle not found. Build it with: "
                    "python -m src.project2_task3_sentiment_infer "
                    "--summary outputs/project2/task3_sentiment/summary.json "
                    "--out outputs/project2/task3_sentiment/best_model_bundle.pkl"
                ),
            }
        ), 404
    try:
        bundle = _get_sentiment_bundle(str(p))
        return jsonify(
            {
                "bundle_exists": True,
                "bundle_path": str(p),
                "bundle_info": bundle_metadata(bundle),
            }
        )
    except Exception as exc:
        return jsonify({"bundle_exists": True, "bundle_path": str(p), "error": str(exc)}), 500


@app.route("/api/sentiment_predict", methods=["POST"])
def api_sentiment_predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))
    bundle_path = str(data.get("bundle_path", DEFAULT_SENTIMENT_BUNDLE_PATH))
    bundle_file = _resolve_project_path(bundle_path)
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    if not bundle_file.exists():
        return jsonify(
            {
                "error": "sentiment bundle not found",
                "bundle_path": str(bundle_file),
                "how_to_build": (
                    "python -m src.project2_task3_sentiment_infer "
                    "--summary outputs/project2/task3_sentiment/summary.json "
                    "--out outputs/project2/task3_sentiment/best_model_bundle.pkl"
                ),
            }
        ), 404
    try:
        bundle = _get_sentiment_bundle(str(bundle_file))
        pred = predict_text(bundle, text)
        pred["bundle_path"] = str(bundle_file)
        return jsonify(pred)
    except Exception as exc:
        return jsonify({"error": str(exc), "bundle_path": str(bundle_file)}), 500


def main() -> None:
    ap = argparse.ArgumentParser(description="Project 2 results UI (spellcheck, n-grams, sentence delimiter, typing assistance).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
