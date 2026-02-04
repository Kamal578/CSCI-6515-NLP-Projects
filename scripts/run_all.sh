#!/usr/bin/env bash
set -euo pipefail

# Root of the project (handles spaces in path)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_TXT="outputs/run_summary.txt"

mkdir -p outputs/stats outputs/plots outputs/bpe outputs/spellcheck data/processed

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Running Task 1: token stats + Zipf plot"
python -m src.task1_stats --corpus_path data/raw/corpus.csv --out_dir outputs/stats --plots_dir outputs/plots --top_n 2000

log "Running Task 2: Heaps' law"
python -m src.heaps --corpus_path data/raw/corpus.csv --out_stats outputs/stats/heaps_params.json --out_plot outputs/plots/heaps.png --step 1000

log "Running Task 3: BPE"
python -m src.task3_bpe --corpus_path data/raw/corpus.csv --out_dir outputs/bpe --num_merges 5000 --min_word_freq 2 --sample_words 30

log "Building filtered vocabulary"
python -m src.build_vocab --corpus_path data/raw/corpus.csv --min_freq 3 --min_len 3 --vocab_path data/processed/vocab.txt --summary_path outputs/stats/vocab_summary.json

log "Generating synthetic spellcheck test set"
python -m src.make_spell_test --corpus_path data/raw/corpus.csv --samples 1000 --out_csv data/processed/spell_test.csv

log "Building confusion matrix + weights"
python -m src.confusion --pairs_csv data/processed/spell_test.csv --out_confusion outputs/spellcheck/confusion.json

log "Evaluating spellchecker (weighted edit distance)"
python -m src.eval_spellcheck --test_csv data/processed/spell_test.csv --confusion outputs/spellcheck/confusion.json --out_summary outputs/spellcheck/spell_eval.json --out_samples outputs/spellcheck/sample_predictions.csv

log "Sentence segmentation over corpus (first N docs)"
python -m src.sentence_segment --corpus_path data/raw/corpus.csv --limit 500 --out outputs/sentences.txt

log "Spellcheck low-frequency tokens (suggestions)"
# sample 200 lowest-frequency vocab entries (proxy for likely errors)
python - <<'PY'
from pathlib import Path
from collections import Counter
from src.spell_utils import load_freqs, filter_vocab

freqs = load_freqs("data/raw/corpus.csv", lowercase=True)
vocab = filter_vocab(freqs, min_freq=1, min_len=3)
rare = [w for w, c in vocab.most_common()][::-1][:200]  # reverse to get rarest 200
Path("outputs/spellcheck/rare_words.txt").write_text("\n".join(rare), encoding="utf-8")
PY
python -m src.spellcheck --corpus_path data/raw/corpus.csv --wordlist outputs/spellcheck/rare_words.txt --out outputs/spellcheck/suggestions.txt

log "Collecting key stats into $OUT_TXT"
python - <<'PY'
import json, pathlib, pandas as pd, datetime
root = pathlib.Path(".")
out = root / "outputs" / "run_summary.txt"
parts = []
parts.append(f"Run timestamp: {datetime.datetime.now().isoformat()}")
def j(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
summary = j(root/"outputs/stats/summary.json")
heaps = j(root/"outputs/stats/heaps_params.json")
bpe = j(root/"outputs/bpe/bpe_summary.json")
vocab = j(root/"outputs/stats/vocab_summary.json")
spell = j(root/"outputs/spellcheck/spell_eval.json")

parts.append("\n[Corpus]")
for k in ("documents","num_tokens","num_types"):
    if k in summary: parts.append(f"{k}: {summary[k]}")

if heaps:
    parts.append("\n[Heaps]")
    parts.append(f"k: {heaps.get('k'):.2f}  beta: {heaps.get('beta'):.3f}")
    parts.append(f"final_N: {heaps.get('final_N')}  final_V: {heaps.get('final_V')}")

if bpe:
    parts.append("\n[BPE]")
    parts.append(f"num_merges: {bpe.get('num_merges')}")
    parts.append(f"bpe_types: {bpe.get('bpe_types')}  bpe_tokens_total: {bpe.get('bpe_tokens_total')}")

if vocab:
    parts.append("\n[Vocab]")
    parts.append(f"types_after_filter: {vocab.get('tokens_after_filter')} (min_freq={vocab.get('min_freq')}, min_len={vocab.get('min_len')})")

if spell:
    parts.append("\n[Spellcheck Eval]")
    parts.append(f"accuracy@1: {spell.get('accuracy@1'):.3f}  accuracy@5: {spell.get('accuracy@5'):.3f}  n={spell.get('total')}")

out.write_text("\n".join(parts), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
PY

log "Done."
