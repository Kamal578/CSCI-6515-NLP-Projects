#!/usr/bin/env bash
set -euo pipefail

# Root of the project (handles spaces in path)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: neither 'python3' nor 'python' is available on PATH." >&2
  exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

CORPUS_PATH="${CORPUS_PATH:-data/raw/corpus.csv}"
TASK1_OUT_DIR="${TASK1_OUT_DIR:-outputs/project2/task1_lm}"
TASK2_OUT_DIR="${TASK2_OUT_DIR:-outputs/project2/task2_smoothing}"
TASK4_OUT_DIR="${TASK4_OUT_DIR:-outputs/project2/task4_lr_sent_gold_actual}"

# Task 4 defaults point to the current checked-in gold labels + pseudo-corpus.
TASK4_LABELS_CSV="${TASK4_LABELS_CSV:-data/processed/task4_dot_labels_from_sent_gold_actual.csv}"
TASK4_CORPUS_PATH="${TASK4_CORPUS_PATH:-data/processed/task4_sent_gold_actual_pseudo_corpus.csv}"

mkdir -p outputs/project2

if [[ ! -f "$CORPUS_PATH" ]]; then
  echo "Error: corpus file not found: $CORPUS_PATH" >&2
  exit 1
fi

log "Project 2 run started"
log "Using Python: $PYTHON_BIN"
log "Corpus: $CORPUS_PATH"

log "Task 1: Unigram/Bigram/Trigram LM + perplexity (MLE baseline)"
"$PYTHON_BIN" -m src.task1_lm \
  --corpus_path "$CORPUS_PATH" \
  --out_dir "$TASK1_OUT_DIR"

log "Task 2: Smoothing comparison (Laplace, Interpolation, Backoff, Kneser-Ney)"
"$PYTHON_BIN" -m src.task2_smoothing \
  --corpus_path "$CORPUS_PATH" \
  --out_dir "$TASK2_OUT_DIR"

if [[ -f "$TASK4_LABELS_CSV" && -f "$TASK4_CORPUS_PATH" ]]; then
  log "Task 4: Logistic regression dot EOS (L1 vs L2) + sentence detection"
  "$PYTHON_BIN" -m src.task4_sentence_lr \
    --labels_csv "$TASK4_LABELS_CSV" \
    --corpus_path "$TASK4_CORPUS_PATH" \
    --compare_rule_baseline \
    --out_dir "$TASK4_OUT_DIR"
else
  log "Task 4 skipped (missing inputs)"
  [[ -f "$TASK4_LABELS_CSV" ]] || log "Missing labels CSV: $TASK4_LABELS_CSV"
  [[ -f "$TASK4_CORPUS_PATH" ]] || log "Missing pseudo-corpus CSV: $TASK4_CORPUS_PATH"
  log "To run Task 4, set TASK4_LABELS_CSV and TASK4_CORPUS_PATH or create the default files."
fi

cat <<EOF

Project 2 outputs
- Task 1: $TASK1_OUT_DIR/summary.json
- Task 2: $TASK2_OUT_DIR/summary.json
- Task 4: $TASK4_OUT_DIR/summary.json
EOF

log "Project 2 run finished"
