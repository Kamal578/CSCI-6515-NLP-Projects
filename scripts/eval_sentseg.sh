#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end sentence segmentation eval pipeline.
# Assumes:
#   - data/raw/corpus.csv exists
#   - You provide a gold sentences-per-line file for the same subset.
#
# Usage:
#   bash scripts/eval_sentseg.sh data/processed/sent_gold.txt 50
#     arg1: path to gold sentences-per-line file (manually segmented)
#     arg2: number of rows from corpus to evaluate (default 50)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GOLD_SENTENCES=${1:-data/processed/sent_gold.txt}
LIMIT=${2:-50}

if [[ ! -f "$GOLD_SENTENCES" ]]; then
  echo "Gold sentences file not found: $GOLD_SENTENCES"
  echo "Please create a manually segmented sentences-per-line file for the same subset of the corpus."
  exit 1
fi

SUBSET_CSV="/tmp/corpus_subset.csv"
PRED_SENTENCES="/tmp/sent_pred.txt"
GOLD_IDX="data/processed/sent_gold_indices.txt"
PRED_IDX="outputs/sentences_pred_indices.txt"
OUT_JSON="outputs/sentseg_eval.json"

mkdir -p data/processed outputs

echo "[1/5] Extracting first $LIMIT rows from corpus to $SUBSET_CSV"
python - <<PY
import pandas as pd
df = pd.read_csv("data/raw/corpus.csv")
df.head(${LIMIT}).to_csv("${SUBSET_CSV}", index=False)
PY

echo "[2/5] Running sentence segmentation on subset -> $PRED_SENTENCES"
python -m src.sentence_segment --corpus_path "$SUBSET_CSV" --limit "$LIMIT" --out "$PRED_SENTENCES"

echo "[3/5] Converting gold sentences to boundary indices -> $GOLD_IDX"
nl -ba "$GOLD_SENTENCES" | awk '{print $1-1}' > "$GOLD_IDX"

echo "[4/5] Converting predicted sentences to boundary indices -> $PRED_IDX"
nl -ba "$PRED_SENTENCES" | awk '{print $1-1}' > "$PRED_IDX"

echo "[5/5] Evaluating"
python -m src.evaluate_segmentation \
  --gold "$GOLD_IDX" \
  --pred "$PRED_IDX" \
  --out "$OUT_JSON"

echo "Done. Metrics saved to $OUT_JSON"
