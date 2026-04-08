#!/usr/bin/env bash
set -euo pipefail

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

VARIANT="${VARIANT:-compare}"
OUT_DIR="${OUT_DIR:-outputs/project4/task2_reading_comprehension}"
CACHE_DIR="${CACHE_DIR:-data/external/project4_cache}"
GLOVE_PATH="${GLOVE_PATH:-}"
BERT_MODEL_NAME="${BERT_MODEL_NAME:-bert-base-uncased}"
TRAIN_JSON="${TRAIN_JSON:-}"
VAL_JSON="${VAL_JSON:-}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"

mkdir -p "$(dirname "$OUT_DIR")"

CMD=(
  "$PYTHON_BIN" -m src.project4_task2_reading_comprehension
  --variant "$VARIANT"
  --out_dir "$OUT_DIR"
  --cache_dir "$CACHE_DIR"
  --bert_model_name "$BERT_MODEL_NAME"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --hidden_size "$HIDDEN_SIZE"
)

if [[ -n "$GLOVE_PATH" ]]; then
  CMD+=(--glove_path "$GLOVE_PATH")
fi
if [[ -n "$TRAIN_JSON" && -n "$VAL_JSON" ]]; then
  CMD+=(--train_json "$TRAIN_JSON" --val_json "$VAL_JSON")
fi
if [[ "$SMOKE" == "1" ]]; then
  CMD+=(--smoke)
fi

log "Project 4 Task 2 run started"
log "Using Python: $PYTHON_BIN"
log "Variant: $VARIANT"
log "Output dir: $OUT_DIR"

"${CMD[@]}"

cat <<EOF

Project 4 Task 2 outputs
- Comparison: $OUT_DIR/comparison.csv
- Notes: $OUT_DIR/report_notes.md
- GloVe summary: $OUT_DIR/glove/summary.json
- BERT summary: $OUT_DIR/bert/summary.json
EOF

log "Project 4 Task 2 run finished"
