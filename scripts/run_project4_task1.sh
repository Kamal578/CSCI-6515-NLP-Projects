#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: neither 'python3' nor 'python' is available on PATH." >&2
  exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

DATASET_NAME="${DATASET_NAME:-hajili/azerbaijani_review_sentiment_classification}"
TRAIN_FILE="${TRAIN_FILE:-}"
TEST_FILE="${TEST_FILE:-}"
MODEL_NAME="${MODEL_NAME:-google-bert/bert-base-multilingual-cased}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/project4/task1_sentiment}"
CACHE_DIR="${CACHE_DIR:-data/external/project4_cache}"
LABEL_MODE="${LABEL_MODE:-score5}"
MAX_LENGTH="${MAX_LENGTH:-192}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
BALANCE_STRATEGY="${BALANCE_STRATEGY:-both}"
DEVICE="${DEVICE:-auto}"
SMOKE="${SMOKE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_VAL_EXAMPLES="${MAX_VAL_EXAMPLES:-}"
MAX_TEST_EXAMPLES="${MAX_TEST_EXAMPLES:-}"

CMD=(
  "$PYTHON_BIN" -m src.project4_task1_sentiment
  --dataset_name "$DATASET_NAME"
  --model_name "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --cache_dir "$CACHE_DIR"
  --label_mode "$LABEL_MODE"
  --max_length "$MAX_LENGTH"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --epochs "$EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --balance_strategy "$BALANCE_STRATEGY"
  --device "$DEVICE"
)

if [[ -n "$TRAIN_FILE" && -n "$TEST_FILE" ]]; then
  CMD+=(--train_file "$TRAIN_FILE" --test_file "$TEST_FILE")
fi
if [[ -n "$MAX_TRAIN_EXAMPLES" ]]; then
  CMD+=(--max_train_examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "$MAX_VAL_EXAMPLES" ]]; then
  CMD+=(--max_val_examples "$MAX_VAL_EXAMPLES")
fi
if [[ -n "$MAX_TEST_EXAMPLES" ]]; then
  CMD+=(--max_test_examples "$MAX_TEST_EXAMPLES")
fi
if [[ "$SMOKE" == "1" ]]; then
  CMD+=(--smoke)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  CMD+=(--trust_remote_code)
fi

log "Project 4 Task 1 run started"
log "Using Python: $PYTHON_BIN"
log "Model: $MODEL_NAME"
log "Output dir: $OUTPUT_DIR"
log "Device: $DEVICE"

"${CMD[@]}"

cat <<EOF

Project 4 Task 1 outputs
- Summary: $OUTPUT_DIR/summary.json
- Model: $OUTPUT_DIR/model
- History: $OUTPUT_DIR/history.csv
- Validation predictions: $OUTPUT_DIR/validation_predictions.json
- Test predictions: $OUTPUT_DIR/test_predictions.json
EOF

log "Project 4 Task 1 run finished"
