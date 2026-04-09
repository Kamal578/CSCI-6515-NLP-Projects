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

VARIANT="${VARIANT:-compare}"
OUT_DIR="${OUT_DIR:-outputs/project4/task2_reading_comprehension}"
CACHE_DIR="${CACHE_DIR:-data/external/project4_cache}"
GLOVE_PATH="${GLOVE_PATH:-}"
BERT_MODEL_NAME="${BERT_MODEL_NAME:-bert-base-uncased}"
TRAIN_JSON="${TRAIN_JSON:-}"
VAL_JSON="${VAL_JSON:-}"
SMOKE="${SMOKE:-0}"
MEDIUM="${MEDIUM:-0}"
LARGE="${LARGE:-0}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
GLOVE_LEARNING_RATE="${GLOVE_LEARNING_RATE:-2e-3}"
BERT_LEARNING_RATE="${BERT_LEARNING_RATE:-5e-4}"
EMBEDDING_DIM="${EMBEDDING_DIM:-}"
DEVICE="${DEVICE:-auto}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_VAL_EXAMPLES="${MAX_VAL_EXAMPLES:-}"

if [[ -z "$GLOVE_PATH" && -f "outputs/project3/task3_glove/vectors.txt" ]]; then
  GLOVE_PATH="outputs/project3/task3_glove/vectors.txt"
  if [[ -z "$EMBEDDING_DIM" ]]; then
    EMBEDDING_DIM="200"
  fi
  log "Using local Project 3 GloVe vectors: $GLOVE_PATH"
fi

if [[ -z "$EMBEDDING_DIM" ]]; then
  EMBEDDING_DIM="100"
fi

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
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --glove_learning_rate "$GLOVE_LEARNING_RATE"
  --bert_learning_rate "$BERT_LEARNING_RATE"
  --hidden_size "$HIDDEN_SIZE"
  --embedding_dim "$EMBEDDING_DIM"
  --device "$DEVICE"
)

if [[ -n "$GLOVE_PATH" ]]; then
  CMD+=(--glove_path "$GLOVE_PATH")
fi
if [[ -n "$TRAIN_JSON" && -n "$VAL_JSON" ]]; then
  CMD+=(--train_json "$TRAIN_JSON" --val_json "$VAL_JSON")
fi
if [[ -n "$MAX_TRAIN_EXAMPLES" ]]; then
  CMD+=(--max_train_examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "$MAX_VAL_EXAMPLES" ]]; then
  CMD+=(--max_val_examples "$MAX_VAL_EXAMPLES")
fi
if [[ "$SMOKE" == "1" ]]; then
  CMD+=(--smoke)
fi
if [[ "$MEDIUM" == "1" ]]; then
  CMD+=(--medium)
fi
if [[ "$LARGE" == "1" ]]; then
  CMD+=(--large)
fi

log "Project 4 Task 2 run started"
log "Using Python: $PYTHON_BIN"
log "Variant: $VARIANT"
log "Output dir: $OUT_DIR"
if [[ "$SMOKE" == "1" ]]; then
  log "Preset: smoke"
elif [[ "$MEDIUM" == "1" ]]; then
  log "Preset: medium"
elif [[ "$LARGE" == "1" ]]; then
  log "Preset: large"
else
  log "Preset: full"
fi

"${CMD[@]}"

cat <<EOF

Project 4 Task 2 outputs
- Comparison: $OUT_DIR/comparison.csv
- Notes: $OUT_DIR/report_notes.md
- GloVe summary: $OUT_DIR/glove/summary.json
- BERT summary: $OUT_DIR/bert/summary.json
EOF

log "Project 4 Task 2 run finished"
