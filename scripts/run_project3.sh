#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: neither python3 nor python found." >&2
  exit 1
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

SMOKE=0
TASKS="1,2,3,4,5,6"
RUN_UI=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --tasks) TASKS="${2:-}"; shift 2 ;;
    --with-ui) RUN_UI=1; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--smoke] [--tasks 1,2,3,4,5,6] [--with-ui]" >&2
      exit 1
      ;;
  esac
done

has_task() {
  local x="$1"
  [[ ",$TASKS," == *",$x,"* ]]
}

COMMON_CORPUS="data/raw/corpus.csv"
TRAIN_CSV="data/external/train.csv"
TEST_CSV="data/external/test.csv"
OUT_ROOT="outputs/project3"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$COMMON_CORPUS" ]]; then
  echo "Missing corpus: $COMMON_CORPUS" >&2
  exit 1
fi

log "Project 3 run started (smoke=$SMOKE tasks=$TASKS)"
log "Using Python: $PYTHON_BIN"

if has_task 1; then
  log "Task 1: Dataset analysis + matrices"
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" -m src.project3_task1_matrices \
      --corpus_path "$COMMON_CORPUS" \
      --out_dir "$OUT_ROOT/task1_matrices" \
      --max_docs 1200 --top_docs 60 --top_terms 80
  else
    "$PYTHON_BIN" -m src.project3_task1_matrices \
      --corpus_path "$COMMON_CORPUS" \
      --out_dir "$OUT_ROOT/task1_matrices"
  fi
fi

if has_task 2; then
  log "Task 2: Word2Vec training + intrinsic evaluation"
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" -m src.project3_task2_word2vec \
      --corpus_path "$COMMON_CORPUS" \
      --out_dir "$OUT_ROOT/task2_word2vec" \
      --shared_corpus_out "$OUT_ROOT/shared/tokenized_corpus.txt" \
      --max_docs 3000 --iters 3 --size 100 --threads 4
  else
    "$PYTHON_BIN" -m src.project3_task2_word2vec \
      --corpus_path "$COMMON_CORPUS" \
      --out_dir "$OUT_ROOT/task2_word2vec" \
      --shared_corpus_out "$OUT_ROOT/shared/tokenized_corpus.txt"
  fi
fi

if has_task 3; then
  log "Task 3: GloVe training + intrinsic evaluation"
  if [[ "$SMOKE" -eq 1 ]]; then
    "$PYTHON_BIN" -m src.project3_task3_glove \
      --corpus_path "$COMMON_CORPUS" \
      --out_dir "$OUT_ROOT/task3_glove" \
      --shared_corpus_out "$OUT_ROOT/shared/tokenized_corpus.txt" \
      --max_docs 3000 --max_iter 6 --vector_size 100 --threads 4
  else
    "$PYTHON_BIN" -m src.project3_task3_glove \
      --corpus_path "$COMMON_CORPUS" \
      --out_dir "$OUT_ROOT/task3_glove" \
      --shared_corpus_out "$OUT_ROOT/shared/tokenized_corpus.txt"
  fi
fi

if has_task 4; then
  log "Task 4: Word2Vec vs GloVe comparison"
  "$PYTHON_BIN" -m src.project3_task4_compare \
    --word2vec_summary "$OUT_ROOT/task2_word2vec/summary.json" \
    --glove_summary "$OUT_ROOT/task3_glove/summary.json" \
    --word2vec_neighbors "$OUT_ROOT/task2_word2vec/nearest_neighbors.csv" \
    --glove_neighbors "$OUT_ROOT/task3_glove/nearest_neighbors.csv" \
    --out_dir "$OUT_ROOT/task4_compare" \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV"
fi

if has_task 5; then
  if [[ ! -f "$TRAIN_CSV" || ! -f "$TEST_CSV" ]]; then
    log "Task 5 skipped: missing sentiment CSVs ($TRAIN_CSV, $TEST_CSV)"
  elif [[ ! -f "$OUT_ROOT/task2_word2vec/vectors.txt" || ! -f "$OUT_ROOT/task3_glove/vectors.txt" ]]; then
    log "Task 5 skipped: missing embedding vectors. Run Tasks 2 and 3 first."
  else
    log "Task 5: RNN/BiRNN/LSTM x 5 feature methods"
    if [[ "$SMOKE" -eq 1 ]]; then
      "$PYTHON_BIN" -m src.project3_task5_dl \
        --train "$TRAIN_CSV" \
        --test "$TEST_CSV" \
        --word2vec_vectors "$OUT_ROOT/task2_word2vec/vectors.txt" \
        --glove_vectors "$OUT_ROOT/task3_glove/vectors.txt" \
        --out_dir "$OUT_ROOT/task5_dl" \
        --smoke
    else
      "$PYTHON_BIN" -m src.project3_task5_dl \
        --train "$TRAIN_CSV" \
        --test "$TEST_CSV" \
        --word2vec_vectors "$OUT_ROOT/task2_word2vec/vectors.txt" \
        --glove_vectors "$OUT_ROOT/task3_glove/vectors.txt" \
        --out_dir "$OUT_ROOT/task5_dl"
    fi
  fi
fi

if has_task 6; then
  log "Task 6: Consolidated report artifacts"
  "$PYTHON_BIN" -m src.project3_task6_report \
    --task1_summary "$OUT_ROOT/task1_matrices/summary.json" \
    --task2_summary "$OUT_ROOT/task2_word2vec/summary.json" \
    --task3_summary "$OUT_ROOT/task3_glove/summary.json" \
    --task4_summary "$OUT_ROOT/task4_compare/summary.json" \
    --task5_summary "$OUT_ROOT/task5_dl/summary.json" \
    --task4_csv "$OUT_ROOT/task4_compare/comparison.csv" \
    --task5_csv "$OUT_ROOT/task5_dl/leaderboard.csv" \
    --out_dir "$OUT_ROOT/task6_report"
fi

cat <<EOF

Project 3 outputs
- Task 1: $OUT_ROOT/task1_matrices/summary.json
- Task 2: $OUT_ROOT/task2_word2vec/summary.json
- Task 3: $OUT_ROOT/task3_glove/summary.json
- Task 4: $OUT_ROOT/task4_compare/summary.json
- Task 5: $OUT_ROOT/task5_dl/summary.json
- Task 6: $OUT_ROOT/task6_report/summary.json
EOF

if [[ "$RUN_UI" -eq 1 ]]; then
  log "Starting Project 3 UI server on http://127.0.0.1:5060"
  exec "$PYTHON_BIN" -m src.project3_results_ui --output-root "$OUT_ROOT" --host 127.0.0.1 --port 5060
fi

log "Project 3 run finished"
