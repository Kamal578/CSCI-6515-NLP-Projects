# Project 4 README

This file documents `Project 4, Task 2`: reading comprehension with a BiDAF span predictor using either pretrained GloVe embeddings or frozen BERT embeddings.

## Task 2 Goal
- Train a reading-comprehension model that predicts answer start/end positions inside a context passage.
- Compare two variants on `SQuAD 1.1`:
  - `glove` — BiDAF with pretrained `glove.6B.100d`
  - `bert` — BiDAF with frozen `bert-base-uncased` contextual word embeddings
- Report `Exact Match (EM)` and `F1`.

## Main Entry Point
```bash
.venv/bin/python -m src.project4_task2_reading_comprehension --variant compare
```

Key flags:
- `--variant glove|bert|compare`
- `--train_json` / `--val_json` to use local SQuAD-format JSON files instead of downloading SQuAD
- `--cache_dir` for dataset/model/GloVe caches
- `--glove_path` to point at an existing `glove.6B.100d.txt`
- `--embedding_dim` to control the BERT projection size and, when using custom GloVe text vectors, the expected comparison dimension
- `--max_train_examples`, `--max_val_examples`, `--epochs`, `--batch_size`
- `--max_question_words`, `--context_window_words`, `--doc_stride_words`
- `--medium` for an intermediate CPU-friendly run
- `--log_every_steps` to print train/eval batch progress
- `--smoke` for a tiny CPU-friendly run

## Smoke Run
Use this first to validate the pipeline quickly:

```bash
.venv/bin/python -m src.project4_task2_reading_comprehension \
  --variant compare \
  --smoke \
  --max_train_examples 16 \
  --max_val_examples 8 \
  --out_dir outputs/project4/task2_reading_comprehension_smoke
```

## Full Run
This is much slower on CPU because the `bert` variant computes frozen contextual embeddings for every batch.

```bash
.venv/bin/python -m src.project4_task2_reading_comprehension \
  --variant compare \
  --epochs 4 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --context_window_words 160 \
  --doc_stride_words 64 \
  --out_dir outputs/project4/task2_reading_comprehension
```

## Medium Run
Use this when smoke is too small but a full compare run is too slow for interactive work:

```bash
MEDIUM=1 bash scripts/run_project4_task2.sh
```

Recommended first-pass workflow on CPU:

```bash
VARIANT=glove MEDIUM=1 bash scripts/run_project4_task2.sh
VARIANT=bert MEDIUM=1 bash scripts/run_project4_task2.sh
```

## Shell Runner
```bash
bash scripts/run_project4_task2.sh
```

Useful environment variables:
- `VARIANT=glove|bert|compare`
- `SMOKE=1`
- `MEDIUM=1`
- `OUT_DIR=...`
- `CACHE_DIR=...`
- `GLOVE_PATH=...`
- `EMBEDDING_DIM=100`
- `DEVICE=auto|cpu|cuda|mps`
- `TRAIN_JSON=...`
- `VAL_JSON=...`
- `MAX_TRAIN_EXAMPLES=...`
- `MAX_VAL_EXAMPLES=...`

## Output Files
- `outputs/project4/task2_reading_comprehension/comparison.csv`
- `outputs/project4/task2_reading_comprehension/report_notes.md`
- `outputs/project4/task2_reading_comprehension/glove/summary.json`
- `outputs/project4/task2_reading_comprehension/glove/history.csv`
- `outputs/project4/task2_reading_comprehension/glove/predictions.json`
- `outputs/project4/task2_reading_comprehension/bert/summary.json`
- `outputs/project4/task2_reading_comprehension/bert/history.csv`
- `outputs/project4/task2_reading_comprehension/bert/predictions.json`

## Notes
- The default dataset is `SQuAD 1.1` loaded through `datasets`.
- The `bert` variant uses `bert-base-uncased` through `transformers`, but keeps all BERT weights frozen.
- GloVe is loaded from `--glove_path` if provided; otherwise the script downloads `glove.6B.zip` and extracts `glove.6B.100d.txt` into the cache directory.
- If you want to reuse the repo’s Project 3 vectors explicitly, you can pass `--glove_path outputs/project3/task3_glove/vectors.txt --embedding_dim 200`. That is optional and is not the default grading path.
- Span evaluation is computed locally in the repo using SQuAD-style normalization, EM, and token F1.
- Project 4 now reuses shared repo helpers from Project 3 for output writing and text-vector parsing so the embedding/file handling is consistent across projects.
- The training loop now prints train/eval progress and writes `history.csv`, `summary.json`, and the best checkpoint incrementally after each epoch, so long runs are observable before completion.
