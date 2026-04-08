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
- `--max_train_examples`, `--max_val_examples`, `--epochs`, `--batch_size`
- `--max_question_words`, `--context_window_words`, `--doc_stride_words`
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

## Shell Runner
```bash
bash scripts/run_project4_task2.sh
```

Useful environment variables:
- `VARIANT=glove|bert|compare`
- `SMOKE=1`
- `OUT_DIR=...`
- `CACHE_DIR=...`
- `GLOVE_PATH=...`
- `TRAIN_JSON=...`
- `VAL_JSON=...`

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
- Span evaluation is computed locally in the repo using SQuAD-style normalization, EM, and token F1.
