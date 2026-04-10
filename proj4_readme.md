# Project 4 README

Project 4 now includes the full assignment workflow:

- `Task 1`: Azerbaijani sentiment analysis with a fine-tuned BERT classifier on `hajili/azerbaijani_review_sentiment_classification`
- `Task 2`: reading comprehension with BiDAF using either trainable GloVe embeddings or frozen BERT embeddings
- `Task 3`: a Streamlit UI with live inference for both tasks
- `Task 4`: report source in `report/project4_report.tex`

## Environment

The repo now expects a local `.venv` on macOS Apple Silicon:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt accelerate sentencepiece safetensors
```

PyTorch with `mps` support is used automatically when available. You can force CPU or MPS with `DEVICE=cpu` or `DEVICE=mps`.

## Task 1: Azerbaijani Sentiment with BERT

Main entry point:

```bash
bash scripts/run_project4_task1.sh
```

Useful environment variables:

- `MODEL_NAME=google-bert/bert-base-multilingual-cased`
- `MODEL_NAME=HPLT/hplt_bert_base_az TRUST_REMOTE_CODE=1`
- `OUTPUT_DIR=outputs/project4/task1_sentiment`
- `LABEL_MODE=score5|sentiment3|binary`
- `DEVICE=auto|cpu|mps|cuda`
- `SMOKE=1`
- `MAX_TRAIN_EXAMPLES=...`
- `MAX_VAL_EXAMPLES=...`
- `MAX_TEST_EXAMPLES=...`
- `GRAD_ACCUMULATION_STEPS=2`

Direct command example:

```bash
.venv/bin/python -m src.project4_task1_sentiment \
  --dataset_name hajili/azerbaijani_review_sentiment_classification \
  --model_name google-bert/bert-base-multilingual-cased \
  --device auto \
  --output_dir outputs/project4/task1_sentiment
```

Artifacts:

- `outputs/project4/task1_sentiment/summary.json`
- `outputs/project4/task1_sentiment/history.csv`
- `outputs/project4/task1_sentiment/model/`
- `outputs/project4/task1_sentiment/validation_predictions.json`
- `outputs/project4/task1_sentiment/test_predictions.json`

Task 1 summary metadata documents:

- model inputs and outputs
- class labels
- maximum input size
- case sensitivity
- morphology and agglutinative adaptation strategy

## Task 2: Reading Comprehension with BiDAF + BERT

Main entry point:

```bash
bash scripts/run_project4_task2.sh
```

Useful environment variables:

- `VARIANT=glove|bert|compare`
- `DEVICE=auto|cpu|mps|cuda`
- `SMOKE=1`
- `MEDIUM=1`
- `GRAD_ACCUMULATION_STEPS=2`
- `MAX_TRAIN_EXAMPLES=...`
- `MAX_VAL_EXAMPLES=...`
- `TRAIN_JSON=...`
- `VAL_JSON=...`

Direct command example:

```bash
.venv/bin/python -m src.project4_task2_reading_comprehension \
  --variant compare \
  --device auto \
  --out_dir outputs/project4/task2_reading_comprehension
```

Artifacts:

- `outputs/project4/task2_reading_comprehension/comparison.csv`
- `outputs/project4/task2_reading_comprehension/report_notes.md`
- `outputs/project4/task2_reading_comprehension/glove/summary.json`
- `outputs/project4/task2_reading_comprehension/glove/history.csv`
- `outputs/project4/task2_reading_comprehension/glove/predictions.json`
- `outputs/project4/task2_reading_comprehension/glove/vocab.json`
- `outputs/project4/task2_reading_comprehension/bert/summary.json`
- `outputs/project4/task2_reading_comprehension/bert/history.csv`
- `outputs/project4/task2_reading_comprehension/bert/predictions.json`

## Task 3: Interactive UI

Launch the Streamlit app:

```bash
bash scripts/run_project4_ui.sh
```

Direct command:

```bash
.venv/bin/streamlit run src/project4_dashboard.py -- \
  --sentiment-root outputs/project4/task1_sentiment \
  --qa-root outputs/project4/task2_reading_comprehension \
  --report-tex report/project4_report.tex
```

The UI provides:

- Azerbaijani sentiment prediction from raw text input
- QA answer extraction from context and question input
- run metrics and artifact inspection
- report preview

## Recommended Apple Silicon Workflow

1. Run Task 1 smoke first:

```bash
SMOKE=1 DEVICE=mps bash scripts/run_project4_task1.sh
```

2. Run Task 2 smoke first:

```bash
SMOKE=1 DEVICE=mps bash scripts/run_project4_task2.sh
```

3. Move to larger runs only after the smoke checks pass.

For `16 GB` RAM, start with:

- Task 1 batch size `8` to `16`
- Task 2 batch size `8` or less for the `bert` variant
- `GRAD_ACCUMULATION_STEPS=2` or higher if memory gets tight
