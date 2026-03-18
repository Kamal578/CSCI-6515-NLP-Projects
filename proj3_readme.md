# Project 3 README

This file tracks implementation and report-ready notes for Project 3.

## Project 3: Word Embeddings, Comparative Analysis, and Deep Learning Classification

Implemented pipeline (Tasks 1..6 + extra UI):
- Task 1: Dataset description + term-document and word-word matrix analysis
- Task 2: Word2Vec training + nearest-neighbor/analogy analysis
- Task 3: GloVe training + nearest-neighbor/analogy analysis
- Task 4: Word2Vec vs GloVe comparative analysis
- Task 5: Deep learning classification (`RNN`, `BiRNN`, `LSTM`) across 5 feature methods
- Task 6: Consolidated report artifacts
- Extra: Results UI dashboard

## Main Entry Points
- `scripts/run_project3.sh` — one-shot Project 3 runner
- `src/project3_task1_matrices.py`
- `src/project3_task2_word2vec.py`
- `src/project3_task3_glove.py`
- `src/project3_task4_compare.py`
- `src/project3_task5_dl.py`
- `src/project3_task6_report.py`
- `src/project3_results_ui.py` + `src/project3_results_ui.html`
- `src/project3_dashboard.py` (interactive Streamlit UI)

## How to Run

### Full run
```bash
bash scripts/run_project3.sh

# Full run + Streamlit UI at end
bash scripts/run_project3.sh --with-ui

# Full run + legacy Flask UI
bash scripts/run_project3.sh --with-legacy-ui
```

### Smoke run
```bash
bash scripts/run_project3.sh --smoke
```

### Run selected tasks
```bash
bash scripts/run_project3.sh --tasks 1,2,3
bash scripts/run_project3.sh --tasks 4,5,6
```

### Launch UI
```bash
# New interactive dashboard (recommended)
streamlit run src/project3_dashboard.py -- --output-root outputs/project3

# Equivalent convenience script
bash scripts/run_project3_ui.sh

# Legacy Flask + HTML dashboard
python3 -m src.project3_results_ui --output-root outputs/project3 --host 127.0.0.1 --port 5060
```
Open:
- Streamlit: `http://localhost:8501`
- Flask UI: `http://127.0.0.1:5060`

The Streamlit UI includes a `Playground` tab where you can type custom words/analogies,
run neighbor search, inspect cosine similarity, blend vectors, and build a 2D semantic map.

## Task 1: Dataset + Matrix Analysis

### Setup/Method
- Dataset: `data/raw/corpus.csv` (`text` column)
- Frequent words threshold: `>=100`
- Rare words threshold: `<=2`
- Visualizations: log-scaled heatmaps on top terms/docs

### Output files
- `outputs/project3/task1_matrices/summary.json`
- `outputs/project3/task1_matrices/word_frequency_distribution.csv`
- `outputs/project3/task1_matrices/term_document_top_matrix.csv`
- `outputs/project3/task1_matrices/word_word_top_matrix.csv`
- `outputs/project3/task1_matrices/term_document_heatmap.png`
- `outputs/project3/task1_matrices/word_word_heatmap.png`

### Final results
- Documents: `31,842`
- Tokens: `11,905,937`
- Vocabulary size: `586,674`
- Frequent words (`>=100`): `12,436`
- Rare words (`<=2`): `398,599`

## Task 2: Word2Vec

### Training config
- Model: skip-gram (`cbow=0`)
- Vector size: `200`
- Window: `5`
- Negative sampling: `10`
- Min count: `5`
- Iterations: `10`

### Output files
- `outputs/project3/task2_word2vec/vectors.txt`
- `outputs/project3/task2_word2vec/nearest_neighbors.csv`
- `outputs/project3/task2_word2vec/analogy_results.csv`
- `outputs/project3/task2_word2vec/summary.json`

### Final results
- Vocab size: `122,930`
- Vector dim: `200`
- Analogy hit@10: `0.6667` (`2/3` evaluated analogies; OOV analogies skipped)

## Task 3: GloVe

### Training config
- Vector size: `200`
- Window size: `10`
- Min count: `5`
- Iterations: `20`
- `x_max=10`, `eta=0.05`, `alpha=0.75`

### Output files
- `outputs/project3/task3_glove/vectors.txt`
- `outputs/project3/task3_glove/vocab.txt`
- `outputs/project3/task3_glove/nearest_neighbors.csv`
- `outputs/project3/task3_glove/analogy_results.csv`
- `outputs/project3/task3_glove/summary.json`

### Final results
- Vocab size: `122,930`
- Vector dim: `200`
- Analogy hit@10: `0.7500` (`3/4` evaluated analogies; OOV analogies skipped)

## Task 4: Comparative Analysis (Word2Vec vs GloVe)

### Output files
- `outputs/project3/task4_compare/comparison.csv`
- `outputs/project3/task4_compare/summary.json`

### Final comparison
| Model | Analogy hit@1 | Analogy hit@10 | Avg neighbor cosine (top3) | Coverage ratio (sentiment vocab) |
|---|---:|---:|---:|---:|
| Word2Vec | `0.6667` | `0.6667` | `0.6619` | `0.0749` |
| GloVe | `0.7500` | `0.7500` | `0.6217` | `0.0749` |

Summary:
- GloVe won on analogy metrics.
- Word2Vec had better average cosine among top-3 neighbors in this run.
- Coverage on sentiment-vocabulary tokens was equal.

## Task 5: Deep Learning Classification

### Experimental setup
- Label mode: `sentiment3` (`negative/neutral/positive`)
- Architectures: `RNN`, `BiRNN`, `LSTM`
- Feature methods: `count_svd`, `tfidf_svd`, `pmi_svd`, `word2vec`, `glove`
- Grid size: `15` DL runs + `1` baseline (`LogReg + TF-IDF`)
- Early stopping + fixed seed (`42`)

### Output files
- `outputs/project3/task5_dl/leaderboard.csv`
- `outputs/project3/task5_dl/metrics.csv`
- `outputs/project3/task5_dl/confusion_matrices.json`
- `outputs/project3/task5_dl/classification_reports.txt`
- `outputs/project3/task5_dl/summary.json`

### Final best and baseline
- Best overall: `lstm__pmi_svd`
  - Accuracy: `0.8225`
  - Macro-F1: `0.5684`
  - Weighted-F1: `0.8594`
- Baseline: `logreg_baseline__tfidf`
  - Accuracy: `0.8271`
  - Macro-F1: `0.5588`
  - Weighted-F1: `0.8582`

Top-ranked runs by macro-F1 are in `outputs/project3/task5_dl/leaderboard.csv`.

## Task 6: Report Artifacts + Presentation Bundle

### Task 6 outputs
- `outputs/project3/task6_report/embedding_comparison_table.csv`
- `outputs/project3/task6_report/dl_results_table.csv`
- `outputs/project3/task6_report/report_artifacts.md`
- `outputs/project3/task6_report/summary.json`

### Presentation bundle
A curated package was prepared for report/presentation use:
- Folder: `outputs/project3/presentation_bundle/`
- Zip: `outputs/project3/presentation_bundle.zip`
- Includes:
  - key figures (Task 1 heatmaps + Task 5 bar chart)
  - main result tables (Tasks 2/3/4/5)
  - task summaries (`summary.json` files)
  - concise presenter notes (`presentation_notes.md`)

## Reproducibility / Notes
- Python dependencies are in `requirements.txt` (includes `torch`, `scipy`, `scikit-learn`, `flask`, `streamlit`, `plotly`, `regex`, `pytest`).
- Imported `word2vec` sources were patched for macOS C compiler compatibility (`fgetc_unlocked`/`fputc_unlocked` fallbacks).
- For grading artifacts, prefer `outputs/project3/task*/summary.json` and `outputs/project3/task5_dl/leaderboard.csv` as primary references.
