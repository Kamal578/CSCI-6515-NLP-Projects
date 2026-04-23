## NLP Course Projects Repository

This repository contains coursework implementations for multiple NLP project assignments.

The codebase currently includes:
- `Project 1`: Azerbaijani Wikipedia corpus collection/cleaning and classic NLP pipeline components (tokenization, Zipf/Heaps, BPE, sentence segmentation, spellchecking).
- `Project 2`: N-gram language modeling + smoothing, sentiment classification experiments, and sentence-boundary detection with logistic regression.
- `Project 3`: Word embedding pipeline (Word2Vec + GloVe), comparative intrinsic analysis, deep-learning sentiment benchmarks (RNN/BiRNN/LSTM), report artifact generation, and results UI.
- `Project 4`: Azerbaijani BERT sentiment fine-tuning, reading comprehension with BiDAF using pretrained GloVe and frozen BERT embeddings on SQuAD 1.1, and a Streamlit interactive UI with live inference.
- `Project 5`: Retrieval-Augmented Generation (RAG) system for ESG report question answering, with local Ollama models, ChromaDB vector store, baseline-vs-RAG evaluation, Docker support, and Streamlit UI.

### Project Readmes
- `proj1_readme.md` — Project 1 setup, pipeline, scripts, and outputs
- `proj2_readme.md` — Project 2 task-by-task methods, commands, outputs, and reported results
- `proj3_readme.md` — Project 3 setup, pipelines, outputs, UI, and report artifacts
- `proj4_readme.md` — Project 4 Task 2 setup, commands, outputs, and QA model comparison
- `proj5_readme.md` — Project 5 setup, commands, outputs, and Streamlit dashboard for all tasks

### Notes on `src/` Naming
- Project 1 modules mostly keep the original generic names (e.g., `task1_stats.py`, `task3_bpe.py`) from the first assignment.
- Project 2 modules are prefixed with `project2_` (e.g., `project2_task1_lm.py`, `project2_task3_sentiment.py`, `project2_task4_sentence_lr.py`) to make cross-project ownership clearer.

### Runners
- `scripts/run_project1.sh` — Project 1 pipeline runner
- `scripts/run_project2.sh` — Project 2 runner (Tasks 1, 2, 3, and 4 when required inputs are present)
- `scripts/run_project3.sh` — Project 3 runner (Tasks 1..6; optional Streamlit UI launch with `--with-ui`, or legacy UI with `--with-legacy-ui`)
- `scripts/run_project4_task1.sh` — Project 4 Task 1 runner for Azerbaijani BERT sentiment fine-tuning
- `scripts/run_project4_task2.sh` — Project 4 Task 2 runner for GloVe/BERT reading comprehension experiments
- `scripts/run_project4_ui.sh` — Project 4 Streamlit dashboard launcher
- Project 5 runners are inside `project5/scripts/` (e.g., `setup.sh`, `prepare_data.sh`, `build_index.sh`, `evaluate.sh`, `run_app.sh`, `docker_up.sh`)

### Project 3 UI
- `streamlit run src/project3_dashboard.py -- --output-root outputs/project3` — interactive Project 3 dashboard
- `python3 -m src.project3_results_ui --output-root outputs/project3 --host 127.0.0.1 --port 5060` — legacy Flask/HTML dashboard

### Project 4 UI
- `bash scripts/run_project4_ui.sh` — interactive Project 4 dashboard
- `streamlit run src/project4_dashboard.py -- --sentiment-root outputs/project4/task1_sentiment --qa-root outputs/project4/task2_reading_comprehension --report-tex report/project4_report.tex` — direct launch command

### Project 5
- Main project directory: `project5/`
- Main documentation: `project5/README.md`
- Root pointers:
  - `proj5_readme.md`
  - `report/project5_report.tex`
  - `slides/project5_slides.tex`

Start with the project-specific README above depending on the assignment you want to run.
