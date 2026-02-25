## NLP Course Projects Repository

This repository contains coursework implementations for multiple NLP project assignments.

The codebase currently includes:
- `Project 1`: Azerbaijani Wikipedia corpus collection/cleaning and classic NLP pipeline components (tokenization, Zipf/Heaps, BPE, sentence segmentation, spellchecking).
- `Project 2`: N-gram language modeling + smoothing, sentiment classification experiments, and sentence-boundary detection with logistic regression.

### Project Readmes
- `proj1_readme.md` — Project 1 setup, pipeline, scripts, and outputs
- `proj2_readme.md` — Project 2 task-by-task methods, commands, outputs, and reported results

### Notes on `src/` Naming
- Project 1 modules mostly keep the original generic names (e.g., `task1_stats.py`, `task3_bpe.py`) from the first assignment.
- Project 2 modules are prefixed with `project2_` (e.g., `project2_task1_lm.py`, `project2_task3_sentiment.py`, `project2_task4_sentence_lr.py`) to make cross-project ownership clearer.

### Runners
- `scripts/run_project1.sh` — Project 1 pipeline runner
- `scripts/run_project2.sh` — Project 2 runner (Tasks 1, 2, 3, and 4 when required inputs are present)

Start with the project-specific README above depending on the assignment you want to run.
