# Azerbaijani Wikipedia NLP Pipeline

End-to-end mini-pipeline for collecting a small Azerbaijani Wikipedia corpus and running core NLP exploratory tasks: tokenization, frequency stats (Zipf), Heaps' law fit, and byte-pair encoding (BPE). Code is script-first to keep the workflow transparent for coursework.

## Repo Layout
- `src/pull_wikipedia.py` — collect and clean Azerbaijani Wikipedia pages via the MediaWiki API.
- `src/tokenize.py` — Unicode-aware tokenization plus Wikipedia-specific cleanup helpers.
- `src/task1_stats.py` — token/type counts, frequency table, optional Zipf plot.
- `src/heaps.py` — Heaps' law (V = k * N^beta) estimation and log-log plot.
- `src/task3_bpe.py` and `src/bpe.py` — train BPE merges and encode the corpus; export merges and token frequencies.
- `data/` — input data; expects `data/raw/corpus.csv` created by the collector.
- `outputs/` — auto-created results (`stats/`, `plots/`, `bpe/`, etc.).
- `notebooks/Main.ipynb` — scratchpad/EDA; mirrors the script workflow.
- `DATASHEET.md` — dataset notes (motivation, licensing, caveats).

## Setup
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```
Dependencies: requests, tqdm, mwparserfromhell, regex, numpy, pandas, matplotlib.

Use Python 3.10+ for best compatibility.

## 1) Collect a Corpus
Fetch and clean Azerbaijani Wikipedia pages into a CSV with one row per document.

Examples:
```
# Random sample of 800 articles
python -m src.pull_wikipedia --random 800 --out data/raw/corpus.csv

# Up to 500 articles from a category
python -m src.pull_wikipedia --category "Azərbaycan" --limit 500 --out data/raw/corpus.csv
```
Key flags:
- `--min_chars` (default 400) drops very short pages after cleaning.
- `--sleep` (default 0.1s) is a politeness delay between API batches.

Output schema: `doc_id, page_id, title, revision_id, timestamp, source, url, text` (UTF-8 CSV).

## 2) Token Stats and Zipf (Task 1)
Compute token frequencies and basic corpus stats using the tokenizer in `src/tokenize.py`.
```
python -m src.task1_stats --corpus_path data/raw/corpus.csv \
    --out_dir outputs/stats --plots_dir outputs/plots --top_n 2000
```
Outputs:
- `outputs/stats/summary.json` — documents, token/type counts, top 20 tokens, lowercase flag.
- `outputs/stats/token_freq.csv` — full frequency table.
- `outputs/plots/zipf.png` — rank-frequency plot (if `--make_zipf_plot` is true).

## 3) Heaps' Law Fit (Task 2)
Estimate Heaps' law parameters k and beta from streamed tokens.
```
python -m src.heaps --corpus_path data/raw/corpus.csv \
    --out_stats outputs/stats/heaps_params.json \
    --out_plot outputs/plots/heaps.png --step 1000
```
Outputs: JSON with k, beta, corpus size, and `heaps.png` log-log fit plot.

## 4) Byte-Pair Encoding (Task 3)
Train a simple BPE model on word tokens and export merges plus encoded token stats.
```
python -m src.task3_bpe --corpus_path data/raw/corpus.csv \
    --out_dir outputs/bpe --num_merges 5000 --min_word_freq 2 --sample_words 30
```
Outputs:
- `merges.txt` — merge rules in order.
- `bpe_token_freq.csv` — BPE token counts.
- `bpe_summary.json` — run metadata plus example word -> BPE segmentations.

## Tokenization Notes
- Uses `regex` with Unicode properties; keeps Azerbaijani letters, apostrophes, hyphens, and numbers (including decimals).
- Light Wikipedia cleanup (`strip_wiki_garbage`) removes category/navigation noise and normalizes punctuation.
- Toggle lowercasing via `--lowercase` where available in scripts.

## Notebook
`notebooks/Main.ipynb` mirrors the scripts with richer explanations and plots. Run after installing the requirements (Jupyter is not auto-installed—use `pip install notebook` if needed).

## Data and Licensing
- Source text: Azerbaijani Wikipedia; content is CC BY-SA. Respect attribution and ShareAlike when redistributing derived corpora.
- See `DATASHEET.md` for open issues (coverage, biases, timestamps, intended use).

## Troubleshooting
- Missing corpus error -> run the collector to create `data/raw/corpus.csv`.
- Slow downloads -> lower `--limit` / `--random` or increase `--sleep` for API politeness.
- Matplotlib backend issues in headless environments -> set `MPLBACKEND=Agg` before running plotting scripts.

## Quickstart (TL;DR)
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.pull_wikipedia --random 500 --out data/raw/corpus.csv
python -m src.task1_stats
python -m src.heaps
python -m src.task3_bpe
```
