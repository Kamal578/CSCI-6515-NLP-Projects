# Spring 2026 Natural Language Processing (CSCI-6515)

# Project 1: Azerbaijani Wikipedia Corpus and NLP Pipeline

## Authors
- Kamal Ahmadov (kahmadov24700@ada.edu.az; kamal.ahmadov@gwu.edu)
- Rufat Guliyev (rguliyev24988@ada.edu.az; rufat.guliyev@gwu.edu)

End-to-end mini-pipeline for collecting, cleaning, and exploring an Azerbaijani Wikipedia corpus. It covers tokenization, frequency stats (Zipf), Heaps' law fit, BPE, sentence segmentation, and spell checking (uniform + weighted edit distance), with reproducible scripts and report-ready outputs.

## Repo Layout
- `src/pull_wikipedia.py` — collect and clean Azerbaijani Wikipedia pages via the MediaWiki API.
- `src/tokenize.py` — Unicode-aware tokenization plus Wikipedia-specific cleanup helpers.
- `src/task1_stats.py` — token/type counts, frequency table, optional Zipf plot.
- `src/heaps.py` — Heaps' law (V = k * N^beta) estimation and log-log plot.
- `src/task3_bpe.py` and `src/bpe.py` — train BPE merges and encode the corpus; export merges and token frequencies.
- `src/sentence_segment.py` — rule-based sentence segmentation CLI (handles abbreviations, decimals, initials, quotes, lowercase continuations).
- `src/spellcheck.py` — Levenshtein/weighted spell checker CLI.
- `src/make_spell_test.py`, `src/confusion.py`, `src/eval_spellcheck.py`, `src/plot_confusion.py` — synthetic spell benchmark, confusion weights, evaluation, and heatmap.
- `src/build_vocab.py` — filtered vocabulary builder.
- `src/serve_spellcheck.py`, `src/spell_ui.html` — lightweight Flask server + web UI for interactive suggestions.
- `data/` — input data; expects `data/raw/corpus.csv` created by the collector.
- `outputs/` — auto-created results (`stats/`, `plots/`, `bpe/`, `spellcheck/`, etc.).
- `notebooks/Main.ipynb` — scratchpad/EDA; mirrors the script workflow.
- `DATASHEET.md` — dataset notes (motivation, licensing, caveats).
- `scripts/run_all.sh` — one-shot pipeline (stats, Heaps, BPE, vocab, spell eval, segmentation, summary).

## Setup
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```
Dependencies: requests, tqdm, mwparserfromhell, regex, numpy, pandas, matplotlib, scikit-learn, flask.

Use Python 3.10+ for best compatibility.

After setup, ensure `data/raw/corpus.csv` exists (see collection step) before running analytics.

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

## Sentence Segmentation
```
python -m src.sentence_segment --corpus_path data/raw/corpus.csv --limit 500 --out outputs/sentences.txt
```
Handles abbreviations, decimals, initials, quotes, and lowercase continuations after periods (e.g., “kv. verst” remains unsplit). Regression tests cover key edge cases.

## Tokenization Notes
- Uses `regex` with Unicode properties; keeps Azerbaijani letters, apostrophes, hyphens, and numbers (including decimals).
- Light Wikipedia cleanup (`strip_wiki_garbage`) removes category/navigation noise and normalizes punctuation.
- Toggle lowercasing via `--lowercase` where available in scripts.

## Data and Licensing
- Source text: Azerbaijani Wikipedia; content is CC BY-SA. Respect attribution and ShareAlike when redistributing derived corpora.
- See `DATASHEET.md` for open issues (coverage, biases, timestamps, intended use).

## Dataset snapshot (latest run)
- Documents: 623
- Tokens: 238,286
- Types: 48,151
- Heaps' law: k=4.57, β=0.750 (`outputs/stats/heaps_params.json`)
- Spellcheck eval (synthetic 1k): Acc@1=0.637, Acc@5=0.801 (`outputs/spellcheck/spell_eval.json`)

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

## One-shot full pipeline
Runs stats, Heaps, BPE, vocab build, synthetic spell benchmark, weighted spell eval, sentence segmentation, rare-word spell suggestions, and writes a textual summary for your report.
```
bash scripts/run_all.sh
```
Key outputs:
- Plots: `outputs/plots/zipf.png`, `outputs/plots/heaps.png`
- Stats: `outputs/stats/summary.json`, `outputs/stats/heaps_params.json`, `outputs/stats/vocab_summary.json`
- BPE: `outputs/bpe/merges.txt`, `outputs/bpe/bpe_summary.json`
- Vocab: `data/processed/vocab.txt`
- Spell: `outputs/spellcheck/spell_eval.json`, `outputs/spellcheck/sample_predictions.csv`, `outputs/spellcheck/confusion.json`, `outputs/spellcheck/confusion_heatmap.png`, `outputs/spellcheck/suggestions.txt`
- Sentences: `outputs/sentences.txt`
- Report-ready summary: `outputs/run_summary.txt`
 - Segmentation eval (if gold indices provided separately): `outputs/sentseg_eval.json`

## Spellcheck example
Provide a wordlist (one word per line) and write suggestions to `outputs/spellcheck`:
```
printf "kvverst\nazrbaycan\n" > /tmp/typos.txt
python -m src.spellcheck --corpus_path data/raw/corpus.csv \
    --wordlist /tmp/typos.txt \
    --out outputs/spellcheck/typos_suggestions.txt
```

## Build a filtered vocabulary
Create a vocab with frequency/length filtering (useful for spellcheck and other tasks):
```
python -m src.build_vocab \
  --corpus_path data/raw/corpus.csv \
  --min_freq 3 \
  --min_len 3 \
  --vocab_path data/processed/vocab.txt \
  --summary_path outputs/stats/vocab_summary.json
```

## Spellcheck evaluation (weighted edit distance)
Create synthetic errors, learn confusion weights, evaluate, and plot:
```
python -m src.make_spell_test --samples 1000
python -m src.confusion --pairs_csv data/processed/spell_test.csv --out_confusion outputs/spellcheck/confusion.json
python -m src.eval_spellcheck --test_csv data/processed/spell_test.csv --confusion outputs/spellcheck/confusion.json
python -m src.plot_confusion --confusion outputs/spellcheck/confusion.json --out outputs/spellcheck/confusion_heatmap.png
```
Metrics: see `outputs/spellcheck/spell_eval.json`; heatmap at `outputs/spellcheck/confusion_heatmap.png`.

Spellcheck internals (what happens):
- Vocab is filtered to Azerbaijani alphabet, min frequency (default 2) and length (default 3) to drop noise.
- Candidates are pruned by length difference (<= `--max_dist`, default 2) before distance scoring.
- Distance = uniform Levenshtein, or weighted edits if `--confusion` is provided (common substitutions get lower cost).
- Ranking = (distance asc, frequency desc); top-k (default 5) returned.
- `scripts/run_all.sh` also scans rare vocab (200 lowest-frequency tokens) to surface likely misspellings automatically.

## Simple web UI for spellcheck
Start the Flask server and open the local page:
```
python -m src.serve_spellcheck  # defaults to http://127.0.0.1:5000
```
Then visit `http://127.0.0.1:5000/` and enter a word to see ranked suggestions (uses corpus vocab; weighted edits if `outputs/spellcheck/confusion.json` exists).

## Sentence segmentation evaluation
Given gold boundary indices and predicted indices (JSON array or newline-separated ints):
```
python -m src.evaluate_segmentation \
  --gold data/processed/sent_gold_indices.txt \
  --pred outputs/sentences_pred_indices.txt \
  --out outputs/sentseg_eval.json
```
Reports Precision/Recall/F1 and Boundary Detection Error Rate (BDER), and writes metrics to JSON.
Helper end-to-end wrapper (builds subset, segments, converts to indices, evaluates):
```
bash scripts/eval_sentseg.sh data/processed/sent_gold.txt 50
```
Gold preparation: manually segment a small subset (e.g., first 50 docs) into one sentence per line in `data/processed/sent_gold.txt`; the helper script will derive indices and evaluate.
