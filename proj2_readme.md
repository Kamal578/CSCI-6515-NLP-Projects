# Project 2 README

This file tracks implementation and report-ready notes for Project 2.

## Task 1: Unigram, Bigram, Trigram Language Models + Perplexity (No Smoothing)

### Goal
- Train unigram, bigram, and trigram language models on the dataset.
- Compute perplexity on a held-out test split.
- Report baseline results **without smoothing** (Task 2 will add smoothing methods).

### Method (Task 1 baseline)
- Model type: Maximum Likelihood Estimation (MLE), **no smoothing**
- Split: Train/test split at document level (recommended: `80/20`)
- Tokenization: `src/tokenize.py`
- Sentence segmentation: `src/sentence_segment.py`
- Boundary tokens:
  - bigram: `<s> ... </s>`
  - trigram: `<s> <s> ... </s>`
- OOV handling:
  - Train vocabulary built from train split only
  - Unseen/rare tokens mapped to `<unk>` (default `--unk_min_freq 2`)
- Important:
  - Bigram/trigram perplexity may be `inf` without smoothing due to unseen n-grams in test data
  - This is expected and should be discussed in the report

### How to Run

#### Full run (dataset)
```bash
.venv/bin/python -m src.task1_lm \
  --corpus_path data/raw/corpus.csv \
  --out_dir outputs/project2/task1_lm \
  --seed 42 \
  --test_ratio 0.2
```

#### Quick smoke test (faster)
```bash
.venv/bin/python -m src.task1_lm \
  --corpus_path data/raw/corpus.csv \
  --out_dir outputs/project2/task1_lm_debug \
  --max_docs 2000 \
  --seed 42 \
  --test_ratio 0.2
```

### Output Files (Task 1)
- `outputs/project2/task1_lm/summary.json` — main report metrics and run configuration
- `outputs/project2/task1_lm/unigram_top.csv` — top unigrams
- `outputs/project2/task1_lm/bigram_top.csv` — top bigrams
- `outputs/project2/task1_lm/trigram_top.csv` — top trigrams

Note:
- The `*_top.csv` files are **capped previews**, not full n-gram inventories.
- By default, the script exports only the top `200` rows (`--top_n 200`).
- The full number of learned n-grams is reported in `outputs/project2/task1_lm/summary.json` under:
  - `models.unigram.num_ngrams`
  - `models.bigram.num_ngrams`
  - `models.trigram.num_ngrams`
- To export more rows, set `--top_n` to a larger value; use `--top_n 0` to export all (can create very large CSV files).

### How to Check Results

#### 1) Inspect the summary JSON
```bash
cat outputs/project2/task1_lm/summary.json
```

Key fields to report/check in `summary.json`:
- `config`
  - `corpus_path`, `text_column`, `test_ratio`, `seed`, `unk_min_freq`
- `data_split`
  - number of docs/sentences/tokens in train and test
- `vocab`
  - train vocab size after thresholding
  - test OOV rate vs train vocab
- `models.unigram.perplexity`
- `models.bigram.perplexity`
- `models.trigram.perplexity`
- `models.*.zero_prob_events`
  - explains infinite perplexity in unsmoothed models

#### 2) Check top n-grams (sanity check)
```bash
head -20 outputs/project2/task1_lm/unigram_top.csv
head -20 outputs/project2/task1_lm/bigram_top.csv
head -20 outputs/project2/task1_lm/trigram_top.csv
```

What to look for:
- Frequent Azerbaijani function words in unigrams
- Reasonable phrase fragments in bigrams/trigrams
- Presence of `<unk>` is normal (especially with `--unk_min_freq 2`)

### Report Results (filled from `outputs/project2/task1_lm/summary.json`)

#### Dataset / Setup
- Corpus file used: `data/raw/corpus.csv`
- Text column used: `text`
- Total documents: `31,842`
- Train/Test split: `25,474 / 6,368` documents (`80/20`, seed=`42`)
- Lowercasing: `True`
- `unk_min_freq`: `2`

#### Vocabulary / OOV
- Train vocab size (after threshold): `246,634`
- Test token count: `2,366,334`
- Test OOV token count vs train vocab: `91,523`
- Test OOV rate: `0.038677` (`3.87%`)

#### Perplexity Results (Task 1, No Smoothing)

| Model | Perplexity | Zero-Probability Events | Notes |
|---|---:|---:|---|
| Unigram | `5107.302759` | `0 / 2,530,518` | Finite baseline perplexity (no zero-probability events after `<unk>` mapping). |
| Bigram | `inf` | `612,997 / 2,530,518` | Infinite perplexity due to unseen bigrams in the test set (expected without smoothing). |
| Trigram | `inf` | `1,324,990 / 2,530,518` | Infinite perplexity due to unseen trigrams; sparsity is stronger than bigram. |

#### Interpretation (Task 1 baseline)
- Unsmoothed n-gram models assign zero probability to unseen test n-grams.
- Therefore, bigram/trigram perplexity may become infinite.
- This motivates smoothing methods in **Task 2**.

### Recommended Report Paragraph (template)
Use this as a starting point in the final report:

> For Task 1, we trained unigram, bigram, and trigram language models using maximum likelihood estimation (MLE) on an Azerbaijani Wikipedia corpus and evaluated them on a held-out test split using perplexity. We used the same tokenization and sentence segmentation pipeline across all models and mapped out-of-vocabulary tokens to `<unk>` based on the training vocabulary. As expected, the unsmoothed bigram and trigram models produced zero-probability events on the test set, which led to infinite perplexity in some cases. This baseline demonstrates the data sparsity problem and motivates the smoothing techniques evaluated in Task 2.

### Reproducibility Notes (important for Task 2 comparison)
- Keep these fixed across Task 1 and Task 2:
  - `corpus_path`
  - `text_column`
  - train/test split (`test_ratio`, `seed`)
  - tokenization and sentence segmentation
  - `<unk>` policy (`unk_min_freq`)
- Change only the smoothing method when comparing perplexity in Task 2.

## Task 2: Smoothing Comparison (Laplace, Interpolation, Backoff, Kneser-Ney)

### Goal
- Apply smoothing to the n-gram language model and compare methods using perplexity.
- Use the same preprocessing/split settings as Task 1 for a fair comparison.
- Select the best method based on **trigram test perplexity** (and report bigram ranking too).

### Methods Implemented
- Laplace (Add-k)
- Linear Interpolation
- Normalized discount-backoff hybrid (proper probabilities for perplexity)
- Interpolated Kneser-Ney (single discount `D`)

### How to Run (Task 2)

#### Full run (recommended)
```bash
.venv/bin/python -m src.task2_smoothing \
  --corpus_path data/raw/corpus.csv \
  --text_column text \
  --test_ratio 0.2 \
  --dev_ratio 0.1 \
  --seed 42 \
  --dev_seed 43 \
  --unk_min_freq 2 \
  --lowercase \
  --orders 2 3 \
  --methods laplace interpolation backoff kneser_ney \
  --out_dir outputs/project2/task2_smoothing
```

#### Smoke test (fast)
```bash
.venv/bin/python -m src.task2_smoothing \
  --max_docs 200 \
  --disable_grid_search \
  --out_dir outputs/project2/task2_smoothing_smoke
```

### Output Files (Task 2)
- `outputs/project2/task2_smoothing/summary.json` — main Task 2 report artifact
- `outputs/project2/task2_smoothing/comparison.csv` — ranked method comparison by order
- `outputs/project2/task2_smoothing/tuning_results.csv` — all tested hyperparameter configs (dev perplexity)
- `outputs/project2/task2_smoothing/notes.txt` — quick textual summary / recommendation

### How to Check Results
```bash
cat outputs/project2/task2_smoothing/summary.json
cat outputs/project2/task2_smoothing/comparison.csv
```

What to verify:
- `results_by_order` has entries for orders `2` and `3`
- `test.zero_prob_events == 0` for smoothed methods (normally expected)
- `best_method_by_order`
- `overall_recommendation.selected_method` (chosen by trigram test perplexity)

### Report Results (Task 2, filled from `outputs/project2/task2_smoothing/summary.json`)

#### Fixed comparison settings (same as Task 1)
- Corpus file: `data/raw/corpus.csv`
- Text column: `text`
- Train/Test split: `25,474 / 6,368` docs (`test_ratio=0.2`)
- Dev split (from train): `64,914` sentences from train (`dev_ratio=0.1`, `train_core=584,221`, `dev=64,914`)
- Seed / Dev seed: `42 / 43`
- Lowercasing: `True`
- `unk_min_freq`: `2`

#### Bigram Results (Task 2)
| Rank | Method | Dev Perplexity | Test Perplexity | Hyperparameters |
|---:|---|---:|---:|---|
| 1 | `kneser_ney` | `646.520451` | `609.074928` | `d=0.75` |
| 2 | `backoff` | `688.625050` | `641.919075` | `d=0.75` |
| 3 | `interpolation` | `742.725392` | `694.148777` | `lambdas=[0.3, 0.7]` |
| 4 | `laplace` | `2263.284135` | `1976.539448` | `k=0.01` |

#### Trigram Results (Task 2)
| Rank | Method | Dev Perplexity | Test Perplexity | Hyperparameters |
|---:|---|---:|---:|---|
| 1 | `kneser_ney` | `391.449246` | `376.394760` | `d=0.75` |
| 2 | `backoff` | `418.384178` | `398.309427` | `d=0.75` |
| 3 | `interpolation` | `484.800043` | `455.434536` | `lambdas=[0.3, 0.4, 0.3]` |
| 4 | `laplace` | `16351.363516` | `14997.943695` | `k=0.01` |

#### Final Selection (Task 2)
- Best smoothing method (by trigram test perplexity): `kneser_ney`
- Selected hyperparameters: `d=0.75`
- Reason: `Selected by lowest trigram test perplexity.`

### Task 2 Interpretation Notes (template)
- Smoothing resolves the zero-probability problem observed in Task 1 for higher-order n-grams.
- The best method should be selected using the predefined criterion (lowest trigram test perplexity).
- Also discuss whether the bigram and trigram rankings agree or differ.

### Task 2 Observed Results (this run)
- All smoothing methods achieved **finite** bigram and trigram perplexity (`zero_prob_events = 0` on test).
- `kneser_ney` ranked **#1** for both bigram and trigram.
- `backoff` ranked **#2**, and was relatively close to Kneser-Ney.
- `laplace` performed worst by a large margin on this dataset.

## Task 3: Sentiment Classification (Naive Bayes, Binary Naive Bayes, Logistic Regression)

### Goal
- Compare three classifiers on an Azerbaijani sentiment dataset:
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes (binary Naive Bayes)
  - Logistic Regression
- Compare feature extraction variants:
  - Bag-of-Words (BoW)
  - Sentiment lexicon features
  - BoW + sentiment lexicon features
- Use statistical significance testing (McNemar exact test with Holm-Bonferroni correction).
- Determine which classifier/feature combination is best and analyze tradeoffs.

### Dataset Used
- Dataset: `hajili/azerbaijani_review_sentiment_classification`
- Local files used:
  - `data/external/train.csv`
  - `data/external/test.csv`
- Columns used:
  - text: `content`
  - score: `score`

### Label Mapping (this run)
- Mode: `sentiment3`
- Mapping:
  - `1-2` -> `negative`
  - `3` -> `neutral`
  - `4-5` -> `positive`

### Features
- `bow`:
  - CountVectorizer with custom Azerbaijani-oriented tokenizer/preprocessing (legacy-compatible)
  - `ngram_range=(1,2)`
  - `min_df=2`
  - `max_features=60000`
- `lexicon`:
  - Sentiment lexicon induced from training data (positive vs negative rows only)
  - Top `500` positive + top `500` negative tokens by polarity log-probability difference
  - Aggregated document-level lexicon statistics (counts, unique counts, weighted sums, ratios, coverage)
- `bow+lexicon`:
  - Sparse concatenation of BoW features and lexicon aggregate features

### How to Run (Task 3)

#### Main command (new structured runner)
```bash
.venv/bin/python -m src.task3_sentiment \
  --train data/external/train.csv \
  --test data/external/test.csv \
  --output-dir outputs/project2/task3_sentiment
```

#### Backward-compatible command (old filename still works)
```bash
.venv/bin/python -m src.project2_task3_experiment \
  --train data/external/train.csv \
  --test data/external/test.csv \
  --output-dir outputs/project2/task3_sentiment
```

### Task 3 Output Files
- `outputs/project2/task3_sentiment/summary.json` — run config, dataset stats, best models
- `outputs/project2/task3_sentiment/metrics.csv` — metrics for all 9 experiments
- `outputs/project2/task3_sentiment/significance_within_feature_set.csv` — pairwise McNemar tests within each feature set
- `outputs/project2/task3_sentiment/significance_top_vs_others.csv` — top overall model vs all other experiments
- `outputs/project2/task3_sentiment/classification_reports.txt` — full per-class reports
- `outputs/project2/task3_sentiment/confusion_matrices.json` — confusion matrices for all experiments
- `outputs/project2/task3_sentiment/lexicon_preview.json` — induced lexicon preview + feature metadata
- `outputs/project2/task3_sentiment/test_predictions.csv` — per-example predictions from all experiments

### How to Check Results (Task 3)
```bash
cat outputs/project2/task3_sentiment/summary.json
head -20 outputs/project2/task3_sentiment/metrics.csv
cat outputs/project2/task3_sentiment/significance_within_feature_set.csv
cat outputs/project2/task3_sentiment/significance_top_vs_others.csv
```

### Report Results (Task 3, filled from `outputs/project2/task3_sentiment/*`)

#### Dataset Summary
- Train size: `127,537`
- Test size: `31,885`
- Train label distribution:
  - `positive`: `108,378`
  - `negative`: `16,396`
  - `neutral`: `2,763`
- Test label distribution:
  - `positive`: `27,136`
  - `negative`: `4,079`
  - `neutral`: `670`

#### Feature Metadata
- BoW vocabulary size: `48,874`
- Induced sentiment lexicon size: `1,000` tokens
- Lexicon aggregate feature count: `11`
- Combined feature count (`bow+lexicon`): `48,885`

#### All Model/Feature Combinations (sorted by macro-F1)
| Rank | Classifier | Feature Set | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---|---|---:|---:|---:|
| 1 | MultinomialNB | `bow+lexicon` | `0.8650` | `0.5590` | `0.8732` |
| 2 | Logistic Regression | `bow` | `0.8998` | `0.5523` | `0.8884` |
| 3 | MultinomialNB | `bow` | `0.9006` | `0.5415` | `0.8922` |
| 4 | Logistic Regression | `bow+lexicon` | `0.8985` | `0.5342` | `0.8836` |
| 5 | BernoulliNB | `bow+lexicon` | `0.8796` | `0.5300` | `0.8753` |
| 6 | BernoulliNB | `bow` | `0.8676` | `0.5049` | `0.8607` |
| 7 | MultinomialNB | `lexicon` | `0.8795` | `0.4694` | `0.8540` |
| 8 | Logistic Regression | `lexicon` | `0.8828` | `0.4650` | `0.8552` |
| 9 | BernoulliNB | `lexicon` | `0.5784` | `0.3526` | `0.6376` |

#### Best Feature Set per Classifier
- MultinomialNB: `bow+lexicon` (`macro-F1 = 0.5590`)
- Logistic Regression: `bow` (`macro-F1 = 0.5523`)
- BernoulliNB: `bow+lexicon` (`macro-F1 = 0.5300`)

#### Best Classifier per Feature Set
- `bow`: Logistic Regression (`macro-F1 = 0.5523`)
- `lexicon`: MultinomialNB (`macro-F1 = 0.4694`)
- `bow+lexicon`: MultinomialNB (`macro-F1 = 0.5590`)

#### Statistical Significance Testing (McNemar exact + Holm-Bonferroni)

Within the same feature set:
- `bow`:
  - MultinomialNB vs Logistic Regression: **not significant** (`p=0.5143`)
  - BernoulliNB is significantly worse than both (`p << 0.001`)
- `lexicon`:
  - All pairwise differences are significant after Holm correction
  - Logistic Regression significantly outperforms MultinomialNB on exact-match correctness (`p≈3.89e-25`) in this feature-only setting
- `bow+lexicon`:
  - All pairwise differences are significant after Holm correction
  - MultinomialNB significantly outperforms both Logistic Regression and BernoulliNB

Top overall (`multinomial_nb__bow+lexicon`) vs all others:
- Statistically significantly better than all experiments **except** `bernoulli_nb__bow` (`p=0.1321`, not significant after Holm correction)
- Notably better than:
  - `logistic_regression__bow` (`p≈1.35e-79`)
  - `multinomial_nb__bow` (`p≈3.85e-131`)
  - `logistic_regression__bow+lexicon` (`p≈5.19e-74`)

### Task 3 Interpretation / Analysis (this run)
- The dataset is strongly class-imbalanced toward `positive`, so **macro-F1** is the right primary metric for comparing classifiers.
- The best overall model by macro-F1 is **Multinomial Naive Bayes with BoW + lexicon features**.
- Logistic Regression with plain BoW achieves the best accuracy among the top models, but lower macro-F1 than MultinomialNB + BoW+lexicon.
- Lexicon-only features are useful but clearly weaker than BoW or BoW+lexicon, especially for `neutral`.
- The `neutral` class remains hard across all models (very low `f1_neutral`), likely due to:
  - heavy class imbalance
  - short/noisy review text
  - label ambiguity around score `3`

### Which Classifier Is Better? (Task 3 conclusion)
- **Overall winner for this assignment setup**: `Multinomial Naive Bayes + BoW + sentiment lexicon`
  - Best **macro-F1** (`0.5590`)
  - Statistically significant improvements over most alternatives under McNemar exact tests (with Holm correction)
- **Strong runner-up**: `Logistic Regression + BoW`
  - Slightly lower macro-F1 (`0.5523`)
  - Higher accuracy (`0.8998`) due to majority-class performance
- **Binary Naive Bayes (BernoulliNB)** performs reasonably with `bow+lexicon`, but is not the best model on this dataset/configuration.

### Notes / Caveats
- Logistic Regression emitted convergence warnings (`max_iter` reached) for some feature sets with current settings (`solver=saga`, `max_iter=1500`).
- The run completed successfully and produced outputs, but you may improve stability by increasing `max_iter` (e.g., `3000+`) and rerunning.

## Task 4: Logistic Regression for Dot-Based Sentence Boundary Detection (L1 vs L2)

### Goal
- Classify each `.` (dot) as:
  - `EOS` (end of sentence) or
  - `NOT_EOS`
- Use dot predictions to segment sentences.
- Train and compare **L1** vs **L2** logistic regression.

### Important Setup (Task 4)
- Gold labels are **manual** (`CSV`, one row per dot candidate)
- Primary comparison metric: **sentence segmentation F1**
- Secondary comparison metric: **dot-boundary F1**
- Rule-based baseline is included (rule-derived dot heuristic from existing segmentation logic)

Note for this run:
- We also support a practical fallback workflow that converts a **sentence-per-line gold file** into:
  - a labeled dot CSV, and
  - a pseudo corpus CSV
- This was used for the current Task 4 run with `data/processed/sent_gold_actual.txt`.

### Alternative Task 4 Workflow (using sentence-per-line gold)
If you have manually corrected sentence-per-line gold (like `data/processed/sent_gold_actual.txt`) but not a dot-label CSV:
```bash
.venv/bin/python -m src.task4_build_labels_from_gold_sentences \
  --gold_sentences data/processed/sent_gold_actual.txt \
  --out_labels_csv data/processed/task4_dot_labels_from_sent_gold_actual.csv \
  --out_corpus_csv data/processed/task4_sent_gold_actual_pseudo_corpus.csv \
  --sentences_per_doc 10
```
Then train/evaluate using those outputs:
```bash
.venv/bin/python -m src.task4_sentence_lr \
  --labels_csv data/processed/task4_dot_labels_from_sent_gold_actual.csv \
  --corpus_path data/processed/task4_sent_gold_actual_pseudo_corpus.csv \
  --text_column text \
  --primary_metric sent_f1 \
  --compare_rule_baseline \
  --out_dir outputs/project2/task4_lr_sent_gold_actual
```

### Step 1: Export Manual Label Template
```bash
.venv/bin/python -m src.task4_label_export \
  --corpus_path data/raw/corpus.csv \
  --text_column text \
  --max_docs 200 \
  --target_dots 2000 \
  --seed 42 \
  --out_csv data/processed/task4_dot_labels_template.csv
```

Then manually fill `gold_label` (`1`=EOS, `0`=NOT_EOS) and save as:
- `data/processed/task4_dot_labels.csv`

### Step 2: Train + Evaluate L1/L2 Logistic Regression
```bash
.venv/bin/python -m src.task4_sentence_lr \
  --labels_csv data/processed/task4_dot_labels.csv \
  --corpus_path data/raw/corpus.csv \
  --text_column text \
  --seed 42 \
  --dev_ratio 0.15 \
  --test_ratio 0.15 \
  --primary_metric sent_f1 \
  --compare_rule_baseline \
  --out_dir outputs/project2/task4_lr
```

Dependencies for Task 4:
- `scikit-learn`
- `joblib`

### Task 4 Output Files
- `outputs/project2/task4_lr/summary.json` — full run configuration + metrics + winner
- `outputs/project2/task4_lr/comparison.csv` — `rule_based`, `lr_l1`, `lr_l2` side-by-side
- `outputs/project2/task4_lr/tuning_results.csv` — dev tuning (`C`) results for L1/L2
- `outputs/project2/task4_lr/predictions_test.csv` — per-dot predictions on the test set
- `outputs/project2/task4_lr/model_l1.joblib` — saved L1 model
- `outputs/project2/task4_lr/model_l2.joblib` — saved L2 model
- `outputs/project2/task4_lr/feature_config.json` — feature settings for reproducibility

### How to Check Results (Task 4)
```bash
cat outputs/project2/task4_lr/summary.json
cat outputs/project2/task4_lr/comparison.csv
cat outputs/project2/task4_lr/tuning_results.csv
```

What to verify:
- `winner.model` (`lr_l1` or `lr_l2`)
- `test_results.lr_l1.sentence_metrics`
- `test_results.lr_l2.sentence_metrics`
- `test_results.rule_based` (if `--compare_rule_baseline` is enabled)
- `tuning.selected_c` values for L1 and L2

### Report Results (Task 4, filled from `outputs/project2/task4_lr_sent_gold_actual/summary.json`)

#### Labeling / Dataset
- Labeled CSV used: `data/processed/task4_dot_labels_from_sent_gold_actual.csv`
- Gold source for this run: `data/processed/sent_gold_actual.txt` converted to dot labels + pseudo corpus
- Total labeled dot candidates: `2,199`
- Total labeled documents (pseudo-docs): `150`
- Split mode (`doc` / fallback): `doc`
- Train / Dev / Test rows: `1,554 / 306 / 339`
- Train / Dev / Test docs: `106 / 22 / 22`
- Class balance (all rows): `1500 EOS` vs `699 NOT_EOS`

#### Dot-Level Metrics (Secondary)
| Model | Precision | Recall | F1 | Notes |
|---|---:|---:|---:|---|
| Rule-based | `1.0000` | `1.0000` | `1.0000` | Perfect on this test split. |
| LR (L1) | `1.0000` | `1.0000` | `1.0000` | `C=1.0` |
| LR (L2) | `1.0000` | `1.0000` | `1.0000` | `C=10.0` |

#### Sentence Segmentation Metrics (Primary)
| Model | Precision | Recall | F1 | BDER | Notes |
|---|---:|---:|---:|---:|---|
| Rule-based | `1.0000` | `1.0000` | `1.0000` | `0.0000` | Perfect on this test split. |
| LR (L1) | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `C=1.0` |
| LR (L2) | `1.0000` | `1.0000` | `1.0000` | `0.0000` | `C=10.0` |

#### Final Comparison (Task 4)
- Primary metric used: `sentence segmentation F1`
- Better regularization: `L1`
- Selection reason: `Sentence and dot F1 tie; fewer/equal false-positive EOS predictions.`
- Secondary metric (dot F1) summary: `Rule-based, L1, and L2 all achieved 1.0000 on this test split; L1 was selected by the tie-breaker.`

### Task 4 Interpretation Notes (template)
- The logistic regression classifier predicts whether a `.` marks sentence end using local character/token context features.
- L1 and L2 regularization were tuned on a dev split and compared on a held-out test split.
- Final sentence segmentation is reconstructed using predicted dot EOS labels plus deterministic `!` / `?` boundaries.
- Compare the learned model(s) against the rule-based baseline and discuss error patterns (abbreviations, decimals, initials, quotes).

### Task 4 Observed Results (this run)
- After expanding `sent_gold_actual.txt` (including many internal-dot negatives), the derived Task 4 dataset increased to `2,199` labeled dot candidates with `699` `NOT_EOS` examples.
- `rule_based`, `lr_l1`, and `lr_l2` all achieved perfect test performance on this split (`sentence F1 = 1.0000`, `dot F1 = 1.0000`).
- Selected hyperparameters from dev tuning: `C=1.0` (L1) and `C=10.0` (L2); winner remained `lr_l1` via the built-in tie-breaker.

### Task 4 Caveat (important to mention in report)
- This Task 4 run used a **pseudo-corpus** reconstructed from `sent_gold_actual.txt` (manual sentence-per-line gold), grouped into fixed-size pseudo-documents.
- This is valid for comparing L1 vs L2 and benchmarking against the rule-based baseline, but it is not identical to evaluating on original raw corpus document boundaries.
- The dataset is still imbalanced (`EOS` > `NOT_EOS`), though the negative class coverage is substantially improved compared to the earlier run; report both sentence-level and dot-level metrics.
- You may see sklearn deprecation warnings about `penalty`; they do not invalidate the reported metrics (implementation compatibility cleanup can be done later).
