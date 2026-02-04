# Dataset Datasheet — Azerbaijani Wikipedia Corpus

Last update: 2026-02-04 (run timestamp: 2026-02-04T23:32:11)

## Motivation
- **Purpose**: Build a course-scale Azerbaijani corpus to study lexical statistics (Zipf/Heaps), train subword models (BPE), prototype sentence segmentation and spell checking, and supply data for class assignments.
- **Creators**: Course team (students) for “Natural Language Processing” (Spring 2026).
- **Funding**: None; academic coursework.

## Composition / Situation
- **Source domain**: Azerbaijani Wikipedia (azwiki), encyclopedic, edited prose.
- **Time of writing**: Mixed; Wikipedia pages contain edits across many years. Collected on 2026-02-04 via MediaWiki API.
- **Register**: Formal/neutral prose; some technical, historical, and geographical content.
- **Document count**: 623 articles (post-cleaning).
- **Size**: 238,286 tokens; 48,151 types (after lowercasing).
- **Variety**: Standard Azerbaijani orthography; includes loanwords, names, numerals, dates.
- **Genres/topics**: Broad (random sampling), so coverage spans geography, history, culture, science; no explicit topical filtering.

## Language Variety
- **Primary language**: Azerbaijani (ISO 639-1: az).
- **Dialects**: Not explicitly controlled; Wikipedia generally uses standard written Azerbaijani.
- **Other languages**: Minimal; lines dominated by English removed via langid (threshold > 0.7).

## Speaker / Author Demographics
- Wikipedia contributors are pseudonymous/anonymous; no consistent demographic metadata is available. Content reflects collective editing norms and potential systemic biases of Wikipedia editors.

## Collection Process
- **Acquisition method**: MediaWiki API calls (\texttt{src/pull\_wikipedia.py}); random-page sampling (main namespace) with optional category mode (not used in the recorded run).
- **Sampling**: Random 623 articles (post length-filtering).
- **Filtering**: Drop pages whose cleaned text < 400 characters.
- **Metadata stored**: \texttt{page\_id}, \texttt{title}, \texttt{revision\_id}, \texttt{timestamp}, \texttt{source}, \texttt{url}, \texttt{text}, \texttt{doc\_id}.
- **Storage format**: UTF-8 CSV at \texttt{data/raw/corpus.csv}.
- **Consent/licensing**: Wikipedia content is CC BY-SA 3.0/4.0; reuse requires attribution and ShareAlike.

## Preprocessing
- **Markup removal**: Templates/tags stripped with \texttt{mwparserfromhell}; category/file links removed; HTML tags dropped.
- **Section filtering**: “References”, “External links”, “Notes”, “See also” removed when detected.
- **Language filtering**: Lines classified as English with langid score > 0.7 removed.
- **Whitespace/normalization**: Collapsed multiple spaces; normalized dashes/quotes in tokenizer.
- **Tokenization**: Unicode regex allowing Azerbaijani letters, internal apostrophes/hyphens, decimals; optional lowercasing (on in stats).
- **Lowercasing**: Enabled for reported counts; raw text preserved.

## Derived Annotations / Artifacts
- **Token frequencies**: \texttt{outputs/stats/token\_freq.csv}; summary in \texttt{outputs/stats/summary.json}.
- **Zipf plot**: \texttt{outputs/plots/zipf.png}.
- **Heaps’ parameters**: \texttt{k}=4.57, $\beta$=0.750; plot at \texttt{outputs/plots/heaps.png}.
- **BPE model**: 5,000 merges; outputs in \texttt{outputs/bpe/}.
- **Vocabulary (filtered)**: 11,491 types with min\_freq≥3, min\_len≥3 (\texttt{data/processed/vocab.txt}).
- **Sentence segmentation output**: \texttt{outputs/sentences.txt} (11,479 sentences from first 500 docs).
- **Spellcheck resources**: synthetic misspell set \texttt{data/processed/spell\_test.csv}; confusion/weights \texttt{outputs/spellcheck/confusion.json}; eval \texttt{outputs/spellcheck/spell\_eval.json} (Acc@1=0.637, Acc@5=0.801); heatmap \texttt{confusion\_heatmap.png}.

## Annotation Process
- No human annotation included in this release.
- Synthetic labels: spellcheck test pairs are automatically corrupted (delete/insert/substitute/transpose) weighted by token frequency.
- Planned future human annotation: sentence boundary gold set (\texttt{data/processed/sent\_gold.txt}) and real-world spelling errors (not yet).

## Uses
- **Intended**: Coursework experiments on tokenization, segmentation, subword modeling, lexical statistics, and spelling correction; demonstrations of algorithm behavior on Azerbaijani text.
- **Out-of-scope / Caution**: Not curated for fairness or safety; not suitable for sensitive-topic analysis; not cleaned for personal data beyond Wikipedia norms; English filtering is heuristic.

## Risks / Biases / Limitations
- Wikipedia biases: topic imbalance, demographic skew of editors, systemic coverage gaps.
- Residual markup/noise may persist; langid filtering may drop or keep some non-AZ lines incorrectly.
- Synthetic spelling benchmark may not reflect real user errors; weighted edits are derived from synthetic confusions.
- No speaker demographics; cannot study sociolinguistic variation.

## Distribution
- **License**: CC BY-SA (inherits from Wikipedia). Redistribution of text or derivatives must include attribution and ShareAlike.
- **Attribution suggestion**: Cite “Azerbaijani Wikipedia dump (retrieved 2026-02-04) via MediaWiki API; processed for NLP coursework.”
- **Files to share**: \texttt{data/raw/corpus.csv}, derived stats/plots, vocab/BPE, spellcheck artifacts, datasheet.

## Maintenance
- Point of contact: Kamal Ahmadov (ahmadov.kamal423@gmail.com).
- Versioning: This snapshot is tied to collection date 2026-02-04 and the outputs produced by \texttt{scripts/run\_all.sh}.
