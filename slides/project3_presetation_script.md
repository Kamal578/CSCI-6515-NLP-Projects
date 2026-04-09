# Project 3 Presentation Script

This script is aligned to `slides/project3_slides.tex` (19 slides).

Use it in two modes:
- `Full mode` (8-10 min): cover every slide.
- `Class mode` (5 min): follow the `Quick path` notes and skip optional details.

## Speaker plan
- `Kamal`: Slides 1-2, 5-9, 13-15, 19
- `Rufat`: Slides 3-4, 10-12, 16-18

---

## Slide-by-slide script

### Slide 1 — Title
`Kamal`

“Good [morning/afternoon]. We are presenting Project 3 for NLP, focused on Azerbaijani word embeddings, comparative analysis, deep learning classification, and an interactive UI. I’m Kamal Ahmadov, and this is Rufat Guliyev.”

Quick path: 10 seconds.

---

### Slide 2 — Outline
`Kamal`

“We will move in this order: motivation and data, Task 1 matrix analysis, Task 2 Word2Vec, Task 3 GloVe, Task 4 comparison, Task 5 deep learning results, then Task 6 report artifacts and our UI.”

Quick path: 10 seconds.

---

### Slide 3 — Motivation
`Rufat`

“Our motivation is simple: sparse bag-of-words is not enough for semantic similarity in Azerbaijani. So we evaluate embeddings in two ways. Intrinsic evaluation checks neighbors and analogy arithmetic. Extrinsic evaluation checks real downstream sentiment performance. The final goal was to complete all assignment tasks with reproducible outputs and a usable interface.”

Quick path: 15-20 seconds.

---

### Slide 4 — Datasets and Pipeline
`Rufat`

“Tasks 1 to 3 use `data/raw/corpus.csv`. Tasks 4 and 5 use `data/external/train.csv` and `test.csv`. The whole pipeline runs from one command, `bash scripts/run_project3.sh`, and writes task-wise outputs under `outputs/project3/task1` through `task6`. We also added a Streamlit dashboard and kept a legacy Flask UI.”

Quick path: 20 seconds.

---

### Slide 5 — Task 1 Method
`Kamal`

“Task 1 builds two matrix views. First, the term-document matrix \(X\), where each cell is token count in a document. Second, the word-word co-occurrence matrix \(C\), built with a context window. We also bucket words by frequency using assignment-specific thresholds: frequent if count is at least 100, rare if count is at most 2.”

Optional detail:
“For presentation clarity, we visualize top slices while still saving full sparse artifacts.”

Quick path: 20 seconds.

---

### Slide 6 — Task 1 Results
`Kamal`

“Our corpus has 31,842 documents, 11.9 million tokens, and 586,674 unique terms. Frequent words are 12,436, and rare words are 398,599. This long-tail pattern is expected for encyclopedic text and justifies subword/embedding methods in later tasks.”

Quick path: 20 seconds.

---

### Slide 7 — Task 1 Visualizations
`Kamal`

“On the left is the term-document heatmap and on the right is the word-word heatmap. The concentration bands show high-frequency function words and strong local co-occurrence structure. These matrix artifacts become the base for our later feature engineering and analysis.”

Quick path: 15-20 seconds.

---

### Slide 8 — Task 2 Word2Vec Setup
`Kamal`

“For Task 2 we trained Word2Vec using skip-gram. The objective maximizes context-word likelihood around each target word. Our configuration is dimension 200, window 5, negative sampling 10, min-count 5, and 10 iterations. We evaluate 10 target words for similarity and run analogy arithmetic using vector offsets \(b-a+c\).”

Quick path: 20 seconds.

---

### Slide 9 — Task 2 Word2Vec Results
`Kamal`

“Word2Vec produced a 122,930-word vocabulary with 200-dimensional vectors. Analogy hit@k is 0.6667, with 3 evaluated analogies out of 5 due to OOV skips. Qualitatively, morphology works well for several tokens, but some neighbors are noisy, which is expected with corpus artifacts and OOV effects.”

Optional line:
“Examples: two analogies were correct, but family-role analogy failed.”

Quick path: 20 seconds.

---

### Slide 10 — Task 3 GloVe Setup
`Rufat`

“Task 3 uses GloVe, which factorizes global co-occurrence information. The loss minimizes weighted reconstruction error between dot products and \(\log X_{ij}\). Our settings are vector size 200, window 10, min-count 5, max-iter 20, with standard \(x_{max}, \eta, \alpha\) values.”

Quick path: 20 seconds.

---

### Slide 11 — Task 3 GloVe Results
`Rufat`

“GloVe also reached 122,930 vocabulary size at 200 dimensions. Analogy hit@k is 0.7500, slightly better than Word2Vec in this run. Similar to Task 2, we observe strong morphological neighbors but still some noisy top neighbors for certain targets.”

Quick path: 20 seconds.

---

### Slide 12 — Task 4 Comparison
`Rufat`

“Task 4 compares both models on shared metrics: analogy score, average top-3 cosine, and sentiment-vocabulary coverage. GloVe wins analogy quality, Word2Vec wins neighborhood cosine, and coverage is equal. So there is no universal winner; model choice depends on task objective.”

Quick path: 20 seconds.

---

### Slide 13 — Task 5 Design
`Kamal`

“Task 5 is our full deep-learning grid: 3 architectures times 5 feature families, plus a baseline. Architectures are RNN, BiRNN, and LSTM. Features are Count-SVD, TFIDF-SVD, PMI-SVD, Word2Vec, and GloVe. Our primary metric is macro-F1, defined as class-wise F1 averaged equally across classes.”

Quick path: 20 seconds.

---

### Slide 14 — Task 5 Results Table
`Kamal`

“The best macro-F1 is from `lstm__pmi_svd` at 0.5684. The logistic-regression TFIDF baseline is close, at 0.5588 macro-F1. This is important: we included a baseline as required and showed that learned sequence models improve macro-F1, but margins are moderate due to class imbalance.”

Quick path: 25 seconds.

---

### Slide 15 — Task 5 Visualization
`Kamal`

“This bar chart shows top runs by macro-F1. PMI-based features dominate the highest ranks in our experiments. So one practical takeaway is that distributional PMI structure remains very effective for this dataset.”

Quick path: 15 seconds.

---

### Slide 16 — Task 6 Deliverables
`Rufat`

“Task 6 packages report-ready artifacts: embedding comparison table, DL results table, and markdown summary notes. We also produced a presentation bundle zip containing figures, tables, and per-task summaries for quick grading and presentation reuse.”

Quick path: 15-20 seconds.

---

### Slide 17 — Extra Task UI
`Rufat`

“For extra credit, we built a Streamlit dashboard. It includes task tabs plus a playground where users can input words, run custom analogies, check cosine similarity, blend vectors, and visualize semantic maps in 2D. The UI is read-only over saved artifacts, so it is fast and reproducible.”

Quick path: 20 seconds.

---

### Slide 18 — Limitations
`Rufat`

“Main limitations are class imbalance, small/OOV-sensitive analogy sets, and noisy tokens in nearest-neighbor lists. Future work includes cleaner analogy benchmarks, improved balancing strategies, and transformer baselines for stronger downstream performance.”

Quick path: 15 seconds.

---

### Slide 19 — Conclusion
`Kamal`

“To conclude: all assignment tasks were completed end-to-end with reproducible outputs and an interactive UI. GloVe and Word2Vec each show different strengths, and our best downstream model was `lstm__pmi_svd`. Thank you — we are ready for questions.”

Quick path: 15 seconds.

---

## 5-minute class mode (recommended order)

If strict 5-minute timing is enforced, present:
1. Slide 1 (title)  
2. Slide 3 (motivation)  
3. Slide 4 (dataset/pipeline)  
4. Slide 6 (Task 1 key numbers)  
5. Slide 9 (Task 2 result)  
6. Slide 11 (Task 3 result)  
7. Slide 12 (comparison)  
8. Slide 14 (Task 5 main table + baseline)  
9. Slide 17 (UI)  
10. Slide 19 (conclusion)

Target pacing:
- Intro + motivation: 45 sec
- Tasks 1-4: 1 min 45 sec
- Task 5: 1 min 15 sec
- Task 6 + UI: 45 sec
- Conclusion: 20-30 sec

---

## Expected Q&A and short answers

### Q1: Why use both Word2Vec and GloVe?
“They capture semantics differently: Word2Vec emphasizes local context prediction, while GloVe uses global co-occurrence statistics. Comparing both is required and gives a stronger analysis.”

### Q2: Why is macro-F1 primary, not accuracy?
“Because class distribution is imbalanced, especially toward positive sentiment. Macro-F1 weights classes equally and better reflects minority-class behavior.”

### Q3: Why is baseline close to the best deep model?
“Dataset imbalance and feature strength of TFIDF make the baseline competitive. Deep models still provided the best macro-F1 in our run.”

### Q4: Biggest error source?
“OOV/noisy tokens in embedding evaluation and neutral-class weakness in sentiment classification.”

### Q5: If you had more time, what would you improve first?
“A larger curated analogy benchmark and transformer-based classification baselines.”
