# Assignment 3: Word Embeddings

## Task 1: Dataset Description & Matrix Operations
* **Dataset Analysis:** Describe your dataset including:
    * Size.
    * Number of distinct words (vocabulary size).
    * Word frequency distribution.
    * Count of frequent vs. rare words.
* **Matrix Creation:** * Construct a **Term-Document** matrix.
    * Construct a **Word-Word** (co-occurrence) matrix.
* **Visualization:** Visualize both matrices in a clear **matrix form**.

---

## Task 2: Word2Vec Modeling
* **Training:** Train a model using [Word2Vec](https://github.com/tmikolov/word2vec) and document your chosen hyperparameters (e.g., skip-gram vs. CBOW, window size, vector dimensions).
* **Synonym Detection:** Identify synonyms or semantically similar words for **10 different target words**. 
    * *Evaluation:* Was the model accurate? Describe the qualitative results.
* **Vector Arithmetic:** Apply mathematical equations to the vector values (e.g., $vec(\text{"king"}) - vec(\text{"man"}) + vec(\text{"woman"})$). 
    * *Analysis:* Are there visible patterns or linguistic regularities? Describe the results.

---

## Task 3: GloVe Modeling (30%)
* **Training:** Train a model using [GloVe](https://github.com/stanfordnlp/GloVe) and describe the chosen parameters.
* **Synonym Detection:** Find similar words for **10 different words**.
    * *Evaluation:* Assess the accuracy and describe the findings.
* **Vector Arithmetic:** Apply mathematical equations to the vector values.
    * *Analysis:* Identify and describe any emerging patterns.

---

## Task 4: Comparative Analysis (10%)
* Compare the performance and embedding quality of **GloVe** versus **Word2Vec** based on your findings in Tasks 2 and 3.

---

## Task 5: Deep Learning Classification (30%)
Apply **RNN**, **Bidirectional RNN**, and **LSTM** architectures to classify text. Compare their performance using the following feature extraction methods:
* Count Vectorizer
* TF-IDF
* PMI (Pointwise Mutual Information)
* Word2Vec
* GloVe

> **Output:** Present all comparative results in a **tabular form**.

---

## Task 6: Reporting & Extra Credit
* **Task 6 (20%):** Write a comprehensive technical report.
* **Extra Task (20%):** Create a **User Interface (UI)** to display program results and model outputs.

---

## Presentations
* **Format:** Each team must give a short explanation of their work.
* **Time Slot:** 10 minutes total (5 minutes for presentation + 5 minutes for Q&A).

---

## Final Report Requirements
* **Length:** Maximum **5 pages** (excluding references).
* **Team Contribution:** Include a section detailing the specific responsibilities of each member.
* **Individual Assessment:** Members will be asked specific questions regarding the theoretical model and the source code.

### Report Structure:
1.  **Motivation:** The problem being tackled and the specific setting/context.
2.  **Method:** Machine learning techniques used and the rationale behind them.
3.  **Experiments:** Description of experiments, outcomes, and error analysis. You must include at least one **baseline**. (Note: Negative results are encouraged and accepted).

---

## Submission & Policies
* **Deliverables:** Submit the final report, all source code, presentation slides, relevant data, and experimental results via Blackboard after the presentation.
* **Late Policy:** 20% deduction per day after the deadline.
* **Attendance Policy:** Group members who do not participate in the presentation will receive a **50% grade deduction**.

**Presentation Date:** 19 March, during class time.