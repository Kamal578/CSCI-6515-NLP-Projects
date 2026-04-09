## Project 4 Presentation Script

Below is a slide-by-slide speaking script matched to [project4_slides.tex](/Users/rufat/ada/nlp/project1/slides/project4_slides.tex).

### Slide 1 — Title Page
**Speaker: Kamal**

“Good afternoon. We are Kamal Ahmadov and Rufat Guliyev, and this is our Project 4 presentation for NLP. Our project combines two parts: first, a theoretical analysis of a fine-tuned BERT model for sentiment analysis, and second, an implemented reading comprehension system using BiDAF with both GloVe and frozen BERT embeddings.”

---

### Slide 2 — Outline
**Speaker: Kamal**

“First I will briefly introduce the project scope and Task 1, which is the BERT sentiment analysis part. Then Rufat will present Task 2, which is the reading comprehension system, including the architecture, experiments, results, and the Streamlit UI. We will finish with the conclusion and questions.”

---

### Slide 3 — Motivation and Scope
**Speaker: Kamal**

“The main motivation of Project 4 is to study how useful transformer-based contextual representations are for downstream NLP tasks. We looked at this from two directions. In Task 1, we analyzed an existing BERT sentiment model to understand what kind of inputs, outputs, and assumptions it has. In Task 2, we built an actual QA system and compared a traditional embedding baseline, GloVe, against frozen BERT embeddings inside the same BiDAF framework. In this repository, we also included the report, experiment artifacts, and a Streamlit UI.”

---

### Slide 4 — Task 1: Selected Open-Source Model
**Speaker: Kamal**

“For Task 1, we selected the open-source Hugging Face model `nlptown/bert-base-multilingual-uncased-sentiment`. It is built on top of multilingual BERT and fine-tuned for 5-class product-review sentiment classification. We chose this model because it is public, well documented, and more relevant for multilingual transfer than an English-only sentiment model. That also makes the Azerbaijani applicability question much more meaningful.”

---

### Slide 5 — Task 1: Inputs, Outputs, and Light Math
**Speaker: Kamal**

“This model takes review text as input, then tokenizes it with WordPiece and converts it into tensors such as input IDs and attention masks. Since it is an uncased model, the text is lowercased before tokenization. The output is a probability distribution over five sentiment classes, from one star to five stars. In the standard BERT classification setup, we take the final representation of the CLS token and pass it through a softmax classifier, which is the equation shown here. Another important property is that the maximum input length is 512 subword positions.”

---

### Slide 6 — Task 1: Can It Work for Azerbaijani?
**Speaker: Kamal**

“Our conclusion for Azerbaijani is yes, but only as a baseline, not as a final solution. It can transfer because the encoder is multilingual and because subword tokenization helps with morphologically rich languages. But direct zero-shot use is still risky, because the sentiment head was not fine-tuned on Azerbaijani, and Azerbaijani is agglutinative, so many sentiment-bearing forms appear with suffix chains. So this kind of model is a good starting point, but reliable Azerbaijani sentiment analysis would still require local fine-tuning.”

**Handoff: Kamal to Rufat**

“Now Rufat will continue with Task 2, the implemented reading comprehension system.”

---

### Slide 7 — Task 2 Goal and Variants
**Speaker: Rufat**

“In Task 2, the goal is different. Given a question and a context passage, the model must predict the answer span inside the context. We implemented two comparable variants. The first is BiDAF with pretrained GloVe embeddings, which serves as the traditional baseline. The second is BiDAF with frozen BERT embeddings from `bert-base-uncased`. Both variants use SQuAD 1.1 as the dataset, and both share the same BiDAF span-prediction architecture. So the main difference is only the source of the input embeddings.”

---

### Slide 8 — Task 2 Architecture
**Speaker: Rufat**

“This slide shows the structure of our QA model. We start with embeddings, then pass them through a two-layer highway network, a contextual bidirectional LSTM, the BiDAF attention-flow mechanism, additional modeling layers, and finally two output heads for the start and end positions. The first formula shows the decoding idea: we choose the span with the best joint start and end score. The second formula shows the training loss, which is the sum of cross-entropy losses for the gold start and end positions. In the BERT variant, we tokenize words into subtokens, take the first subtoken representation for each word, and then project it into the BiDAF input space.”

---

### Slide 9 — Task 2 Experimental Setup
**Speaker: Rufat**

“The experiment reported in the slides uses the normal Task 2 model settings, but on a larger CPU-feasible slice of SQuAD. We trained on 8192 examples and validated on 2048 examples. After windowing, that became 9390 training windows and 3069 validation windows. We used 4 epochs, batch size 16, hidden size 64, context window 160 words, and stride 64 words. For evaluation, we used Exact Match and token-level F1, shown here. We selected the best checkpoint by validation F1, so the saved EM is the EM at the best-F1 epoch.”

---

### Slide 10 — Task 2 Results
**Speaker: Rufat**

“These are the refreshed results from the larger rerun. BiDAF with the traditional word-vector baseline reached an EM of 0.1216 and an F1 of 0.1819. BiDAF with frozen BERT reached an EM of 0.2642 and an F1 of 0.3600. So in this configuration, frozen BERT clearly outperformed the baseline, with a gain of about 0.14 in EM and about 0.18 in F1. This is the result we expected conceptually, because contextual embeddings give the model much richer information than static vectors.”

---

### Slide 11 — Task 2 Error Analysis and Engineering Notes
**Speaker: Rufat**

“The results are much better now, but there is still room for improvement. This is still a capped subset rather than the full SQuAD training set, and the BERT encoder is frozen, so it cannot adapt fully to the QA objective. We also found an important debugging issue during development. Some contexts contained standalone Unicode formatting tokens, combining-mark tokens, and even a replacement-character token. Our word tokenizer kept them, but BERT dropped them, which broke word-level pooling. We fixed this by filtering out tokens made only of control, format, or combining-mark characters, and also dropping standalone replacement-character tokens. After that, the larger compare run completed successfully.”

---

### Slide 12 — Extra Task: Streamlit UI
**Speaker: Rufat**

“As the extra task, we built a Streamlit dashboard for Project 4. The UI includes a Project 4 overview, Task 1 model-analysis panels, Task 2 metric boards, a prediction inspector that compares GloVe and BERT outputs side by side, and an artifact explorer with run commands and saved summaries. This makes the project much easier to demonstrate and also makes the saved results easier to inspect interactively.”

---

### Slide 13 — Conclusion
**Speaker: Kamal**

“To conclude, Project 4 produced both a theoretical and an implemented result. In Task 1, we showed that a multilingual fine-tuned BERT sentiment model has clear inputs, outputs, and limitations, and that it is only partially suitable for Azerbaijani without extra fine-tuning. In Task 2, we implemented BiDAF in PyTorch, integrated frozen BERT embeddings, trained the QA system on SQuAD-style data, and evaluated it with EM and F1. In the refreshed larger experiment, frozen BERT clearly outperformed the traditional baseline, which supports the value of contextual embeddings in this setup. Finally, the Streamlit UI completed the project by making the outputs easier to inspect and present.”

---

### Slide 14 — Questions
**Speaker: Rufat**

“Thank you for listening. We are ready for your questions.”

---

## Suggested Timing Split
- **Kamal**
  - Slides 1–6
  - Slide 13
- **Rufat**
  - Slides 7–12
  - Slide 14

## Very Short Backup Version
If you are running out of time, compress like this:
- Kamal: Slides 1–3 in about 45 seconds
- Kamal: Slides 4–6 in about 1 minute 15 seconds
- Rufat: Slides 7–12 in about 2 minutes 15 seconds
- Kamal + Rufat: Slides 13–14 in about 30–40 seconds

If you want, I can also turn this into a **rehearsal version** with pauses, emphasis words, and likely Q&A questions per slide.
