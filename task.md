Reading Comprehension System

Objective: In this part of the project, you will focus on developing a system capable of answering questions based on a given context passage. You will implement and integrate a Bidirectional Attention Flow (BiDAF) model and leverage the contextual understanding capabilities of a pre-trained BERT-Base model to achieve this.

Tasks:

BiDAF Implementation: (10%)
Implement the BiDAF architecture using either TensorFlow or PyTorch.
The model should take a question and a context passage as input.
The model should output the start and end positions of the answer within the context passage.
2.             BERT-Base Integration: (20%)

Utilize a pre-trained BERT-Base model.
Generate contextualized word embeddings for both the question and the context.
Integrate these embeddings into the BiDAF model.
Analyze how BERT embeddings affect the model's performance compared to traditional word embeddings (like GloVe or Word2Vec).
3.             Training and Evaluation: (20)

Train the BiDAF model (with or without BERT embeddings) on a suitable reading comprehension dataset (e.g., SQuAD, CoQA).
Evaluate the model's performance using metrics such as Exact Match (EM) and F1-score.
