# ITC6110 — Natural Language Processing  
## Group Project Report  
**Module:** ITC 6110 — Natural Language Processing  
**Term:** Spring Semester 2025  
**Team Members:** [NAME 1] · [NAME 2] · [NAME 3]  
**Submission Date:** [DATE]  

---

## Abstract

This report documents an end-to-end NLP pipeline applied to the BBC News dataset (2,225 articles, 5 categories). We cover data collection, preprocessing, feature engineering, topic modelling, text classification using both traditional machine learning and deep learning approaches, a Retrieval-Augmented Generation (RAG) conversational system, and an interactive Streamlit web application. Our best classifier, fine-tuned DistilBERT, achieves 97.53% accuracy and 0.974 macro F1. The RAG system uses FAISS vector search with Flan-T5-base for answer generation, evaluated via ROUGE-L.

---

## 1. Data Collection

### 1.1 Dataset

We used the **BBC News dataset** (Greene & Cunningham, 2006), sourced from HuggingFace Hub (`SetFit/bbc-news`). The dataset comprises **2,225 news articles** collected from the BBC website between 2004 and 2005, labelled across five mutually exclusive categories:

| Category      | Train | Test | Total |
|---------------|-------|------|-------|
| Sport         | [N]   | [N]  | [N]   |
| Business      | [N]   | [N]  | [N]   |
| Politics      | [N]   | [N]  | [N]   |
| Tech          | [N]   | [N]  | [N]   |
| Entertainment | [N]   | [N]  | [N]   |
| **Total**     | **1,225** | **1,000** | **2,225** |

The dataset is pre-split into 1,225 training articles and 1,000 test articles. Articles average approximately 390 words in length.

### 1.2 Dataset Justification

[FILL IN: 2–3 sentences on why BBC News was chosen — multi-class, clean labels, sufficient size, diverse topics covering NLP tasks from topic modelling to RAG]

### 1.3 Loading the Data

The dataset was loaded using the HuggingFace `datasets` library and saved locally as `data/raw/bbc_news.csv` for reproducibility:

```python
from datasets import load_dataset
import pandas as pd
ds = load_dataset("SetFit/bbc-news")
df = pd.concat([pd.DataFrame(ds['train']).assign(split='train'),
                pd.DataFrame(ds['test']).assign(split='test')])
df.to_csv('data/raw/bbc_news.csv', index=False)
```

---

## 2. Data Preprocessing and Normalisation

### 2.1 Pipeline Overview

Text preprocessing was implemented in Notebook 1 (`01_data_features.ipynb`). The following steps were applied in sequence:

1. **Lowercasing** — all text converted to lowercase
2. **URL and HTML removal** — regex patterns removed hyperlinks and tags
3. **Punctuation and special character removal** — non-alphabetic characters stripped
4. **Stop word removal** — using NLTK's English stop word list
5. **Lemmatisation** — WordNetLemmatizer applied to reduce inflected forms to base lemma
6. **Tokenisation** — whitespace tokenisation after cleaning

### 2.2 Missing and Duplicate Values

[FILL IN: report how many missing values / duplicates were found and handled. Check NB1 output]

### 2.3 Stemming vs Lemmatisation

We chose **lemmatisation** over stemming because lemmatised tokens preserve valid English words, which improves the quality of downstream embeddings and topic models. Stemming produces truncated forms (e.g. "running" → "run" vs "runn") that can conflate unrelated terms.

### 2.4 Design Decision: Two Text Representations

Two versions of the text were retained throughout the pipeline:
- `text` — original, uncleaned text (used by DistilBERT, whose subword tokeniser performs better on natural language)
- `text_processed` — lemmatised, stop-word-free tokens (used by Word2Vec, LSTM, TF-IDF, LDA)

[FILL IN: optionally add a before/after preprocessing example for one article]

---

## 3. Feature Engineering and Text Visualisation

### 3.1 TF-IDF

A TF-IDF matrix was constructed over the full corpus (2,225 × 10,000 features) using scikit-learn's `TfidfVectorizer`. This sparse representation captures term importance relative to the corpus and serves as input for the classical ML classifiers in Section 4.2.

### 3.2 Word2Vec Embeddings

A **Skip-gram Word2Vec** model was trained on the BBC corpus using Gensim (embedding size = 100, window = 5, min_count = 2, epochs = 10). Document-level vectors were obtained by averaging the word vectors of all tokens in each article, producing a (2,225 × 100) matrix.

**Nearest neighbour example:** querying the trained model for words most similar to *"football"* returns:

[FILL IN: paste the top-5 similar words output from NB1]

### 3.3 t-SNE Visualisation

t-SNE (perplexity = 30, max_iter = 1,000) was applied to the 100-dimensional Word2Vec document vectors to project them into 2D for visualisation. Figure 1 shows the resulting scatter plot coloured by category.

**Figure 1:** t-SNE projection of Word2Vec document embeddings (5 BBC categories)  
![t-SNE](../outputs/figures/03_tsne_w2v.png)

[FILL IN: 2–3 sentences describing what you observe — which categories cluster well, which overlap and why]

### 3.4 Word Frequency Visualisation

[FILL IN: describe any word cloud or frequency figures generated — figures 01, 02]

---

## 4. Model Building

### 4.1 Unsupervised Learning — Topic Modelling (LDA)

#### 4.1.1 Algorithm

Latent Dirichlet Allocation (LDA) was implemented using Gensim. The optimal number of topics *K* was selected by maximising the **C_v coherence score** across K ∈ {3, 4, 5, 6, 7, 8}.

**Figure 2:** Coherence score vs. number of topics  
![Coherence](../outputs/figures/07_lda_coherence.png)

[FILL IN: state which K was chosen and what C_v score it achieved]

#### 4.1.2 Top Words per Topic

[FILL IN: paste the top-10 keywords per topic table from NB2 output]

#### 4.1.3 Interactive Visualisation

An interactive pyLDAvis visualisation was generated (`outputs/figures/lda_vis.html`), showing topic distances and term saliency.

**Figure 3:** LDA topic visualisation (pyLDAvis)  
![LDA](../outputs/figures/08_lda_topics.png)

#### 4.1.4 Discussion

[FILL IN: discuss how well the discovered topics map to the 5 known BBC categories. Which topics are clean, which overlap?]

---

### 4.2 Supervised Learning

#### 4.2.1 Task 1 — Text Classification

##### Traditional ML Models

Three classifiers were trained using a scikit-learn `Pipeline` (TF-IDF refitted on train split only to prevent data leakage):

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Linear SVM | **97.00%** | [F1] |
| Logistic Regression | 96.78% | [F1] |
| Naive Bayes | 96.46% | [F1] |

All models used 5-fold stratified cross-validation for hyperparameter selection.

[FILL IN: any brief comment on why SVM outperforms the others on this task]

##### Deep Learning — Bidirectional LSTM

A two-layer Bidirectional LSTM was trained from scratch on lemmatised token sequences (vocabulary size = 15,000, max sequence length = 200 tokens, embedding dimension = 128, hidden dimension = 256). Training used the Adam optimiser (lr = 1e-3) with ReduceLROnPlateau scheduling over 8 epochs.

| Metric | Value |
|--------|-------|
| Test Accuracy | 80.26% |
| Macro F1 | 0.793 |

**Figure 4:** BiLSTM training curves  
![LSTM curves](../outputs/figures/14_lstm_training_curves.png)

**Figure 5:** BiLSTM confusion matrix  
![LSTM CM](../outputs/figures/15_lstm_confusion_matrix.png)

##### Deep Learning — DistilBERT (Fine-tuned)

DistilBERT (`distilbert-base-uncased`) was fine-tuned for sequence classification using the HuggingFace `Trainer` API. The model was trained on the original (unprocessed) text, as BERT's subword tokeniser performs better on natural language than lemmatised input. Training used 4 epochs, batch size 16, learning rate 2e-5 with linear warmup.

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.53%** |
| Macro F1 | **0.974** |

**Figure 6:** DistilBERT confusion matrix  
![BERT CM](../outputs/figures/16_distilbert_confusion_matrix.png)

##### Model Comparison

**Figure 7:** Accuracy and Macro F1 across all models  
![Comparison](../outputs/figures/17_model_comparison.png)

| Model | Type | Accuracy | Macro F1 |
|-------|------|----------|----------|
| DistilBERT (fine-tuned) | Deep Learning | **97.53%** | **0.974** |
| Linear SVM | Traditional ML | 97.00% | [F1] |
| Logistic Regression | Traditional ML | 96.78% | [F1] |
| Naive Bayes | Traditional ML | 96.46% | [F1] |
| BiLSTM (random embeddings) | Deep Learning | 80.26% | 0.793 |

**Key finding:** A fine-tuned transformer (DistilBERT) achieves the best results, marginally outperforming well-tuned classical methods. Notably, the BiLSTM — despite being a deep learning model — underperforms classical methods when trained without pre-trained embeddings, illustrating that pre-training, not architecture depth, drives performance gains.

##### XAI — LIME

[FILL IN: describe the LIME explanation generated for one sport article. Reference figures 11/12 (lime_explanation_sport.html). What features drove the prediction? Were they intuitive?]

**Figure 8:** LIME explanation for a Sport article  
![LIME](../outputs/figures/12_lime_sport_explanation.png)

---

#### 4.2.2 Task 2 — Retrieval-Augmented Generation (RAG)

##### System Architecture

The RAG pipeline consists of two components:

**Retriever:** Sentence embeddings were generated using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). All 2,225 articles were encoded and stored in a FAISS `IndexFlatIP` index with L2-normalised vectors, providing exact cosine similarity search.

**Generator:** Flan-T5-base (`google/flan-t5-base`) was used for answer generation via `AutoModelForSeq2SeqLM`. The prompt template provides retrieved article snippets as context and instructs the model to answer only from the provided sources.

```
You are a knowledgeable news assistant. Answer using only the articles provided.

Article 1 [sport]: <snippet>
Article 2 [politics]: <snippet>
...

Question: <user query>
Answer:
```

##### Evaluation

The system was evaluated on 10 manually curated questions drawn from topics present in the BBC corpus. Performance was measured using **ROUGE-L** (longest common subsequence overlap between generated and expected answers).

**Figure 9:** ROUGE-L scores per question  
![ROUGE](../outputs/figures/18_rag_rouge_scores.png)

[FILL IN: paste the mean ROUGE-L score from NB3 output]

| Metric | Score |
|--------|-------|
| Mean ROUGE-L | [FILL] |
| Best question ROUGE-L | [FILL] |
| Worst question ROUGE-L | [FILL] |

[FILL IN: 2–3 sentences discussing the ROUGE-L results. Note that Flan-T5-base is a modest generator and ROUGE-L penalises paraphrase — scores in the 0.10–0.35 range are expected and not indicative of poor retrieval quality]

---

#### 4.2.3 Task 3 — Streamlit UI and Deployment

##### Application Design

An interactive Streamlit application (`app/app.py`) was developed with two tabs:
- **Article Classifier** — user pastes any news text; the Linear SVM pipeline predicts the category and displays per-class confidence scores as a bar chart
- **Sports Q&A (RAG)** — user enters a natural language question; the system retrieves the top-k most relevant BBC articles and generates an answer using Flan-T5-base

[FILL IN: add 1–2 screenshots of the running app here]

##### Deployment

[FILL IN: HuggingFace Spaces URL once deployed — e.g. https://huggingface.co/spaces/USERNAME/bbc-news-assistant]

[FILL IN: brief description of deployment steps — upload app.py, requirements.txt, data files to HF Spaces]

---

## 5. Conclusions and Future Work

### 5.1 Summary of Findings

[FILL IN: 3–4 bullet points summarising the key takeaways — what worked, what surprised you, what the numbers mean]

### 5.2 Future Work

[FILL IN: 3–4 genuine directions — e.g. use pre-trained GloVe/FastText embeddings in LSTM to close the gap with classical methods; experiment with BERTopic for topic modelling; use a larger LLM (Llama 3, Mistral) for RAG generation; extend to multilingual corpus]

---

## References

1. Greene, D. & Cunningham, P. (2006). *Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering*. Proc. 23rd International Conference on Machine Learning (ICML), pp. 377–384.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
3. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. ICLR.
4. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation*. Journal of Machine Learning Research, 3, 993–1022.
5. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
6. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.
7. Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models (Flan-T5)*. arXiv:2210.11416.
8. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-Scale Similarity Search with GPUs*. IEEE Transactions on Big Data.

---

*Word count target: 5,000 ± 500 words. Use appendix for any content that exceeds the limit.*
