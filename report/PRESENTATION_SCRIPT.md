# Presentation Narration Script — BBC News NLP Pipeline

A speakable script, split into 3 roughly equal parts so each team member can own one part end-to-end (swap sections between people freely — what matters is that whoever presents a part has actually read the matching section of `PROJECT_WALKTHROUGH.md`). Assumes ~12–15 minutes total plus Q&A; trim the italicised "optional detail" lines first if you need to shorten.

Where a screenshot/figure/live-demo is expected, it's marked **[SHOW: ...]**.

---

## Part A — Data, Preprocessing & Feature Engineering
*(Notebook 1 — presenter: ___________)*

**Opening / framing:**

> "Our project is an end-to-end NLP pipeline built on the BBC News dataset — 2,225 news articles from 2004 and 2005, labelled by BBC editors into five categories: business, entertainment, politics, sport, and tech. We chose it because it's multi-class rather than just binary sentiment, it's already cleanly labelled so we didn't need to do manual annotation, and it's rich enough to support every stage the assignment asks for — from topic modelling to classification to a retrieval-based chatbot."

**Data collection:**

> "The data comes from HuggingFace's `SetFit/bbc-news` dataset, which itself traces back to a well-known 2006 academic paper on document clustering. It arrives pre-split into 1,225 training and 1,000 test articles."

**[SHOW: class distribution / pie chart, `01_class_distribution.png`]**

> "The categories are reasonably balanced — sport and business are the largest, tech and entertainment slightly smaller, but no category dominates."

**Cleaning:**

> "Before anything else, we checked for missing values — there were none — but found 99 duplicate or near-duplicate articles, which we removed. That left 2,126 articles, which is the number everything downstream is actually built on."

**Preprocessing:**

> "We built a standard text-normalisation pipeline: lowercase everything, strip URLs and HTML tags, expand contractions so 'don't' and 'do not' aren't treated as different words, remove punctuation and stopwords, and then lemmatise — reducing words to their dictionary form. We chose lemmatisation over stemming deliberately: stemming can mangle words into non-words, which hurts readability of everything downstream, especially our topic model's keyword lists and our word-embedding neighbours."

*Optional detail: "We actually kept two versions of the text throughout the whole project — the original raw text, and this cleaned version. That turned out to matter later: our from-scratch models want the cleaned version, but our pretrained transformer model, DistilBERT, actually performs better on the original, uncleaned text, because it has its own subword tokenizer that already knows how to handle punctuation and capitalisation."*

**Feature engineering / embeddings:**

> "We built three feature representations from this cleaned text: a TF-IDF matrix for our classical machine learning models, a Word2Vec model trained from scratch on our own corpus, and averaged Word2Vec vectors to get one embedding per article."

**Nearest-neighbour demo (rubric requirement):**

> "As one deliverable, we built a function that takes any word and returns its most similar words by embedding distance. For example, querying 'olympic' returns words like 'medallist', 'athens', 'radcliffe' and 'medal' — all genuinely Olympic-related vocabulary the model learned purely from context, without ever being told what these words mean."

**[SHOW: t-SNE plot, `05_tsne_w2v.png`]**

> "To visualise the embedding space, we projected our 100-dimensional Word2Vec document vectors down to 2D using t-SNE, coloured by category. [Look at the actual plot before presenting and describe, in your own words, which categories cluster tightly and which overlap — this will differ slightly run to run, so speak to what you actually see on the slide.]"

---

## Part B — Topic Modelling, Traditional ML & Explainability
*(Notebook 2 — presenter: ___________)*

**Transition in:**

> "With our features built, we moved to the modelling stage, which has two halves: unsupervised topic discovery, and supervised classification."

**LDA topic modelling:**

> "For unsupervised learning, we used Latent Dirichlet Allocation — LDA — to discover topics in the corpus without using the category labels at all. To choose how many topics to look for, we trained models for every K from 2 to 12 and scored each using coherence — a metric for how interpretable the topics are — and found K equals 10 gave the best score."

**[SHOW: coherence plot, `07_lda_coherence.png`]**

> "Interestingly, the best number of topics was 10, not 5 — double our number of known categories. Looking at the actual topic keywords, this makes sense: LDA split several of our categories into finer sub-themes. For example, 'sport' split into an athletics/Olympics topic and a separate football/club topic; 'entertainment' split into music and film. That's not a failure — it's the unsupervised method finding real structure inside our categories that the human labels don't capture."

**[SHOW: pyLDAvis interactive visualisation, `outputs/figures/lda_vis.html`, if doing a live demo]**

**Traditional ML classification:**

> "For supervised classification, we trained three classical models on TF-IDF features — Logistic Regression, Naive Bayes, and a linear Support Vector Machine — using the dataset's original train/test split, and refitting TF-IDF on the training data only, to avoid any information leaking from the test set."

> "All three performed extremely well — between 96.5% and 97% accuracy. The Linear SVM was the best of the three, at 97.0% accuracy and 0.969 macro F1."

**[SHOW: confusion matrices, `09_confusion_matrices.png`]**

**Explainability (LIME):**

> "The assignment also asks us to explain our model's decisions, not just report accuracy. We used LIME — Local Interpretable Model-agnostic Explanations — which explains one prediction at a time by showing which words pushed the model toward its answer."

**[SHOW: LIME explanation, `11_lime_sport_correct.png` or the interactive HTML]**

> "Here's a correctly classified sport article — LIME highlights the specific words that drove the sport prediction. [Name 2-3 of the actual highlighted words from the figure.] We also looked at a misclassified example to understand where the model gets confused, and separately plotted the top words driving each category overall, across the whole model — not just one article — giving us both a local and a global view of what the model has learned."

---

## Part C — Deep Learning, RAG Chatbot & Live App
*(Notebook 3 — presenter: ___________)*

**Transition in:**

> "Finally, we moved from classical machine learning to deep learning, built a retrieval-based conversational AI system, and deployed everything as a live web app."

**BiLSTM:**

> "First, we trained a bidirectional LSTM completely from scratch — no pretrained embeddings — to see how a 'from scratch' deep learning model compares to classical machine learning. It reached 81.6% accuracy — clearly worse than our TF-IDF classifiers, and in fact beaten by Naive Bayes, the simplest model we tried. Hold onto that, because it turns out to be the most informative result we got."

**DistilBERT:**

> "Then we fine-tuned DistilBERT — a compressed, pretrained transformer — instead of training from scratch. This reached 97.2% accuracy and 0.971 macro F1, which puts it top of our table."

**[SHOW: model comparison chart, `17_model_comparison.png`]**

> "But I want to be careful about that headline, because the margin matters. DistilBERT beats our Linear SVM by 0.21 percentage points — on a test set of 932 articles, that's a difference of **two articles**. That's well within the range that shifts between random runs, so we don't claim the transformer is meaningfully better. The honest reading is that a 67-million-parameter transformer and a linear model over word counts are **tied**."

> "Now look at the whole table together, because this is our main finding. Our two *deep learning* models are at opposite ends — DistilBERT at 97.2%, the LSTM at 81.6%. That's a 15.7-point gap between two neural networks. Meanwhile DistilBERT and a bag-of-words SVM differ by 0.2. So 'is it deep learning?' explains almost nothing about these results. What explains them is **pre-training**. DistilBERT starts from representations learned on billions of words and only has to learn the mapping onto five labels. The LSTM starts from random noise and has to learn what words *mean* and how to classify, simultaneously, from 1,194 articles."

> "And the reason the transformer can't pull further ahead is that this task doesn't need what transformers are good at. Their advantage is understanding context and word sense — but BBC categories are so lexically distinct that the word 'shares' or the word 'scored' basically settles the answer on its own. With a ceiling around 97%, there's no headroom left for contextual understanding to prove its value."

**RAG system:**

> "For the second deep learning task, we built a Retrieval-Augmented Generation system — a chatbot that answers questions grounded in our article corpus, rather than making things up. It has two parts: a retriever, using sentence-transformer embeddings and a FAISS vector index to find the most relevant articles for a question, and a generator, using a pretrained model called Flan-T5, which reads those retrieved articles and writes an answer using only that context."

> "We evaluated it on ten manually written questions about topics from the corpus — 'Who is Roger Federer?', 'What is the Premier League?' — scoring the generated answers against expected answers with ROUGE-L. Our mean was 0.155, with a best of 0.31 and two questions scoring zero."

**[SHOW: ROUGE-L per-question chart, `18_rag_rouge_scores.png`]**

> "That number looks bad, so let me explain why we think the metric is measuring the wrong thing. ROUGE-L scores the longest common subsequence against a reference answer — it rewards *wording overlap*, not correctness. Our reference for 'What sport does Tiger Woods play?' was the full sentence 'Tiger Woods is a professional golfer.' But an instruction-tuned model asked a direct question answers tersely — it says 'golf'. That answer is completely correct, and it scores **exactly zero**, because 'golf' and 'golfer' are different tokens with no common subsequence. Both of our zero-scoring questions are short-factoid questions of exactly that shape."

> "We can check retrieval separately, and retrieval is sound. When we query 'Who won the football championship?', it returns two relevant football articles — and a review of the video game *Championship Manager*, tagged tech. That third hit isn't a bug; it's evidence the retriever is matching on **meaning rather than keywords**, because it understood 'championship' in a sense our query never disambiguated."

> "We do have two genuine weaknesses, and we'd rather name them than defend the number. Flan-T5-base is small, so its answers are terse. And more importantly, some of our own questions aren't answerable from this corpus — it's BBC news from 2004 and 2005, so asking who won the Formula One championship has no clean answer in any single document. That's a curation mistake on our side. What we should have done is measure retrieval and generation separately — Recall@k for the retriever — and score answers with something like BERTScore that credits correct paraphrases."

**Streamlit app + deployment:**

> "Finally, everything is wrapped in an interactive web app built with Streamlit, with two tabs: one where you paste any news text and get a live category prediction with confidence scores, and one where you can ask the RAG chatbot a question and see both its answer and the source articles it retrieved."

**[SHOW: live demo of the app, or screenshots if live demo risks failing — https://huggingface.co/spaces/Geoanto/bbc-news-assistant]**

> "It's deployed publicly on HuggingFace Spaces, so anyone can try it from a browser — no local setup required."

**Closing / conclusions:**

> "To summarise: we built a complete pipeline from raw text to a deployed, working chatbot — unsupervised topic discovery, three classical classifiers, two deep learning approaches, an explainability layer, and a retrieval-based question-answering system."

> "Our main takeaway is that **pre-training, not depth, is what separates these models** — our two neural networks sit at opposite ends of the results table, while a transformer and a bag-of-words SVM are tied. And that finding had a practical consequence: we deployed the SVM, not the transformer, because a 0.2-point difference that isn't even real doesn't justify a 250-megabyte model. The best model on the leaderboard isn't always the right model to ship."

> "For future work, the cleanest next experiment is to initialise that same LSTM from pretrained GloVe or FastText embeddings — same architecture, same capacity, only the initialisation changes. If we're right that pre-training drives the results, most of that 15-point gap should close. We'd also rebuild the RAG evaluation to measure retrieval and generation separately, and test our 'no headroom' explanation on a harder corpus — something like sarcasm or fine-grained sentiment, where context actually matters and a transformer should pull away."

---

## Delivery notes

- **Practice the handoffs.** The weakest moment in team presentations is usually the 2-second silence between speakers — agree on a one-line "and now X will cover..." handoff line for each transition.
- **Everyone should be able to answer questions outside their own section.** Read the whole `PROJECT_WALKTHROUGH.md`, not just your part — examiners often ask the quietest-looking person a question about someone else's slide.
- **All numbers in this script now match the fresh Notebook 3 run of 2026-07-16** (DistilBERT 97.21/0.971, BiLSTM 81.55/0.810, mean ROUGE-L 0.155). If the notebook is re-run again, re-check them — DistilBERT in particular wobbles by ~0.2 points between runs, which is itself the reason we describe it as tied with the SVM.
- **Expect the ROUGE-L question.** 0.155 looks like failure and someone will say so. The answer is in Part C and in the walkthrough: the metric scores a correct one-word answer at zero. Whoever presents Part C should be ready to give the Tiger Woods example without notes.
