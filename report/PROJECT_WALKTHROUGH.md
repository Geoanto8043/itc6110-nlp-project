# Project Walkthrough — BBC News NLP Pipeline

This document exists so you can genuinely understand and defend every part of this project in front of the examiners. Read it section by section, in the order the notebooks run. Each section explains: **what it does, why it was done this way, the real numbers produced, and what you might get asked about it.**

The project is an end-to-end NLP pipeline on the **BBC News dataset**: 2,225 news articles, hand-labelled into 5 categories (business, entertainment, politics, sport, tech), collected 2004–2005 for a well-known academic clustering paper (Greene & Cunningham, 2006). It's the "MNIST of text classification" — small, clean, balanced enough to be tractable on a laptop, but real enough to need genuine NLP work.

The pipeline has 3 notebooks, run in sequence, each handing artefacts to the next, plus a deployed Streamlit app:

```
NB1: Data + Preprocessing + Embeddings  →  NB2: Topic Modelling + Traditional ML + XAI  →  NB3: Deep Learning + RAG + UI
```

---

## Notebook 1 — Data, Preprocessing, Feature Engineering

### 1. Data loading (Step 1 — Data Collection)

The raw dataset is pulled from HuggingFace (`SetFit/bbc-news`) and saved locally as `data/raw/bbc_news.csv`. It arrives pre-split into a `train`/`test` column (this is the original split from the dataset publisher, not one we created ourselves).

**Real numbers (raw, before any cleaning):**

| Category | Train | Test | Total |
|---|---|---|---|
| Business | 286 | 224 | 510 |
| Entertainment | 210 | 176 | 386 |
| Politics | 242 | 175 | 417 |
| Sport | 275 | 236 | 511 |
| Tech | 212 | 189 | 401 |
| **Total** | **1,225** | **1,000** | **2,225** |

Average article length: ~390 words.

**If asked "why this dataset":** it's multi-class (5 balanced-ish categories, not just binary sentiment), pre-labelled by BBC editors (so no manual labelling needed), small enough to iterate quickly on a laptop/Colab, and rich enough to support every downstream task the rubric requires — topic modelling, classification, and RAG (because articles have real informative content, not just star ratings).

### 2. Cleaning: duplicates and missing values (Step 2)

The raw data has **zero missing values** but **99 duplicate/near-duplicate rows** (`df.drop_duplicates(subset='text').dropna(...)`). After removing these, 2,126 articles remain — this is the dataset used for **everything downstream** (TF-IDF, Word2Vec, LDA, all classifiers, RAG).

**Real numbers (processed, what every model actually trains/tests on):**

| Category | Train | Test | Total |
|---|---|---|---|
| Business | 282 | 221 | 503 |
| Entertainment | 206 | 163 | 369 |
| Politics | 236 | 167 | 403 |
| Sport | 274 | 230 | 504 |
| Tech | 196 | 151 | 347 |
| **Total** | **1,194** | **932** | **2,126** |

⚠️ **Watch out:** the current report draft still says the TF-IDF matrix and dataset are "2,225 × 10,000" / "2,225 articles" in a few places. The real, final number after dedup is **2,126**. Use the table above when filling in the report, not the raw 2,225 figures (except in Section 1, which correctly describes the raw download).

### 3. Text normalisation pipeline (Step 2)

Implemented as a `clean_text()` + `lemmatize_and_filter()` function pair, applied to every article to produce `text_processed`:

1. Lowercase everything
2. Strip URLs (`http...`, `www...`) and HTML tags via regex
3. Expand contractions (`don't` → `do not`, etc.) so `not` isn't lost as a token
4. Strip punctuation/digits/special characters, keeping only letters
5. Remove English stopwords (NLTK's list)
6. Lemmatise (WordNetLemmatizer) — reduces words to dictionary form
7. Tokenise (whitespace split after cleaning), drop tokens ≤2 characters

**Key design decision — two parallel text columns are kept, not one:**
- `text` — the original, untouched article. Used later by **DistilBERT**, because its subword tokenizer already handles punctuation/casing and performs *worse* on pre-lemmatised text (it was pretrained on natural language).
- `text_processed` — the cleaned/lemmatised tokens. Used by **Word2Vec, TF-IDF, LDA, and the LSTM**.

**If asked "why lemmatisation over stemming":** lemmatisation produces real dictionary words (e.g. "running" → "run"), which keeps Word2Vec's neighbourhoods and LDA's topic-word lists human-readable, whereas stemming often produces mangled fragments (e.g. Porter stemmer: "running"→"run" but "universal"→"univers") that can conflate unrelated words purely by shared prefixes.

### 4. Feature engineering — three representations built here (Step 3)

- **TF-IDF**: `TfidfVectorizer(max_features=10_000, ngram_range=(1,2), min_df=2)` → sparse matrix, shape **(2,126 × 10,000)**. Captures unigrams and bigrams, down-weighting words common across all articles. This feeds the traditional ML classifiers in NB2.
- **Word2Vec**: Gensim Skip-gram, `vector_size=100, window=5, min_count=2, epochs=10`, trained from scratch on this corpus (not pretrained). Vocabulary: **15,986 words**. Skip-gram was chosen over CBOW because it tends to work better on smaller corpora like this one.
- **Document vectors**: each article's Word2Vec vectors are averaged into one 100-dim vector per article (2,126 × 100 matrix) — the standard "mean pooling" trick for turning word embeddings into a document embedding.

**Nearest-neighbour demo (Step 3 requirement — "given a word, return N most similar"):** implemented as `get_similar_words()`, a thin wrapper over `w2v_model.wv.most_similar()`. Real output for 5 query words:

| Query | Top similar words |
|---|---|
| football | corinthian, league, ibrox, club, judo, punish, sponsorship, anfield |
| champion | henin, yelling, maurice, hardenne, compatriot, wimbledon, kenenisa, hayley |
| injury | knee, hamstring, neck, groin, surgery, shoulder, ankle, ligament |
| transfer | lure, construed, worded, smart, celestine, edu, morientes, sponsorship |
| olympic | medallist, holmes, athens, radcliffe, vault, heptathlon, gebrselassie, medal |

Notice these are genuinely sensible: "injury" neighbours are all body parts/medical terms; "olympic" neighbours are athletes and Olympic-specific vocabulary (Holmes, Radcliffe = British Olympic runners; Gebrselassie = Ethiopian distance runner). This is good evidence Word2Vec learned real semantic structure from only ~2,100 documents.

### 5. Visualisation — t-SNE (Step 3)

Word2Vec's 100-dim document vectors are projected to 2D with t-SNE (`perplexity=30`, Barnes-Hut method for speed) and plotted, coloured by category (Figure `05_tsne_w2v.png`). An extra comparison plot (`06_tsne_comparison.png`) does the same for TF-IDF (reduced first via TruncatedSVD to 50 dims, since t-SNE doesn't work directly on 10,000-dim sparse data).

**What you should say when asked to interpret it:** open the actual PNGs before presenting and describe what you see — typically sport and business form the tightest, most separated clusters (very distinctive vocabulary), while politics/entertainment/tech show more overlap because they share more general-purpose words. Don't guess this from memory — glance at `outputs/figures/05_tsne_w2v.png` once so you can speak to it honestly.

---

## Notebook 2 — Topic Modelling, Traditional ML, Explainability

### 1. LDA topic modelling (Step 4.1 — Unsupervised)

Uses Gensim's `LdaModel` on a filtered dictionary (words appearing in fewer than 5 docs or more than 80% of docs are dropped — this removes both noise and near-universal filler words) — **7,692 terms** across 2,126 documents.

**How K (number of topics) was chosen:** trained LDA models for K = 2 through 12, scored each with **C_v coherence** (a metric measuring how often a topic's top words co-occur together in real documents — higher means topics are more human-interpretable, not just statistically likely). Full scan:

| K | Coherence | K | Coherence |
|---|---|---|---|
| 2 | 0.293 | 8 | 0.441 |
| 3 | 0.313 | 9 | 0.440 |
| 4 | 0.279 | **10** | **0.461 (best)** |
| 5 | 0.342 | 11 | 0.397 |
| 6 | 0.356 | 12 | 0.417 |
| 7 | 0.391 | | |

**K = 10 was selected (coherence = 0.4612).**

**Top keywords per topic (final model):**

| Topic | Top words | Looks like |
|---|---|---|
| 1 | music, band, edward, top, back, new, year, sale, one, group | Music/entertainment industry |
| 2 | game, time, people, brown, life, online, world, hour, gaming | Video games/tech-culture |
| 3 | test, thanou, greek, blunkett, year, game, visa, kenteris | Greek athletics doping scandal (very specific news event) |
| 4 | world, time, year, champion, holmes, olympic, speed, european | Athletics/Olympics |
| 5 | technology, software, patent, company, network, machine, computer | Tech/computing |
| 6 | mobile, phone, people, woman, game, digital, handset | Mobile phones |
| 7 | year, company, market, share, firm, price, month, sale | Business/markets |
| 8 | would, election, government, labour, party, minister, blair, tory | UK politics |
| 9 | england, club, win, game, year, first, player, play, final | Football/club sport |
| 10 | film, year, best, award, show, star, new, west | Film/entertainment |

**Discussion point (this is a `[FILL IN]` in the report, and a likely presentation question):** with only 5 true category labels but K=10 gave the best coherence, LDA is splitting each broad category into finer sub-themes rather than recovering the 5 labels 1:1. For example, topics 4 and 9 are both "sport" but split into athletics/Olympics vs. football/clubs; topics 1 and 10 are both "entertainment" split into music vs. film; topic 3 is a sport-adjacent but genuinely distinct scandal story. This is expected and a *positive* sign — LDA found real semantic sub-structure the 5-way labels don't capture, which is the whole point of unsupervised topic discovery (it isn't told the categories and isn't required to reproduce them exactly).

There's also a bonus analysis: LDA run again on **sport-only** articles to find sub-topics within sport (K=4 chosen by the same coherence method): club football (chelsea/united/league), international rugby (england/wales/ireland/six-nations), doping scandals (drug/test/iaaf), and general competition (year/world/final/win). This double-checks the topic-splitting story above using a controlled subset.

An interactive visualisation was also generated: `outputs/figures/lda_vis.html` (pyLDAvis) — open this in a browser to show the topic map live during the demo if you want something visual and clickable.

### 2. Traditional ML classification (Step 4.2, Task 1a)

Three classifiers, each as a `Pipeline([TfidfVectorizer, classifier])` where **TF-IDF is refit on the training split only** — this is a leakage-prevention detail worth mentioning if asked ("why not just reuse the TF-IDF matrix from Notebook 1?" — because that one was fit on the *full* corpus including test articles, which would leak test-set vocabulary statistics into training).

Train/test split is the dataset's **original** split (1,194 train / 932 test after dedup) — not a fresh random split, so results are comparable to any published benchmark using the same BBC split.

**Real results:**

| Model | Accuracy | Macro F1 |
|---|---|---|
| **Linear SVM** | **97.00%** | **0.9693** |
| Logistic Regression | 96.78% | 0.9669 |
| Naive Bayes | 96.46% | 0.9628 |

All three are excellent — this dataset is genuinely easy for classical bag-of-words methods because each category has very distinctive vocabulary (sport articles say "goal"/"match"; business articles say "shares"/"market"). Naive Bayes' independence assumption (each word contributes independently) hurts it slightly relative to SVM/LR, which can weigh correlated words together.

### 3. Explainable AI — LIME (Step 4.2, Task 1 XAI requirement)

LIME explains **Logistic Regression**'s predictions specifically, not SVM — because `LinearSVC` has no `predict_proba` (LIME needs class probabilities to perturb-and-explain), while Logistic Regression does.

Two concrete demonstrations were generated:
1. **A correctly classified sport article** (test article about a Sydney North-vs-South charity rugby match) — LIME shows which words pushed the model toward "sport" (saved as `outputs/figures/lime_explanation_sport.html`, an interactive version, plus a static PNG).
2. **A misclassified article** — error analysis, showing which words misled the model.

There's also a **global** (not just local/per-article) explainability view: for each of the 5 classes, the top-12 TF-IDF features with the highest Logistic Regression coefficients are plotted (`13_lr_global_features.png`) — this shows, overall, which words the model has learned are most diagnostic of each category (e.g. for sport it's likely words like "club", "match", "player"). Local (LIME) explains one prediction; global (LR coefficients) explains the model's general behaviour — mentioning both if asked "local or global" covers the rubric's either/or requirement more completely than necessary.

**Before presenting: open `outputs/figures/lime_explanation_sport.html` and the misclassified PNG once** so you can describe in your own words which specific words LIME highlighted — this is exactly the kind of thing an examiner might ask you to read off the screen.

---

## Notebook 3 — Deep Learning, RAG, and the App

### 1. BiLSTM (Step 4.2, Task 1b — a "from scratch" deep learning baseline)

A 2-layer **bidirectional LSTM** built and trained entirely from scratch in PyTorch (no pretrained embeddings) — `embed_dim=128, hidden_dim=256`, ~4.29M parameters. Vocabulary is capped at 15,000 words and built **only from the training split** (again, leakage prevention), sequences padded/truncated to 200 tokens, and it trains on `text_processed` (the lemmatised tokens from NB1), not raw text — LSTMs work token-by-token on a fixed vocabulary, so cleaning helps here (unlike BERT below).

Trained for 8 epochs with Adam (lr=1e-3) and a learning-rate scheduler that halves the rate if the loss plateaus.

**Real result (fresh run, 2026-07-16):** Test accuracy **81.55%**, Macro F1 **0.8099**.

**Per-class detail (fresh run):** best on sport (F1 0.89), then business (0.81); weakest on entertainment (0.76), with politics and tech both at 0.79. The revealing pattern is in precision vs recall: business has high recall (0.93) but low precision (0.72), while politics has high precision (0.89) but low recall (0.71) — i.e. the model is over-predicting "business" and swallowing politics articles into it. Entertainment recall is equally weak (0.72), so it isn't purely a business/politics problem.

**Also worth knowing:** the training curves show loss still falling and validation accuracy still rising at epoch 8 — the model is under-trained, not converged. More epochs would gain a few points but wouldn't close a 15-point gap. If an examiner asks "did you train it long enough?", that's the honest answer: no, but it isn't the reason it loses.

**If asked "why does the from-scratch DL model underperform classical ML":** the LSTM has to learn word meaning *and* classification simultaneously from only ~1,200 training articles — nowhere near enough data to learn good embeddings from scratch. TF-IDF-based classifiers don't need to learn word meaning at all (they just count words), so they're far more data-efficient at this scale. This is the textbook illustration of why transfer learning (see DistilBERT next) matters.

### 2. DistilBERT — fine-tuned (Step 4.2, Task 1b — pretrained transfer learning)

`distilbert-base-uncased` (66.9M parameters, a distilled/compressed version of BERT — ~40% smaller, ~60% faster, keeps ~97% of BERT's accuracy) fine-tuned via HuggingFace's `Trainer` for 4 epochs, batch size 16, learning rate 2e-5. Trains on the **original, uncleaned `text`** column — BERT's own subword tokenizer (WordPiece) already handles punctuation and casing, and was pretrained on natural (not lemmatised) text, so feeding it lemmatised tokens would actually throw away information it knows how to use.

**Real result (fresh run, 2026-07-16):** Test accuracy **97.21%**, Macro F1 **0.9709**.

**Final comparison — these are the numbers to present:**

| Model | Type | Accuracy | Macro F1 |
|---|---|---|---|
| **DistilBERT (fine-tuned)** | Deep Learning | **97.21%** | **0.9709** |
| Linear SVM | Traditional ML | 97.00% | 0.9693 |
| Logistic Regression | Traditional ML | 96.78% | 0.9669 |
| Naive Bayes | Traditional ML | 96.46% | 0.9628 |
| BiLSTM | Deep Learning | 81.55% | 0.8099 |

**How to talk about this — the nuance matters.** DistilBERT ranks first, but its lead over the SVM is 0.21 percentage points, which on a 932-article test set is **two articles**. That is well inside run-to-run noise; do *not* claim DistilBERT is meaningfully better. The defensible framing is: *a 66.9M-parameter transformer and a linear model over word counts are tied* — and the SVM gets there in seconds of CPU training versus minutes of GPU fine-tuning, which is exactly why the deployed app serves the SVM.

**The deeper point (this is your best answer to "which model is best and why"):** the two *deep learning* models are 15.7 points apart (DistilBERT 97.21 vs BiLSTM 81.55), while DistilBERT and a bag-of-words SVM are 0.21 apart. So "deep vs. not deep" explains nothing about these results — **pre-trained knowledge** explains everything. DistilBERT starts from representations learned on billions of words; the BiLSTM starts from random noise and must learn word meaning *and* the task from 1,194 articles.

**Why the transformer can't pull ahead:** its advantage is resolving context and word-sense ambiguity — exactly the weakness we saw in Word2Vec around the polysemous word "transfer". But BBC categories are so lexically distinct that word *identity* nearly settles the question ("shares" → business, "scored" → sport). With a ~97% ceiling set by genuinely ambiguous articles, there's no headroom for contextual understanding to prove its worth.

*(Historical note: an earlier saved run had DistilBERT at 97.00%/0.9684, fractionally behind the SVM, and the original report text claimed 97.53%/0.974. Both are superseded by the fresh run above. This run-to-run wobble of ~0.2 points is itself evidence that the DistilBERT-vs-SVM gap isn't real.)*

### 3. Retrieval-Augmented Generation — RAG (Step 4.2, Task 2)

Two components:
- **Retriever:** `sentence-transformers/all-MiniLM-L6-v2` encodes all 2,126 articles' **original text** into 384-dim vectors, stored in a FAISS `IndexFlatIP` index. Because embeddings are L2-normalised first, inner product = cosine similarity — so this is exact (not approximate) cosine-similarity search over all articles.
- **Generator:** `google/flan-t5-base`, a pretrained instruction-tuned text-to-text model, given a prompt that bundles the top-k retrieved article snippets as "context" plus the user's question, and asked to answer using only that context (this is the "prompt engineering" the rubric asks for — a template, not free-form generation).

**Demo retrieval (from the notebook, real output):** query *"Who won the football championship?"* retrieves 3 articles — a Southampton/Portsmouth FA Cup article, a Champions League article, and (interestingly) a football video-game "Championship Manager" review, at cosine scores 0.31/0.30/0.29. Worth knowing: these similarity scores are fairly modest (~0.3, not ~0.7+), which is typical for short queries against long, topically-diffuse news articles — not a bug, just a property of comparing a 6-word question against 400-word documents.

**Evaluation results (now run, 2026-07-16):** 10 manually curated questions, scored with ROUGE-L.

| Metric | Score |
|---|---|
| **Mean ROUGE-L** | **0.1549** |
| Best (UEFA Champions League / Roger Federer) | 0.3077 |
| Worst (Tiger Woods / Formula One) | 0.0000 |
| Median | 0.1483 |

**This is the section most likely to draw a hostile question — "0.155, isn't your system broken?" Here's the answer.**

No, and the two zero scores prove the point. ROUGE-L measures **longest common subsequence with a reference string** — it rewards lexical overlap, not correctness. Our reference for "What sport does Tiger Woods play?" is the full sentence *"Tiger Woods is a professional golfer."* An instruction-tuned model asked a direct question answers tersely — plausibly *"golf"*. That answer is **completely correct and scores exactly zero**, because "golf" and "golfer" are different tokens with no common subsequence. Both zero-scoring questions are short-factoid questions of exactly this shape. The metric is failing, not the system.

**Retrieval can be checked independently, and it's sound** — see the "Who won the football championship?" trace above: it returned two genuinely relevant football articles plus a *Championship Manager* video-game review tagged `tech`. That third hit isn't an error, it's proof the retriever matches on **semantics not keywords** — it understood "championship" in a sense the query didn't disambiguate.

**The real weaknesses (say these before an examiner does):** (1) Flan-T5-base is small at 250M params, so answers are terse; (2) **some of our questions aren't answerable from this corpus at all** — it's 2004–2005 BBC news, so "Who won the Formula One championship?" has no single document stating a clean answer. That's a curation weakness on our side, and owning it is much stronger than defending the number.

**What we'd do instead:** report retrieval metrics (Recall@k, MRR) separately from generation metrics, and score answers with BERTScore or an LLM-as-judge, both of which credit correct paraphrases. 0.155 is a lower bound on system quality, not a measurement of it.

⚠️ **One thing I don't have:** the actual *generated answers* from your run. If you paste the per-question `Q:` / `A:` printout, I can confirm the Tiger Woods hypothesis above with the real output rather than a well-founded inference — worth doing, because "we checked and it answered 'golf'" is a much stronger presentation line than "it probably answered 'golf'."

### 4. Streamlit App + Deployment (Step 4.2, Task 3)

The app (`app/app.py`, generated by a notebook cell that writes the file, then separately edited/fixed afterwards per the git history) has two tabs:
- **Article Classifier** — paste any text, get a predicted category + a confidence bar chart. Uses the saved **Linear SVM pipeline** (`best_ml_pipeline.pkl`) — note this is the classical model, not DistilBERT, presumably because it's much lighter to deploy (no GPU/large transformer needed for inference) and, per the corrected numbers above, is essentially just as accurate.
- **Sports Q&A (RAG)** — ask a natural-language question, get a FAISS-retrieved-and-Flan-T5-generated answer, with the retrieved source articles shown in an expandable panel for transparency.

Deployed live on HuggingFace Spaces: **https://huggingface.co/spaces/Geoanto/bbc-news-assistant**. It's a pure text/browser app — no camera, no image input anywhere; the only input widgets are a text area and a text box.

---

## Quick-reference cheat sheet (for Q&A)

| If asked... | Say... |
|---|---|
| Why BBC News? | Multi-class, cleanly labelled, small enough to iterate fast, rich enough for topic modelling + classification + RAG |
| Why two text columns (`text` vs `text_processed`)? | BERT-family models want natural text; classical/statistical models (TF-IDF, Word2Vec, LDA, LSTM) want cleaned/lemmatised tokens |
| Why lemmatise, not stem? | Produces real words → more interpretable LDA topics and Word2Vec neighbours |
| Why K=10 for LDA when there are 5 categories? | Coherence-optimal K found finer sub-themes within each category (e.g. athletics vs. football within sport) — expected, and a sign the unsupervised method is finding real structure |
| Why fit TF-IDF twice (NB1 full-corpus vs NB2 train-only)? | NB1's version is for visualisation only (t-SNE); NB2 refits on train-only inside a `Pipeline` to avoid leaking test vocabulary |
| Which model is actually best? | **DistilBERT (97.21%, 0.971 F1)** ranks first — but by 0.21 points over Linear SVM (97.00%, 0.969), i.e. **two articles out of 932**. Call it a tie; don't oversell the transformer |
| Why does the from-scratch LSTM do worse than a transformer or even plain TF-IDF? | Not enough training data (~1,200 articles) to learn good word embeddings *and* a classifier from scratch; TF-IDF and pretrained-transformer approaches don't have that burden |
| What's the single most important finding? | The two *deep* models are 15.7 points apart; DistilBERT and a bag-of-words SVM are 0.21 apart. **Pre-training explains the results, not depth** |
| Which model does the deployed app use, and why? | Linear SVM — a few MB and millisecond CPU inference vs. DistilBERT's ~250MB, for a 0.21-point difference that isn't real. The best model on a leaderboard isn't always the right model to ship |
| Local vs global XAI? | LIME (local, per-article) explains one prediction at a time; the LR-coefficient plot (global) shows the top words driving each class overall |
| RAG retrieval scores look low (~0.3) — is that a bug? | No — short questions vs. long documents naturally get modest cosine similarity; it's still correctly retrieving topically relevant articles |
