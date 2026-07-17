# NLP Report — Verified Facts Checklist (for merging the partner's 7,500-word version)

**How to use this:** the partner's longer version becomes the base document. Do NOT merge prose.
Instead, check the partner's doc against each fact below and correct any that are wrong or vague —
these are all verified against the actual data/notebook outputs (2026-07-16). Then run the ROUGE cell.

## 1. Dataset counts (a common place to get it wrong)
- Dataset AS DOWNLOADED: **2,225 articles** (train 1,225 / test 1,000). ✅ correct to say 2,225 in intro/dataset sections.
- After deduplication: **2,126 unique articles**. (99 rows removed from 2,225.)
- Category counts (original): sport 511, business 510, politics 417, tech 401, entertainment 386.

## 2. The deduplication story (VERIFIED — partner's version likely wrong/vague here)
- `df.duplicated().sum()` (all columns) = **57 exact duplicates**.
- Removal used `drop_duplicates(subset='text')` → removed **99 rows total**.
- The 99 = 57 exact duplicates + **42 articles present in BOTH train and test split** (identical text, different split label).
- All 42 extra rows are cross-split (verified: 42/42). NOT "differing metadata" — specifically train↔test leakage.
- Result: **2,126 unique articles.**
- The point: text-level dedup prevents train/test leakage that would inflate test accuracy.

## 3. Feature-matrix dimensions (built on the 2,126 DEDUPLICATED corpus, NOT 2,225)
- TF-IDF matrix: **2,126 × 10,000** ✅ (verified from tfidf_matrix.pkl)
- Word2Vec doc vectors: **2,126 × 100** ✅ (verified from doc_vectors_w2v.npy)
- RAG / FAISS index: **2,126 articles / ntotal=2,126** ✅ (verified from faiss_index.index)
- ⚠️ If the partner's version says 2,225 for any of these three, it's WRONG → fix to 2,126.

## 4. LDA (VERIFIED from NB2 output)
- Best K = **10**, coherence C_v = **0.4612**.
- 10 topics recovered (see report §4.1.2 table). Sport is the most fragmented category.

## 5. Word2Vec nearest neighbours (VERIFIED from NB1 output)
- football → league 0.685, ibrox 0.681, club 0.677, anfield 0.660
- champion → henin 0.793, wimbledon 0.787, compatriot 0.788
- injury → knee 0.784, hamstring 0.779, groin 0.730, surgery 0.720

## 6. LIME sport-article explanation (VERIFIED from lime_explanation_sport.html)
- Toward sport: match +0.123, rugby +0.117, team +0.032, olympic +0.023, player +0.020, lion +0.020, game +0.019, coach +0.017, squad +0.013
- Against: charity, said, july, new (generic, non-topical)

## 7. Model results (confirmed)
- Linear SVM: 97.00% acc / macro F1 0.9693
- Logistic Regression: 96.78% / 0.9669
- Naive Bayes: 96.46% / 0.9628
- BiLSTM: 80.26% / 0.793
- DistilBERT: 97.53% / 0.974
- SVM wins because classes are near-linearly-separable in sparse TF-IDF space (max-margin generalises best).

## 8. STILL MISSING — ROUGE-L (run NB3's RAG-eval cell, ideally in Colab)
- Three cells need filling: mean / best / worst ROUGE-L. Currently marked [run NB3].
- Expected range 0.10–0.35 (Flan-T5-base + paraphrase penalty). Interpretation prose already written.

## 9. Other confirmed facts
- HF Spaces URL: https://huggingface.co/spaces/Geoanto/bbc-news-assistant
- DistilBERT uses ORIGINAL text; LSTM uses lemmatised text (design decision — defend it).
- RAG: FAISS IndexFlatIP + L2-normalised = exact cosine; Flan-T5-base generator on CPU.

---
**Merge procedure when partner's doc arrives:**
1. Save the partner's version as the new base (it's the longer/authoritative one).
2. Read through it once against sections 1–9 above; correct any wrong/vague fact (targeted edits, both .md and .docx).
3. Run the ROUGE cell, fill section 8.
4. Add the two app screenshots (Tab 1 classifier, Tab 2 RAG).
5. Final word-count + read-aloud pass.
