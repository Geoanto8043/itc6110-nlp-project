# ITC6110 Project — Remaining Tasks

## Report (`report/ITC6110_report.md`)

### Needs team input (discussion/interpretation):
- [ ] Team member names + submission date (title page)
- [ ] 2–3 sentences: why BBC News was chosen (Section 1.2)
- [ ] 2–3 sentences: t-SNE scatter plot interpretation (Section 3.3)
- [ ] 2–3 sentences: LIME explanation for sport article (Section 4.2.1 XAI)
- [ ] Discussion: how LDA topics map to the 5 BBC categories (Section 4.1.4)
- [ ] 1 sentence: why SVM outperforms LR and NB (Section 4.2.1)
- [ ] 2–3 sentences: ROUGE-L results interpretation (Section 4.2.2)
- [ ] 3–4 bullet points: key conclusions (Section 5.1)
- [ ] 3–4 bullet points: future work directions (Section 5.2)

### Needs copy-paste from notebook outputs:
- [ ] Per-category article counts table (run `df.groupby(['category','split']).size()` in NB1)
- [ ] Missing/duplicate value counts from NB1
- [ ] Word2Vec top-5 nearest neighbours for a sample word (from NB1 output)
- [ ] LDA: which K was chosen + C_v coherence score (from NB2 output)
- [ ] LDA top-10 keywords per topic table (from NB2 output)
- [ ] Traditional ML Macro F1 scores for SVM/LR/NB (check `data/processed/ml_results_summary.csv`)
- [ ] Mean / best / worst ROUGE-L scores (from NB3 output)

### Needs screenshots:
- [ ] Screenshot of Streamlit app — Tab 1 (Classifier) working
- [ ] Screenshot of Streamlit app — Tab 2 (RAG Q&A) working

### Already confirmed and written in:
- [x] HuggingFace Spaces URL: https://huggingface.co/spaces/Geoanto/bbc-news-assistant
- [x] BiLSTM: acc=80.26%, macro_f1=0.793
- [x] DistilBERT: acc=97.53%, macro_f1=0.974
- [x] Linear SVM: acc=97.00%
- [x] Logistic Regression: acc=96.78%
- [x] Naive Bayes: acc=96.46%
- [x] All figures 01–18 referenced

---

## Presentation (not started)
- [ ] Draft slide outline (structure mirrors report sections)
- [ ] Each of 3 team members presents a section
- [ ] Include live demo of Streamlit app

---

## Submission checklist
- [x] Code on GitHub: https://github.com/Geoanto8043/itc6110-nlp-project
- [x] App deployed: https://huggingface.co/spaces/Geoanto/bbc-news-assistant
- [ ] Report submitted via Turnitin (~5,000 words)
- [ ] Oral presentation prepared
