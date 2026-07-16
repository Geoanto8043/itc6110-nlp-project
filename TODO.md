# ITC6110 Project — Remaining Tasks

## ⚠️ Known landmine — read before re-running Notebook 3

**The `app_code` cell near the end of NB3 overwrites `app/app.py`, and what it writes is broken.**
Re-running NB3 on 2026-07-16 silently reverted three committed fixes and introduced a syntax error:

- `app_code` is a plain (non-raw) `'''...'''` string, so its `\n\n` escapes expand into real newlines
  when written to disk → **unterminated string literal at line 89, the file will not parse**
- Reverts commit `c6ab758` — puts back `clf_pipeline.predict_proba(...)`, which `LinearSVC` does not have
  (crashes the classifier tab)
- Reverts commit `4415cf7` — restores the full CUDA `torch` wheel and drops `scikit-learn` from
  `app/requirements.txt` (2.5 GB image; SVM pipeline cannot be unpickled)

`app/app.py` and `app/requirements.txt` were restored from git on 2026-07-16.
**If you re-run NB3, do not commit those two files** — check `git diff app/` first.

- [ ] Fix the `app_code` cell properly (make it a raw string and fold in the three fixes), or delete the
      cell so `app/app.py` is maintained by hand as the source of truth

## ⚠️ Notebook outputs are stale

`notebooks/03_dl_rag_ui.ipynb` was **not saved** after the 2026-07-16 run. Its stored outputs still show
the old numbers (BiLSTM 81.12% / DistilBERT 97.00%), which **contradict the report** (81.55% / 97.21%).

- [ ] Re-open NB3, re-run, and **save it** so the stored outputs match the report — if the notebooks are
      submitted as evidence alongside the report, the numbers must agree

---

## Report (`report/ITC6110_report.md`)

Fully rewritten on 2026-07-16 against the fresh NB3 run, then restructured with appendices to control
length. All metrics, tables, figure links and discussion sections are filled in and verified.

- [ ] Team member names + submission date (title page — 4 placeholders)
- [ ] Screenshot of Streamlit app — Tab 1 (Classifier) working
- [ ] Screenshot of Streamlit app — Tab 2 (RAG Q&A) working
- [ ] *(Optional)* Paste the RAG run's generated answers — would let us confirm, rather than infer, that
      "What sport does Tiger Woods play?" scored 0.0 because the model correctly answered "golf" against
      a reference of "…is a professional golfer". Verified beats inferred if an examiner pushes on 0.155.
- [ ] *(Optional)* Trim ~190 more words — main body is 5,687 vs. a 5,500 ceiling (3.4% over; the 1,383-word
      appendices hold supporting tables and extended analysis and are normally excluded from the count)

### Done
- [x] All model metrics filled in from the 2026-07-16 NB3 re-run
- [x] Per-category counts (raw + post-deduplication), missing/duplicate counts
- [x] Word2Vec nearest-neighbour table; t-SNE interpretation (written from the actual figure)
- [x] LDA K selection (K=10, C_v=0.4612), full coherence scan, top-10 keywords per topic, discussion
- [x] LIME description (written from the actual figure: `match` +0.123, `rugby` +0.117)
- [x] ROUGE-L results + discussion of why the metric understates the system
- [x] Conclusions and future work
- [x] Fixed 3 broken figure links (`03_tsne_w2v` → `05_tsne_w2v`, `08_lda_topics` → `08_topic_per_category`,
      `12_lime_sport_explanation` → `11_lime_sport_correct`); all 18 links verified to resolve

---

## Supporting docs
- `report/PROJECT_WALKTHROUGH.md` — section-by-section explanation of the pipeline + Q&A cheat sheet
- `report/PRESENTATION_SCRIPT.md` — speakable narration split into 3 parts, one per team member

---

## Authoritative results (2026-07-16 re-run)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| DistilBERT (fine-tuned) | 97.21% | 0.9709 |
| Linear SVM | 97.00% | 0.9693 |
| Logistic Regression | 96.78% | 0.9669 |
| Naive Bayes | 96.46% | 0.9628 |
| BiLSTM (from scratch) | 81.55% | 0.8099 |

RAG mean ROUGE-L: 0.1549 · Corpus: 2,126 after dedup (1,194 train / 932 test)

> Earlier figures (DistilBERT 97.53%/0.974, BiLSTM 80.26%/0.793) were stale — do not reuse.
> The DistilBERT/SVM gap is 0.21 points = 2 articles out of 932, and it flipped sign between runs.
> Do not present it as "the transformer won."

---

## Presentation
- [ ] Build slides (structure mirrors `PRESENTATION_SCRIPT.md`)
- [ ] Assign the 3 parts to team members
- [ ] Rehearse handoffs between speakers
- [ ] Prepare for the "your ROUGE-L is 0.155, is it broken?" question — answer is in Part C
- [ ] Include live demo of Streamlit app (have screenshots as fallback)

---

## Submission checklist
- [x] Code on GitHub: https://github.com/Geoanto8043/itc6110-nlp-project
- [x] App deployed: https://huggingface.co/spaces/Geoanto/bbc-news-assistant
- [ ] Verify the deployed Space still works (NB3 re-run reverted its fixes locally — confirm the live
      Space was not redeployed from the broken files)
- [ ] Report submitted via Turnitin
- [ ] Oral presentation prepared
