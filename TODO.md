# ITC6110 Project — Remaining Tasks

## 🚨 FULL-PROJECT CONSISTENCY AUDIT (2026-07-22)

Scope: report ↔ code, report internal (citations/figures/cross-refs), report ↔ deck, and a sweep
for claims that could be exposed under questioning.

### Must fix before submitting / presenting

- [ ] **Deck slide 12 states a falsehood.** *"Both of our zero-scoring questions are short-factoid
      questions of exactly that shape."* Only ONE zero is a metric artefact ("golf", correct). The
      other — Formula One → **"ivanovic"** — is simply a wrong answer. If an examiner asks "what was
      the other zero?", the current slide leads you into a trap. `PRESENTATION_SCRIPT.md` Part C has
      been rewritten with the correct, stronger framing; update the slide to match.
- [ ] **`.docx` claims five-fold cross-validation that the code never performs** (see next section).
- [ ] **`.docx` RAG section repeats the same "both zeros" overclaim** plus *"retrieval succeeds on
      every question"*. Port the corrected §4.2.2 / §5.1 from the `.md`.
- [ ] **Deck slide 6 is titled "Topic Modelling (Supervised)"** — supervised classification is not
      topic modelling; the two are different tasks and the deck already covers real topic modelling
      on slide 5. Retitle to "Text Classification (Supervised)". This is the kind of terminology slip
      an examiner picks up on immediately.
- [ ] **Deck slide 12 contains leftover placeholder text** — "Instance 1" / "Instance 2".
- [ ] **Deck slide 11 is empty** (between RAG and RAG Continued — probably wants `18_rag_rouge_scores.png`).
- [ ] **Deck has no results table.** Slides 6/8/9 quote individual figures but the five-model
      comparison never appears. `17_model_comparison.png` on slide 9 would fix it.

### Verified clean — no action

- **All 30 hyperparameter claims in the report match the notebook code exactly** (TF-IDF
  max_features/ngram/min_df; Word2Vec size/window/min_count/sg/epochs; t-SNE perplexity/max_iter;
  LDA filter_extremes; LSTM vocab/len/dims/dropout/lr/epochs/clipping/scheduler; DistilBERT
  epochs/batch/lr/decay/max_len; MiniLM; IndexFlatIP; Flan-T5; 400-char snippets). Zero contradictions.
- **All 24 headline metrics match** across notebooks, `.md`, `.docx` and deck. No stale numbers
  anywhere in the deck; slide roundings (97.2, 81.6, 96.5, 0.969, 0.971, 0.155, 0.31) are all correct.
- **Figures:** 18 declared, sequential, no duplicates, every referenced file exists, none unused.
- **Cross-references:** all 17 "Section N"/"Appendix X.Y" pointers resolve.
- **Citations:** all 13 references now cited in text and all in-text citations appear in the list.

### Fixed in this pass

- [x] Three references were listed but never cited — **Devlin (2019)**, **Mikolov (2013)**,
      **Johnson (2019)**. Now cited at DistilBERT, Word2Vec and FAISS respectively. (The `.docx`
      already cited all three; the `.md` was the weaker document here.)
- [x] Added an explicit **"On hyperparameters"** note to §4.2.1 stating that no search was performed
      and models use defaults (`C=1.0`, `alpha=0.1`). This is the honest counterpart to deleting the
      false CV claim, and it pre-empts "how did you choose C?" in Q&A.
- [x] `PRESENTATION_SCRIPT.md` Part C rewritten around the verified answers.
- [x] Walkthrough: replaced stale advice telling you the report still says 2,225 (it doesn't any more).

### Known asymmetry (decide, not a bug)

- The `.md` has **18 figures and Appendices A–D**; the `.docx` has **12 figures and no appendices**,
  so figure numbers differ between the two documents. Only one will be submitted — just don't cite
  `.md` figure numbers aloud while the `.docx` is on screen.
- The `.docx` integrates figures into the prose ("The confusion matrix (Figure 7) shows…"); the `.md`
  captions them but the body text refers to only one. If the `.md` is ever submitted, add those.

---

## DOCX vs notebooks — metric audit detail (2026-07-22)

Full audit of `report/ITC6110_report.docx` against the notebooks' saved outputs.
**All 24 headline metrics match the notebooks exactly.** Issues found, worst first:

- [ ] **FALSE METHODOLOGY CLAIM in the .docx — must be removed.** §4.2.1 states: *"Five-fold
      stratified cross-validation on the training split was used for hyperparameter selection."*
      **Notebook 2 performs no cross-validation of any kind** — no `GridSearchCV`, `cross_val_score`,
      `StratifiedKFold`, or `cv=` anywhere. The three classifiers are fit once with hardcoded
      hyperparameters (`C=1.0`, `alpha=0.1`). Claiming unperformed methodology in an assessed report
      is a serious problem and it is trivially checkable from the notebook. The `.md` does **not**
      contain this claim — delete the sentence from the `.docx`.
- [ ] **.docx RAG discussion is contradicted by the saved run.** It claims *"Both zero-scoring
      questions are of this short-factoid form, a strong sign the metric, not the system, produced
      them"* and *"the retriever surfaces relevant articles for every question"*. The saved answers
      show only ONE zero is a metric artefact ("golf", correct). The other — "Who won the Formula One
      championship?" → **"ivanovic"** — is simply wrong. Four of ten answers are factually wrong
      ("American sprinter" for Phelps, "rugby world cup" for Six Nations, "autumn series" for Ashes).
      The `.md` was corrected on 2026-07-22; port that corrected §4.2.2 + §5.1 into the `.docx`.
- [x] Title page reconciled — `.md` now matches `.docx` (Term Spring Semester 2026, 22 July 2026,
      name order George · Orestis · Samuel). The `.md` previously said 2025 with no date.
- [x] `Grootendorst (2022)` added to `.md` references — it cites BERTopic in future work (docx had it)
- [ ] **`.docx` has no appendices.** The `.md` carries Appendices A–D (per-question ROUGE table with
      generated answers, full coherence scan, topic table, sport-only cross-check, retrieval trace,
      UX evaluation). Decide whether to include them in the submitted `.docx` — Appendix C.1 is the
      evidence base for the whole ROUGE argument.
- [ ] Minor: the two documents interpret t-SNE differently. `.docx` says *"sport and tech form tight,
      well-isolated clusters"*; the `.md` (written from the figure) says tech bleeds into business and
      that business/politics are the adjacent pair. Reconcile to one reading — the figure supports the
      `.md`'s version.
- [ ] Minor: the §2.4 before/after preprocessing examples differ between the two documents
      (`.docx` uses a Bupa Great Ireland Run fragment; `.md` uses a ringtone-regulation article).
      Not an error, but pick one for consistency.

**Verified consistent** (docx = notebooks, no action): all five models' accuracy and macro F1;
LDA K=10 / C_v=0.4612 / K=5=0.342; corpus 2,225→2,126 with 99 = 57 exact + 42 cross-split;
1,194/932 splits; TF-IDF 2,126×10,000; Word2Vec neighbour similarities; LIME weights
(match +0.123, rugby +0.117); RAG mean 0.1549 / best 0.3077 / worst 0.0000.

---

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

- [x] `report/ITC6110_report.docx` regenerated by George — verified to carry the fresh numbers (97.21 / 81.55)
- [x] Team member names on title page (George Antonakis · Samuel Cook · Orestis Pappas)
- [ ] **Submission date** on title page (still `[DATE]`)
- [ ] Screenshot of Streamlit app — Tab 1 (Classifier) working
- [ ] Screenshot of Streamlit app — Tab 2 (RAG Q&A) working
      ⚠️ Blocked: the deployed Space is in an error state ("Launch timed out, workload was not healthy
      after 30 min"). Fix it before screenshotting — see the deployment section below.
- [ ] Re-export the `.docx` once the names/date/screenshots are final (current export predates them)
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

## Presentation — `report/NLP BBC Project .pptx` (15 slides, built from PRESENTATION_SCRIPT.md)
- [x] Slides drafted by George; narration in `report/PRESENTATION_SCRIPT.md` maps to them
- [ ] **Slide 11 is completely empty** — sits between "RAG" (10) and "RAG Continued" (12), so it was
      probably meant to hold `outputs/figures/18_rag_rouge_scores.png`. Fill it or delete it.
- [ ] Slide 13 (Streamlit) still needs the two app screenshots pasted in
- [ ] Add the results table to the deck — no model numbers appear on any slide except 81.6% and the
      0.21-point margin. The comparison table or `17_model_comparison.png` would carry slide 9.
- [ ] Assign the 3 parts to team members
- [ ] Rehearse handoffs between speakers
- [ ] Prepare for the "your ROUGE-L is 0.155, is it broken?" question — answer is in Part C.
      Now verifiable, not just arguable: the saved NB3 run shows the model answered **"golf"**.

---

## Submission checklist
- [x] Code on GitHub: https://github.com/Geoanto8043/itc6110-nlp-project
- [x] App deployed: https://huggingface.co/spaces/Geoanto/bbc-news-assistant
- [ ] Verify the deployed Space still works (NB3 re-run reverted its fixes locally — confirm the live
      Space was not redeployed from the broken files)
- [ ] Report submitted via Turnitin
- [ ] Oral presentation prepared
