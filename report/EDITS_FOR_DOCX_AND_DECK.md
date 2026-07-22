# Exact edits for `ITC6110_report.docx` and `NLP BBC Project .pptx`

Both files are binary and were edited outside this repo, so these are copy-paste instructions.
Each item gives the **exact text to find** and the **exact text to replace it with**.

Source of truth for every number below: the saved outputs in `notebooks/03_dl_rag_ui.ipynb`
(re-run 2026-07-16) and `notebooks/02_unsupervised_ml.ipynb`.

---

# PART 1 — `report/ITC6110_report.docx`

## Edit 1.1 — DELETE the cross-validation claim (§4.2.1, "Traditional ML Models")

**Priority: highest.** The code performs no cross-validation of any kind — no `GridSearchCV`,
no `cross_val_score`, no `StratifiedKFold`, no `cv=` anywhere in Notebook 2. The three
classifiers are fit once with hardcoded hyperparameters. This claim is checkable in under a
minute by anyone who opens the notebook.

**FIND** (the sentence immediately after the three-model results table):

> Five-fold stratified cross-validation on the training split was used for hyperparameter selection; the accuracy and macro-F1 figures above are the final scores on the held-out 932-article test set.

**REPLACE WITH:**

> The figures above are final scores on the held-out 932-article test set. We did not perform a hyperparameter search: each classifier uses library defaults or standard settings (`LinearSVC(C=1.0)`, `MultinomialNB(alpha=0.1)`, `LogisticRegression(C=1.0)`). This is a deliberate limitation rather than an oversight — all three models sit within 0.54 percentage points of each other and close to the dataset's apparent ceiling, so tuning would be optimising differences we argue below are not meaningful. The reported figures should therefore be read as untuned baselines; a search over `C`, the TF-IDF vocabulary size or the n-gram range could plausibly shift them by a few tenths of a point.

---

## Edit 1.2 — REPLACE the RAG discussion paragraph (§4.2.2, after the ROUGE-L results table)

The current paragraph says both zero scores are metric artefacts and that retrieval succeeds on
every question. The saved run refutes both claims. Only one zero is a metric artefact; four of
ten answers are factually wrong.

**FIND** (the whole paragraph beginning "A mean ROUGE-L of 0.155 looks low"):

> A mean ROUGE-L of 0.155 looks low, but substantially reflects a metric that is measuring the wrong thing rather than a broken system. […] ROUGE-L should therefore be read here as a proxy for lexical overlap, not a full measure of answer correctness — a limitation a semantic metric such as BERTScore would address.

**REPLACE WITH:**

> A mean ROUGE-L of 0.155 looks like failure, but inspecting the generated answers shows the metric and the system are failing in different places, and that ROUGE-L cannot tell the two apart.
>
> ROUGE-L rewards literal longest-common-subsequence overlap and penalises valid paraphrase. Our references are full sentences, while an instruction-tuned model asked a direct question answers tersely. Asked "What sport does Tiger Woods play?", the system answered "golf" — completely correct — and scored exactly 0.0000, because it shares no common subsequence with the reference "Tiger Woods is a professional golfer". Six of the ten answers are substantively correct, yet the highest score awarded to any of them is 0.3077.
>
> The second zero is a different matter and we do not excuse it. Asked "Who won the Formula One championship?", the system answered "ivanovic" — a name drawn from a retrieved sports article, and simply wrong. Four of the ten answers are factually wrong on the same pattern: Michael Phelps is described as a sprinter rather than a swimmer, the Six Nations as a "rugby world cup", and the Ashes as an "autumn series". These are genuine failures of a 250M-parameter generator working from truncated context.
>
> The two zeros therefore have opposite causes — one a correct answer the metric cannot recognise, one an incorrect answer — and ROUGE-L assigns them the identical score of 0.0000. That is the substance of our objection: not that the number is harsh, but that ranking these ten answers by ROUGE-L produces an ordering largely unrelated to ranking them by correctness. The headline figure should be read as a lower bound on lexical overlap, not as a measure of whether the system works. A semantic metric such as BERTScore, or separate retrieval metrics (Recall@k, MRR), would measure what we actually care about.

---

## Edit 1.3 — FIX the §5.1 conclusion bullet on RAG

**FIND:**

> The RAG system retrieves reliably, with exact cosine search over FAISS surfacing relevant articles for every evaluation question; generation quality is bounded by the compact Flan-T5-base model rather than by retrieval.

**REPLACE WITH:**

> The RAG evaluation measured the wrong thing, though the system has real faults too. Six of ten answers are substantively correct despite a mean ROUGE-L of 0.155, and the correct answer "golf" scores zero — but four answers are also factually wrong, the signature failure of a compact generator. Retrieval is the stronger half of the pipeline: exact cosine search over FAISS returns topically relevant articles, including a semantically correct match the query never disambiguated. The lesson is that retrieval and generation should have been measured separately rather than through a single overlap score.

---

## Edit 1.4 — OPTIONAL but recommended: reconcile the t-SNE reading (§3.3)

The docx and the `.md` currently describe the same figure differently. The figure
(`outputs/figures/05_tsne_w2v.png`) supports the `.md`'s reading: sport is the isolated cluster;
tech sits lower-left and bleeds into business; business and politics are the adjacent pair.

**FIND:** "sport and tech form tight, well-isolated clusters"
**REPLACE WITH:** "sport forms the most isolated cluster, separated from every other category by a clear margin, while tech is cohesive but sits adjacent to business"

---

## Edit 1.5 — Check the title page

The docx reads **Term: Spring Semester 2026** and **Submission Date: 22 July 2026**. The `.md`
has been updated to match. **Confirm the submission date is real** and not a placeholder — it is
the one figure in this project I could not verify against anything.

Name order in the docx is George Antonakis · Orestis Pappas · Samuel Cook; the `.md` now matches.

---

# PART 2 — `report/NLP BBC Project .pptx`

## Edit 2.1 — Slide 6 title (terminology error)

**FIND:** `Topic Modelling (Supervised)`
**REPLACE WITH:** `Text Classification (Supervised)`

Topic modelling is unsupervised by definition — slide 5 already covers it (LDA). Slide 6 is
supervised classification into the five known categories. Leaving the current title invites the
question "isn't topic modelling unsupervised?" before you have said anything else.

---

## Edit 2.2 — Slide 12, replace the false bullet

**FIND:**

> Both of our zero-scoring questions are short-factoid questions of exactly that shape.

**REPLACE WITH:**

> Our other zero is a different story: "Who won the Formula One championship?" → "ivanovic", which is simply wrong.
> Four of ten answers are factually wrong — a real limitation of a small generator.
> One zero is the metric failing, the other is the system failing — and ROUGE-L gives them the same score.

That last line is the strongest thing on the slide. It converts a weak number into a sharp
methodological point, and it is defensible because it is true.

---

## Edit 2.3 — Slide 12, delete placeholder text

**DELETE** the two stray text boxes reading `Instance 1` and `Instance 2`.

---

## Edit 2.4 — Slide 11 is empty. Fill it.

It sits between "RAG" (10) and "RAG Continued" (12). Suggested content:

**Title:** `RAG — Evaluation Results`

**Insert image:** `outputs/figures/18_rag_rouge_scores.png`

**Bullets:**
> Mean ROUGE-L 0.155 · best 0.31 · two questions scored 0.0000
> "What sport does Tiger Woods play?" → "golf" — correct, scored 0.0000
> Six of ten answers correct; the highest score any of them earned was 0.31

This sets up slide 12's argument with evidence, so the argument lands as a finding rather than
an excuse.

---

## Edit 2.5 — Add the results table (recommended, slide 8 or 9)

No slide currently shows the five-model comparison; individual numbers appear in prose but the
comparison that carries your headline finding is never shown. Either insert
`outputs/figures/17_model_comparison.png`, or add this table:

| Model | Type | Accuracy | Macro F1 |
|---|---|---|---|
| DistilBERT (fine-tuned) | Deep Learning | 97.21% | 0.971 |
| Linear SVM | Traditional ML | 97.00% | 0.969 |
| Logistic Regression | Traditional ML | 96.78% | 0.967 |
| Naive Bayes | Traditional ML | 96.46% | 0.963 |
| BiLSTM (from scratch) | Deep Learning | 81.55% | 0.810 |

The 15.7-point gap between the two deep models, next to the 0.21-point gap between DistilBERT
and the SVM, is the whole argument in one image. It is much more persuasive seen than described.

---

## Edit 2.6 — Slide 13 still needs the two app screenshots

Blocked until the HuggingFace Space is running again. The docx has the same gap
("[Screenshot to be inserted from the live HuggingFace Space before submission.]").

---

# PART 3 — After the edits

1. Re-export the `.docx` and commit it, so the repo copy matches what is submitted.
2. Read `report/PROJECT_WALKTHROUGH.md` → **"Danger zone"** before presenting. It covers the five
   questions where the honest answer differs from what an earlier draft implied.
3. Whoever presents Part C should be able to give the two-zeros answer without notes. It is the
   most likely hostile question and, answered correctly, the most impressive one.
