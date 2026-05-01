# Text Mining — Development Plan

**Group size:** 4 members (M1, M2, M3, M4 — assign names on first sync)
**Working window:** 2026-05-01 → 2026-06-05 (5 weeks, internal deadline; official deadline is 15 June 2026 — gives us a 10-day buffer for slippage / oral defense prep).
**Goal:** Hit all six graded criteria in `REQUIREMENTS.md` plus at least one of the +2.0 extras and Extra Challenge 2 (agentic workflow).

---

## 1. What's already on the table from class

Mapping the syllabus we've already had to the project criteria:

| Class material                                   | Project criterion served |
|--------------------------------------------------|--------------------------|
| L1 Introduction to NLP                           | 4.1 framing |
| Lab 1 — Preprocessing + BoW + KNN                | 4.3 preprocessing, 4.4 BoW, 4.5 traditional ML (KNN) |
| Lab 2 Extra — TF-IDF + NB + LR + MLP + bigrams   | 4.4 BoW variant (TF-IDF + n-grams), 4.5 traditional ML (NB, LR, MLP) |
| Lab 2 — Word2Vec + GloVe + LR-on-embeddings      | 4.4 word2vec, 4.5 LR on embeddings |
| L2 EXTRA n-grams + TF-IDF (PDF)                  | 4.4 BoW theory |
| L2 Word Embeddings (PDF)                         | 4.4 word2vec theory |
| **Not yet covered — to come in May classes**     | **Transformer Encoder** (4.4 + 4.5 — required), **decoder model** (extra), agentic workflows |

> Implication: weeks 1–3 reuse class material directly; week 4 lines up with whichever lab covers Transformers; week 5 polishes and adds the agent.

---

## 2. Roles (4 people)

Assignments are *primary owners*; everyone reviews everyone else's PRs.

| Member | Primary area | Backup |
|--------|--------------|--------|
| **M1 — Data & EDA lead** | Data exploration, preprocessing pipeline, class-imbalance handling, train/val split | Documentation |
| **M2 — Classical ML lead** | BoW/TF-IDF + n-gram features, KNN/LR/NB/MLP/RandomForest/XGBoost variants | Word2Vec |
| **M3 — Embeddings lead** | Word2Vec / GloVe / FastText averaging + weighted-by-IDF; Skip-Gram if needed | Transformer fine-tune |
| **M4 — Transformer & Agent lead** | DistilBERT / RoBERTa fine-tuning, decoder-LLM extra, agentic workflow | Evaluation tables |

Cross-cutting (shared):
- **Evaluation harness** (M1+M2): one notebook utility that takes `(y_true, y_pred, model_name)` → row in a results DataFrame with Recall (macro), Precision (macro), Accuracy, F1 (macro + per-class).
- **Report writing** (everyone): each owner drafts the section for their area; M1 consolidates.

---

## 3. Repo layout (proposed)

```
group_xx/
├── data/
│   ├── train.csv                  # symlink or copy from Project Corpora
│   └── test.csv
├── notebooks/
│   ├── 00_eda.ipynb               # M1
│   ├── 01_preprocessing.ipynb     # M1
│   ├── 02_bow_tfidf_classical.ipynb   # M2
│   ├── 03_word_embeddings.ipynb   # M3
│   ├── 04_transformers.ipynb      # M4
│   ├── 05_decoder_extra.ipynb     # M4 (extra)
│   ├── 06_agent.ipynb             # M4 (extra challenge 2)
│   ├── tm_tests_xx.ipynb          # consolidated experiments — DELIVERABLE
│   └── tm_final_xx.ipynb          # final pipeline + best model — DELIVERABLE
├── src/
│   ├── preprocessing.py           # shared cleaning fn
│   ├── features.py                # vectorisers, embedding builders
│   ├── evaluate.py                # metric helpers
│   └── agent.py                   # agentic workflow
├── outputs/
│   ├── results.csv                # rolling leaderboard of model runs
│   └── pred_xx.csv                # FINAL DELIVERABLE
├── report/
│   └── report_xx.tex / .docx
├── REQUIREMENTS.md
├── DEVELOPMENT.md
└── README.md
```

Use one shared **GitHub** (or Azure DevOps) repo. Branch per task, PR review by another member before merge into `main`. No notebook commits with cell outputs over 1 MB — strip outputs before merging (`nbstripout` pre-commit).

---

## 4. Five-week plan

Each week ends with a **30-min sync** (proposed: Friday 17:00 GMT) — review demo, decide next week's tasks, refresh the leaderboard.

### Week 1 — May 1 → 8: Foundation
**Goal:** working dataset, EDA done, preprocessing pipeline frozen, dummy baseline submitted to `pred_xx.csv` end-to-end.

| Owner | Task |
|-------|------|
| M1    | Load `train.csv` / `test.csv`; tweet-length histogram, token counts, class balance, hashtag/URL/cashtag freq, top-n words per class, word clouds per class. |
| M1    | Detect duplicates, near-duplicates, empty rows, encoding issues, language outliers. |
| M2    | Stratified train/val split (80/20, `random_state=42`); also wire 5-fold StratifiedKFold for later. |
| M3    | Implement `preprocessing.py` — single `clean(text, options)` covering: lowercasing, URL/cashtag/mention/hashtag handling, regex normalisation, stopword removal (NLTK + custom finance stopwords), tokenisation, stemming (Porter), lemmatisation (WordNet). **≥4 techniques satisfied** (criterion 4.3). |
| M4    | Set up `evaluate.py` (macro/weighted P/R/F1, accuracy, per-class report, confusion matrix). Build a results-logging helper that appends a row to `outputs/results.csv`. |
| All   | Sanity baseline: majority-class predictor on `test.csv` → `pred_xx.csv` with placeholder labels just to verify the submission path. |

**Definition of done:** EDA notebook signed off by 2 members; preprocessing covers ≥4 techniques and has unit-style smoke tests on toy strings; baseline F1 logged.

### Week 2 — May 9 → 15: Classical ML Foundation
**Goal:** all classical-ML variants trained, BoW + TF-IDF features locked in.

| Owner | Task |
|-------|------|
| M2    | BoW (Count) + TF-IDF unigrams; TF-IDF unigrams+bigrams (covers ≥1 BoW variant — criterion 4.4 part 1). |
| M2    | Train **≥2 traditional-ML variants per algorithm**, hyper-tune via GridSearchCV / RandomizedSearchCV: KNN, Logistic Regression, Multinomial NB, MLP, Random Forest, XGBoost. Log every run. |
| M3    | Class-imbalance experiments: `class_weight='balanced'`, oversampling (RandomOverSampler), undersampling, focal loss for MLP. Pick the best handler and freeze it. |
| M1    | Add error-analysis notebook section: confusion matrices, top misclassified tweets per class, pattern-spotting (sarcasm, ambiguous tickers). |
| M4    | Spike: try Hugging Face `datasets` + DistilBERT zero-shot on 200 samples just to check tokenisation, GPU access, environment. **No training yet.** |

**Definition of done:** ≥6 classical-ML configurations on the leaderboard; best-of-classical F1 macro identified.

### Week 3 — May 16 → 22: Word Embeddings
**Goal:** word2vec / GloVe path complete (criterion 4.4 part 2 + criterion 4.5).

| Owner | Task |
|-------|------|
| M3    | Train **Word2Vec (gensim)** on the train corpus: try CBOW vs Skip-Gram, dim 100/200, min_count tuning. |
| M3    | Load **GloVe Twitter 100** (the lab uses it) — straight comparison vs custom word2vec on the same downstream classifier. |
| M3    | Build sentence embeddings two ways: **mean pooling** and **TF-IDF-weighted mean pooling**. The latter is a non-trivial improvement over the lab. |
| M2    | Train ≥2 classifiers on each embedding type (LR + MLP, or LR + RandomForest). Add to the leaderboard. |
| M4    | Set up Hugging Face fine-tuning skeleton (`Trainer` API or PyTorch loop), tokeniser, dataloaders. Validate on 500 samples. |
| M1    | Start drafting the Data Exploration and Preprocessing sections of the report. |

**Definition of done:** word2vec + GloVe each plug into the classifier harness; embedding-based leaderboard rows present; transformer infra ready.

### Week 4 — May 23 → 29: Transformers
**Goal:** Transformer Encoder feature + Transformer Encoder classifier (criterion 4.4 part 3 + criterion 4.5 part 2). Lock the *final* model.

| Owner | Task |
|-------|------|
| M4    | **DistilBERT-base-uncased** fine-tune (3 epochs, lr 2e-5, batch 16, weight decay 0.01) — required Transformer variant 1. |
| M4    | **Twitter-RoBERTa-base-sentiment** fine-tune — required Transformer variant 2 (specifically pretrained on tweets, often the strongest). |
| M3    | **FinBERT** fine-tune — counts as +0.5 extra Transformer Encoder method (finance-domain pretraining). |
| M3    | **DeBERTa-v3-base** fine-tune — counts as +0.5 extra Transformer Encoder method, often state-of-the-art. |
| M2    | Use frozen Transformer embeddings (CLS or mean of last hidden states) as features into LR / XGBoost — counts as a Transformer-Encoder-as-feature variant. |
| M1    | Build the consolidated comparison table across BoW / TF-IDF / w2v / GloVe / DistilBERT / RoBERTa / FinBERT / DeBERTa, with macro-F1 and per-class. |
| All   | Pick the **single final model** (likely RoBERTa-Twitter or DeBERTa) for `tm_final_xx.ipynb`. |

**Definition of done:** ≥2 transformer fine-tunes complete + ≥1 transformer-as-feature run; all four required-criterion families fully met; the +1.0 to +2.0 extras attempted; final model chosen.

### Week 5 — May 30 → Jun 5: Extras, Polish, Submission
**Goal:** decoder extra, agent, report, deliverable hygiene.

| Owner | Task |
|-------|------|
| M4    | **Decoder extra** (+1.0): zero/few-shot classification with a small instruction-tuned decoder (e.g. `Qwen2.5-1.5B-Instruct` locally) — prompt-engineered labelling, evaluated on val set. |
| M4    | **Extra Challenge 2 — agentic workflow** (+1.5): LangChain-based agent with tools `[classify_with_roberta, classify_with_finbert, classify_with_decoder, fetch_metrics]`. The agent receives a natural-language prompt ("classify this tweet" / "compare models on tweet X" / "rerun evaluation on val set") and decides which tool(s) to call. Conversational REPL in the notebook. |
| M2    | Write `tm_final_xx.ipynb`: single end-to-end pipeline (preprocess → tokenise → final transformer → predict → write `pred_xx.csv`). Verify it runs top-to-bottom on a clean kernel. |
| M3    | Generate `pred_xx.csv` with two columns exactly: `id`, predicted label. |
| M1    | Finish `report_xx.pdf` (≤15 pages). Ensure each extra is **flagged "EXTRA WORK"** as required. |
| All   | Final review pass: file naming (`tm_tests_xx`, `tm_final_xx`, `pred_xx`, `report_xx`), folder name `group_xx`, zip, submit on Moodle. **Internal deadline Jun 5; safety buffer to Jun 15.** |

**Definition of done:** every checkbox in `REQUIREMENTS.md §11` ticked; submission package zipped and uploaded.

---

## 5. Modelling / experimental policy

- **Random seed:** `42` everywhere (`numpy`, `random`, `torch`, sklearn `random_state`).
- **Splits:** stratified 80/20 train/val from `train.csv`. K-fold (5) only for final-table robustness, not for hyper-tuning every model (would blow the budget).
- **Primary metric:** **macro-F1** (handout doesn't fix one — macro-F1 is the standard for 3-class imbalanced sentiment). Report all four required metrics in the final table.
- **No data leakage:** every vectoriser / embedding model fitted on **train only**, then `transform` applied to val and test. Codified in `features.py`.
- **Logging:** every model run appends to `outputs/results.csv` with columns `timestamp, owner, features, model, hyperparams, val_acc, val_p, val_r, val_f1_macro, val_f1_per_class, notes`.
- **Reproducibility:** pin versions in `requirements.txt`; commit notebook outputs only after a clean run.
- **Compute:** local CPU for classical ML; Colab T4 / Kaggle for transformer fine-tuning. Keep tokenisation cached on disk to avoid re-paying the cost.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------:|-------:|-----------|
| Transformer fine-tune doesn't beat TF-IDF + LR | Medium | Low | TF-IDF + LR is a known-strong baseline on financial tweets; fine-tuning two transformers gives us at least one win. |
| GPU/Colab quota exhausted in week 4 | Medium | Medium | Pre-tokenise; checkpoint after each epoch; M3 has Kaggle as backup. |
| Class imbalance dominates loss → Bullish always wins | High at first | Medium | `class_weight`, `WeightedRandomSampler`, focal loss; track per-class F1 not just macro. |
| Agentic workflow becomes a time sink in W5 | High | Medium | Keep agent scope minimal: 3 tools, one routing decision. Stop at +1.5 ceiling — don't gold-plate. |
| Notebook merge conflicts | High | Low | One owner per notebook; `nbstripout`; favour `src/*.py` modules over notebook code. |
| Member illness / exam clash | Medium | High | Backup column in §2; weekly sync surfaces blockers; 10-day buffer between Jun 5 and the official Jun 15. |

---

## 7. Definition of done (project-wide)

A merge to `main` qualifies as "done" when:
1. The notebook runs top-to-bottom on a clean kernel.
2. All metrics are appended to `outputs/results.csv`.
3. The owner has updated their report section.
4. A second member has reviewed the diff.
5. No `train.csv` or `test.csv` content is hard-coded into the report or committed twice.

The whole project is "done" when:
- `group_xx.zip` contains `tm_tests_xx.ipynb`, `tm_final_xx.ipynb`, `pred_xx.csv`, `report_xx.pdf`.
- `tm_final_xx.ipynb` produces `pred_xx.csv` reproducibly from a fresh kernel.
- Every checkbox in `REQUIREMENTS.md §11` is ticked.
- Submission uploaded on Moodle **before midnight Jun 5** (internal) — Jun 15 (hard).
