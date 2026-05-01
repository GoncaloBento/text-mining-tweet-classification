# Text Mining — Project Requirements

**Course:** Text Mining — NOVA IMS, Spring Semester 2025/2026
**Handout version:** 1.0
**Source:** `Project Guidelines - Text Mining 2025-2026 v1.pdf`

---

## 1. Project Summary

Build an NLP classification model that takes a tweet as input and predicts market sentiment as one of three classes:

| Label | Class   |
|------:|---------|
| 0     | Bearish |
| 1     | Bullish |
| 2     | Neutral |

**Tech stack (required / suggested):** Python 3, NLTK, scikit-learn, Hugging Face, LangChain.
The problem has multiple valid solutions — there is no single "correct" answer.

---

## 2. Group Rules & Deadline

- Group size: **1 to 4 students**. Groups larger than 4 are **not graded**.
- **Delivery deadline:** midnight, **15 June 2026** (Project Report + Notebooks + Predictions).
- The handout also references "23h:59 of the 23rd of January" in the delivery section — this looks like leftover text from a previous edition; treat **15 June 2026** as the binding deadline and confirm with the instructor if needed.

---

## 3. Data — Starting Corpora

Located in `Project Corpora-20260429/`:

| File        | Rows | Columns        | Notes                                       |
|-------------|-----:|----------------|---------------------------------------------|
| `train.csv` | 9543 | `text`, `label`| `label` ∈ {0, 1, 2}. May be split into train/validation. |
| `test.csv`  |  299 | `id`, `text`   | No label column. Predict 0/1/2 for each row. The instructors compare predictions against the true labels. |

---

## 4. Graded Solution Requirements (17.0 pts core)

| # | Criterion             | Points | What it requires |
|--:|-----------------------|-------:|------------------|
| 1 | Data Exploration      | 2.00   | Analyse the corpora; produce conclusions and visuals (bar charts, word clouds, etc.) that contextualise the data. |
| 2 | Corpus Split          | 0.50   | Apply a train/validation split, or K-Fold cross-validation. |
| 3 | Data Preprocessing    | 3.00   | Implement **≥ 4** preprocessing techniques from class (stop words, regex, lemmatisation, stemming, …). |
| 4 | Feature Engineering   | 5.50   | Implement and experiment with **≥ 1 variation each** of: **BoW**, **word2vec**, **Transformer Encoder**. |
| 5 | Classification Models | 4.50   | Implement and test **≥ 2 variations each** of: **Traditional ML** (KNN, MLP, Logistic Regression, Random Forest, XGBoost, …) **and** **Transformer Encoders**. |
| 6 | Evaluation & Analysis | 1.50   | Compare models using at least **Recall, Precision, Accuracy, F1**; explain what the metrics mean in the problem context. |

---

## 5. Extra Work (up to +2.0 pts)

| Area                   | Points | Rule |
|------------------------|-------:|------|
| Feature Engineering    | +0.50 each | Each additional Transformer Encoder method beyond the required one. **Max 2** extras → up to +1.0. |
| Classification Models  | +1.00      | Correctly use a **decoder model** for classification. |

> **Important:** Any extra work must be **clearly flagged as such in the PDF report** — otherwise it will not count.

---

## 6. Deliverables

All files inside a folder named `group_xx/` (zip if needed), submitted via Moodle.

| File                    | Description |
|-------------------------|-------------|
| `tm_tests_xx.ipynb`     | Notebook following the section-4 structure with **all techniques experimented with** and their evaluation. |
| `tm_final_xx.ipynb`     | **Ready-to-run** final solution: a **single pipeline with a single classification model**. |
| `pred_xx.csv`           | Two columns only: **`id`** of test set + **predicted label** (0/1/2). |
| `report_XX.pdf`         | Written report (see §7). **Max 15 pages.** |

`xx` / `XX` = group number.

---

## 7. Report Structure & Weighting

PDF report (max 15 pages). Other structures are accepted, but this is the recommended one — and the % column shows how each section feeds back into the criteria of §4:

| Section                      | Share of criterion |
|------------------------------|-------------------|
| 1. Data Exploration          | 50% of 4.1 |
| 2. Data Preprocessing        | 25% of 4.2 + 25% of 4.3 |
| 3. Feature Engineering       | 30% of 4.4 |
| 4. Classification Models     | 30% of 4.5 |
| 5. Evaluation and Results    | 50% of 4.6 |

---

## 8. Penalties

- **−0.5 pt** for each page over the 15-page report limit.
- **−1.0 pt** for each half-day late.
- **Up to −1.0 pt** for failing to comply with the delivery guide (file naming, folder structure, etc.).
- Students may be **randomly chosen for an oral defense** to verify understanding (anti-plagiarism / anti-misuse-of-GenAI).

---

## 9. Extra Challenge 1 — Leaderboard

Top-3 groups by test-set performance get bonus points:

| Rank | Bonus    |
|-----:|---------:|
| 1st  | +1.00 pt |
| 2nd  | +0.50 pt |
| 3rd  | +0.25 pt |

---

## 10. Extra Challenge 2 — Agentic AI Workflow (up to +1.5 pts)

Design and implement an **agentic AI-based workflow** that orchestrates the classification pipeline using tools or multiple models. Requirements:

- **Conversational interface** that accepts a prompt.
- Performs **at least one non-trivial decision or coordination task**, e.g.:
  - choosing between alternative models,
  - comparing outputs from multiple classifiers,
  - routing tweets to different models,
  - automating evaluation.
- A simple single-LLM prompt that just classifies a tweet **does not qualify**.
- The agent must be **described in detail in the report** to count.

> Final grade is **capped at 20** — extras can compensate for losses elsewhere but cannot push the total above 20.

---

## 11. Personal Checklist

Core (17.0 pts):
- [ ] **Data Exploration** — distributions, length stats, word clouds, class balance.
- [ ] **Corpus split** — train/val (or K-Fold).
- [ ] **Preprocessing** — at least 4 techniques applied (e.g. lowercasing, regex cleanup, stopword removal, lemmatisation, stemming, tokenisation).
- [ ] **Feature engineering** — BoW variant, word2vec variant, Transformer Encoder variant.
- [ ] **Classifiers** — ≥2 traditional ML variants AND ≥2 Transformer Encoder variants.
- [ ] **Evaluation** — Recall, Precision, Accuracy, F1 reported and discussed per class.

Extras (up to +4.5 pts, capped at 20 overall):
- [ ] Extra Transformer Encoder method #1 (+0.5)
- [ ] Extra Transformer Encoder method #2 (+0.5)
- [ ] Decoder-model classification (+1.0)
- [ ] Compete on leaderboard (Extra Challenge 1)
- [ ] Agentic workflow with conversational interface (Extra Challenge 2, +1.5)

Deliverables:
- [ ] `tm_tests_xx.ipynb`
- [ ] `tm_final_xx.ipynb` (single pipeline, single model)
- [ ] `pred_xx.csv` (`id`, predicted label)
- [ ] `report_XX.pdf` (≤ 15 pages, extra work flagged)
- [ ] All inside `group_xx/`, zipped, uploaded to Moodle before **midnight 15 June 2026**.
