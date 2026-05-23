---
name: sentiment_model_experiments
description: Trains, tunes, and evaluates 10 classical ML configurations (GridSearchCV KNN, regularized LR, smoothed NB) on optimized TF-IDF features and logs to results.csv.
---

# Sentiment Classification Model Experiments Skill

This skill documents execution instructions, pipeline automation, hyperparameter exploration spaces, and rolling leaderboard parameters for our classical TF-IDF model architecture suite.

## 📋 Skill Prerequisites
- Local virtual environment `.venv` active.
- Complete baseline codebase including `src/experiment.py`, `src/evaluate.py`, and `src/train_val_split.py`.

---

## 🛠️ Execution & Leaderboard Logging

### 1. Trigger Full Pipeline Run
Run the entire baseline experiment suite (which splits the dataset, preprocesses text, vectorizes TF-IDF representations, trains all 10 model variants, and generates comparison reports):
```bash
# Windows
.\.venv\Scripts\python.exe src/experiment.py

# macOS/Linux
./.venv/bin/python src/experiment.py
```

### 2. Verify rolling leaderboard
The pipeline writes to [results.csv](file:///c:/Users/filip/TextMining-Corpora/outputs/results.csv) in a strictly **idempotent** way. Verify that runs are updated in-place:
1. Locate the file `outputs/results.csv`.
2. Inspect line items; ensure there are no duplicate parameter combinations for identical models.
3. Validate that a second execution printed `[LOGGED] Idempotent Update` for all configurations.

---

## 🔬 Parameter Settings & Grid Search Details

The model explorer runs 10 specific configurations:
- **TF-IDF Representations (Logistic Regression)**
  - Unigram Baseline: `ngram_range=(1,1)`, `min_df=1`
  - Raw Bigram Baseline: `ngram_range=(1,2)`, `min_df=1`
  - Optimized Bigrams: `ngram_range=(1,2)`, `min_df=2`, `max_features=25000` (drops 81% of vocabulary noise).
- **K-Nearest Neighbors (KNN) on Optimized TF-IDF**
  - Variant 1: $k=3$ neighbors.
  - Variant 2: $k=7$ neighbors.
  - Grid Search CV: Fits `GridSearchCV(cv=3, scoring='f1_macro')` over `n_neighbors: [3, 7]` to choose the optimal neighbor parameter.
- **Logistic Regression Variants on Optimized TF-IDF**
  - Lasso (L1 Regularization): `penalty='l1'`, `solver='saga'`, `C=1.0` (performs sparse feature selection).
  - Ridge (L2 Regularization): `penalty='l2'`, `solver='lbfgs'`, `C=1.0` (standard baseline).
- **Multinomial Naive Bayes Variants on Optimized TF-IDF**
  - Variant 1 (Laplace): Smoothing parameter $\alpha=1.0$.
  - Variant 2 (Lidstone): Smoothing parameter $\alpha=0.1$ (allows sparse words to contribute effectively).

---

## 📝 Jupyter Notebook Auto-Generation
The runner automatically translates execution findings into a formal Jupyter Notebook located at [02_bow_tfidf_classical.ipynb](file:///c:/Users/filip/TextMining-Corpora/notebooks/02_bow_tfidf_classical.ipynb). Do not modify this notebook by hand; always edit `src/experiment.py` and run the script to regenerate it programmatically.
