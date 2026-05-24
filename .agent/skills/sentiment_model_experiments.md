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

## 🔬 Parameter Settings & Classifier Details

The model explorer evaluates 16 specific configurations:
- **TF-IDF Representations (Logistic Regression)**
  - Unigram Baseline: `ngram_range=(1,1)`, `min_df=1`
  - Raw Bigram Baseline: `ngram_range=(1,2)`, `min_df=1`
  - Optimized Bigrams: `ngram_range=(1,2)`, `min_df=2`, `max_features=25000` (drops 81% of vocabulary noise).
- **k-Nearest Neighbors (KNN) on Optimized TF-IDF**
  - Variant 1: $k=3$ neighbors.
  - Variant 2: $k=7$ neighbors.
  - Grid Search CV: Fits `GridSearchCV(cv=3, scoring='f1_macro')` over `n_neighbors: [3, 7]`.
- **Logistic Regression Variants on Optimized TF-IDF**
  - Lasso (L1 Regularization): `penalty='l1'`, `solver='saga'`, `C=1.0` (sparse feature selection).
  - Ridge (L2 Regularization): `penalty='l2'`, `solver='lbfgs'`, `C=1.0` (champ baseline).
- **Multinomial Naive Bayes Variants on Optimized TF-IDF**
  - Laplace Smoothing: $\alpha=1.0$.
  - Lidstone Smoothing: $\alpha=0.1$ (empirically competitive).
- **Multi-Layer Perceptron (MLP) Neural Networks**
  - Single Layer: `hidden_layer_sizes=(100,)`, `learning_rate_init=0.01`, `early_stopping=True`.
  - Double Layer: `hidden_layer_sizes=(100, 50)`, `learning_rate_init=0.001`, `early_stopping=True`.
- **Random Forest Baselines**
  - Shallow Forest: `n_estimators=100`, `max_depth=10`, `class_weight='balanced'`.
  - Deep Forest: `n_estimators=200`, `max_depth=20`, `class_weight='balanced'`.
- **XGBoost Baselines**
  - Shallow Gradient Booster: `max_depth=3`, `learning_rate=0.1`, `n_estimators=150`.
  - Deep Gradient Booster: `max_depth=6`, `learning_rate=0.05`, `n_estimators=200`.

---

## ⚙️ Automated Champion Retrainer & Predictor (`src/autotune.py`)
To automatically resolve Moodle submission guidelines, the autotuning pipeline does the following:
1. Reads `outputs/results.csv` to find the configuration with the highest **Macro F1-Score**.
2. Instantiates the champion pipeline.
3. Refits the entire pipeline on **100% of the training data** ($N=9,543$) to maximize signal utilization.
4. Generates predictions on the unlabeled test fold (`data/test.csv`) and writes them directly to `outputs/pred_best.csv`.

---

## 📝 Jupyter Notebook Auto-Generation
The runner automatically translates execution findings into a formal Jupyter Notebook located at [02_bow_tfidf_classical.ipynb](file:///c:/Users/filip/TextMining-Corpora/notebooks/02_bow_tfidf_classical.ipynb). Do not modify this notebook by hand; always edit `src/experiment.py` and run the script to regenerate it programmatically.

