---
name: feature_vectorization
description: Fits CountVectorizer (BoW unigrams) and TfidfVectorizer (TF-IDF unigrams) strictly on training text and exports CSR sparse matrices (.npz) under outputs/.
---

# Classical Feature Vectorization (BoW & TF-IDF) Skill

This skill documents execution instructions, data partitioning guidelines, and sparse file outputs for classical frequency-based vectorizations in our sentiment classification corpus.

## 📋 Skill Prerequisites
- Local virtual environment `.venv` active.
- Complete baseline codebase including `src/features.py`, `src/train_val_split.py`, and `src/preprocessing.py`.
- Training data file `data/train.csv` present in the workspace.

---

## 🛠️ Execution & Sparse File Generation

### 1. Trigger Feature Extraction Pipeline
Run the feature extraction suite (which splits the dataset, preprocesses text, and fits both BoW and TF-IDF unigram models):
```bash
# Windows
.\.venv\Scripts\python.exe src/features.py

# macOS/Linux
./.venv/bin/python src/features.py
```

### 2. Verify Output Sparse Files
The pipeline writes four compressed sparse row (CSR) matrices inside the `outputs/` folder. Verify that they are present:
1. `outputs/X_train_bow.npz` (CountVectorizer training features)
2. `outputs/X_val_bow.npz` (CountVectorizer validation features)
3. `outputs/X_train_tfidf_uni.npz` (TfidfVectorizer unigram training features)
4. `outputs/X_val_tfidf_uni.npz` (TfidfVectorizer unigram validation features)

---

## 🔒 Rigorous Data Partitioning Rules
To comply with scientific validation and prevent **data leakage**:
- **Fit Only on Train**: Vectorizers (`CountVectorizer` and `TfidfVectorizer`) must be fitted on `X_train_preprocessed` only. They must never see or be fitted on validation or test text.
- **Out-of-Sample Transform**: Validation and test sets must only be transformed via the already-fitted vectorizers (`vectorizer.transform(X_val_preprocessed)`).

---

## 📊 Summary of Baseline Model Performance
Evaluating these unigram vectors using a balanced Logistic Regression classifier yields the following baseline scores:

- **BoW (CountVectorizer) Unigrams**:
  - Accuracy: `0.7737`
  - Macro F1-Score: `0.6990`
- **TF-IDF (TfidfVectorizer) Unigrams**:
  - Accuracy: `0.7685`
  - Macro F1-Score: `0.6957`
