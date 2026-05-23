#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
notebooks/run_tfidf_experiment.py

Runs the TF-IDF unigrams vs unigrams+bigrams comparison experiment,
evaluates using a Logistic Regression baseline, logs metrics to outputs/results.csv,
and generates the formal notebooks/02_bow_tfidf_classical.ipynb notebook.
"""

import os
import sys
import json

# Ensure project root is in the path to support src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import custom src packages
from src.train_val_split import stratified_split
from src.preprocessing import preprocess_tweet
from src.evaluate import evaluate_and_log

def main():
    print("=" * 60)
    print("RUNNING TF-IDF UNIGRAMS vs UNIGRAMS+BIGRAMS EXPERIMENT...")
    print("=" * 60)

    # 1. Load data
    train_path = "data/train.csv"
    if not os.path.exists(train_path):
        print(f"[ERROR] Training CSV not found at: {train_path}. Make sure the data is dropped in.")
        return

    print("Loading data...")
    train_df = pd.read_csv(train_path)

    # 2. Train/Val Split
    print("Splitting train/validation sets...")
    X_train, X_val, y_train, y_val = stratified_split(train_df)
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

    # 3. Preprocess Texts
    print("Applying text preprocessing pipeline (lemmatization enabled)...")
    X_train_preprocessed = X_train.apply(lambda t: preprocess_tweet(t, return_str=True))
    X_val_preprocessed = X_val.apply(lambda t: preprocess_tweet(t, return_str=True))

    # 4. TF-IDF Unigrams (ngram_range=(1,1))
    print("Fitting TF-IDF (1, 1) Vectorizer...")
    vec_uni = TfidfVectorizer(ngram_range=(1, 1))
    X_train_uni = vec_uni.fit_transform(X_train_preprocessed)
    X_val_uni = vec_uni.transform(X_val_preprocessed)
    vocab_size_uni = len(vec_uni.vocabulary_)
    print("   |-- Unigrams Vocabulary Dimension: " + str(vocab_size_uni))

    # 5. TF-IDF Unigrams+Bigrams (ngram_range=(1,2))
    print("Fitting TF-IDF (1, 2) Vectorizer...")
    vec_bi = TfidfVectorizer(ngram_range=(1, 2))
    X_train_bi = vec_bi.fit_transform(X_train_preprocessed)
    X_val_bi = vec_bi.transform(X_val_preprocessed)
    vocab_size_bi = len(vec_bi.vocabulary_)
    print("   |-- Unigrams+Bigrams Vocabulary Dimension: " + str(vocab_size_bi))

    # Calculate growth
    growth_ratio = (vocab_size_bi / vocab_size_uni) if vocab_size_uni > 0 else 0
    print("   |-- Vocabulary size grew by " + f"{growth_ratio:.2f}x" + " when adding bigrams.")

    # 6. Evaluate Logistic Regression (LR) on Unigrams
    print("\nTraining Logistic Regression on TF-IDF (1, 1)...")
    lr_uni = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_uni.fit(X_train_uni, y_train)
    y_pred_uni = lr_uni.predict(X_val_uni)
    metrics_uni = evaluate_and_log(
        y_val, y_pred_uni,
        model_name="Logistic Regression Baseline",
        feature_desc="TF-IDF (1,1)",
        params="max_iter=1000, class_weight='balanced', random_state=42"
    )

    # 7. Evaluate Logistic Regression (LR) on Unigrams+Bigrams
    print("\nTraining Logistic Regression on TF-IDF (1, 2)...")
    lr_bi = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_bi.fit(X_train_bi, y_train)
    y_pred_bi = lr_bi.predict(X_val_bi)
    metrics_bi = evaluate_and_log(
        y_val, y_pred_bi,
        model_name="Logistic Regression Baseline",
        feature_desc="TF-IDF (1,2)",
        params="max_iter=1000, class_weight='balanced', random_state=42"
    )

    # 8. Comparison Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} | {'TF-IDF (1,1)':<15} | {'TF-IDF (1,2)':<15} | {'Difference':<15}")
    print("-" * 76)
    print(f"{'Vocabulary Dimension':<25} | {vocab_size_uni:<15} | {vocab_size_bi:<15} | {f'+{vocab_size_bi-vocab_size_uni} ({(growth_ratio-1)*100:.1f}%)':<15}")
    acc_uni_str = f"{metrics_uni['accuracy']:.4f}"
    acc_bi_str = f"{metrics_bi['accuracy']:.4f}"
    acc_diff_str = f"{metrics_bi['accuracy']-metrics_uni['accuracy']:+.4f}"
    
    prec_uni_str = f"{metrics_uni['precision_macro']:.4f}"
    prec_bi_str = f"{metrics_bi['precision_macro']:.4f}"
    prec_diff_str = f"{metrics_bi['precision_macro']-metrics_uni['precision_macro']:+.4f}"
    
    rec_uni_str = f"{metrics_uni['recall_macro']:.4f}"
    rec_bi_str = f"{metrics_bi['recall_macro']:.4f}"
    rec_diff_str = f"{metrics_bi['recall_macro']-metrics_uni['recall_macro']:+.4f}"
    
    f1_uni_str = f"{metrics_uni['f1_macro']:.4f}"
    f1_bi_str = f"{metrics_bi['f1_macro']:.4f}"
    f1_diff_str = f"{metrics_bi['f1_macro']-metrics_uni['f1_macro']:+.4f}"

    print(f"{'Accuracy':<25} | {acc_uni_str:<15} | {acc_bi_str:<15} | {acc_diff_str:<15}")
    print(f"{'Precision (Macro)':<25} | {prec_uni_str:<15} | {prec_bi_str:<15} | {prec_diff_str:<15}")
    print(f"{'Recall (Macro)':<25} | {rec_uni_str:<15} | {rec_bi_str:<15} | {rec_diff_str:<15}")
    print(f"{'F1-Score (Macro)':<25} | {f1_uni_str:<15} | {f1_bi_str:<15} | {f1_diff_str:<15}")
    print("=" * 76)

    # 9. Generate 02_bow_tfidf_classical.ipynb
    print("\nGenerating Jupyter Notebook: notebooks/02_bow_tfidf_classical.ipynb...")
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 2. TF-IDF Representation & Baseline Evaluation\n",
                    "**Nova IMS — Text Mining 2025/2026**\n",
                    "\n",
                    "This notebook implements the Bag-of-Words TF-IDF vectorization and compares the performance of:\n",
                    "1. **TF-IDF Unigrams** (`ngram_range=(1,1)`)\n",
                    "2. **TF-IDF Unigrams + Bigrams** (`ngram_range=(1,2)`)\n",
                    "\n",
                    "Both representations are evaluated using a **Logistic Regression** baseline classifier with class weights balanced to handle target sentiment distribution."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "# Ensure project src is in the system path\n",
                    "sys.path.append(os.path.abspath('..'))\n",
                    "\n",
                    "import pandas as pd\n",
                    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "\n",
                    "from src.train_val_split import stratified_split\n",
                    "from src.preprocessing import preprocess_tweet\n",
                    "from src.evaluate import evaluate_and_log"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 💾 1. Load Data & Perform Split"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_df = pd.read_csv('../data/train.csv')\n",
                    "X_train, X_val, y_train, y_val = stratified_split(train_df)\n",
                    "print(f\"Train set size: {len(X_train)} | Validation set size: {len(X_val)}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🧹 2. Apply Custom Preprocessing\n",
                    "We preprocess the text using the custom pipeline from `src/preprocessing.py`, utilizing WordNet Lemmatization while preserving placeholders."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Preprocessing training set...\")\n",
                    "X_train_preprocessed = X_train.apply(lambda t: preprocess_tweet(t, return_str=True))\n",
                    "print(\"Preprocessing validation set...\")\n",
                    "X_val_preprocessed = X_val.apply(lambda t: preprocess_tweet(t, return_str=True))\n",
                    "print(\"Preprocessing complete!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔬 3. Feature Extraction (TF-IDF)\n",
                    "We fit both vectorizers strictly on the **training set only** to prevent data leakage."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# TF-IDF Unigrams\n",
                    "vec_uni = TfidfVectorizer(ngram_range=(1, 1))\n",
                    "X_train_uni = vec_uni.fit_transform(X_train_preprocessed)\n",
                    "X_val_uni = vec_uni.transform(X_val_preprocessed)\n",
                    "vocab_size_uni = len(vec_uni.vocabulary_)\n",
                    "\n",
                    "# TF-IDF Unigrams + Bigrams\n",
                    "vec_bi = TfidfVectorizer(ngram_range=(1, 2))\n",
                    "X_train_bi = vec_bi.fit_transform(X_train_preprocessed)\n",
                    "X_val_bi = vec_bi.transform(X_val_preprocessed)\n",
                    "vocab_size_bi = len(vec_bi.vocabulary_)\n",
                    "\n",
                    "print(f\"Unigrams Vocab size: {vocab_size_uni}\")\n",
                    "print(f\"Unigrams + Bigrams Vocab size: {vocab_size_bi}\")\n",
                    "print(f\"Vocabulary size grew by {vocab_size_bi / vocab_size_uni:.2f}x when adding bigrams.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## ⚖️ 4. Model Baseline Evaluation (Logistic Regression)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Model A: Unigrams Baseline"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "lr_uni = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_uni.fit(X_train_uni, y_train)\n",
                    "y_pred_uni = lr_uni.predict(X_val_uni)\n",
                    "\n",
                    "# Evaluate and log metrics to outputs/results.csv\n",
                    "metrics_uni = evaluate_and_log(\n",
                    "    y_val, y_pred_uni,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,1)\",\n",
                    "    params=\"max_iter=1000, class_weight='balanced', random_state=42\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Model B: Unigrams + Bigrams"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "lr_bi = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_bi.fit(X_train_bi, y_train)\n",
                    "y_pred_bi = lr_bi.predict(X_val_bi)\n",
                    "\n",
                    "# Evaluate and log metrics to outputs/results.csv\n",
                    "metrics_bi = evaluate_and_log(\n",
                    "    y_val, y_pred_bi,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2)\",\n",
                    "    params=\"max_iter=1000, class_weight='balanced', random_state=42\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 5. Summary & Comparison\n",
                    "\n",
                    "| Metric | TF-IDF (1,1) | TF-IDF (1,2) | Difference |\n",
                    "| :--- | :---: | :---: | :---: |\n",
                    f"| **Vocabulary Size** | {vocab_size_uni:,} | {vocab_size_bi:,} | {vocab_size_bi-vocab_size_uni:+,} ({(vocab_size_bi/vocab_size_uni-1)*100:+.1f}%) |\n",
                    f"| **Accuracy** | {metrics_uni['accuracy']:.4f} | {metrics_bi['accuracy']:.4f} | {metrics_bi['accuracy']-metrics_uni['accuracy']:+.4f} |\n",
                    f"| **Precision (Macro)** | {metrics_uni['precision_macro']:.4f} | {metrics_bi['precision_macro']:.4f} | {metrics_bi['precision_macro']-metrics_uni['precision_macro']:+.4f} |\n",
                    f"| **Recall (Macro)** | {metrics_uni['recall_macro']:.4f} | {metrics_bi['recall_macro']:.4f} | {metrics_bi['recall_macro']-metrics_uni['recall_macro']:+.4f} |\n",
                    f"| **F1-Score (Macro)** | {metrics_uni['f1_macro']:.4f} | {metrics_bi['f1_macro']:.4f} | {metrics_bi['f1_macro']-metrics_uni['f1_macro']:+.4f} |\n",
                    "\n",
                    "### Core Observations:\n",
                    "1. **Vocabulary Expansion**: Adding bigrams exponentially increases the features. Bigrams capture multi-word sentiment indicators (e.g. \"price cut\", \"earnings beat\", \"underperform needham\").\n",
                    "2. **Baseline Boost**: Evaluating the F1-Score and general performance tells us how bigrams compare to unigram baselines under class balancing. Check `outputs/results.csv` for rolling leaderboard logs."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    notebook_path = "notebooks/02_bow_tfidf_classical.ipynb"
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=2)
    print(f"[SUCCESS] Formal jupyter notebook generated at {notebook_path} successfully!")

if __name__ == '__main__':
    main()
