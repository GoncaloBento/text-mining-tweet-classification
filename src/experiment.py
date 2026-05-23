#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/experiment.py

Modular, idempotent text mining experiment runner.
Executes sentiment classification experiments comparing:
1. TF-IDF Unigrams baseline
2. TF-IDF Unigrams + Bigrams raw baseline
3. TF-IDF Unigrams + Bigrams optimized (min_df=2, max_features=25000)

Evaluates all features using a balanced class weight Logistic Regression model,
idempotently logs scores to outputs/results.csv, and automatically generates
the notebooks/02_bow_tfidf_classical.ipynb Jupyter notebook.
"""

import os
import sys
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Ensure project root is in sys.path to allow running directly or as module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom src modules
from src.train_val_split import stratified_split
from src.preprocessing import preprocess_tweet
from src.evaluate import evaluate_and_log

def run_tfidf_pipeline(
    X_train_preprocessed, X_val_preprocessed, y_train, y_val,
    ngram_range=(1, 1), min_df=1, max_features=None,
    feature_desc="TF-IDF"
):
    """
    Fits TF-IDF vectorizer on training text, transforms validation set,
    trains a Logistic Regression classifier, and logs metrics idempotently.
    """
    # Create parameter string for logging
    params_str = f"ngram_range={ngram_range}, min_df={min_df}, max_features={max_features}"
    
    # 1. Feature Extraction
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    X_train_vec = vec.fit_transform(X_train_preprocessed)
    X_val_vec = vec.transform(X_val_preprocessed)
    vocab_size = len(vec.vocabulary_)
    
    # 2. Model Training
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_vec, y_train)
    y_pred = lr.predict(X_val_vec)
    
    # 3. Evaluation & Logging
    metrics = evaluate_and_log(
        y_val, y_pred,
        model_name="Logistic Regression Baseline",
        feature_desc=feature_desc,
        params=params_str
    )
    
    return vocab_size, metrics

def run_model_pipeline(
    X_train_vec, X_val_vec, y_train, y_val,
    model, model_name, feature_desc, params_str
):
    """
    Fits a standard classifier model on vectorized features, predicts on the validation set,
    and logs the resulting metrics using evaluate_and_log idempotently.
    """
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_val_vec)
    
    metrics = evaluate_and_log(
        y_val, y_pred,
        model_name=model_name,
        feature_desc=feature_desc,
        params=params_str
    )
    
    return metrics

def generate_notebook(
    vocab_uni, vocab_bi, vocab_opt,
    metrics_uni, metrics_bi, metrics_opt,
    metrics_knn3, metrics_knn7, metrics_knn_grid,
    metrics_lr_l1, metrics_lr_l2,
    metrics_nb_1, metrics_nb_01,
    metrics_mlp_shallow, metrics_mlp_deep,
    metrics_rf_shallow, metrics_rf_deep,
    metrics_xgb_shallow, metrics_xgb_deep,
    best_knn_k
):
    """Generates the notebooks/02_bow_tfidf_classical.ipynb notebook file programmatically."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 2. TF-IDF Representation & Comprehensive Multi-Algorithm Exploration\n",
                    "**Nova IMS — Text Mining 2025/2026**\n",
                    "\n",
                    "This notebook implements Bag-of-Words TF-IDF vectorization and systematically evaluates:\n",
                    "1. **TF-IDF Representation Baselines** (Unigrams, raw Unigrams+Bigrams, and Optimized Unigrams+Bigrams with Logistic Regression)\n",
                    "2. **K-Nearest Neighbors (KNN)** variants ($k=3$, $k=7$, and GridSearchCV-tuned optimal $k$)\n",
                    "3. **Logistic Regression (LR)** variants (L1 Regularization SAGA solver and L2 Regularization LBFGS solver)\n",
                    "4. **Multinomial Naive Bayes (Multinomial NB)** variants (alpha=1.0 Laplace smoothing and alpha=0.1 Lidstone smoothing)\n",
                    "5. **Multi-Layer Perceptron (MLP)** neural networks (shallow vs deep layers & learning rate adjustments)\n",
                    "6. **Random Forest (RF)** ensemble variants (shallow vs deep maximum depths)\n",
                    "7. **XGBoost (XGB)** gradient boosting trees (differing depths & learning rates)\n",
                    "\n",
                    "All multi-algorithm variants are evaluated on our standardized **Optimized TF-IDF feature space** to ensure clean, rigorous model architecture comparisons."
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
                    "from sklearn.neighbors import KNeighborsClassifier\n",
                    "from sklearn.naive_bayes import MultinomialNB\n",
                    "from sklearn.model_selection import GridSearchCV\n",
                    "from sklearn.neural_network import MLPClassifier\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from xgboost import XGBClassifier\n",
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
                    "We preprocess the text using our custom pipeline from `src/preprocessing.py`, incorporating lemmatization, smart punctuation normalizations, and emoji handling."
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
                    "## ⚖️ 3. TF-IDF Representation Baselines\n",
                    "We first analyze the N-gram boundaries using Logistic Regression classifiers."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Model A: Unigrams Baseline (`ngram_range=(1,1)`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "vec_uni = TfidfVectorizer(ngram_range=(1, 1))\n",
                    "X_train_uni = vec_uni.fit_transform(X_train_preprocessed)\n",
                    "X_val_uni = vec_uni.transform(X_val_preprocessed)\n",
                    "\n",
                    "lr_uni = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_uni.fit(X_train_uni, y_train)\n",
                    "y_pred_uni = lr_uni.predict(X_val_uni)\n",
                    "\n",
                    "metrics_uni = evaluate_and_log(\n",
                    "    y_val, y_pred_uni,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,1)\",\n",
                    "    params=\"ngram_range=(1, 1), min_df=1, max_features=None\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Model B: Unigrams + Bigrams Raw (`ngram_range=(1,2)`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "vec_bi = TfidfVectorizer(ngram_range=(1, 2))\n",
                    "X_train_bi = vec_bi.fit_transform(X_train_preprocessed)\n",
                    "X_val_bi = vec_bi.transform(X_val_preprocessed)\n",
                    "\n",
                    "lr_bi = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_bi.fit(X_train_bi, y_train)\n",
                    "y_pred_bi = lr_bi.predict(X_val_bi)\n",
                    "\n",
                    "metrics_bi = evaluate_and_log(\n",
                    "    y_val, y_pred_bi,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2)\",\n",
                    "    params=\"ngram_range=(1, 2), min_df=1, max_features=None\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Model C: Unigrams + Bigrams Optimized (`ngram_range=(1,2), min_df=2, max_features=25000`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "vec_opt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)\n",
                    "X_train_opt = vec_opt.fit_transform(X_train_preprocessed)\n",
                    "X_val_opt = vec_opt.transform(X_val_preprocessed)\n",
                    "\n",
                    "lr_opt = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_opt.fit(X_train_opt, y_train)\n",
                    "y_pred_opt = lr_opt.predict(X_val_opt)\n",
                    "\n",
                    "metrics_opt = evaluate_and_log(\n",
                    "    y_val, y_pred_opt,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"ngram_range=(1, 2), min_df=2, max_features=25000\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🤖 4. Multi-Algorithm Exploration (Trained on Optimized TF-IDF Features)\n",
                    "All subsequent models are trained on the pre-vectorized optimized feature spaces: `X_train_opt` and `X_val_opt`."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 👥 A. K-Nearest Neighbors (KNN) Exploration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# KNN k=3\n",
                    "knn3 = KNeighborsClassifier(n_neighbors=3)\n",
                    "knn3.fit(X_train_opt, y_train)\n",
                    "y_pred_knn3 = knn3.predict(X_val_opt)\n",
                    "metrics_knn3 = evaluate_and_log(\n",
                    "    y_val, y_pred_knn3,\n",
                    "    model_name=\"KNN Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"n_neighbors=3\"\n",
                    ")\n",
                    "\n",
                    "# KNN k=7\n",
                    "knn7 = KNeighborsClassifier(n_neighbors=7)\n",
                    "knn7.fit(X_train_opt, y_train)\n",
                    "y_pred_knn7 = knn7.predict(X_val_opt)\n",
                    "metrics_knn7 = evaluate_and_log(\n",
                    "    y_val, y_pred_knn7,\n",
                    "    model_name=\"KNN Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"n_neighbors=7\"\n",
                    ")\n",
                    "\n",
                    "# KNN GridSearchCV\n",
                    "knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [3, 7]}, cv=3, scoring='f1_macro', n_jobs=-1)\n",
                    "knn_grid.fit(X_train_opt, y_train)\n",
                    "print(f\"Best KNN parameter found: {knn_grid.best_params_}\")\n",
                    "y_pred_knn_grid = knn_grid.predict(X_val_opt)\n",
                    "metrics_knn_grid = evaluate_and_log(\n",
                    "    y_val, y_pred_knn_grid,\n",
                    "    model_name=\"KNN Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=f\"GridSearchCV Best, n_neighbors={knn_grid.best_params_['n_neighbors']}\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### ⚖️ B. Logistic Regression Regularization Variants"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# L1 Lasso Saga\n",
                    "lr_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_l1.fit(X_train_opt, y_train)\n",
                    "y_pred_l1 = lr_l1.predict(X_val_opt)\n",
                    "metrics_l1 = evaluate_and_log(\n",
                    "    y_val, y_pred_l1,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"penalty=l1, solver=saga, C=1.0, class_weight=balanced\"\n",
                    ")\n",
                    "\n",
                    "# L2 Ridge Lbfgs\n",
                    "lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)\n",
                    "lr_l2.fit(X_train_opt, y_train)\n",
                    "y_pred_l2 = lr_l2.predict(X_val_opt)\n",
                    "metrics_l2 = evaluate_and_log(\n",
                    "    y_val, y_pred_l2,\n",
                    "    model_name=\"Logistic Regression Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"penalty=l2, solver=lbfgs, C=1.0, class_weight=balanced\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 🔔 C. Multinomial Naive Bayes Smoothing Variants"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Multinomial NB alpha=1.0\n",
                    "nb_1 = MultinomialNB(alpha=1.0)\n",
                    "nb_1.fit(X_train_opt, y_train)\n",
                    "y_pred_nb_1 = nb_1.predict(X_val_opt)\n",
                    "metrics_nb_1 = evaluate_and_log(\n",
                    "    y_val, y_pred_nb_1,\n",
                    "    model_name=\"Multinomial NB Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"alpha=1.0\"\n",
                    ")\n",
                    "\n",
                    "# Multinomial NB alpha=0.1\n",
                    "nb_01 = MultinomialNB(alpha=0.1)\n",
                    "nb_01.fit(X_train_opt, y_train)\n",
                    "y_pred_nb_01 = nb_01.predict(X_val_opt)\n",
                    "metrics_nb_01 = evaluate_and_log(\n",
                    "    y_val, y_pred_nb_01,\n",
                    "    model_name=\"Multinomial NB Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"alpha=0.1\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 🧠 D. Multi-Layer Perceptron (MLP) Neural Network Exploration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# MLP Shallow & Fast\n",
                    "mlp_shallow = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, early_stopping=True, random_state=42)\n",
                    "mlp_shallow.fit(X_train_opt, y_train)\n",
                    "y_pred_mlp_s = mlp_shallow.predict(X_val_opt)\n",
                    "metrics_mlp_shallow = evaluate_and_log(\n",
                    "    y_val, y_pred_mlp_s,\n",
                    "    model_name=\"MLP Classifier\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"hidden_layer_sizes=(100,), learning_rate_init=0.01, early_stopping=True\"\n",
                    ")\n",
                    "\n",
                    "# MLP Deep & Regularized\n",
                    "mlp_deep = MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.001, early_stopping=True, random_state=42)\n",
                    "mlp_deep.fit(X_train_opt, y_train)\n",
                    "y_pred_mlp_d = mlp_deep.predict(X_val_opt)\n",
                    "metrics_mlp_deep = evaluate_and_log(\n",
                    "    y_val, y_pred_mlp_d,\n",
                    "    model_name=\"MLP Classifier\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"hidden_layer_sizes=(100, 50), learning_rate_init=0.001, early_stopping=True\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 🌲 E. Random Forest Ensemble Exploration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Random Forest Shallow\n",
                    "rf_shallow = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)\n",
                    "rf_shallow.fit(X_train_opt, y_train)\n",
                    "y_pred_rf_s = rf_shallow.predict(X_val_opt)\n",
                    "metrics_rf_shallow = evaluate_and_log(\n",
                    "    y_val, y_pred_rf_s,\n",
                    "    model_name=\"Random Forest Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"n_estimators=100, max_depth=10, class_weight=balanced\"\n",
                    ")\n",
                    "\n",
                    "# Random Forest Deep\n",
                    "rf_deep = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)\n",
                    "rf_deep.fit(X_train_opt, y_train)\n",
                    "y_pred_rf_d = rf_deep.predict(X_val_opt)\n",
                    "metrics_rf_deep = evaluate_and_log(\n",
                    "    y_val, y_pred_rf_d,\n",
                    "    model_name=\"Random Forest Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"n_estimators=200, max_depth=20, class_weight=balanced\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 🚀 F. XGBoost Gradient Boosting Exploration"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# XGBoost Conservative Depth\n",
                    "xgb_shallow = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=150, random_state=42, n_jobs=-1)\n",
                    "xgb_shallow.fit(X_train_opt, y_train)\n",
                    "y_pred_xgb_s = xgb_shallow.predict(X_val_opt)\n",
                    "metrics_xgb_shallow = evaluate_and_log(\n",
                    "    y_val, y_pred_xgb_s,\n",
                    "    model_name=\"XGBoost Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"max_depth=3, learning_rate=0.1, n_estimators=150\"\n",
                    ")\n",
                    "\n",
                    "# XGBoost Aggressive Depth\n",
                    "xgb_deep = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=200, random_state=42, n_jobs=-1)\n",
                    "xgb_deep.fit(X_train_opt, y_train)\n",
                    "y_pred_xgb_d = xgb_deep.predict(X_val_opt)\n",
                    "metrics_xgb_deep = evaluate_and_log(\n",
                    "    y_val, y_pred_xgb_d,\n",
                    "    model_name=\"XGBoost Baseline\",\n",
                    "    feature_desc=\"TF-IDF (1,2) Optimized\",\n",
                    "    params=\"max_depth=6, learning_rate=0.05, n_estimators=200\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 5. Comprehensive Exploration Leaderboard Summary\n",
                    "\n",
                    "The following rolling leaderboard details the performance of all 16 evaluated configurations:\n",
                    "\n",
                    "| Model & Variant | Feature Space | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |\n",
                    "| :--- | :---: | :---: | :---: | :---: | :---: |\n",
                    f"| **Model A: TF-IDF (1,1) (LR)** | Unigram Baseline | {metrics_uni['accuracy']:.4f} | {metrics_uni['precision_macro']:.4f} | {metrics_uni['recall_macro']:.4f} | {metrics_uni['f1_macro']:.4f} |\n",
                    f"| **Model B: TF-IDF (1,2) (LR)** | Raw N-Grams | {metrics_bi['accuracy']:.4f} | {metrics_bi['precision_macro']:.4f} | {metrics_bi['recall_macro']:.4f} | {metrics_bi['f1_macro']:.4f} |\n",
                    f"| **Model C: TF-IDF (1,2) Opt (LR)** | Pruned Vocabulary | {metrics_opt['accuracy']:.4f} | {metrics_opt['precision_macro']:.4f} | {metrics_opt['recall_macro']:.4f} | {metrics_opt['f1_macro']:.4f} |\n",
                    f"| **Model D1: KNN (k=3)** | Pruned Vocabulary | {metrics_knn3['accuracy']:.4f} | {metrics_knn3['precision_macro']:.4f} | {metrics_knn3['recall_macro']:.4f} | {metrics_knn3['f1_macro']:.4f} |\n",
                    f"| **Model D2: KNN (k=7)** | Pruned Vocabulary | {metrics_knn7['accuracy']:.4f} | {metrics_knn7['precision_macro']:.4f} | {metrics_knn7['recall_macro']:.4f} | {metrics_knn7['f1_macro']:.4f} |\n",
                    f"| **Model D3: KNN (CV Best, k={best_knn_k})** | Pruned Vocabulary | {metrics_knn_grid['accuracy']:.4f} | {metrics_knn_grid['precision_macro']:.4f} | {metrics_knn_grid['recall_macro']:.4f} | {metrics_knn_grid['f1_macro']:.4f} |\n",
                    f"| **Model E1: Logistic Reg. L1** | Pruned Vocabulary | {metrics_lr_l1['accuracy']:.4f} | {metrics_lr_l1['precision_macro']:.4f} | {metrics_lr_l1['recall_macro']:.4f} | {metrics_lr_l1['f1_macro']:.4f} |\n",
                    f"| **Model E2: Logistic Reg. L2** | Pruned Vocabulary | {metrics_lr_l2['accuracy']:.4f} | {metrics_lr_l2['precision_macro']:.4f} | {metrics_lr_l2['recall_macro']:.4f} | {metrics_lr_l2['f1_macro']:.4f} |\n",
                    f"| **Model F1: Multinomial NB (1.0)** | Pruned Vocabulary | {metrics_nb_1['accuracy']:.4f} | {metrics_nb_1['precision_macro']:.4f} | {metrics_nb_1['recall_macro']:.4f} | {metrics_nb_1['f1_macro']:.4f} |\n",
                    f"| **Model F2: Multinomial NB (0.1)** | Pruned Vocabulary | {metrics_nb_01['accuracy']:.4f} | {metrics_nb_01['precision_macro']:.4f} | {metrics_nb_01['recall_macro']:.4f} | {metrics_nb_01['f1_macro']:.4f} |\n",
                    f"| **Model G1: MLP (Shallow)** | Pruned Vocabulary | {metrics_mlp_shallow['accuracy']:.4f} | {metrics_mlp_shallow['precision_macro']:.4f} | {metrics_mlp_shallow['recall_macro']:.4f} | {metrics_mlp_shallow['f1_macro']:.4f} |\n",
                    f"| **Model G2: MLP (Deep)** | Pruned Vocabulary | {metrics_mlp_deep['accuracy']:.4f} | {metrics_mlp_deep['precision_macro']:.4f} | {metrics_mlp_deep['recall_macro']:.4f} | {metrics_mlp_deep['f1_macro']:.4f} |\n",
                    f"| **Model H1: Random Forest (Shallow)** | Pruned Vocabulary | {metrics_rf_shallow['accuracy']:.4f} | {metrics_rf_shallow['precision_macro']:.4f} | {metrics_rf_shallow['recall_macro']:.4f} | {metrics_rf_shallow['f1_macro']:.4f} |\n",
                    f"| **Model H2: Random Forest (Deep)** | Pruned Vocabulary | {metrics_rf_deep['accuracy']:.4f} | {metrics_rf_deep['precision_macro']:.4f} | {metrics_rf_deep['recall_macro']:.4f} | {metrics_rf_deep['f1_macro']:.4f} |\n",
                    f"| **Model I1: XGBoost (Depth=3)** | Pruned Vocabulary | {metrics_xgb_shallow['accuracy']:.4f} | {metrics_xgb_shallow['precision_macro']:.4f} | {metrics_xgb_shallow['recall_macro']:.4f} | {metrics_xgb_shallow['f1_macro']:.4f} |\n",
                    f"| **Model I2: XGBoost (Depth=6)** | Pruned Vocabulary | {metrics_xgb_deep['accuracy']:.4f} | {metrics_xgb_deep['precision_macro']:.4f} | {metrics_xgb_deep['recall_macro']:.4f} | {metrics_xgb_deep['f1_macro']:.4f} |\n",
                    "\n",
                    "### 💡 Key Strategic Observations:\n",
                    "1. **Tree-Based Models & Sparsity**: Ensembles like Random Forest and XGBoost require careful depth configurations to navigate highly sparse text bag-of-words dimensions efficiently.\n",
                    "2. **Neural Architectures**: MLPs offer strong non-linear boundary learning but can require early stopping safeguards to limit validation loss decay.\n",
                    "3. **Automatic Selection**: Using our automated autotuning script (`src/autotune.py`), any agent can parse these exact validation scores and dynamically compile optimal predictions on the full test sets."
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

def main():
    print("=" * 60)
    print("RUNNING IDEMPOTENT TF-IDF PIPELINE EXPERIMENTS...")
    print("=" * 60)

    # 1. Load Data
    train_path = "data/train.csv"
    if not os.path.exists(train_path):
        print(f"[ERROR] Training CSV not found at: {train_path}")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)

    # 2. Train/Val Split
    X_train, X_val, y_train, y_val = stratified_split(train_df)

    # 3. Preprocess Texts
    print("Applying text preprocessing pipeline (lemmatization enabled)...")
    X_train_pre = X_train.apply(lambda t: preprocess_tweet(t, return_str=True))
    X_val_pre = X_val.apply(lambda t: preprocess_tweet(t, return_str=True))

    # Experiment A: Unigrams Baseline
    print("\n--- EXPERIMENT A: TF-IDF UNIGRAMS BASELINE ---")
    vocab_uni, metrics_uni = run_tfidf_pipeline(
        X_train_pre, X_val_pre, y_train, y_val,
        ngram_range=(1, 1), min_df=1, max_features=None,
        feature_desc="TF-IDF (1,1)"
    )

    # Experiment B: Unigrams + Bigrams Raw
    print("\n--- EXPERIMENT B: TF-IDF UNIGRAMS + BIGRAMS RAW ---")
    vocab_bi, metrics_bi = run_tfidf_pipeline(
        X_train_pre, X_val_pre, y_train, y_val,
        ngram_range=(1, 2), min_df=1, max_features=None,
        feature_desc="TF-IDF (1,2)"
    )

    # Experiment C: Unigrams + Bigrams Optimized
    print("\n--- EXPERIMENT C: TF-IDF UNIGRAMS + BIGRAMS OPTIMIZED ---")
    vocab_opt, metrics_opt = run_tfidf_pipeline(
        X_train_pre, X_val_pre, y_train, y_val,
        ngram_range=(1, 2), min_df=2, max_features=25000,
        feature_desc="TF-IDF (1,2) Optimized"
    )

    # Set up static pre-vectorized optimized matrices for all subsequent baseline models
    print("\nVectorizing training & validation text using Optimized TF-IDF vectorizer...")
    vec_opt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
    X_train_opt = vec_opt.fit_transform(X_train_pre)
    X_val_opt = vec_opt.transform(X_val_pre)

    # --- KNN EXPLORATION ---
    print("\n--- EXPERIMENT D1: KNN k=3 BASELINE ---")
    metrics_knn3 = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=KNeighborsClassifier(n_neighbors=3),
        model_name="KNN Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="n_neighbors=3"
    )

    print("\n--- EXPERIMENT D2: KNN k=7 BASELINE ---")
    metrics_knn7 = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=KNeighborsClassifier(n_neighbors=7),
        model_name="KNN Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="n_neighbors=7"
    )

    print("\n--- TUNING: KNN GRIDSEARCHCV ---")
    knn_cv = GridSearchCV(
        KNeighborsClassifier(),
        param_grid={'n_neighbors': [3, 7]},
        cv=3,
        scoring='f1_macro',
        n_jobs=-1
    )
    knn_cv.fit(X_train_opt, y_train)
    best_knn_k = knn_cv.best_params_['n_neighbors']
    print(f"Optimal n_neighbors found: {best_knn_k}")

    y_pred_knn_grid = knn_cv.predict(X_val_opt)
    metrics_knn_grid = evaluate_and_log(
        y_val, y_pred_knn_grid,
        model_name="KNN Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params=f"GridSearchCV Best, n_neighbors={best_knn_k}"
    )

    # --- LOGISTIC REGRESSION VARIANTS ---
    print("\n--- EXPERIMENT E1: LOGISTIC REGRESSION L1 (LASSO) ---")
    metrics_lr_l1 = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=LogisticRegression(penalty='l1', solver='saga', max_iter=1000, class_weight='balanced', random_state=42),
        model_name="Logistic Regression Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="penalty=l1, solver=saga, C=1.0, class_weight=balanced"
    )

    print("\n--- EXPERIMENT E2: LOGISTIC REGRESSION L2 (RIDGE) ---")
    metrics_lr_l2 = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42),
        model_name="Logistic Regression Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="penalty=l2, solver=lbfgs, C=1.0, class_weight=balanced"
    )

    # --- NAIVE BAYES VARIANTS ---
    print("\n--- EXPERIMENT F1: MULTINOMIAL NB (alpha=1.0) ---")
    metrics_nb_1 = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=MultinomialNB(alpha=1.0),
        model_name="Multinomial NB Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="alpha=1.0"
    )

    print("\n--- EXPERIMENT F2: MULTINOMIAL NB (alpha=0.1) ---")
    metrics_nb_01 = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=MultinomialNB(alpha=0.1),
        model_name="Multinomial NB Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="alpha=0.1"
    )

    # --- MLP NEURAL NETWORK EXPLORATION ---
    print("\n--- EXPERIMENT G1: MLP (SHALLOW) ---")
    metrics_mlp_shallow = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, early_stopping=True, random_state=42),
        model_name="MLP Classifier",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="hidden_layer_sizes=(100,), learning_rate_init=0.01, early_stopping=True"
    )

    print("\n--- EXPERIMENT G2: MLP (DEEP) ---")
    metrics_mlp_deep = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.001, early_stopping=True, random_state=42),
        model_name="MLP Classifier",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="hidden_layer_sizes=(100, 50), learning_rate_init=0.001, early_stopping=True"
    )

    # --- RANDOM FOREST ENSEMBLE ---
    print("\n--- EXPERIMENT H1: RANDOM FOREST (SHALLOW) ---")
    metrics_rf_shallow = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        model_name="Random Forest Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="n_estimators=100, max_depth=10, class_weight=balanced"
    )

    print("\n--- EXPERIMENT H2: RANDOM FOREST (DEEP) ---")
    metrics_rf_deep = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1),
        model_name="Random Forest Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="n_estimators=200, max_depth=20, class_weight=balanced"
    )

    # --- XGBOOST GRADIENT BOOSTING ---
    print("\n--- EXPERIMENT I1: XGBOOST (max_depth=3, lr=0.1) ---")
    metrics_xgb_shallow = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=150, random_state=42, n_jobs=-1),
        model_name="XGBoost Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="max_depth=3, learning_rate=0.1, n_estimators=150"
    )

    print("\n--- EXPERIMENT I2: XGBOOST (max_depth=6, lr=0.05) ---")
    metrics_xgb_deep = run_model_pipeline(
        X_train_opt, X_val_opt, y_train, y_val,
        model=XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=200, random_state=42, n_jobs=-1),
        model_name="XGBoost Baseline",
        feature_desc="TF-IDF (1,2) Optimized",
        params_str="max_depth=6, learning_rate=0.05, n_estimators=200"
    )

    # Output Console Summary Comparison Table
    print("\n" + "=" * 105)
    print("COMPREHENSIVE BASELINE EXPLORATION LEADERBOARD")
    print("=" * 105)
    print(f"{'Model & Variant':<35} | {'Accuracy':<10} | {'Prec. (Macro)':<14} | {'Recall (Macro)':<14} | {'F1 (Macro)':<12}")
    print("-" * 105)
    
    leaderboard_data = [
        ("TF-IDF (1,1) LR Baseline", metrics_uni),
        ("TF-IDF (1,2) LR Raw Baseline", metrics_bi),
        ("TF-IDF (1,2) LR Optimized", metrics_opt),
        ("KNN (k=3) Baseline", metrics_knn3),
        ("KNN (k=7) Baseline", metrics_knn7),
        (f"KNN (GridSearchCV Best k={best_knn_k})", metrics_knn_grid),
        ("Logistic Reg. L1 (Lasso SAGA)", metrics_lr_l1),
        ("Logistic Reg. L2 (Ridge LBFGS)", metrics_lr_l2),
        ("Multinomial NB (alpha=1.0 Laplace)", metrics_nb_1),
        ("Multinomial NB (alpha=0.1 Lidstone)", metrics_nb_01),
        ("MLP Shallow (hidden_layers=100)", metrics_mlp_shallow),
        ("MLP Deep (hidden_layers=100,50)", metrics_mlp_deep),
        ("Random Forest Shallow (depth=10)", metrics_rf_shallow),
        ("Random Forest Deep (depth=20)", metrics_rf_deep),
        ("XGBoost Shallow (depth=3, lr=0.1)", metrics_xgb_shallow),
        ("XGBoost Deep (depth=6, lr=0.05)", metrics_xgb_deep),
    ]

    for label, metrics in leaderboard_data:
        print(f"{label:<35} | {metrics['accuracy']:.4f}     | {metrics['precision_macro']:.4f}        | {metrics['recall_macro']:.4f}        | {metrics['f1_macro']:.4f}")
    print("=" * 105)

    # Generate Notebook
    generate_notebook(
        vocab_uni, vocab_bi, vocab_opt,
        metrics_uni, metrics_bi, metrics_opt,
        metrics_knn3, metrics_knn7, metrics_knn_grid,
        metrics_lr_l1, metrics_lr_l2,
        metrics_nb_1, metrics_nb_01,
        metrics_mlp_shallow, metrics_mlp_deep,
        metrics_rf_shallow, metrics_rf_deep,
        metrics_xgb_shallow, metrics_xgb_deep,
        best_knn_k
    )
if __name__ == '__main__':
    main()
