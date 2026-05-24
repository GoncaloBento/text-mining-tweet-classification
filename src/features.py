#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/features.py

Feature engineering module. Implements:
1. CountVectorizer (Bag of Words unigrams)
2. TfidfVectorizer (TF-IDF unigrams)
Fits vectorizers strictly on the training set and saves the sparse matrices
as .npz files inside the outputs/ directory.
"""

import os
import sys
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.train_val_split import stratified_split
from src.preprocessing import preprocess_tweet

def extract_and_save_features():
    print("=" * 60)
    print("RUNNING FEATURE EXTRACTION PIPELINE (BoW & TF-IDF UNIGRAMS)...")
    print("=" * 60)

    # 1. Load Data
    train_path = os.path.join(project_root, "data", "train.csv")
    if not os.path.exists(train_path):
        print(f"[ERROR] Training CSV not found at: {train_path}")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    print(f"Loaded training data from {train_path} ({len(train_df)} rows).")

    # 2. Train/Val Split
    X_train, X_val, y_train, y_val = stratified_split(train_df)
    print(f"Performed stratified split: Train={len(X_train)} | Val={len(X_val)}")

    # 3. Preprocess Texts
    print("Applying text preprocessing pipeline (lemmatization enabled)...")
    X_train_pre = X_train.apply(lambda t: preprocess_tweet(t, return_str=True))
    X_val_pre = X_val.apply(lambda t: preprocess_tweet(t, return_str=True))
    print("Text preprocessing complete.")

    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # 4. Bag-of-Words (CountVectorizer)
    print("\n--- Fitting CountVectorizer (BoW unigrams) ---")
    bow_vec = CountVectorizer(ngram_range=(1, 1))
    X_train_bow = bow_vec.fit_transform(X_train_pre)
    X_val_bow = bow_vec.transform(X_val_pre)
    print(f"BoW Vocabulary Size: {len(bow_vec.vocabulary_)}")

    # Save BoW matrices
    train_bow_path = os.path.join(outputs_dir, "X_train_bow.npz")
    val_bow_path = os.path.join(outputs_dir, "X_val_bow.npz")
    sp.save_npz(train_bow_path, X_train_bow)
    sp.save_npz(val_bow_path, X_val_bow)
    print(f"Saved BoW sparse matrices to:\n  - {train_bow_path}\n  - {val_bow_path}")

    # 5. TF-IDF unigrams (TfidfVectorizer)
    print("\n--- Fitting TfidfVectorizer (TF-IDF unigrams) ---")
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
    X_train_tfidf = tfidf_vec.fit_transform(X_train_pre)
    X_val_tfidf = tfidf_vec.transform(X_val_pre)
    print(f"TF-IDF Vocabulary Size: {len(tfidf_vec.vocabulary_)}")

    # Save TF-IDF matrices
    train_tfidf_path = os.path.join(outputs_dir, "X_train_tfidf_uni.npz")
    val_tfidf_path = os.path.join(outputs_dir, "X_val_tfidf_uni.npz")
    sp.save_npz(train_tfidf_path, X_train_tfidf)
    sp.save_npz(val_tfidf_path, X_val_tfidf)
    print(f"Saved TF-IDF sparse matrices to:\n  - {train_tfidf_path}\n  - {val_tfidf_path}")

    print("=" * 60)
    print("FEATURE EXTRACTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    extract_and_save_features()
