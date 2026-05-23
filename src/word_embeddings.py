#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/word_embeddings.py

Loads pre-trained GloVe-Twitter-100 embeddings and trains a custom Word2Vec model.
Compares Out-of-Vocabulary (OOV) rates on the validation set.
Vectorizes training & validation sets via mean pooling.
Trains and evaluates Logistic Regression L2 on both representations,
logging performance results to outputs/results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import gensim.downloader as api

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.train_val_split import stratified_split
from src.preprocessing import preprocess_tweet
from src.evaluate import evaluate_and_log

TRAIN_CSV_PATH = "data/train.csv"

def calculate_oov(tokenized_sentences, embedding_model):
    """
    Computes unique word OOV rate and token OOV rate for a tokenized corpus.
    """
    if hasattr(embedding_model, 'key_to_index'):
        vocab = embedding_model.key_to_index
    elif hasattr(embedding_model, 'wv') and hasattr(embedding_model.wv, 'key_to_index'):
        vocab = embedding_model.wv.key_to_index
    else:
        vocab = embedding_model

    all_tokens = [tok for sent in tokenized_sentences for tok in sent]
    unique_tokens = set(all_tokens)

    if not unique_tokens:
        return 0.0, 0.0, set()

    oov_unique = {tok for tok in unique_tokens if tok not in vocab}
    oov_tokens = [tok for tok in all_tokens if tok not in vocab]

    unique_oov_rate = len(oov_unique) / len(unique_tokens)
    token_oov_rate = len(oov_tokens) / len(all_tokens)

    return unique_oov_rate, token_oov_rate, oov_unique

def vectorize_document(tokens, vocab, vectors, vector_size=100):
    """
    Vectorizes a single document by averaging its token embedding vectors.
    """
    valid_vectors = [vectors[tok] for tok in tokens if tok in vocab]
    if not valid_vectors:
        return np.zeros(vector_size)
    return np.mean(valid_vectors, axis=0)

def vectorize_corpus(tokenized_sentences, embedding_model, vector_size=100):
    """
    Vectorizes a full corpus of tokenized sentences using mean pooling.
    """
    if hasattr(embedding_model, 'key_to_index'):
        vocab = embedding_model.key_to_index
        vectors = embedding_model
    elif hasattr(embedding_model, 'wv') and hasattr(embedding_model.wv, 'key_to_index'):
        vocab = embedding_model.wv.key_to_index
        vectors = embedding_model.wv
    else:
        vocab = embedding_model
        vectors = embedding_model

    return np.array([vectorize_document(sent, vocab, vectors, vector_size) for sent in tokenized_sentences])


def main():
    print("=" * 60)
    print("RUNNING WORD EMBEDDINGS COMPARISON PIPELINE...")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"[ERROR] Training CSV not found at {TRAIN_CSV_PATH}")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # 2. Train/Val Split
    print("[INFO] Creating stratified train/validation split...")
    X_train_raw, X_val_raw, y_train, y_val = stratified_split(train_df)

    # 3. Preprocess Texts to Token Lists
    print("[INFO] Preprocessing tweets into token lists (lemmatizer active)...")
    X_train_tokens = X_train_raw.apply(lambda t: preprocess_tweet(t, return_str=False)).tolist()
    X_val_tokens = X_val_raw.apply(lambda t: preprocess_tweet(t, return_str=False)).tolist()

    # 4. Train Custom Word2Vec Model
    print("[INFO] Training custom Word2Vec model (size=100, window=5, min_count=2)...")
    w2v_model = Word2Vec(
        sentences=X_train_tokens,
        vector_size=100,
        window=5,
        min_count=2,
        seed=42,
        workers=4
    )
    print("[SUCCESS] Custom Word2Vec model trained successfully!")
    print(f"[INFO] Word2Vec Vocabulary Size: {len(w2v_model.wv.key_to_index)}")

    # 5. Load pre-trained GloVe Twitter Embeddings
    print("[INFO] Loading pre-trained GloVe Twitter 100-dimensional embeddings...")
    print("[INFO] This might download up to 400MB if not cached locally...")
    try:
        glove_model = api.load("glove-twitter-100")
        print("[SUCCESS] Pre-trained GloVe model loaded successfully!")
        print(f"[INFO] GloVe Vocabulary Size: {len(glove_model.key_to_index)}")
    except Exception as e:
        print(f"[ERROR] Failed to load GloVe-Twitter-100 embeddings: {str(e)}")
        sys.exit(1)

    # 6. Compute Out-of-Vocabulary (OOV) Statistics on Validation Fold
    print("[INFO] Computing Out-of-Vocabulary (OOV) statistics on validation set...")
    w2v_uniq_oov, w2v_tok_oov, w2v_oov_words = calculate_oov(X_val_tokens, w2v_model)
    glove_uniq_oov, glove_tok_oov, glove_oov_words = calculate_oov(X_val_tokens, glove_model)

    print("-" * 60)
    print("OUT-OF-VOCABULARY (OOV) COMPARISON:")
    print("-" * 60)
    print(f"Custom Word2Vec Unique Word OOV Rate: {w2v_uniq_oov * 100:.2f}%")
    print(f"Custom Word2Vec Token OOV Rate:       {w2v_tok_oov * 100:.2f}%")
    print(f"GloVe-Twitter-100 Unique Word OOV Rate: {glove_uniq_oov * 100:.2f}%")
    print(f"GloVe-Twitter-100 Token OOV Rate:       {glove_tok_oov * 100:.2f}%")
    print("-" * 60)

    # Sample OOV Words
    print("[INFO] Sample OOV Words in Word2Vec:")
    print(list(w2v_oov_words)[:15])
    print("[INFO] Sample OOV Words in GloVe-Twitter-100:")
    print(list(glove_oov_words)[:15])
    print("-" * 60)

    # 7. Vectorize Dataset via Mean Pooling
    print("[INFO] Creating mean pooled vector representations...")
    X_train_w2v = vectorize_corpus(X_train_tokens, w2v_model, 100)
    X_val_w2v = vectorize_corpus(X_val_tokens, w2v_model, 100)

    X_train_glove = vectorize_corpus(X_train_tokens, glove_model, 100)
    X_val_glove = vectorize_corpus(X_val_tokens, glove_model, 100)

    # 8. Train and Evaluate Downstream Logistic Regression L2 on Custom Word2Vec
    print("[INFO] Downstream Evaluation: Custom Word2Vec Mean Pooling + LR L2...")
    clf_w2v = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
    clf_w2v.fit(X_train_w2v, y_train)
    y_pred_w2v = clf_w2v.predict(X_val_w2v)

    evaluate_and_log(
        y_val, y_pred_w2v,
        model_name="Logistic Regression Baseline",
        feature_desc="Word2Vec Mean Pooling",
        params="vector_size=100, window=5, min_count=2, penalty=l2, class_weight=balanced"
    )

    # 9. Train and Evaluate Downstream Logistic Regression L2 on GloVe
    print("[INFO] Downstream Evaluation: GloVe-Twitter-100 Mean Pooling + LR L2...")
    clf_glove = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
    clf_glove.fit(X_train_glove, y_train)
    y_pred_glove = clf_glove.predict(X_val_glove)

    evaluate_and_log(
        y_val, y_pred_glove,
        model_name="Logistic Regression Baseline",
        feature_desc="GloVe-Twitter-100 Mean Pooling",
        params="vector_size=100, pre-trained, penalty=l2, class_weight=balanced"
    )

    print("\n" + "=" * 60)
    print("WORD EMBEDDINGS COMPARISON PIPELINE COMPLETE!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
