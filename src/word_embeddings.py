#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/word_embeddings.py

Loads pre-trained GloVe-Twitter-100 embeddings and trains custom Word2Vec models.
Compares Out-of-Vocabulary (OOV) rates on the validation set.
Vectorizes training & validation sets via mean pooling and TF-IDF weighted pooling.
Trains and evaluates Logistic Regression and MLP on all representations,
logging performance results to outputs/results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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
    Mean pooling.
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

    return np.array([
        vectorize_document(sent, vocab, vectors, vector_size)
        for sent in tokenized_sentences
    ])


def vectorize_document_tfidf(tokens, vocab, vectors, tfidf_row, tfidf_vocab, vector_size=100):
    """
    Vectorizes a single document using TF-IDF weighted mean pooling.
    """
    weighted_vectors = []
    weights = []

    for tok in tokens:
        if tok in vocab and tok in tfidf_vocab:
            weight = tfidf_row[0, tfidf_vocab[tok]]

            if weight > 0:
                weighted_vectors.append(vectors[tok] * weight)
                weights.append(weight)

    if not weighted_vectors:
        return np.zeros(vector_size)

    return np.sum(weighted_vectors, axis=0) / np.sum(weights)


def vectorize_corpus_tfidf(tokenized_sentences, embedding_model, tfidf_matrix, tfidf_vectorizer, vector_size=100):
    """
    Vectorizes a full corpus using TF-IDF weighted mean pooling.
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

    tfidf_vocab = tfidf_vectorizer.vocabulary_

    return np.array([
        vectorize_document_tfidf(
            tokens=sent,
            vocab=vocab,
            vectors=vectors,
            tfidf_row=tfidf_matrix[i],
            tfidf_vocab=tfidf_vocab,
            vector_size=vector_size
        )
        for i, sent in enumerate(tokenized_sentences)
    ])


def print_embedding_diagnostics(name, X_train, X_val):
    """
    Prints shape and norm diagnostics required for TM-020.
    """
    print("\n" + "-" * 60)
    print(f"DIAGNOSTICS: {name}")
    print("-" * 60)
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Train norm:  {np.linalg.norm(X_train):.4f}")
    print(f"Val norm:    {np.linalg.norm(X_val):.4f}")
    print(f"NaNs:        {np.isnan(X_train).sum() + np.isnan(X_val).sum()}")


def train_and_evaluate_classifier(
    X_train,
    X_val,
    y_train,
    y_val,
    classifier,
    model_name,
    feature_desc,
    params
):
    """
    Trains a classifier and logs evaluation metrics.
    """
    print("\n" + "=" * 60)
    print(f"[INFO] Training {model_name} on {feature_desc}")
    print("=" * 60)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)

    evaluate_and_log(
        y_val,
        y_pred,
        model_name=model_name,
        feature_desc=feature_desc,
        params=params
    )


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
    X_train_tokens = X_train_raw.apply(
        lambda t: preprocess_tweet(t, return_str=False)
    ).tolist()

    X_val_tokens = X_val_raw.apply(
        lambda t: preprocess_tweet(t, return_str=False)
    ).tolist()

    # TF-IDF uses strings, so we join the preprocessed tokens
    X_train_str = [" ".join(tokens) for tokens in X_train_tokens]
    X_val_str = [" ".join(tokens) for tokens in X_val_tokens]

    print("[INFO] Fitting TF-IDF vectorizer for weighted pooling...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train_str)
    X_val_tfidf_matrix = tfidf_vectorizer.transform(X_val_str)

    # 4. Train Word2Vec models: CBOW vs Skip-Gram, 100 vs 200 dimensions
    configs = [
        {"name": "CBOW_100", "vector_size": 100, "sg": 0},
        {"name": "SKIPGRAM_100", "vector_size": 100, "sg": 1},
        {"name": "CBOW_200", "vector_size": 200, "sg": 0},
        {"name": "SKIPGRAM_200", "vector_size": 200, "sg": 1},
    ]

    models = {}

    for config in configs:
        print("\n" + "=" * 60)
        print(f"[INFO] Training {config['name']}")
        print("=" * 60)

        model = Word2Vec(
            sentences=X_train_tokens,
            vector_size=config["vector_size"],
            window=5,
            min_count=2,
            sg=config["sg"],
            seed=42,
            workers=4
        )

        models[config["name"]] = model

        print(f"[SUCCESS] {config['name']} trained.")
        print(f"[INFO] Vocabulary size: {len(model.wv.key_to_index)}")

    # Select one 100-dimensional custom Word2Vec model for downstream comparison with GloVe-100
    w2v_model = models["CBOW_100"]

    # 5. Evaluate embeddings qualitatively with similar words
    sanity_words = ["good", "bad", "love", "hate", "movie", "market", "stock"]

    similarity_results = []

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"SIMILAR WORDS - {model_name}")
        print("=" * 60)

        for word in sanity_words:
            if word not in model.wv.key_to_index:
                print(f"{word}: OOV")
                continue

            print(f"\nMost similar to '{word}':")

            for similar_word, score in model.wv.most_similar(word, topn=10):
                print(f"{similar_word:<15} {score:.4f}")

                similarity_results.append({
                    "model": model_name,
                    "query_word": word,
                    "similar_word": similar_word,
                    "similarity": score
                })

    os.makedirs("outputs", exist_ok=True)

    pd.DataFrame(similarity_results).to_csv(
        "outputs/word2vec_similarity.csv",
        index=False
    )

    # 6. Load pre-trained GloVe Twitter Embeddings
    print("[INFO] Loading pre-trained GloVe Twitter 100-dimensional embeddings...")
    print("[INFO] This might download up to 400MB if not cached locally...")

    try:
        glove_model = api.load("glove-twitter-100")
        print("[SUCCESS] Pre-trained GloVe model loaded successfully!")
        print(f"[INFO] GloVe Vocabulary Size: {len(glove_model.key_to_index)}")
    except Exception as e:
        print(f"[ERROR] Failed to load GloVe-Twitter-100 embeddings: {str(e)}")
        sys.exit(1)

    # 7. Compute Out-of-Vocabulary (OOV) Statistics on Validation Fold
    print("[INFO] Computing Out-of-Vocabulary (OOV) statistics on validation set...")

    w2v_uniq_oov, w2v_tok_oov, w2v_oov_words = calculate_oov(X_val_tokens, w2v_model)
    glove_uniq_oov, glove_tok_oov, glove_oov_words = calculate_oov(X_val_tokens, glove_model)

    print("-" * 60)
    print("OUT-OF-VOCABULARY (OOV) COMPARISON:")
    print("-" * 60)
    print(f"Custom Word2Vec Unique Word OOV Rate:   {w2v_uniq_oov * 100:.2f}%")
    print(f"Custom Word2Vec Token OOV Rate:         {w2v_tok_oov * 100:.2f}%")
    print(f"GloVe-Twitter-100 Unique Word OOV Rate: {glove_uniq_oov * 100:.2f}%")
    print(f"GloVe-Twitter-100 Token OOV Rate:       {glove_tok_oov * 100:.2f}%")
    print("-" * 60)

    print("[INFO] Sample OOV Words in Word2Vec:")
    print(list(w2v_oov_words)[:15])

    print("[INFO] Sample OOV Words in GloVe-Twitter-100:")
    print(list(glove_oov_words)[:15])
    print("-" * 60)

    pd.DataFrame([
        {
            "model": "Custom Word2Vec CBOW_100",
            "unique_oov_rate": w2v_uniq_oov,
            "token_oov_rate": w2v_tok_oov,
            "sample_oov_words": list(w2v_oov_words)[:15]
        },
        {
            "model": "GloVe-Twitter-100",
            "unique_oov_rate": glove_uniq_oov,
            "token_oov_rate": glove_tok_oov,
            "sample_oov_words": list(glove_oov_words)[:15]
        }
    ]).to_csv("outputs/oov_comparison.csv", index=False)

    # 8. Vectorize Dataset via Mean Pooling and TF-IDF Weighted Pooling
    print("[INFO] Creating mean pooled vector representations...")

    X_train_w2v_mean = vectorize_corpus(X_train_tokens, w2v_model, 100)
    X_val_w2v_mean = vectorize_corpus(X_val_tokens, w2v_model, 100)

    X_train_glove_mean = vectorize_corpus(X_train_tokens, glove_model, 100)
    X_val_glove_mean = vectorize_corpus(X_val_tokens, glove_model, 100)

    print("[INFO] Creating TF-IDF weighted pooled vector representations...")

    X_train_w2v_tfidf = vectorize_corpus_tfidf(
        X_train_tokens,
        w2v_model,
        X_train_tfidf_matrix,
        tfidf_vectorizer,
        100
    )

    X_val_w2v_tfidf = vectorize_corpus_tfidf(
        X_val_tokens,
        w2v_model,
        X_val_tfidf_matrix,
        tfidf_vectorizer,
        100
    )

    X_train_glove_tfidf = vectorize_corpus_tfidf(
        X_train_tokens,
        glove_model,
        X_train_tfidf_matrix,
        tfidf_vectorizer,
        100
    )

    X_val_glove_tfidf = vectorize_corpus_tfidf(
        X_val_tokens,
        glove_model,
        X_val_tfidf_matrix,
        tfidf_vectorizer,
        100
    )

    embedding_sets = [
        ("Word2Vec Mean Pooling", X_train_w2v_mean, X_val_w2v_mean),
        ("Word2Vec TF-IDF Weighted Pooling", X_train_w2v_tfidf, X_val_w2v_tfidf),
        ("GloVe-Twitter-100 Mean Pooling", X_train_glove_mean, X_val_glove_mean),
        ("GloVe-Twitter-100 TF-IDF Weighted Pooling", X_train_glove_tfidf, X_val_glove_tfidf),
    ]

    print("\n" + "=" * 60)
    print("TM-020 DIAGNOSTICS: SHAPES AND NORMS")
    print("=" * 60)

    for name, X_train_emb, X_val_emb in embedding_sets:
        print_embedding_diagnostics(name, X_train_emb, X_val_emb)

    print("\n[INFO] Small sample validation:")
    print(f"Original tweet: {X_train_raw.iloc[0]}")
    print(f"Tokens: {X_train_tokens[0]}")
    print(f"Word2Vec Mean first 5 dims: {X_train_w2v_mean[0][:5]}")
    print(f"Word2Vec TF-IDF first 5 dims: {X_train_w2v_tfidf[0][:5]}")

    # 9. Train and Evaluate Downstream Classifiers
    print("\n" + "=" * 60)
    print("TM-021 CLASSIFIER EVALUATION")
    print("=" * 60)

    for feature_desc, X_train_emb, X_val_emb in embedding_sets:

        train_and_evaluate_classifier(
            X_train=X_train_emb,
            X_val=X_val_emb,
            y_train=y_train,
            y_val=y_val,
            classifier=LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            model_name="Logistic Regression",
            feature_desc=feature_desc,
            params="penalty=l2, solver=lbfgs, max_iter=1000, class_weight=balanced"
        )

        train_and_evaluate_classifier(
            X_train=X_train_emb,
            X_val=X_val_emb,
            y_train=y_train,
            y_val=y_val,
            classifier=MLPClassifier(
                hidden_layer_sizes=(128,),
                activation='relu',
                solver='adam',
                max_iter=300,
                random_state=42
            ),
            model_name="MLP",
            feature_desc=feature_desc,
            params="hidden_layer_sizes=(128,), activation=relu, solver=adam, max_iter=300"
        )

    print("\n" + "=" * 60)
    print("WORD EMBEDDINGS COMPARISON PIPELINE COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
