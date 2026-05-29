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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/word_embeddings.py

TM-018:
- Train Word2Vec models: CBOW/Skip-Gram, 100/200 dimensions
- Evaluate similar words

TM-019:
- Load GloVe-Twitter-100
- Compare OOV rates against custom Word2Vec

TM-020:
- Build sentence embeddings with:
  - mean pooling
  - TF-IDF weighted pooling

TM-021:
- Train and evaluate:
  - Logistic Regression
  - MLP
"""

import os
import sys
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
import gensim.downloader as api

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.train_val_split import stratified_split
from src.preprocessing import preprocess_tweet
from src.evaluate import evaluate_and_log


TRAIN_CSV_PATH = "data/train.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "outputs/models"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_vocab_and_vectors(embedding_model):
    """Supports both Gensim Word2Vec and GloVe KeyedVectors."""
    if hasattr(embedding_model, "key_to_index"):
        return embedding_model.key_to_index, embedding_model

    if hasattr(embedding_model, "wv") and hasattr(embedding_model.wv, "key_to_index"):
        return embedding_model.wv.key_to_index, embedding_model.wv

    raise ValueError("Unsupported embedding model format.")


def calculate_oov(tokenized_sentences, embedding_model):
    """Computes unique-word and token-level OOV rates."""
    vocab, _ = get_vocab_and_vectors(embedding_model)

    all_tokens = [tok for sent in tokenized_sentences for tok in sent]
    unique_tokens = set(all_tokens)

    if not unique_tokens:
        return 0.0, 0.0, set()

    oov_unique = {tok for tok in unique_tokens if tok not in vocab}
    oov_tokens = [tok for tok in all_tokens if tok not in vocab]

    unique_oov_rate = len(oov_unique) / len(unique_tokens)
    token_oov_rate = len(oov_tokens) / len(all_tokens)

    return unique_oov_rate, token_oov_rate, oov_unique


# ---------------------------------------------------------------------
# TM-018: Word2Vec training + similar words
# ---------------------------------------------------------------------

def train_word2vec_models(X_train_tokens):
    """Train CBOW and Skip-Gram Word2Vec models with 100 and 200 dimensions."""
    configs = [
        {"name": "CBOW_100", "vector_size": 100, "sg": 0},
        {"name": "SKIPGRAM_100", "vector_size": 100, "sg": 1},
        {"name": "CBOW_200", "vector_size": 200, "sg": 0},
        {"name": "SKIPGRAM_200", "vector_size": 200, "sg": 1},
    ]

    os.makedirs(MODEL_DIR, exist_ok=True)
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
            workers=4,
        )

        models[config["name"]] = model

        model_path = os.path.join(MODEL_DIR, f"{config['name']}.model")
        model.save(model_path)

        print(f"[SUCCESS] {config['name']} trained.")
        print(f"[INFO] Vocabulary size: {len(model.wv.key_to_index)}")
        print(f"[INFO] Saved model to {model_path}")

    return models


def evaluate_similar_words(models, sanity_words=None, topn=10):
    """Evaluate embeddings qualitatively with most_similar()."""
    if sanity_words is None:
        sanity_words = ["good", "bad", "love", "hate", "market", "stock"]

    results = []

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"SIMILAR WORDS - {model_name}")
        print("=" * 60)

        for word in sanity_words:
            if word not in model.wv.key_to_index:
                print(f"{word}: OOV")
                continue

            print(f"\nMost similar to '{word}':")

            for similar_word, score in model.wv.most_similar(word, topn=topn):
                print(f"{similar_word:<20} {score:.4f}")

                results.append({
                    "model": model_name,
                    "query_word": word,
                    "similar_word": similar_word,
                    "similarity": score,
                })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "word2vec_similarity.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\n[INFO] Similarity results saved to {output_path}")


# ---------------------------------------------------------------------
# TM-020: sentence embeddings
# ---------------------------------------------------------------------

def mean_pool_document(tokens, vocab, vectors, vector_size):
    """Simple mean pooling over token vectors."""
    valid_vectors = [vectors[token] for token in tokens if token in vocab]

    if not valid_vectors:
        return np.zeros(vector_size)

    return np.mean(valid_vectors, axis=0)


def mean_pool_corpus(tokenized_sentences, embedding_model, vector_size):
    """Create sentence embeddings using mean pooling."""
    vocab, vectors = get_vocab_and_vectors(embedding_model)

    return np.array([
        mean_pool_document(tokens, vocab, vectors, vector_size)
        for tokens in tokenized_sentences
    ])


def tfidf_weighted_pool_document(
    tokens,
    vocab,
    vectors,
    tfidf_row,
    tfidf_vocab,
    vector_size,
):
    """Weighted mean pooling using TF-IDF weights."""
    weighted_vectors = []
    weights = []

    for token in tokens:
        if token in vocab and token in tfidf_vocab:
            weight = tfidf_row[0, tfidf_vocab[token]]

            if weight > 0:
                weighted_vectors.append(vectors[token] * weight)
                weights.append(weight)

    if not weighted_vectors:
        return np.zeros(vector_size)

    return np.sum(weighted_vectors, axis=0) / np.sum(weights)


def tfidf_weighted_pool_corpus(
    tokenized_sentences,
    embedding_model,
    tfidf_matrix,
    tfidf_vectorizer,
    vector_size,
):
    """Create sentence embeddings using TF-IDF weighted mean pooling."""
    vocab, vectors = get_vocab_and_vectors(embedding_model)
    tfidf_vocab = tfidf_vectorizer.vocabulary_

    return np.array([
        tfidf_weighted_pool_document(
            tokens=tokens,
            vocab=vocab,
            vectors=vectors,
            tfidf_row=tfidf_matrix[i],
            tfidf_vocab=tfidf_vocab,
            vector_size=vector_size,
        )
        for i, tokens in enumerate(tokenized_sentences)
    ])


def print_embedding_diagnostics(name, X_train, X_val):
    """Print shape and norm checks required for TM-020."""
    print(f"\n{name}")
    print("-" * 60)
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Train norm:  {np.linalg.norm(X_train):.4f}")
    print(f"Val norm:    {np.linalg.norm(X_val):.4f}")
    print(f"NaNs:        {np.isnan(X_train).sum() + np.isnan(X_val).sum()}")


# ---------------------------------------------------------------------
# TM-021: classifiers
# ---------------------------------------------------------------------

def train_and_evaluate_classifier(
    X_train,
    X_val,
    y_train,
    y_val,
    classifier,
    model_name,
    feature_desc,
    params,
):
    """Train classifier and log metrics to outputs/results.csv."""
    print("\n" + "=" * 60)
    print(f"Training {model_name} on {feature_desc}")
    print("=" * 60)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_val)

    evaluate_and_log(
        y_val,
        y_pred,
        model_name=model_name,
        feature_desc=feature_desc,
        params=params,
    )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RUNNING WORD EMBEDDINGS PIPELINE")
    print("=" * 60)

    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"[ERROR] Training CSV not found at {TRAIN_CSV_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # 2. Train/validation split
    print("[INFO] Creating stratified train/validation split...")
    X_train_raw, X_val_raw, y_train, y_val = stratified_split(train_df)

    # 3. Preprocess
    print("[INFO] Preprocessing tweets into token lists...")
    X_train_tokens = X_train_raw.apply(
        lambda t: preprocess_tweet(t, return_str=False)
    ).tolist()

    X_val_tokens = X_val_raw.apply(
        lambda t: preprocess_tweet(t, return_str=False)
    ).tolist()

    X_train_str = [" ".join(tokens) for tokens in X_train_tokens]
    X_val_str = [" ".join(tokens) for tokens in X_val_tokens]

    # 4. TM-018: train Word2Vec models
    models = train_word2vec_models(X_train_tokens)
    evaluate_similar_words(models)

    # For TM-019/020/021 we use CBOW_100 because GloVe-Twitter-100 has 100 dimensions.
    w2v_model = models["CBOW_100"]

    # 5. TM-019: load GloVe
    print("\n" + "=" * 60)
    print("[INFO] Loading GloVe-Twitter-100")
    print("=" * 60)

    try:
        glove_model = api.load("glove-twitter-100")
        print("[SUCCESS] GloVe-Twitter-100 loaded.")
        print(f"[INFO] GloVe vocabulary size: {len(glove_model.key_to_index)}")
    except Exception as e:
        print(f"[ERROR] Failed to load GloVe-Twitter-100: {str(e)}")
        sys.exit(1)

    # 6. TM-019: OOV comparison
    print("\n" + "=" * 60)
    print("OOV COMPARISON")
    print("=" * 60)

    w2v_uniq_oov, w2v_tok_oov, w2v_oov_words = calculate_oov(
        X_val_tokens,
        w2v_model,
    )

    glove_uniq_oov, glove_tok_oov, glove_oov_words = calculate_oov(
        X_val_tokens,
        glove_model,
    )

    print(f"Custom Word2Vec unique OOV rate: {w2v_uniq_oov * 100:.2f}%")
    print(f"Custom Word2Vec token OOV rate:  {w2v_tok_oov * 100:.2f}%")
    print(f"GloVe unique OOV rate:           {glove_uniq_oov * 100:.2f}%")
    print(f"GloVe token OOV rate:            {glove_tok_oov * 100:.2f}%")

    oov_results = pd.DataFrame([
        {
            "model": "Custom Word2Vec CBOW_100",
            "unique_oov_rate": w2v_uniq_oov,
            "token_oov_rate": w2v_tok_oov,
            "sample_oov_words": list(w2v_oov_words)[:20],
        },
        {
            "model": "GloVe-Twitter-100",
            "unique_oov_rate": glove_uniq_oov,
            "token_oov_rate": glove_tok_oov,
            "sample_oov_words": list(glove_oov_words)[:20],
        },
    ])

    oov_path = os.path.join(OUTPUT_DIR, "oov_comparison.csv")
    oov_results.to_csv(oov_path, index=False)
    print(f"[INFO] OOV comparison saved to {oov_path}")

    # 7. Fit TF-IDF vectorizer for TM-020
    print("\n[INFO] Fitting TF-IDF vectorizer on training corpus...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train_str)
    X_val_tfidf_matrix = tfidf_vectorizer.transform(X_val_str)

    # 8. TM-020: sentence embeddings
    print("\n" + "=" * 60)
    print("TM-020: SENTENCE EMBEDDINGS")
    print("=" * 60)

    X_train_w2v_mean = mean_pool_corpus(
        X_train_tokens,
        w2v_model,
        vector_size=100,
    )
    X_val_w2v_mean = mean_pool_corpus(
        X_val_tokens,
        w2v_model,
        vector_size=100,
    )

    X_train_w2v_tfidf = tfidf_weighted_pool_corpus(
        X_train_tokens,
        w2v_model,
        X_train_tfidf_matrix,
        tfidf_vectorizer,
        vector_size=100,
    )
    X_val_w2v_tfidf = tfidf_weighted_pool_corpus(
        X_val_tokens,
        w2v_model,
        X_val_tfidf_matrix,
        tfidf_vectorizer,
        vector_size=100,
    )

    X_train_glove_mean = mean_pool_corpus(
        X_train_tokens,
        glove_model,
        vector_size=100,
    )
    X_val_glove_mean = mean_pool_corpus(
        X_val_tokens,
        glove_model,
        vector_size=100,
    )

    X_train_glove_tfidf = tfidf_weighted_pool_corpus(
        X_train_tokens,
        glove_model,
        X_train_tfidf_matrix,
        tfidf_vectorizer,
        vector_size=100,
    )
    X_val_glove_tfidf = tfidf_weighted_pool_corpus(
        X_val_tokens,
        glove_model,
        X_val_tfidf_matrix,
        tfidf_vectorizer,
        vector_size=100,
    )

    embedding_sets = [
        ("Word2Vec Mean Pooling", X_train_w2v_mean, X_val_w2v_mean),
        ("Word2Vec TF-IDF Weighted Pooling", X_train_w2v_tfidf, X_val_w2v_tfidf),
        ("GloVe Mean Pooling", X_train_glove_mean, X_val_glove_mean),
        ("GloVe TF-IDF Weighted Pooling", X_train_glove_tfidf, X_val_glove_tfidf),
    ]

    for name, X_train_emb, X_val_emb in embedding_sets:
        print_embedding_diagnostics(name, X_train_emb, X_val_emb)

    print("\n[INFO] Small sample validation:")
    print(f"Original tweet: {X_train_raw.iloc[0]}")
    print(f"Tokens: {X_train_tokens[0]}")
    print(f"Word2Vec mean first 5 dims: {X_train_w2v_mean[0][:5]}")
    print(f"Word2Vec TF-IDF first 5 dims: {X_train_w2v_tfidf[0][:5]}")

    # 9. TM-021: classifiers
    print("\n" + "=" * 60)
    print("TM-021: CLASSIFIER EVALUATION")
    print("=" * 60)

    for feature_desc, X_train_emb, X_val_emb in embedding_sets:

        train_and_evaluate_classifier(
            X_train=X_train_emb,
            X_val=X_val_emb,
            y_train=y_train,
            y_val=y_val,
            classifier=LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            model_name="Logistic Regression",
            feature_desc=feature_desc,
            params="penalty=l2, solver=lbfgs, class_weight=balanced",
        )

        train_and_evaluate_classifier(
            X_train=X_train_emb,
            X_val=X_val_emb,
            y_train=y_train,
            y_val=y_val,
            classifier=MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                solver="adam",
                max_iter=300,
                random_state=42,
            ),
            model_name="MLP",
            feature_desc=feature_desc,
            params="hidden_layer_sizes=(128,), activation=relu, solver=adam, max_iter=300",
        )

    print("\n" + "=" * 60)
    print("WORD EMBEDDINGS PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
