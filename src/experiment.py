import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.evaluate import evaluate_and_log
from src.config import SEED, LABEL_NAMES


def run_tfidf_pipeline(
    X_train_preprocessed, X_val_preprocessed, y_train, y_val,
    ngram_range=(1, 1), min_df=1, max_features=None,
    feature_desc="TF-IDF",
) -> tuple[int, dict]:
    """Fits TF-IDF + LR L2, evaluates on val, logs idempotently. Returns (vocab_size, metrics)."""
    params_str = f"ngram_range={ngram_range}, min_df={min_df}, max_features={max_features}"

    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features)
    X_train_vec = vec.fit_transform(X_train_preprocessed)
    X_val_vec = vec.transform(X_val_preprocessed)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(X_train_vec, y_train)

    metrics = evaluate_and_log(
        y_val, lr.predict(X_val_vec),
        model_name="Logistic Regression Baseline",
        feature_desc=feature_desc,
        params=params_str,
    )
    return len(vec.vocabulary_), metrics


def run_model_pipeline(
    X_train_vec, X_val_vec, y_train, y_val,
    model, model_name: str, feature_desc: str, params_str: str,
) -> dict:
    """Fits a classifier on pre-vectorized features, evaluates on val, logs idempotently."""
    model.fit(X_train_vec, y_train)
    return evaluate_and_log(
        y_val, model.predict(X_val_vec),
        model_name=model_name,
        feature_desc=feature_desc,
        params=params_str,
    )


def run_majority_baseline(y_train, y_val) -> dict:
    """Evaluates the majority-class baseline on the validation set and logs metrics."""
    majority_class = int(pd.Series(y_train).value_counts().idxmax())
    y_pred = np.full(len(y_val), majority_class)
    return evaluate_and_log(
        y_val, y_pred,
        model_name="Majority Class Baseline",
        feature_desc="N/A",
        params=f"majority_class={majority_class} ({LABEL_NAMES[majority_class]})",
    )
