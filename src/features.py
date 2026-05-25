import os
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import (
    TRAIN_CSV_PATH,
    BOW_TRAIN_PATH, BOW_VAL_PATH,
    TFIDF_UNI_TRAIN_PATH, TFIDF_UNI_VAL_PATH,
    TFIDF_OPT_TRAIN_PATH, TFIDF_OPT_VAL_PATH,
)
from src.preprocessing import stratified_split
from src.preprocessing import preprocess_tweet
from src.utils import log_info, log_success, print_header


def extract_and_save_features() -> None:
    """Fits BoW, TF-IDF unigram, and optimized TF-IDF (1,2) vectorizers and saves sparse matrices."""
    print_header("FEATURE EXTRACTION PIPELINE (BoW & TF-IDF)")

    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Training CSV not found at: {TRAIN_CSV_PATH}")

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    log_info(f"Loaded {len(train_df)} rows from {TRAIN_CSV_PATH}")

    X_train, X_val, y_train, y_val = stratified_split(train_df)
    log_info(f"Split: Train={len(X_train)} | Val={len(X_val)}")

    log_info("Preprocessing texts ...")
    X_train_pre = X_train.apply(lambda t: preprocess_tweet(t, return_str=True))
    X_val_pre = X_val.apply(lambda t: preprocess_tweet(t, return_str=True))

    os.makedirs(os.path.dirname(BOW_TRAIN_PATH), exist_ok=True)

    bow_vec = CountVectorizer(ngram_range=(1, 1))
    X_train_bow = bow_vec.fit_transform(X_train_pre)
    X_val_bow = bow_vec.transform(X_val_pre)
    log_info(f"BoW vocab size: {len(bow_vec.vocabulary_)}")
    sp.save_npz(BOW_TRAIN_PATH, X_train_bow)
    sp.save_npz(BOW_VAL_PATH, X_val_bow)
    log_success(f"BoW saved → {BOW_TRAIN_PATH}, {BOW_VAL_PATH}")

    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
    X_train_tfidf = tfidf_vec.fit_transform(X_train_pre)
    X_val_tfidf = tfidf_vec.transform(X_val_pre)
    log_info(f"TF-IDF unigram vocab size: {len(tfidf_vec.vocabulary_)}")
    sp.save_npz(TFIDF_UNI_TRAIN_PATH, X_train_tfidf)
    sp.save_npz(TFIDF_UNI_VAL_PATH, X_val_tfidf)
    log_success(f"TF-IDF unigrams saved → {TFIDF_UNI_TRAIN_PATH}, {TFIDF_UNI_VAL_PATH}")

    tfidf_opt_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
    X_train_tfidf_opt = tfidf_opt_vec.fit_transform(X_train_pre)
    X_val_tfidf_opt = tfidf_opt_vec.transform(X_val_pre)
    log_info(f"TF-IDF optimized vocab size: {len(tfidf_opt_vec.vocabulary_)}")
    sp.save_npz(TFIDF_OPT_TRAIN_PATH, X_train_tfidf_opt)
    sp.save_npz(TFIDF_OPT_VAL_PATH, X_val_tfidf_opt)
    log_success(f"TF-IDF optimized saved → {TFIDF_OPT_TRAIN_PATH}, {TFIDF_OPT_VAL_PATH}")

    log_success("Feature extraction complete.")
