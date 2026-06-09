import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RESULTS_CSV_PATH, SEED, TRAIN_CSV_PATH, TEST_CSV_PATH
from src.utils import log_info, log_warning, print_header, log_success
from src.preprocessing import preprocess_tweet
from src.evaluate import save_submission


def _build_vectorizer(feature_desc: str) -> TfidfVectorizer:
    fd = feature_desc.lower()
    if "optimized" in fd or "opt" in fd:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
    if "(1,2)" in fd:
        return TfidfVectorizer(ngram_range=(1, 2))
    return TfidfVectorizer(ngram_range=(1, 1))


def _build_classifier(model_name: str, params: str):
    name = model_name.lower()
    p = params.lower()

    if "logistic regression" in name:
        penalty = "l1" if "penalty=l1" in p else "l2"
        solver = "saga" if penalty == "l1" else "lbfgs"
        return LogisticRegression(penalty=penalty, solver=solver, max_iter=1000,
                                  class_weight="balanced", random_state=SEED)

    if "knn" in name:
        m = re.search(r"n_neighbors=(\d+)", p)
        k = int(m.group(1)) if m else 3
        return KNeighborsClassifier(n_neighbors=k)

    if "multinomial nb" in name:
        m = re.search(r"alpha=([\d.]+)", p)
        alpha = float(m.group(1)) if m else 1.0
        return MultinomialNB(alpha=alpha)

    if "mlp" in name:
        if "100, 50" in p or "100,50" in p:
            return MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.001,
                                 early_stopping=True, random_state=SEED)
        return MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01,
                             early_stopping=True, random_state=SEED)

    if "random forest" in name:
        m = re.search(r"max_depth=(\d+)", p)
        depth = int(m.group(1)) if m else 10
        n_est = 200 if depth >= 20 else 100
        return RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)

    if "xgboost" in name:
        md = re.search(r"max_depth=(\d+)", p)
        lr = re.search(r"learning_rate=([\d.]+)", p)
        depth = int(md.group(1)) if md else 3
        rate = float(lr.group(1)) if lr else 0.1
        n_est = 200 if depth >= 6 else 150
        return XGBClassifier(max_depth=depth, learning_rate=rate, n_estimators=n_est,
                             random_state=SEED, n_jobs=-1)

    log_warning(f"Unknown model '{model_name}', defaulting to LR L2.")
    return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000,
                              class_weight="balanced", random_state=SEED)


def get_best_model_config() -> pd.Series:
    """Returns the leaderboard row with the highest f1_macro."""
    if not os.path.exists(RESULTS_CSV_PATH):
        raise FileNotFoundError(f"Leaderboard not found at {RESULTS_CSV_PATH}. Run experiments first.")
    df = pd.read_csv(RESULTS_CSV_PATH)
    if df.empty:
        raise ValueError("Leaderboard is empty. Run experiments first.")

    best = df.sort_values("f1_macro", ascending=False).iloc[0]
    print_header("BEST PIPELINE FROM LEADERBOARD")
    log_info(f"Model    : {best['model_name']}")
    log_info(f"Features : {best['feature_description']}")
    log_info(f"Params   : {best['parameters']}")
    log_info(f"Val F1   : {float(best['f1_macro']):.4f}")
    return best


def instantiate_pipeline(config: pd.Series) -> tuple:
    """Returns (vectorizer, classifier) reconstructed from a leaderboard row."""
    vec = _build_vectorizer(config["feature_description"])
    clf = _build_classifier(config["model_name"], config["parameters"])
    log_info(f"Vectorizer : {vec.__class__.__name__} {vec.get_params()}")
    log_info(f"Classifier : {clf.__class__.__name__}")
    return vec, clf

if __name__ == "__main__":
    best_config = get_best_model_config()
    model_name_upper = str(best_config["model_name"]).upper()
    feat_desc = str(best_config["feature_description"])

    # Check if Deep Learning Model
    if "HF FINE-TUNE" in feat_desc.upper() or "BERT" in model_name_upper:
        log_info("Deep Learning model detected. Dispatching to HF trainer...")
        if "DISTILBERT" in model_name_upper:
            import src.distilbert_trainer as trainer_module
        elif "FINBERT" in model_name_upper:
            import src.finbert_trainer as trainer_module
        elif "ROBERTA" in model_name_upper:
            import src.roberta_trainer as trainer_module
        else:
            raise ValueError(f"Unsupported HF model: {best_config['model_name']}")
            
        trainer = trainer_module.run_trainer(n_samples=None)
        tokenizer = trainer_module.load_tokenizer()
        preds = trainer_module.predict_test_set(trainer, tokenizer)
        test_df = pd.read_csv(TEST_CSV_PATH)
        save_submission(test_df, preds)
    else:
        log_info("Classical ML model detected. Using scikit-learn logic...")
        vec, clf = instantiate_pipeline(best_config)
        
        log_info("Loading 100% training data...")
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        y_train = train_df["label"]
        
        log_info("Preprocessing...")
        X_train_pre = train_df["text"].apply(lambda t: preprocess_tweet(t, return_str=True))
        
        log_info("Vectorizing...")
        X_train_vec = vec.fit_transform(X_train_pre)
        
        log_info("Fitting classical model...")
        clf.fit(X_train_vec, y_train)
        
        log_info("Predicting test data...")
        test_df = pd.read_csv(TEST_CSV_PATH)
        X_test_pre = test_df["text"].apply(lambda t: preprocess_tweet(t, return_str=True))
        X_test_vec = vec.transform(X_test_pre)
        preds = clf.predict(X_test_vec)
        
        save_submission(test_df, preds)

