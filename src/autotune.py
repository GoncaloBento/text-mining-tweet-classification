#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/autotune.py

Automated ML baseline hyperparameter search orchestrator.
1. Parsesoutputs/results.csv to find the model variant with the highest validation Macro F1 score.
2. Dynamically reconstructs the winning vectorizer and model architecture.
3. Fits the winning configuration on 100% of the training data (train + val combined).
4. Generates predictions on the test set (data/test.csv) and saves outputs to outputs/pred_best.csv.
"""

import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess_tweet

RESULTS_CSV_PATH = "outputs/results.csv"
TRAIN_CSV_PATH = "data/train.csv"
TEST_CSV_PATH = "data/test.csv"
OUTPUT_PRED_PATH = "outputs/pred_best.csv"

def get_best_model_config():
    """Reads leaderboard results and extracts the row with the highest f1_macro."""
    if not os.path.exists(RESULTS_CSV_PATH):
        print(f"[ERROR] Leaderboard file not found at {RESULTS_CSV_PATH}. Run experiment.py first!")
        sys.exit(1)
        
    df = pd.read_csv(RESULTS_CSV_PATH)
    if len(df) == 0:
        print("[ERROR] Leaderboard is empty. Run experiment.py first!")
        sys.exit(1)
        
    # Sort by f1_macro (our primary performance metric)
    df_sorted = df.sort_values(by="f1_macro", ascending=False)
    best_run = df_sorted.iloc[0]
    
    print("\n" + "=" * 80)
    print("[LEADERBOARD] AUTO-TUNING: LOCATED WINNING PIPELINE CONFIGURATION")
    print("=" * 80)
    print(f"Model Class:        {best_run['model_name']}")
    print(f"Feature Space:      {best_run['feature_description']}")
    print(f"Parameters:         {best_run['parameters']}")
    print(f"Validation F1-Macro: {best_run['f1_macro']:.4f}")
    print(f"Validation Accuracy: {best_run['accuracy']:.4f}")
    print("=" * 80)
    
    return best_run

def instantiate_pipeline(config):
    """Dynamically reconstructs and returns the vectorizer and classifier model based on config."""
    feat_desc = str(config["feature_description"]).lower()
    
    # 1. Instantiate Vectorizer
    if "optimized" in feat_desc or "opt" in feat_desc:
        print("[INFO] Reconstructing Vectorizer: TF-IDF (1,2) Optimized (ngram_range=(1,2), min_df=2, max_features=25000)")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
    elif "(1,2)" in feat_desc:
        print("[INFO] Reconstructing Vectorizer: TF-IDF (1,2) Raw (ngram_range=(1,2), min_df=1)")
        vec = TfidfVectorizer(ngram_range=(1, 2))
    else:
        print("[INFO] Reconstructing Vectorizer: TF-IDF (1,1) Unigrams (ngram_range=(1,1), min_df=1)")
        vec = TfidfVectorizer(ngram_range=(1, 1))
        
    # 2. Instantiate Classifier
    model_name = str(config["model_name"]).lower()
    params = str(config["parameters"]).lower()
    
    if "logistic regression" in model_name:
        if "penalty=l1" in params or "l1" in params:
            print("[INFO] Reconstructing Classifier: L1 Lasso Logistic Regression")
            model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, class_weight='balanced', random_state=42)
        else:
            print("[INFO] Reconstructing Classifier: L2 Ridge Logistic Regression")
            model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
            
    elif "knn" in model_name:
        if "n_neighbors=7" in params or "=7" in params:
            print("[INFO] Reconstructing Classifier: KNN Classifier (k=7)")
            model = KNeighborsClassifier(n_neighbors=7)
        else:
            print("[INFO] Reconstructing Classifier: KNN Classifier (k=3)")
            model = KNeighborsClassifier(n_neighbors=3)
            
    elif "multinomial nb" in model_name:
        if "alpha=0.1" in params or "0.1" in params:
            print("[INFO] Reconstructing Classifier: Multinomial NB (alpha=0.1)")
            model = MultinomialNB(alpha=0.1)
        else:
            print("[INFO] Reconstructing Classifier: Multinomial NB (alpha=1.0)")
            model = MultinomialNB(alpha=1.0)
            
    elif "mlp" in model_name:
        if "(100, 50)" in params or "100,50" in params:
            print("[INFO] Reconstructing Classifier: Deep MLP Classifier (hidden_layer_sizes=(100, 50))")
            model = MLPClassifier(hidden_layer_sizes=(100, 50), learning_rate_init=0.001, early_stopping=True, random_state=42)
        else:
            print("[INFO] Reconstructing Classifier: Shallow MLP Classifier (hidden_layer_sizes=(100,))")
            model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.01, early_stopping=True, random_state=42)
            
    elif "random forest" in model_name:
        if "max_depth=20" in params or "20" in params:
            print("[INFO] Reconstructing Classifier: Deep Random Forest Classifier (n_estimators=200, max_depth=20)")
            model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        else:
            print("[INFO] Reconstructing Classifier: Shallow Random Forest Classifier (n_estimators=100, max_depth=10)")
            model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
            
    elif "xgboost" in model_name:
        if "max_depth=6" in params or "6" in params:
            print("[INFO] Reconstructing Classifier: Deep XGBoost Classifier (max_depth=6, lr=0.05)")
            model = XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=200, random_state=42, n_jobs=-1)
        else:
            print("[INFO] Reconstructing Classifier: Shallow XGBoost Classifier (max_depth=3, lr=0.1)")
            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=150, random_state=42, n_jobs=-1)
            
    else:
        print(f"[WARNING] Unknown model name '{config['model_name']}'. Defaulting to standard Ridge Logistic Regression.")
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        
    return vec, model

def main():
    print("=" * 60)
    print("AUTOMATED HYPERPARAMETER RETRAINING & INFERENCE SYSTEM")
    print("=" * 60)
    
    # 1. Retrieve the winning configuration
    best_config = get_best_model_config()
    vec, model = instantiate_pipeline(best_config)
    
    # 2. Load 100% Training Dataset
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"[ERROR] Training CSV not found at {TRAIN_CSV_PATH}")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    
    # 3. Apply Text Preprocessing on 100% Training Dataset
    print("\n[STEP 1/4] Preprocessing 100% training text data...")
    X_train_pre = train_df["text"].apply(lambda t: preprocess_tweet(t, return_str=True))
    y_train = train_df["label"]
    
    # 4. Vectorize Complete Training Set
    print("[STEP 2/4] Vectorizing preprocessed texts...")
    X_train_vec = vec.fit_transform(X_train_pre)
    
    # 5. Fit Winning Classifier Model
    print(f"[STEP 3/4] Fitting winning model on 100% of training data (N={len(train_df)})...")
    model.fit(X_train_vec, y_train)
    print("[SUCCESS] Fitting complete!")
    
    # 6. Load Test Dataset for Inference
    if not os.path.exists(TEST_CSV_PATH):
        print(f"[ERROR] Test CSV not found at {TEST_CSV_PATH}")
        sys.exit(1)
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # 7. Preprocess & Vectorize Test Data
    print("\n[STEP 4/4] Running inference on test set (N={})...".format(len(test_df)))
    X_test_pre = test_df["text"].apply(lambda t: preprocess_tweet(t, return_str=True))
    X_test_vec = vec.transform(X_test_pre)
    
    # 8. Generate Predictions
    y_pred_test = model.predict(X_test_vec)
    
    # 9. Output final formatted CSV predictions
    output_df = pd.DataFrame({
        "id": test_df["id"],
        "label": y_pred_test
    })
    
    os.makedirs(os.path.dirname(OUTPUT_PRED_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PRED_PATH, index=False)
    print("\n" + "=" * 80)
    print(f"[SUCCESS] OPTIMIZED PREDICTIONS GENERATED SUCCESSFULLY!")
    print(f"File Saved: {OUTPUT_PRED_PATH}")
    print(f"Total Rows: {len(output_df)}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
