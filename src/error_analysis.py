#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/error_analysis.py

Performs deep error analysis on the champion classical baseline model:
- Fits Logistic Regression L2 on TF-IDF (1,2) Optimized features.
- Generates and saves confusion matrix plots to outputs/confusion_matrix.png.
- Extracts prediction probabilities and isolates the top 20 most confident/severe
  misclassifications for each actual sentiment class (Bearish, Bullish, Neutral).
- Outputs human-readable (TXT) and agent-parseable (JSON) analytical reports.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.train_val_split import stratified_split
from src.preprocessing import preprocess_tweet
from src.evaluate import compute_metrics

TRAIN_CSV_PATH = "data/train.csv"
CONF_MATRIX_PLOT_PATH = "outputs/confusion_matrix.png"
MISCLASSIFIED_TXT_PATH = "outputs/misclassified_report.txt"
MISCLASSIFIED_JSON_PATH = "outputs/misclassified_analysis.json"

LABEL_MAPPING = {0: "Bearish", 1: "Bullish", 2: "Neutral"}

def main():
    print("=" * 60)
    print("RUNNING SENTIMENT ERROR ANALYSIS PIPELINE...")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"[ERROR] Training CSV not found at {TRAIN_CSV_PATH}")
        sys.exit(1)
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # 2. Train/Val Split
    X_train_raw, X_val_raw, y_train, y_val = stratified_split(train_df)
    
    # Keep copies of raw validation series with original indices
    val_indices = X_val_raw.index

    # 3. Preprocess Texts
    print("[INFO] Preprocessing training & validation texts (lemmatizer active)...")
    X_train_pre = X_train_raw.apply(lambda t: preprocess_tweet(t, return_str=True))
    X_val_pre = X_val_raw.apply(lambda t: preprocess_tweet(t, return_str=True))

    # 4. TF-IDF Vectorization (Standardized Champion Configuration)
    print("[INFO] Vectorizing features using Optimized TF-IDF representation...")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
    X_train_vec = vec.fit_transform(X_train_pre)
    X_val_vec = vec.transform(X_val_pre)

    # 5. Fit model
    print("[INFO] Training champion classifier: Logistic Regression L2...")
    model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_vec, y_train)

    # 6. Predict on Validation Set
    y_pred = model.predict(X_val_vec)
    y_proba = model.predict_proba(X_val_vec)
    
    metrics = compute_metrics(y_val, y_pred)
    print("-" * 60)
    print(f"Validation F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print("-" * 60)

    # 7. Generate and Save Confusion Matrix Plot
    print(f"[INFO] Plotting confusion matrix heatmap to {CONF_MATRIX_PLOT_PATH}...")
    cm = confusion_matrix(y_val, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Bearish (0)", "Bullish (1)", "Neutral (2)"],
        yticklabels=["Bearish (0)", "Bullish (1)", "Neutral (2)"]
    )
    plt.title("Confusion Matrix Heatmap\nLogistic Regression Baseline (Optimized TF-IDF)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(CONF_MATRIX_PLOT_PATH), exist_ok=True)
    plt.savefig(CONF_MATRIX_PLOT_PATH, dpi=150)
    plt.close()
    print("[SUCCESS] Heatmap plot saved successfully!")

    # 8. Isolate Misclassifications & Confidences
    print("[INFO] Analyzing misclassification confidences...")
    
    error_list = []
    
    # Reconstruct validation rows with prediction details
    for idx, (original_idx, actual, predicted, probas) in enumerate(zip(val_indices, y_val, y_pred, y_proba)):
        if actual != predicted:
            raw_text = X_val_raw.loc[original_idx]
            clean_text = X_val_pre.iloc[idx]
            
            # Confidence of the wrong prediction is the max proba
            max_proba = float(probas[predicted])
            actual_proba = float(probas[actual])
            
            error_list.append({
                "original_index": int(original_idx),
                "tweet_text": str(raw_text),
                "cleaned_text": str(clean_text),
                "actual_label": int(actual),
                "actual_name": LABEL_MAPPING[actual],
                "predicted_label": int(predicted),
                "predicted_name": LABEL_MAPPING[predicted],
                "confidence": max_proba,
                "actual_class_proba": actual_proba,
                "probabilities": {LABEL_MAPPING[i]: float(p) for i, p in enumerate(probas)}
            })

    # Group errors by actual class and sort by confidence descending
    errors_df = pd.DataFrame(error_list)
    top_errors_per_class = {}
    
    for actual_class in [0, 1, 2]:
        class_name = LABEL_MAPPING[actual_class]
        class_errors = errors_df[errors_df["actual_label"] == actual_class]
        class_errors_sorted = class_errors.sort_values(by="confidence", ascending=False)
        top_errors_per_class[class_name] = class_errors_sorted.head(20).to_dict(orient="records")

    # 9. Output Human-Readable TXT Report
    print(f"[INFO] Writing human-readable report to {MISCLASSIFIED_TXT_PATH}...")
    with open(MISCLASSIFIED_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("SENTIMENT CLASSIFICATION DEEP ERROR ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Model Configuration: Logistic Regression Baseline (L2 Regularization)\n")
        f.write(f"Standard Feature Space: TF-IDF (1,2) Optimized (N_features=25,000)\n")
        f.write(f"Validation Set Size: N={len(y_val)}\n")
        f.write(f"Total Errors Found: N={len(errors_df)} (Error Rate: {len(errors_df)/len(y_val)*100:.2f}%)\n\n")
        
        f.write("CONFUSION MATRIX DETAILS:\n")
        f.write(f"Bearish (0) - Predicted: Bearish={cm[0][0]}, Bullish={cm[0][1]}, Neutral={cm[0][2]}\n")
        f.write(f"Bullish (1) - Predicted: Bearish={cm[1][0]}, Bullish={cm[1][1]}, Neutral={cm[1][2]}\n")
        f.write(f"Neutral (2) - Predicted: Bearish={cm[2][0]}, Bullish={cm[2][1]}, Neutral={cm[2][2]}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("TOP 20 MOST CONFIDENT/SEVERE MISCLASSIFICATIONS BY ACTUAL CLASS\n")
        f.write("=" * 100 + "\n\n")
        
        for class_name, errors in top_errors_per_class.items():
            f.write(f"{"#" * 40}\n")
            f.write(f"ACTUAL CLASS: {class_name.upper()}\n")
            f.write(f"{"#" * 40}\n\n")
            
            if not errors:
                f.write("No errors found in this class.\n\n")
                continue
                
            for i, err in enumerate(errors, 1):
                f.write(f"{i}. Tweet Text: \"{err['tweet_text']}\"\n")
                f.write(f"   Clean Text: \"{err['cleaned_text']}\"\n")
                f.write(f"   Original DF Index: {err['original_index']}\n")
                f.write(f"   Actual: {err['actual_name']} (p={err['actual_class_proba']:.4f})\n")
                f.write(f"   Predicted: {err['predicted_name']} (Confidence={err['confidence']:.4f})\n")
                f.write(f"   Probas: {err['probabilities']}\n")
                f.write("-" * 80 + "\n")
            f.write("\n")

    # 10. Output Programmatic JSON Report
    print(f"[INFO] Writing agent-parseable JSON analysis to {MISCLASSIFIED_JSON_PATH}...")
    with open(MISCLASSIFIED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(top_errors_per_class, f, indent=2)

    print("\n" + "=" * 80)
    print("[SUCCESS] SENTIMENT ERROR ANALYSIS COMPLETE!")
    print(f"Heatmap:      {CONF_MATRIX_PLOT_PATH}")
    print(f"Human Report: {MISCLASSIFIED_TXT_PATH}")
    print(f"JSON Report:  {MISCLASSIFIED_JSON_PATH}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
