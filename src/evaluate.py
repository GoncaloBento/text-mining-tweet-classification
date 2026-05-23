#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/evaluate.py

Evaluation utilities for the Text Mining sentiment models.
Implements unified scoring (Recall, Precision, Accuracy, F1-Score)
and logs all model runs to outputs/results.csv for rolling leaderboard.
Includes IDEMPOTENT logging to avoid duplicate rows for the same run.
"""

import os
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

RESULTS_CSV_PATH = "outputs/results.csv"

def compute_metrics(y_true, y_pred) -> dict:
    """
    Computes standard classification metrics: Accuracy, Precision, Recall, and F1-Score (Macro).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    }

def log_model_run(model_name: str, feature_desc: str, metrics: dict, params: str = ""):
    """
    Logs a model run to outputs/results.csv in an IDEMPOTENT way.
    If a run with the same model_name, feature_description, and parameters already exists,
    it updates the metrics and timestamp of that existing row instead of appending.
    """
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    
    headers = [
        "timestamp", "model_name", "feature_description", 
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "parameters"
    ]
    
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "feature_description": feature_desc,
        "accuracy": f"{metrics['accuracy']:.4f}",
        "precision_macro": f"{metrics['precision_macro']:.4f}",
        "recall_macro": f"{metrics['recall_macro']:.4f}",
        "f1_macro": f"{metrics['f1_macro']:.4f}",
        "parameters": params
    }
    
    existing_rows = []
    updated = False
    
    # Read existing rows if file exists
    if os.path.exists(RESULTS_CSV_PATH) and os.path.getsize(RESULTS_CSV_PATH) > 0:
        try:
            with open(RESULTS_CSV_PATH, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictWriter(f, fieldnames=headers)
                # Read rows (skipping header row)
                raw_reader = csv.reader(f)
                header_row = next(raw_reader, None)
                if header_row:
                    for r in raw_reader:
                        if len(r) == len(headers):
                            row_dict = dict(zip(headers, r))
                            # Check for unique run match
                            if (row_dict["model_name"] == model_name and 
                                row_dict["feature_description"] == feature_desc and 
                                row_dict["parameters"] == params):
                                # Update existing row
                                existing_rows.append(new_row)
                                updated = True
                            else:
                                existing_rows.append(row_dict)
        except Exception as e:
            print(f"[WARNING] Error reading leaderboard CSV: {str(e)}. Resetting leaderboard file.")
            existing_rows = []
            
    if not updated:
        existing_rows.append(new_row)
        
    # Write back all rows
    with open(RESULTS_CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(existing_rows)
        
    if updated:
        print(f"[LOGGED] Idempotent Update: Overwrote existing run in {RESULTS_CSV_PATH} successfully!")
    else:
        print(f"[LOGGED] Added new run to {RESULTS_CSV_PATH} successfully!")

def evaluate_and_log(y_true, y_pred, model_name: str, feature_desc: str, params: str = "") -> dict:
    """
    Evaluates model predictions, prints classification report & confusion matrix,
    and logs metrics to outputs/results.csv in an idempotent way.
    """
    metrics = compute_metrics(y_true, y_pred)
    
    print("=" * 60)
    print(f"MODEL EVALUATION: {model_name} ({feature_desc})")
    print("=" * 60)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Bearish", "Bullish", "Neutral"], zero_division=0))
    print("-" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("=" * 60)
    
    log_model_run(model_name, feature_desc, metrics, params)
    return metrics
