#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/evaluate.py

Evaluation utilities for the Text Mining sentiment models.
Implements unified scoring (Recall, Precision, Accuracy, F1-Score)
and logs all model runs to outputs/results.csv for rolling leaderboard.
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
    Logs a model run to outputs/results.csv. Creates the file and headers if it doesn't exist.
    """
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    
    file_exists = os.path.exists(RESULTS_CSV_PATH)
    headers = [
        "timestamp", "model_name", "feature_description", 
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "parameters"
    ]
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "feature_description": feature_desc,
        "accuracy": f"{metrics['accuracy']:.4f}",
        "precision_macro": f"{metrics['precision_macro']:.4f}",
        "recall_macro": f"{metrics['recall_macro']:.4f}",
        "f1_macro": f"{metrics['f1_macro']:.4f}",
        "parameters": params
    }
    
    with open(RESULTS_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists or os.path.getsize(RESULTS_CSV_PATH) == 0:
            writer.writeheader()
        writer.writerow(row)
        
    print(f"[LOGGED] Model run added to {RESULTS_CSV_PATH} successfully!")

def evaluate_and_log(y_true, y_pred, model_name: str, feature_desc: str, params: str = "") -> dict:
    """
    Evaluates model predictions, prints classification report & confusion matrix,
    and logs metrics to outputs/results.csv.
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
