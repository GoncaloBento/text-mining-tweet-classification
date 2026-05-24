#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
src/evaluate.py

Evaluation utilities for the Text Mining sentiment models.
Supports both Filipe's (leaderboard, idempotent logging) and Bento's 
(scikit-learn estimator, owner tracking, notes) evaluation workflows.
Logs runs to a unified leaderboard in outputs/results.csv.
"""

import os
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

RESULTS_CSV_PATH = "outputs/results.csv"

# Unified headers
HEADERS = [
    "timestamp", "owner", "model_name", "feature_description", 
    "accuracy", "precision_macro", "recall_macro", "f1_macro", 
    "parameters", "f1_per_class", "notes"
]

def compute_metrics(y_true, y_pred) -> dict:
    """Computes standard metrics: Accuracy, Precision, Recall, and F1 (Macro)."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class F1 score (Bearish=0, Bullish=1, Neutral=2)
    # Average=None returns F1 for Bearish, Bullish, Neutral
    from sklearn.metrics import f1_score
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Handle if some classes are missing in smaller evaluations safely
    f1_per_class_dict = {"Bearish": 0.0, "Bullish": 0.0, "Neutral": 0.0}
    if len(f1_per_class) >= 1: f1_per_class_dict["Bearish"] = float(f1_per_class[0])
    if len(f1_per_class) >= 2: f1_per_class_dict["Bullish"] = float(f1_per_class[1])
    if len(f1_per_class) >= 3: f1_per_class_dict["Neutral"] = float(f1_per_class[2])
    
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_per_class": f1_per_class_dict
    }

def log_model_run(
    model_name: str, 
    feature_desc: str, 
    metrics: dict, 
    params: str = "", 
    owner: str = "Filipe", 
    notes: str = ""
):
    """
    Logs a model run to outputs/results.csv in an IDEMPOTENT way.
    If a run with the same model_name, feature_description, and parameters already exists,
    it updates the metrics and timestamp of that existing row instead of appending.
    """
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "owner": owner,
        "model_name": model_name,
        "feature_description": feature_desc,
        "accuracy": f"{metrics['accuracy']:.4f}",
        "precision_macro": f"{metrics['precision_macro']:.4f}",
        "recall_macro": f"{metrics['recall_macro']:.4f}",
        "f1_macro": f"{metrics['f1_macro']:.4f}",
        "parameters": params,
        "f1_per_class": str(metrics["f1_per_class"]),
        "notes": notes
    }
    
    existing_rows = []
    updated = False
    
    # Read existing rows if file exists
    if os.path.exists(RESULTS_CSV_PATH) and os.path.getsize(RESULTS_CSV_PATH) > 0:
        try:
            with open(RESULTS_CSV_PATH, mode='r', newline='', encoding='utf-8') as f:
                # Read rows
                raw_reader = csv.reader(f)
                header_row = next(raw_reader, None)
                if header_row:
                    for r in raw_reader:
                        if len(r) == len(HEADERS):
                            row_dict = dict(zip(HEADERS, r))
                            # Check for unique run match
                            if (row_dict["model_name"] == model_name and 
                                row_dict["feature_description"] == feature_desc and 
                                row_dict["parameters"] == params and
                                row_dict["owner"] == owner):
                                # Update existing row
                                existing_rows.append(new_row)
                                updated = True
                            else:
                                existing_rows.append(row_dict)
                        else:
                            # Handle fallback mapping for old 8-column format if any row wasn't migrated yet
                            old_headers = [
                                "timestamp", "model_name", "feature_description", 
                                "accuracy", "precision_macro", "recall_macro", "f1_macro", "parameters"
                            ]
                            if len(r) == len(old_headers):
                                row_dict = dict(zip(old_headers, r))
                                # Map to unified headers
                                mapped_row = {h: "" for h in HEADERS}
                                for k, v in row_dict.items():
                                    mapped_row[k] = v
                                mapped_row["owner"] = "Filipe" # Old runs were all Filipe's
                                
                                if (mapped_row["model_name"] == model_name and 
                                    mapped_row["feature_description"] == feature_desc and 
                                    mapped_row["parameters"] == params and
                                    mapped_row["owner"] == owner):
                                    existing_rows.append(new_row)
                                    updated = True
                                else:
                                    existing_rows.append(mapped_row)
        except Exception as e:
            print(f"[WARNING] Error reading leaderboard CSV: {str(e)}. Resetting leaderboard file.")
            existing_rows = []
            
    if not updated:
        existing_rows.append(new_row)
        
    # Write back all rows
    with open(RESULTS_CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(existing_rows)
        
    if updated:
        print(f"[LOGGED] Idempotent Update: Overwrote existing run in {RESULTS_CSV_PATH} successfully!")
    else:
        print(f"[LOGGED] Added new run to {RESULTS_CSV_PATH} successfully!")

# Filipe's Interface
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
    
    log_model_run(model_name, feature_desc, metrics, params, owner="Filipe")
    
    # Return metrics mapping for backward compatibility
    return metrics

# Bento's Interface
def evaluate_model(
    y_true,
    y_pred,
    owner: str,
    model,
    notes: str = " ",
    output_path: str = "outputs/results.csv",
):
    """
    Bento's original interface. Computes metrics, prints classification report
    and confusion matrix, and appends the results to outputs/results.csv.
    """
    # 1. Safely extract model name and hyperparameters
    if hasattr(model, "__class__") and hasattr(model, "get_params"):
        model_name = model.__class__.__name__
        hyperparameters = str(model.get_params())
    else:
        model_name = str(model)
        hyperparameters = ""

    # 2. Compute metrics
    metrics = compute_metrics(y_true, y_pred)

    # 3. Print report & matrix
    print("Classification Report")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    # 4. Log to unified CSV safely (corrects Bento's column shifting bug!)
    log_model_run(
        model_name=model_name,
        feature_desc="N/A",
        metrics=metrics,
        params=hyperparameters,
        owner=owner,
        notes=notes
    )

    # Bento's original return format
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "owner": owner,
        "model": model_name,
        "hyperparams": hyperparameters,
        "val_accuracy": metrics["accuracy"],
        "val_p_macro": metrics["precision_macro"],
        "val_r_macro": metrics["recall_macro"],
        "val_f1_macro": metrics["f1_macro"],
        "val_f1_per_class": metrics["f1_per_class"],
        "notes": notes,
    }

def save_results(results: dict, output_path: str = "outputs/results.csv"):
    """
    Deprecated fallback function to prevent any import failures.
    We log using log_model_run inside evaluate_model instead.
    """
    pass
