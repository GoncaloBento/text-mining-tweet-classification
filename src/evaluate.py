import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import RESULTS_CSV_PATH, RESULTS_HEADERS, LABEL_NAMES, OUTPUT_PRED_PATH
from src.utils import log_info, log_warning, log_success, print_header, print_separator


def compute_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = {
        name: float(per_class[i]) if i < len(per_class) else 0.0
        for i, name in LABEL_NAMES.items()
    }
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_per_class": f1_per_class,
    }


def log_model_run(
    model_name: str,
    feature_desc: str,
    metrics: dict,
    params: str = "",
    owner: str = "",
    notes: str = "",
) -> None:
    """Logs a model run to results.csv idempotently — updates if the same key exists."""
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
        "f1_per_class": str(metrics.get("f1_per_class", "")),
        "notes": notes,
    }

    def _is_match(row: dict) -> bool:
        return (
            row["model_name"] == model_name
            and row["feature_description"] == feature_desc
            and row["parameters"] == params
            and row["owner"] == owner
        )

    existing_rows = []
    updated = False

    if os.path.exists(RESULTS_CSV_PATH) and os.path.getsize(RESULTS_CSV_PATH) > 0:
        try:
            with open(RESULTS_CSV_PATH, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    if len(r) != len(RESULTS_HEADERS):
                        continue
                    row = dict(zip(RESULTS_HEADERS, r))
                    if _is_match(row):
                        existing_rows.append(new_row)
                        updated = True
                    else:
                        existing_rows.append(row)
        except Exception as e:
            log_warning(f"Error reading leaderboard CSV: {e}. Resetting.")
            existing_rows = []

    if not updated:
        existing_rows.append(new_row)

    with open(RESULTS_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADERS)
        writer.writeheader()
        writer.writerows(existing_rows)

    log_info(f"{'Updated' if updated else 'Added'} run in {RESULTS_CSV_PATH}")


def evaluate_and_log(
    y_true, y_pred,
    model_name: str,
    feature_desc: str,
    params: str = "",
    owner: str = "",
) -> dict:
    """Evaluates predictions, prints report, and logs to results.csv."""
    metrics = compute_metrics(y_true, y_pred)

    print_header(f"MODEL EVALUATION: {model_name} ({feature_desc})")
    log_info(f"Accuracy          : {metrics['accuracy']:.4f}")
    log_info(f"Precision (Macro) : {metrics['precision_macro']:.4f}")
    log_info(f"Recall (Macro)    : {metrics['recall_macro']:.4f}")
    log_info(f"F1 (Macro)        : {metrics['f1_macro']:.4f}")
    print_separator()
    print(classification_report(y_true, y_pred, target_names=list(LABEL_NAMES.values()), zero_division=0))
    print_separator()
    print(confusion_matrix(y_true, y_pred))

    log_model_run(model_name, feature_desc, metrics, params, owner=owner)
    return metrics


def evaluate_model(y_true, y_pred, owner: str, model, notes: str = "") -> dict:
    """Bento's interface — computes metrics, prints report, and logs to results.csv."""
    model_name = model.__class__.__name__ if hasattr(model, "__class__") else str(model)
    hyperparameters = str(model.get_params()) if hasattr(model, "get_params") else ""

    metrics = compute_metrics(y_true, y_pred)
    print(classification_report(y_true, y_pred, zero_division=0))
    print(confusion_matrix(y_true, y_pred))

    log_model_run(
        model_name=model_name,
        feature_desc="N/A",
        metrics=metrics,
        params=hyperparameters,
        owner=owner,
        notes=notes,
    )

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


def save_submission(
    test_df: pd.DataFrame,
    predictions,
    output_path: str = OUTPUT_PRED_PATH,
    id_col: str = "id",
) -> pd.DataFrame:
    """Saves id + label predictions to CSV."""
    submission = pd.DataFrame({"id": test_df[id_col], "label": predictions})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    log_success(f"Predictions saved to {output_path} ({len(submission)} rows)")
    return submission
