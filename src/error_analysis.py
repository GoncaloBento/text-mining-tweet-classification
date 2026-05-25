import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.evaluate import compute_metrics
from src.config import (
    CONF_MATRIX_PLOT_PATH, MISCLASSIFIED_TXT_PATH, MISCLASSIFIED_JSON_PATH,
    LABEL_NAMES as LABEL_MAPPING,
)
from src.utils import log_info, log_success, print_separator


def _plot_confusion_matrix(cm, path: str, model_name: str, feature_desc: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Bearish (0)", "Bullish (1)", "Neutral (2)"],
        yticklabels=["Bearish (0)", "Bullish (1)", "Neutral (2)"],
    )
    plt.title(f"Confusion Matrix\n{model_name} ({feature_desc})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    log_success(f"Confusion matrix saved to {path}")


def _collect_errors(y_val, y_pred, y_proba, X_val_raw, X_val_pre) -> tuple[pd.DataFrame, dict]:
    errors = []
    for (orig_idx, raw_text), clean_text, actual, predicted, probas in zip(
        X_val_raw.items(), X_val_pre, y_val, y_pred, y_proba
    ):
        if actual == predicted:
            continue
        errors.append({
            "original_index": int(orig_idx),
            "tweet_text": str(raw_text),
            "cleaned_text": str(clean_text),
            "actual_label": int(actual),
            "actual_name": LABEL_MAPPING[actual],
            "predicted_label": int(predicted),
            "predicted_name": LABEL_MAPPING[predicted],
            "confidence": float(probas[predicted]),
            "actual_class_proba": float(probas[actual]),
            "probabilities": {LABEL_MAPPING[i]: float(p) for i, p in enumerate(probas)},
        })

    errors_df = pd.DataFrame(errors)
    top_per_class = {
        LABEL_MAPPING[cls]: (
            errors_df[errors_df["actual_label"] == cls]
            .sort_values("confidence", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
        for cls in [0, 1, 2]
    }
    return errors_df, top_per_class


def _write_txt_report(
    errors_df: pd.DataFrame, top_per_class: dict, cm, y_val,
    path: str, model_name: str, feature_desc: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("SENTIMENT CLASSIFICATION ERROR ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Model : {model_name} | Features : {feature_desc}\n")
        f.write(f"Val size : {len(y_val)} | Errors : {len(errors_df)} ({len(errors_df)/len(y_val)*100:.2f}%)\n\n")

        f.write("CONFUSION MATRIX:\n")
        for i, name in LABEL_MAPPING.items():
            row = ", ".join(f"{LABEL_MAPPING[j]}={cm[i][j]}" for j in range(3))
            f.write(f"  {name} → {row}\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write("TOP 20 MOST CONFIDENT MISCLASSIFICATIONS BY CLASS\n")
        f.write("=" * 100 + "\n\n")

        for class_name, errs in top_per_class.items():
            f.write(f"{'#' * 40}\nACTUAL CLASS: {class_name.upper()}\n{'#' * 40}\n\n")
            if not errs:
                f.write("No errors in this class.\n\n")
                continue
            for i, err in enumerate(errs, 1):
                f.write(f"{i}. \"{err['tweet_text']}\"\n")
                f.write(f"   Clean  : \"{err['cleaned_text']}\"\n")
                f.write(f"   Actual : {err['actual_name']} (p={err['actual_class_proba']:.4f})"
                        f"  →  Predicted : {err['predicted_name']} (p={err['confidence']:.4f})\n")
                f.write(f"   Probas : {err['probabilities']}\n")
                f.write("-" * 80 + "\n")
            f.write("\n")
    log_success(f"TXT report saved to {path}")


def _write_json_report(top_per_class: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(top_per_class, f, indent=2)
    log_success(f"JSON report saved to {path}")


def run_error_analysis(
    y_val, y_pred, y_proba,
    X_val_raw: pd.Series, X_val_pre: pd.Series,
    model_name: str = "Logistic Regression",
    feature_desc: str = "TF-IDF",
) -> None:
    """Plots confusion matrix and writes misclassification reports for any fitted model."""
    metrics = compute_metrics(y_val, y_pred)
    print_separator()
    log_info(f"Val F1 Macro : {metrics['f1_macro']:.4f}")
    log_info(f"Val Accuracy : {metrics['accuracy']:.4f}")
    print_separator()

    cm = confusion_matrix(y_val, y_pred)
    _plot_confusion_matrix(cm, CONF_MATRIX_PLOT_PATH, model_name, feature_desc)

    log_info("Collecting misclassifications ...")
    errors_df, top_per_class = _collect_errors(y_val, y_pred, y_proba, X_val_raw, X_val_pre)
    _write_txt_report(errors_df, top_per_class, cm, y_val,
                      MISCLASSIFIED_TXT_PATH, model_name, feature_desc)
    _write_json_report(top_per_class, MISCLASSIFIED_JSON_PATH)
