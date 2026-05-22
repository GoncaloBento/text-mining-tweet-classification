#import needed libraries
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def evaluate_model(
    y_true,
    y_pred,
    owner:str,
    model,
    notes: str= " ",
    output_path: str = "outputs/results.csv",
):
    """
    Evaluates a classification model using accuracy, precision, recall and F1-score.

    This function computes macro and weighted metrics, prints a per-class
    classification report, prints the confusion matrix, and appends the main
    results to a CSV file.

    Parameters
    ----------
    y_true : array-like
        True labels.

    y_pred : array-like
        Predicted labels produced by the model.

    model_name : str
        Name of the model or experiment being evaluated.

    output_path : str, optional
        Path where the results CSV file will be saved.
        Default is "outputs/results.csv".

    Returns
    -------
    dict
        Dictionary containing the calculated evaluation metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    #A good weighted F1 score could lead to a fallacious interpretation since attributes a bigger weight to most frequent classes. Macro gives the same weight to each class.
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_per_class_dict = {"Bearish": f1_per_class[0], "Bullish": f1_per_class[1], "Neutral": f1_per_class[2]}

    #precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    #recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    #f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    model_name = model.__class__.__name__
    hyperparameters = model.get_params()

    results = {
        "timestamp":timestamp,
        "owner":owner,
        "model": model_name,
        "hyperparams": hyperparameters,
        "val_accuracy": accuracy,
        "val_p_macro": precision_macro,
        "val_r_macro": recall_macro,
        "val_f1_macro": f1_macro,
        "val_f1_per_class": f1_per_class_dict,
        "notes": notes,
        #"precision_weighted": precision_weighted,
        #"recall_weighted": recall_weighted,
        #"f1_weighted": f1_weighted,
    }

    print("Classification Report")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    save_results(results, output_path)

    return results


def save_results(
    results: dict,
    output_path: str = "outputs/results.csv",
):
    """
    Appends model evaluation results to a CSV file.

    If the output folder does not exist, it is created automatically.
    If the CSV file already exists, the new results are appended as a new row.

    Parameters
    ----------
    results : dict
        Dictionary containing the model evaluation metrics.

    output_path : str, optional
        Path where the results CSV file will be saved.
        Default is "outputs/results.csv".

    Returns
    -------
    None
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([results])

    if output_path.exists():
        results_df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(output_path, index=False)