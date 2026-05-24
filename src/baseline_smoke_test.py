#import needed libraries
import pandas as pd


def get_majority_class(
    train_df: pd.DataFrame,
    label_col: str = "label"
):
    """
    Finds the most frequent class in the training dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe containing the target labels.

    label_col : str, optional
        Name of the column containing the labels.
        Default is "label".

    Returns
    -------
    int
        Majority class label.
    """

    return train_df[label_col].value_counts().idxmax()


def create_majority_predictions(
    test_df: pd.DataFrame,
    majority_class: int
):
    """
    Creates predictions by assigning the majority class to every test sample.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe.

    majority_class : int
        Label of the most frequent class in the training dataset.

    Returns
    -------
    list
        List of predicted labels.
    """

    return [majority_class] * len(test_df)


def create_submission_file(
    test_df: pd.DataFrame,
    predictions,
    output_path: str,
    id_col: str = "id"
):
    """
    Creates a submission CSV file with id and predicted label columns.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing the id column.

    predictions : list or array-like
        Predicted labels for the test samples.

    output_path : str
        Path where the submission CSV file will be saved.

    id_col : str, optional
        Name of the id column in the test dataframe.
        Default is "id".

    Returns
    -------
    pd.DataFrame
        Submission dataframe.
    """

    submission = pd.DataFrame({
        "id": test_df[id_col],
        "label": predictions
    })

    submission.to_csv(output_path, index=False)

    return submission


def run_majority_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: str = "../outputs/Baseline_Smoke_Test/baseline_smoke_test.csv",
    label_col: str = "label",
    id_col: str = "id",
    verbose: bool = True,
):
    """
    Runs the full majority-class baseline pipeline.

    This function finds the most frequent class in the training set,
    assigns that class to every sample in the test set, and saves
    the predictions in the required submission format.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe containing labels.

    test_df : pd.DataFrame
        Test dataframe containing ids.

    output_path : str
        Path where the prediction CSV file will be saved.

    label_col : str, optional
        Name of the label column in the training dataframe.
        Default is "label".

    id_col : str, optional
        Name of the id column in the test dataframe.
        Default is "id".

    Returns
    -------
    pd.DataFrame
        Submission dataframe with columns id and label.
    """

    try:
        majority_class = get_majority_class(
            train_df=train_df,
            label_col=label_col
        )

        predictions = create_majority_predictions(
            test_df=test_df,
            majority_class=majority_class
        )

        submission = create_submission_file(
            test_df=test_df,
            predictions=predictions,
            output_path=output_path,
            id_col=id_col
        )

        if verbose:
            print("The results were saved inside the outputs/Baseline_Smoke_Test folder ✅")
    
    except Exception as e:
        print(f"An error ocurred while running the function: {e}")

    return submission