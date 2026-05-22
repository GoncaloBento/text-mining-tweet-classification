#import needed variables
from src.config import SEED
from src.config import VAL_SIZE
from src.config import K_Fold_n_splits

#import needed libraries
import pandas as pd

#import needed functions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold



def stratified_split(dataset: pd.DataFrame, test_size: float = VAL_SIZE, seed=SEED):
    """
    Splits the dataset into stratified training and validation sets.

    This function separates the input text data and target labels,
    then applies a stratified train/validation split to preserve
    the original class distribution in both subsets.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataframe containing tweet texts and labels.

    test_size : float, optional
        Proportion of the dataset to allocate to the validation set.
        Default is VAL_SIZE.

    seed : int, optional
        Random seed used to ensure reproducibility of the split.
        Default is SEED.

    Returns
    -------
    X_train : pd.Series
        Training text samples.

    X_val : pd.Series
        Validation text samples.

    y_train : pd.Series
        Training labels.

    y_val : pd.Series
        Validation labels.
    """
        
    X = dataset['text']
    y = dataset['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return X_train, X_val, y_train, y_val


def create_stratified_kfold(n_splits: int = K_Fold_n_splits, seed = SEED):
    """
    Creates a Stratified K-Fold cross-validation splitter.

    This function initializes a StratifiedKFold object that preserves
    the class distribution across all folds while performing
    cross-validation splits.

    Parameters
    ----------
    n_splits : int, optional
        Number of folds used in cross-validation.
        Default is K_Fold_n_splits.

    seed : int, optional
        Random seed used to ensure reproducibility when shuffling data.
        Default is SEED.

    Returns
    -------
    StratifiedKFold
        Configured StratifiedKFold object ready for cross-validation.
    """
    
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    return skf