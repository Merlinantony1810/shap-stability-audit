"""
Preprocessing: validation and the train/test split.

The split is created once and reused across the baseline and every
reformatted variant, so that the encoding of the protected attribute is
the only thing that differs between comparisons.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def validate(X, y):
    """
    Check the data is usable before training.

    Returns a dict of findings rather than printing them, so callers can
    log, assert, or display them as they choose.
    """
    findings = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "missing_by_column": X.isnull().sum().to_dict(),
        "total_missing": int(X.isnull().sum().sum()),
        "class_balance": y.value_counts().to_dict(),
        "non_numeric_columns": X.select_dtypes(exclude="number").columns.tolist(),
    }
    return findings


def split(X, y, test_size=None, random_state=None):
    """
    Stratified train/test split.

    Stratified because the target is imbalanced (roughly 69/31 here). A
    plain random split can shift the class ratio between train and test,
    which would change the SHAP importances for reasons unrelated to the
    encoding being audited.
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_SEED

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )