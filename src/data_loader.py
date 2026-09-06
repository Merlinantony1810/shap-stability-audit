"""
Data loading and validation.

Takes any CSV and splits it into features (X) and a binary target (y).
Nothing here is specific to one dataset — the path and target column are
arguments, which is what lets the same code run on the recruitment data,
Adult, German Credit, or anything else.
"""

import pandas as pd


def load_dataset(csv_path, target_col):
    """
    Load a CSV and split it into features and a binary target.

    Args:
        csv_path:   path to the CSV file
        target_col: name of the column to use as the prediction target

    Returns:
        (X, y) where X is a DataFrame of features and y is a Series
        containing the target

    Raises:
        ValueError: if the target column is missing, or is not binary
    """
    df = pd.read_csv(csv_path)

    # Fail with a useful message rather than a bare KeyError. When you
    # point this at a new dataset, the most common mistake is guessing
    # the target column name wrong — so list what is actually there.
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {csv_path}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # The audit assumes binary classification throughout: SHAP values are
    # taken for the positive class, so a multi-class target would silently
    # produce meaningless results rather than an error.
    n_classes = df[target_col].nunique()
    if n_classes != 2:
        raise ValueError(
            f"Target column '{target_col}' has {n_classes} unique values; "
            f"this tool expects exactly 2 (binary classification)."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y