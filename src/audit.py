"""
The SHAP stability audit.

Question: not "is this model biased?" but "can the SHAP explanation of
how a protected attribute drives predictions be trusted?"

For each variant: reformat the column, retrain an identical model,
recompute SHAP importance, compare the ranking to the baseline.
"""

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from src import config
from src.train_model import train_random_forest


def compute_shap_importance(model, X_test):
    """
    Global SHAP importance: mean absolute SHAP value per feature.

    Uses exact TreeExplainer rather than a sampling estimator. This
    matters: the audit trains and explains ten models per run, and a
    sampling-based explainer would add its own estimation variance to the
    very quantity being measured — confounding instability caused by
    re-encoding with instability caused by the explainer itself.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary classifiers return per-class values. Take the positive class.
    # Newer SHAP returns a 3D array (rows, features, classes); older
    # versions return a list of 2D arrays.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    importance = np.abs(shap_values).mean(axis=0)

    return pd.Series(
        importance, index=X_test.columns
    ).sort_values(ascending=False)


def collapse_to_canonical(importance, original_col):
    """
    Sum derived columns back into the original attribute.

    One-hot encoding turns one column into several, so rankings would
    otherwise have different lengths and not be comparable. Summing a
    feature's one-hot components to estimate its total impact is the same
    convention Hwang et al. (2025) apply.
    """
    derived = [
        c for c in importance.index
        if c.startswith(f"{original_col}_")
    ]

    if not derived:
        return importance

    total = importance[derived].sum()
    collapsed = importance.drop(labels=derived)
    collapsed[original_col] = total

    return collapsed.sort_values(ascending=False)