"""
Reformatting functions.

Each variant represents a plausible, non-adversarial thing a data
engineering team might do to a column: banding an age field, one-hot
encoding a category, standardising a scale, imputing missing values.
None is contrived to break SHAP — that ordinariness is the point. If
routine choices destabilise an explanation, the fragility is an
operational risk, not just an attack surface.

Every function takes the column name as an argument rather than assuming
one. That is what lets the same code audit 'Age' here, 'age' in German
Credit, and 'AGE' somewhere else.
"""

import numpy as np
import pandas as pd

from src import config

AGE_VARIANTS = [
    "binned_ordinal",
    "binned_onehot",
    "reversed_scale",
    "standardized",
    "coarse_binary",
]

GENDER_VARIANTS = [
    "swapped_encoding",
    "onehot",
    "string_labels_then_encoded",
    "imputed_missing",
]


def reformat_continuous(X, variant, col, n_bins=3):
    """
    Reformat a continuous column (e.g. Age).

    Bin edges come from pd.qcut — quantiles of the actual data — rather
    than fixed values. Hardcoded edges are the bug that broke the earlier
    version of this tool on a dataset with a wider range: rows outside the
    edges became NaN silently. Quantiles adapt to whatever distribution
    they are given.
    """
    X = X.copy()

    if variant == "baseline":
        return X

    if variant == "binned_ordinal":
        X[col] = pd.qcut(X[col], q=n_bins, labels=False, duplicates="drop")
        return X

    if variant == "binned_onehot":
        banded = pd.qcut(X[col], q=n_bins, labels=False, duplicates="drop")
        dummies = pd.get_dummies(banded, prefix=col).astype(int)
        X = X.drop(columns=[col])
        return pd.concat([X, dummies], axis=1)

    if variant == "reversed_scale":
        # Same information, inverted direction — as if one data feed
        # recorded the scale the other way round.
        X[col] = X[col].max() - X[col] + X[col].min()
        return X

    if variant == "standardized":
        X[col] = (X[col] - X[col].mean()) / X[col].std()
        return X

    if variant == "coarse_binary":
        # Maximum information loss: two bands. The fewest buckets
        # possible, and the variant expected to be most disruptive.
        X[col] = (X[col] > X[col].median()).astype(int)
        return X

    raise ValueError(
        f"Unknown continuous variant '{variant}'. "
        f"Expected one of: {AGE_VARIANTS}"
    )


def reformat_categorical(X, variant, col, missing_rate=0.15):
    """
    Reformat a binary categorical column (e.g. Gender).

    Args:
        missing_rate: fraction of values blanked out before imputation in
                      the 'imputed_missing' variant.
    """
    X = X.copy()

    if variant == "baseline":
        return X

    if variant == "swapped_encoding":
        X[col] = 1 - X[col]
        return X

    if variant == "onehot":
        dummies = pd.get_dummies(X[col], prefix=col).astype(int)
        X = X.drop(columns=[col])
        return pd.concat([X, dummies], axis=1)

    if variant == "string_labels_then_encoded":
        # Round-trip through strings, then re-encode alphabetically —
        # what happens when a categorical is exported and reimported.
        labels = X[col].map({0: "Male", 1: "Female"})
        X[col] = labels.map({"Female": 0, "Male": 1})
        return X

    if variant == "imputed_missing":
        # Simulates a self-disclosure field with partial non-response:
        # blank out a fraction of values, then fill with the mode.
        #
        # An earlier version only called fillna, which was a no-op on data
        # with no missing values — it scored a perfect rho of 1.0 by
        # comparing an unchanged column against itself. Introducing the
        # missingness first is what makes the variant test anything.
        rng = np.random.default_rng(config.RANDOM_SEED)
        mask = rng.random(len(X)) < missing_rate
        X.loc[mask, col] = np.nan
        X[col] = X[col].fillna(X[col].mode()[0]).astype(int)
        return X

    raise ValueError(
        f"Unknown categorical variant '{variant}'. "
        f"Expected one of: {GENDER_VARIANTS}"
    )