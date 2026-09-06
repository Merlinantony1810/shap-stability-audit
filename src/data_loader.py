"""
Data loading, validation and automatic encoding.

Takes an arbitrary tabular CSV and prepares it for the audit: resolves
column names case-insensitively, drops constant and identifier columns,
encodes string columns, and encodes the target to 0/1.

What it deliberately does NOT do is guess which column is the protected
attribute, or untangle a column that encodes two things at once (as
German Credit does with sex and marital status). Those are modelling
decisions with consequences, and a tool that makes them silently is
worse than one that stops and asks.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# A string column with more unique values than this is treated as an
# identifier (names, IDs, free text) rather than a categorical feature.
MAX_CATEGORY_CARDINALITY = 20


def resolve_column(df, name):
    """
    Find a column case- and whitespace-insensitively.

    'Age', 'age', 'AGE' and ' Age ' all resolve to whatever the file
    actually calls it. Returns the real column name.
    """
    target = name.strip().lower()
    lookup = {c.strip().lower(): c for c in df.columns}

    if target not in lookup:
        raise ValueError(
            f"Column '{name}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )
    return lookup[target]


def encode_features(X, verbose=True):
    """
    Make every column numeric, or drop it.

    Returns (X_encoded, report) where report records what happened to
    each column, so the caller can show the user rather than silently
    transforming their data.
    """
    X = X.copy()
    report = {"encoded": [], "dropped": [], "constant": [], "numeric": []}

    for col in list(X.columns):
        # A column with one distinct value carries no information. Real
        # datasets contain these — IBM Attrition has 'Over18' (always
        # 'Y'), 'EmployeeCount' and 'StandardHours' — and keeping them
        # just adds useless features to the ranking.
        if X[col].nunique() <= 1:
            X = X.drop(columns=[col])
            report["constant"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(X[col]):
            report["numeric"].append(col)
            continue

        n_unique = X[col].nunique()

        if n_unique > MAX_CATEGORY_CARDINALITY:
            # Too many distinct values to be a useful category — almost
            # always a name, ID or free-text field.
            X = X.drop(columns=[col])
            report["dropped"].append((col, n_unique))
            continue

        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        report["encoded"].append((col, n_unique))

    if verbose:
        for col in report["constant"]:
            print(f"  dropped '{col}' (constant — one value throughout)")
        for col, n in report["encoded"]:
            print(f"  encoded '{col}' ({n} categories)")
        for col, n in report["dropped"]:
            print(f"  dropped '{col}' ({n} unique values — looks like an identifier)")

    return X, report


def encode_target(y):
    """Encode a binary target of any type to 0/1."""
    n_classes = y.nunique()
    if n_classes != 2:
        raise ValueError(
            f"Target has {n_classes} unique values; this tool expects "
            f"exactly 2 (binary classification). Found: "
            f"{sorted(y.unique().tolist())[:10]}"
        )

    if pd.api.types.is_numeric_dtype(y) and set(y.unique()) == {0, 1}:
        return y

    encoded = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    mapping = dict(zip(y, encoded))
    print(f"  encoded target: {mapping}")
    return encoded


def infer_variant_type(X, col, max_categories=10):
    """
    Decide whether a column takes the continuous or categorical battery.

    Continuous: numeric with many distinct values (Age, income, scores).
    Categorical: few distinct values, or non-numeric (Gender, race).
    """
    if X[col].nunique() <= max_categories:
        return "categorical"
    if pd.api.types.is_numeric_dtype(X[col]):
        return "continuous"
    return "categorical"


def load_dataset(csv_path, target_col, verbose=True):
    """
    Load any tabular CSV and prepare it for the audit.

    Returns:
        (X, y, report) — X all-numeric, y binary 0/1, report describing
        what was encoded or dropped.

    Raises:
        ValueError: if the file will not parse as a table, the target is
                    missing, or the target is not binary.
    """
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
    except Exception as e:
        raise ValueError(
            f"Could not read '{csv_path}' as a table. This tool needs "
            f"tabular data — a CSV with rows and columns. "
            f"Underlying error: {e}"
        )

    if df.shape[1] < 2:
        raise ValueError(
            f"Only {df.shape[1]} column(s) found. This tool needs tabular "
            f"data with at least a target and one feature. Check the file "
            f"is a CSV and not, for example, plain text or JSON."
        )

    if verbose:
        print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = resolve_column(df, target_col)

    y = encode_target(df[target_col])
    X, report = encode_features(df.drop(columns=[target_col]), verbose=verbose)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after encoding.")

    return X, y, report