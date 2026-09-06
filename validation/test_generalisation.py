"""
External validation on UCI Adult Census Income.

A single-dataset finding is a weaker claim than one replicated on
independent data. Adult is chosen deliberately: Hwang et al. (2025) used
census data, so this returns the tool to the domain where the underlying
sensitivity was first reported.

Also a genuine portability test. This dataset has a much wider age range
(17-90) than the recruitment data (20-50), which is exactly the kind of
distribution shift that breaks hardcoded assumptions.

    python -m validation.test_generalisation
"""

import pandas as pd

from src import config
from src.audit import run_audit
from src.preprocessing import split
from src.reformat import AGE_VARIANTS, reformat_continuous

ADULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "adult/adult.data"
)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

# Subsampled and capped because exact TreeExplainer scales badly. This is
# a deliberate deviation from the main pipeline, disclosed rather than
# hidden: the audit logic is identical, only the model capacity differs.
SUBSAMPLE_N = 6000
RF_KWARGS = {"max_depth": 12, "n_estimators": 150}


def load_adult():
    df = pd.read_csv(ADULT_URL, names=COLUMNS, skipinitialspace=True)

    # Keep the numeric columns plus a binary-encoded sex, so the feature
    # set is comparable in spirit to the recruitment data.
    df = df[["age", "education_num", "hours_per_week", "capital_gain",
             "capital_loss", "fnlwgt", "sex", "income"]].copy()

    df["sex"] = (df["sex"] == "Female").astype(int)
    df["income"] = (df["income"] == ">50K").astype(int)

    df = df.sample(n=SUBSAMPLE_N, random_state=config.RANDOM_SEED)

    y = df["income"]
    X = df.drop(columns=["income"])
    return X, y


def main():
    print(f"Downloading Adult and subsampling to {SUBSAMPLE_N} rows...")
    X, y = load_adult()

    print(f"  age range: {X['age'].min()}-{X['age'].max()} "
          f"(recruitment data was 20-50)")
    print(f"  {X.shape[0]} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = split(X, y)

    print("\nAuditing 'age'...")
    results, baseline, _ = run_audit(
        X_train, X_test, y_train, y_test,
        protected_col="age",              # note: lowercase, unlike 'Age'
        variants=AGE_VARIANTS,
        reformat_fn=reformat_continuous,
        make_plots=False,
        **RF_KWARGS,
    )

    out = config.PROCESSED_DATA_DIR / "adult_generalisation_results.csv"
    results.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n{results[['variant', 'spearman_rho', 'protected_baseline_rank', 'protected_variant_rank', 'magnitude_change_pct', 'stable']].to_string(index=False)}")
    print(f"\nBaseline ranking:\n{baseline.round(4).to_string()}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()