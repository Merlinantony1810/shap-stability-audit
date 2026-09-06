"""
German Credit: a third domain, and a documented representation problem.

Two reasons this dataset earns its place.

1. Different domain entirely — credit scoring, not employment. Also
   Annex III high-risk under the EU AI Act, so the regulatory framing
   carries over directly.

2. The protected attribute cannot be cleanly represented. Attribute 9
   encodes sex and marital status jointly:

       A91  male,   divorced/separated
       A92  female, divorced/separated/married
       A93  male,   single
       A94  male,   married/widowed
       A95  female, single

   To audit sex at all, a Gender column has to be DERIVED, and the
   mapping is not neutral:

   - It is lossy asymmetrically. Male marital status is recoverable
     (divorced / single / married); female is not — A92 collapses
     divorced, separated and married into one code.
   - There is no A96. No "female, married/widowed" code exists separate
     from A92. The scheme was never symmetric.
   - The derived column therefore carries residual marital information.
     If its SHAP importance shifts under reformatting, some of that may
     be marital status moving rather than sex.

   This is the project's thesis appearing in the wild: a real dataset
   where the protected attribute has no clean representation, so the
   analyst's encoding choice is baked into any explanation drawn from it.
   The audit is run on `age` (clean, numeric) and the sex problem is
   documented rather than silently resolved.

    python -m validation.test_german_credit
"""

import pandas as pd

from src import config
from src.audit import run_audit
from src.data_loader import encode_features, encode_target
from src.preprocessing import split
from src.reformat import AGE_VARIANTS, reformat_continuous

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits", "job",
    "dependents", "telephone", "foreign_worker", "credit_risk",
]

# The derivation. Documented here rather than buried, because it is a
# modelling decision, not a data-cleaning step.
SEX_MAP = {
    "A91": "male", "A93": "male", "A94": "male",
    "A92": "female", "A95": "female",
}


def load_german_credit(path, derive_sex=True):
    df = pd.read_csv(path, sep=" ", names=COLUMNS, header=None)

    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    status_counts = df["personal_status_sex"].value_counts().sort_index()
    print("\n  personal_status_sex distribution:")
    for code, n in status_counts.items():
        print(f"    {code}: {n:4d}  ({SEX_MAP.get(code, '?')})")

    if derive_sex:
        df["derived_sex"] = (
            df["personal_status_sex"].map(SEX_MAP).eq("female").astype(int)
        )
        n_female = int(df["derived_sex"].sum())
        print(f"\n  derived_sex: {n_female} female, "
              f"{len(df) - n_female} male")
        print("  NOTE: derived from an entangled column — see module docstring")
        df = df.drop(columns=["personal_status_sex"])

    y = encode_target(df["credit_risk"])
    X, _ = encode_features(df.drop(columns=["credit_risk"]))

    return X, y


def main():
    path = config.RAW_DATA_DIR / "german.data"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download german.data from "
            "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
        )

    X, y = load_german_credit(path)

    X_train, X_test, y_train, y_test = split(X, y)

    print("\nAuditing 'age'...")
    results, baseline, _ = run_audit(
        X_train, X_test, y_train, y_test,
        protected_col="age",
        variants=AGE_VARIANTS,
        reformat_fn=reformat_continuous,
        make_plots=False,
        max_depth=12, n_estimators=150,
    )

    out = config.PROCESSED_DATA_DIR / "german_credit_results.csv"
    results.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n{results[['variant', 'spearman_rho', 'protected_baseline_rank', 'protected_variant_rank', 'magnitude_change_pct', 'stable']].to_string(index=False)}")
    print(f"\nBaseline ranking (top 8):\n{baseline.head(8).round(4).to_string()}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()