"""
Pipeline entrypoint: runs the full audit and writes results to disk.

    python main.py
"""

import json

import pandas as pd

from src import config
from src.audit import run_audit
from src.data_loader import load_dataset
from src.preprocessing import split, validate
from src.reformat import (
    AGE_VARIANTS,
    GENDER_VARIANTS,
    reformat_categorical,
    reformat_continuous,
)


def main():
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, _ = load_dataset(
        config.RAW_DATA_DIR / "recruitment_data.csv",
        config.DEFAULT_TARGET_COL,
    )

    findings = validate(X, y)
    print(f"  {findings['n_rows']} rows, {findings['n_features']} features, "
          f"{findings['total_missing']} missing values")

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"  train {X_train.shape}, test {X_test.shape}")

    print("\nAuditing Age...")
    age_results, baseline, age_detail = run_audit(
        X_train, X_test, y_train, y_test,
        protected_col=config.DEFAULT_AGE_COL,
        variants=AGE_VARIANTS,
        reformat_fn=reformat_continuous,
    )

    print("Auditing Gender...")
    gender_results, _, gender_detail = run_audit(
        X_train, X_test, y_train, y_test,
        protected_col=config.DEFAULT_GENDER_COL,
        variants=GENDER_VARIANTS,
        reformat_fn=reformat_categorical,
    )

    results = pd.concat([age_results, gender_results], ignore_index=True)

    matrix_path = config.PROCESSED_DATA_DIR / "audit_results_matrix.csv"
    results.to_csv(matrix_path, index=False)

    detail = {
        "baseline": baseline.to_dict(),
        "Age": age_detail,
        "Gender": gender_detail,
    }
    detail_path = config.REPORTS_DIR / "shap_importance_detail.json"
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)

    print(f"\n{'=' * 70}")
    print(results[[
        "attribute", "variant", "spearman_rho",
        "protected_baseline_rank", "protected_variant_rank",
        "magnitude_change_pct", "stable",
    ]].to_string(index=False))
    print(f"{'=' * 70}")
    print(f"\nMatrix -> {matrix_path}")
    print(f"Detail -> {detail_path}")


if __name__ == "__main__":
    main()