"""
Seed robustness check.

A single-seed result could be an artefact of one particular Random Forest
fit. This re-runs the full audit at five seeds, varying only the model's
random state while holding the train/test split fixed — so the comparison
isolates model stochasticity specifically, rather than confounding it
with a different data split.

    python -m validation.test_seed_stability
"""

import pandas as pd

from src import config
from src.audit import run_audit
from src.data_loader import load_dataset
from src.preprocessing import split
from src.reformat import (
    AGE_VARIANTS,
    GENDER_VARIANTS,
    reformat_categorical,
    reformat_continuous,
)

SEEDS = [0, 7, 42, 123, 2024]


def main():
    X, y = load_dataset(
        config.RAW_DATA_DIR / "recruitment_data.csv",
        config.DEFAULT_TARGET_COL,
    )

    # Split fixed across all seeds — only the model seed varies.
    X_train, X_test, y_train, y_test = split(X, y)

    all_runs = []

    for seed in SEEDS:
        print(f"seed {seed}...")

        for protected_col, variants, fn in [
            (config.DEFAULT_AGE_COL, AGE_VARIANTS, reformat_continuous),
            (config.DEFAULT_GENDER_COL, GENDER_VARIANTS, reformat_categorical),
        ]:
            results, _, _ = run_audit(
                X_train, X_test, y_train, y_test,
                protected_col=protected_col,
                variants=variants,
                reformat_fn=fn,
                random_state=seed,
                make_plots=False,
            )
            results["seed"] = seed
            all_runs.append(results)

    raw = pd.concat(all_runs, ignore_index=True)

    summary = raw.groupby(["attribute", "variant"]).agg(
        rho_mean=("spearman_rho", "mean"),
        rho_min=("spearman_rho", "min"),
        rho_max=("spearman_rho", "max"),
        baseline_rank_min=("protected_baseline_rank", "min"),
        baseline_rank_max=("protected_baseline_rank", "max"),
        variant_rank_min=("protected_variant_rank", "min"),
        variant_rank_max=("protected_variant_rank", "max"),
        mag_change_mean=("magnitude_change_pct", "mean"),
        n_stable=("stable", "sum"),
    ).round(4).reset_index()

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(config.PROCESSED_DATA_DIR / "seed_stability_raw.csv", index=False)
    summary.to_csv(config.PROCESSED_DATA_DIR / "seed_stability_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\n{len(SEEDS)} seeds x 9 variants = {len(raw)} runs")


if __name__ == "__main__":
    main()