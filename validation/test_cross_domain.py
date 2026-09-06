"""
Cross-domain validation.

Runs the identical audit on datasets from different domains to test
whether the finding is a property of the recruitment data or of the
method. Uses the real audit code, not a reimplementation.

Model capacity is capped per dataset rather than globally: exact
TreeExplainer scales badly on wide data, but capping a small dataset
unnecessarily changes its results (the recruitment run gave rho 0.9515
capped versus 0.9152 uncapped, with Age at rank 8 rather than 7). The
'capped' column records which runs were limited.

    python -m validation.test_cross_domain
"""

import pandas as pd

from src import config
from src.audit import run_audit
from src.data_loader import load_dataset
from src.preprocessing import split
from src.reformat import AGE_VARIANTS, reformat_continuous

DATASETS = [
    {
        "name": "Recruitment (hiring)",
        "file": "recruitment_data.csv",
        "target": "HiringDecision",
        "protected": "Age",
        "rf_kwargs": {},                                    # small enough to run uncapped
    },
    {
        "name": "IBM HR (attrition)",
        "file": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "target": "attrition",        # lowercase on purpose: tests the resolver
        "protected": "Age",
        "rf_kwargs": {"max_depth": 12, "n_estimators": 150},
    },
]


def main():
    rows = []

    for spec in DATASETS:
        path = config.RAW_DATA_DIR / spec["file"]
        if not path.exists():
            print(f"skipping {spec['name']} — {spec['file']} not found")
            continue

        print(f"\n{spec['name']}")
        X, y, _ = load_dataset(path, spec["target"])
        X_train, X_test, y_train, y_test = split(X, y)

        results, baseline, _ = run_audit(
            X_train, X_test, y_train, y_test,
            protected_col=spec["protected"],
            variants=AGE_VARIANTS,
            reformat_fn=reformat_continuous,
            make_plots=False,
            **spec["rf_kwargs"],
        )

        results["dataset"] = spec["name"]
        results["n_features"] = X.shape[1]
        results["capped"] = bool(spec["rf_kwargs"])
        rows.append(results)

    combined = pd.concat(rows, ignore_index=True)

    out = config.PROCESSED_DATA_DIR / "cross_domain_results.csv"
    combined.to_csv(out, index=False)

    # The comparison that matters: how the same transformation behaves
    # as the number of features grows.
    coarse = combined[combined["variant"] == "coarse_binary"][[
        "dataset", "n_features", "capped", "spearman_rho",
        "protected_baseline_rank", "protected_variant_rank",
        "magnitude_change_pct", "stable",
    ]]

    pd.set_option("display.width", 220)
    print(f"\n{'=' * 80}")
    print("coarse_binary across domains:")
    print(coarse.to_string(index=False))
    print(f"{'=' * 80}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()