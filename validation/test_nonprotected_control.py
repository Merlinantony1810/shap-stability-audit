"""
Control: is the effect specific to protected attributes?

The obvious objection to the main finding is that it may be
unremarkable — coarsening any continuous feature destroys information and
will reduce its contribution to a tree model, protected or not.

This applies the identical coarse_binary transformation to five
NON-protected continuous features. If they behave the same way, the
claim was never "coarsening hurts protected attributes"; it is that the
damage is invisible to the global stability metric, whichever feature it
happens to. That generality is what makes an attribute-level check belong
in the standard audit procedure rather than being an Age-specific quirk.

    python -m validation.test_nonprotected_control
"""

import pandas as pd

from src import config
from src.audit import run_audit
from src.data_loader import load_dataset
from src.preprocessing import split
from src.reformat import reformat_continuous

CONTROL_FEATURES = [
    "ExperienceYears",
    "InterviewScore",
    "SkillScore",
    "PersonalityScore",
    "DistanceFromCompany",
]


def main():
    X, y = load_dataset(
        config.RAW_DATA_DIR / "recruitment_data.csv",
        config.DEFAULT_TARGET_COL,
    )
    X_train, X_test, y_train, y_test = split(X, y)

    rows = []

    # Age first, as the reference case, then the five controls.
    for feature in [config.DEFAULT_AGE_COL] + CONTROL_FEATURES:
        print(f"coarsening {feature}...")

        results, _, _ = run_audit(
            X_train, X_test, y_train, y_test,
            protected_col=feature,
            variants=["coarse_binary"],
            reformat_fn=reformat_continuous,
            make_plots=False,
        )

        row = results.iloc[0].to_dict()
        row["protected_attribute"] = feature == config.DEFAULT_AGE_COL
        rows.append(row)

    control = pd.DataFrame(rows)[[
        "attribute", "protected_attribute", "spearman_rho",
        "protected_baseline_rank", "protected_variant_rank",
        "magnitude_change_pct", "stable",
    ]]

    out = config.PROCESSED_DATA_DIR / "nonprotected_control_results.csv"
    control.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(f"\n{control.to_string(index=False)}")
    print(f"\nAll cleared rho >= {config.STABILITY_THRESHOLD}: "
          f"{bool(control['stable'].all())}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()