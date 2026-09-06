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

def compare_rankings(baseline_importance, variant_importance, protected_col,
                     threshold=None, top_k=None):
    """
    Compare a variant's SHAP ranking to the baseline on three metrics.

    Three, not one, because the central hypothesis is that a single
    aggregate metric can mask exactly the failure an audit exists to
    catch. The global correlation can pass comfortably while the one
    attribute under audit is the outlier driving what little
    disagreement there is.
    """
    if threshold is None:
        threshold = config.STABILITY_THRESHOLD
    if top_k is None:
        top_k = config.TOP_K

    # Compare only features present in both, so one-hot variants that
    # have been canonicalised line up with the baseline.
    shared = [f for f in baseline_importance.index
              if f in variant_importance.index]

    # 1. Global rank correlation. Spearman rather than raw magnitude
    #    because SHAP values are not comparable in scale across
    #    independently retrained models.
    rho, p_value = spearmanr(
        baseline_importance[shared].rank(ascending=False),
        variant_importance[shared].rank(ascending=False),
    )

    # 2. Top-k overlap — did the headline features change?
    baseline_top = set(baseline_importance.index[:top_k])
    variant_top = set(variant_importance.index[:top_k])
    overlap = len(baseline_top & variant_top) / top_k

    # 3. The protected attribute's own rank. This is the metric that
    #    encodes the thesis: it is the one the other two can hide.
    baseline_rank = list(baseline_importance.index).index(protected_col) + 1
    variant_rank = list(variant_importance.index).index(protected_col) + 1

    baseline_mag = baseline_importance[protected_col]
    variant_mag = variant_importance[protected_col]
    mag_change = (variant_mag - baseline_mag) / baseline_mag

    return {
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p_value), 6),
        "top_k_overlap": overlap,
        "protected_baseline_rank": baseline_rank,
        "protected_variant_rank": variant_rank,
        "rank_shift": variant_rank - baseline_rank,
        "baseline_magnitude": round(float(baseline_mag), 6),
        "variant_magnitude": round(float(variant_mag), 6),
        "magnitude_change_pct": round(float(mag_change) * 100, 1),
        "stable": bool(rho >= threshold),
    }

def run_audit(X_train, X_test, y_train, y_test, protected_col,
              variants, reformat_fn, random_state=None, **rf_kwargs):
    """
    Run the full audit: baseline plus every variant.

    The model is retrained for each variant rather than reused. Reusing
    the baseline model and only perturbing its input at inference time
    would test a different question — how a fixed model responds to
    out-of-distribution input — rather than the intended scenario, where
    a team encodes a column differently upstream, trains its own model on
    that encoding, and reads the resulting SHAP output.
    """
    # Baseline: same data, same model settings, no reformatting.
    baseline_model = train_random_forest(
        X_train, y_train, random_state=random_state, **rf_kwargs
    )
    baseline_importance = compute_shap_importance(baseline_model, X_test)

    results = []
    detail = {"baseline": baseline_importance.to_dict()}

    for variant in variants:
        X_train_v = reformat_fn(X_train, variant, protected_col)
        X_test_v = reformat_fn(X_test, variant, protected_col)

        model_v = train_random_forest(
            X_train_v, y_train, random_state=random_state, **rf_kwargs
        )
        importance_v = compute_shap_importance(model_v, X_test_v)

        # One-hot variants produce extra columns; sum them back so the
        # rankings are the same length and comparable.
        importance_v = collapse_to_canonical(importance_v, protected_col)

        comparison = compare_rankings(
            baseline_importance, importance_v, protected_col
        )
        comparison["attribute"] = protected_col
        comparison["variant"] = variant

        results.append(comparison)
        detail[variant] = importance_v.to_dict()

    results_df = pd.DataFrame(results)

    # Put the identifying columns first for readability.
    cols = ["attribute", "variant"] + [
        c for c in results_df.columns if c not in ("attribute", "variant")
    ]

    return results_df[cols], baseline_importance, detail