"""
Exploratory data analysis.

The group-rate checks below are descriptive only. They show how hiring
rates differ across Gender and Age bands in this dataset — they are NOT
a fairness conclusion about the model, which would need disparate-impact
ratios, calibration, and a great deal more care. They are here to
motivate the audit that follows, not to substitute for one.
"""

import matplotlib
matplotlib.use("Agg")          # no display needed; write straight to file
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

EDA_DIR = config.FIGURES_DIR / "eda"


def _save(fig, name):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


def plot_target_balance(y, target_col="HiringDecision"):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["Not hired (0)", "Hired (1)"], counts.values,
           color=["#95a5a6", "#2980b9"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(y):.1%})", ha="center", va="bottom")
    ax.set_ylabel("Candidates")
    ax.set_title("Target balance")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, "target_balance")


def plot_distribution(X, col):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(X[col], bins=30, color="#2980b9", edgecolor="white")
    ax.axvline(X[col].median(), color="#e74c3c", linestyle="--",
               label=f"median = {X[col].median():.0f}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.set_title(f"{col} distribution")
    ax.legend()
    return _save(fig, f"{col.lower()}_distribution")


def plot_correlation_heatmap(X):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Feature correlations")
    return _save(fig, "correlation_heatmap")


def plot_rate_by_group(X, y, col, bins=None, labels=None):
    """
    Selection rate by group. Descriptive only — see module docstring.
    """
    df = X.copy()
    df["_target"] = y

    if bins is not None:
        df["_group"] = pd.cut(df[col], bins=bins, labels=labels)
    else:
        df["_group"] = df[col]

    rates = df.groupby("_group", observed=True)["_target"].agg(["mean", "size"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rates.index.astype(str), rates["mean"], color="#2980b9")
    for i, (rate, n) in enumerate(zip(rates["mean"], rates["size"])):
        ax.text(i, rate, f"{rate:.1%}\n(n={n})", ha="center", va="bottom")
    ax.set_ylabel("Selection rate")
    ax.set_title(f"Selection rate by {col} (descriptive only)")
    ax.set_ylim(0, rates["mean"].max() * 1.3)
    return _save(fig, f"rate_by_{col.lower()}")


def run(X, y):
    """Generate all EDA figures."""
    print("Generating EDA figures...")
    plot_target_balance(y)
    plot_distribution(X, config.DEFAULT_AGE_COL)
    plot_correlation_heatmap(X)
    plot_rate_by_group(X, y, config.DEFAULT_GENDER_COL)
    plot_rate_by_group(
        X, y, config.DEFAULT_AGE_COL,
        bins=[19, 30, 40, 51],
        labels=["20-30", "31-40", "41-50"],
    )