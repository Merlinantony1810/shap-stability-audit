"""
Model evaluation.

Establishes that the model being audited is predictively credible. A weak
model's explanation would be uninteresting on its own terms, and any
instability could be dismissed as noise from a poor fit rather than a
property of the explanation method.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from src import config

EVAL_DIR = config.FIGURES_DIR / "evaluation"


def evaluate(model, X_test, y_test, name="random_forest"):
    """Compute metrics and save confusion matrix + ROC curve."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix — {name}")
    fig.savefig(EVAL_DIR / f"confusion_matrix_{name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2980b9",
            label=f"AUC = {metrics['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve — {name}")
    ax.legend()
    fig.savefig(EVAL_DIR / f"roc_curve_{name}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  accuracy {metrics['accuracy']}, AUC {metrics['roc_auc']}")
    print(classification_report(y_test, y_pred, digits=3))

    return metrics