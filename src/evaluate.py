"""
Model Evaluation
================
Comprehensive, fraud-specific evaluation suite covering:
  • Precision / Recall / F1 (recall is the primary business metric)
  • ROC-AUC and Average Precision (PR-AUC)
  • Confusion matrix
  • SHAP feature importance

All plots are saved to models/reports/ for inclusion in the README / portfolio.
"""

import os
import json
import logging
import numpy as np
import pandas as pd

# Use non-interactive backend so plots work in headless/server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

logger = logging.getLogger(__name__)

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(_ROOT, "models", "reports")

# ── Colour palette (consistent across all plots) ──────────────────────────────
_PALETTE = {
    "fraud":   "#EF4444",
    "legit":   "#10B981",
    "blue":    "#4F46E5",
    "warning": "#F59E0B",
    "bg":      "#0F172A",
    "card":    "#1E293B",
    "text":    "#E2E8F0",
}
plt.rcParams.update({
    "figure.facecolor": _PALETTE["bg"],
    "axes.facecolor":   _PALETTE["card"],
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  _PALETTE["text"],
    "xtick.color":      _PALETTE["text"],
    "ytick.color":      _PALETTE["text"],
    "text.color":       _PALETTE["text"],
    "grid.color":       "#334155",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "legend.facecolor": _PALETTE["card"],
    "legend.edgecolor": "#334155",
})


# ── Top-level evaluator ───────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test,
    y_test,
    feature_names: list,
    model_name:    str = "model",
) -> dict:
    """
    Run the complete evaluation suite for one model.

    Generates four plots (confusion matrix, ROC, PR curve, SHAP summary)
    and returns a metrics dict suitable for JSON serialisation.

    Args:
        model:         Fitted sklearn/xgboost estimator.
        X_test:        Test features (DataFrame or ndarray).
        y_test:        True labels.
        feature_names: Ordered list of feature column names.
        model_name:    Identifier used for file names and log messages.

    Returns:
        dict with precision, recall, f1, roc_auc, avg_precision, and counts.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics = {
        "precision":       float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":          float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":              float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":         float(roc_auc_score(y_test, y_prob)),
        "avg_precision":   float(average_precision_score(y_test, y_prob)),
        "total_test":      int(len(y_test)),
        "fraud_total":     int(y_test.sum()),
        "fraud_detected":  int(((y_pred == 1) & (y_test == 1)).sum()),
        "fraud_missed":    int(((y_pred == 0) & (y_test == 1)).sum()),
        "false_alarms":    int(((y_pred == 1) & (y_test == 0)).sum()),
    }
    metrics["recall_pct"] = round(
        metrics["fraud_detected"] / max(metrics["fraud_total"], 1) * 100, 2
    )

    _log_metrics(metrics, model_name)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_confusion_matrix(y_test, y_pred, model_name)
    _plot_roc_curve(y_test, y_prob, model_name)
    _plot_precision_recall(y_test, y_prob, model_name)
    _plot_shap_summary(model, X_test, feature_names, model_name)

    # ── Save per-model JSON ───────────────────────────────────────────────────
    save_evaluation_report(metrics, model_name)

    return metrics


def _log_metrics(metrics: dict, model_name: str):
    pad = 30
    logger.info(f"\n{'─'*pad} {model_name.upper()} {'─'*pad}")
    logger.info(f"  Precision    : {metrics['precision']:.4f}")
    logger.info(f"  Recall       : {metrics['recall']:.4f}  ← PRIMARY METRIC")
    logger.info(f"  F1 Score     : {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC       : {metrics['avg_precision']:.4f}")
    logger.info(
        f"  Fraud recall : {metrics['fraud_detected']}/{metrics['fraud_total']} "
        f"({metrics['recall_pct']}%)"
    )
    logger.info(f"  False alarms : {metrics['false_alarms']:,}")


# ── Individual plot functions ─────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=sns.color_palette("Blues", as_cmap=True),
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        linewidths=1, linecolor="#334155",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {_fmt(model_name)}", fontsize=14, pad=12)
    plt.tight_layout()
    _save(f"{model_name}_confusion_matrix.png")


def _plot_roc_curve(y_test, y_prob, model_name: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=_PALETTE["blue"],  lw=2.5, label=f"ROC-AUC = {auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.12, color=_PALETTE["blue"])
    ax.plot([0, 1], [0, 1], "w--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve — {_fmt(model_name)}", fontsize=14, pad=12)
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    _save(f"{model_name}_roc_curve.png")


def _plot_precision_recall(y_test, y_prob, model_name: str):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=_PALETTE["legit"], lw=2.5, label=f"PR-AUC = {avg_prec:.4f}")
    ax.fill_between(recall, precision, alpha=0.12, color=_PALETTE["legit"])
    baseline = y_test.mean()
    ax.axhline(baseline, color=_PALETTE["warning"], linestyle="--", lw=1,
               label=f"Baseline (fraud rate = {baseline:.3f})")
    ax.set_xlabel("Recall (Fraud Detection Rate)", fontsize=12)
    ax.set_ylabel("Precision",                    fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {_fmt(model_name)}", fontsize=14, pad=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    _save(f"{model_name}_pr_curve.png")


def _plot_shap_summary(model, X_test, feature_names: list, model_name: str, max_samples: int = 300):
    """Generate SHAP beeswarm summary (skips gracefully if SHAP fails)."""
    try:
        from sklearn.linear_model import LogisticRegression

        n = min(max_samples, len(X_test))
        X_sample = (
            X_test.iloc[:n] if isinstance(X_test, pd.DataFrame)
            else pd.DataFrame(X_test[:n], columns=feature_names)
        )

        if isinstance(model, LogisticRegression):
            background = pd.DataFrame(
                np.zeros((1, len(feature_names))), columns=feature_names
            )
            explainer  = shap.LinearExplainer(model, background)
        else:
            explainer = shap.TreeExplainer(model)

        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # fraud class

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_vals, X_sample, show=False, max_display=15,
                          plot_type="dot")
        plt.title(f"SHAP Feature Importance — {_fmt(model_name)}", fontsize=14)
        plt.tight_layout()
        _save(f"{model_name}_shap_summary.png", bbox_inches="tight")
        logger.info(f"SHAP summary saved for {model_name}")

    except Exception as exc:
        logger.warning(f"SHAP plot skipped for {model_name}: {exc}")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fmt(name: str) -> str:
    return name.replace("_", " ").title()


def _save(filename: str, **kwargs):
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, **kwargs)
    plt.close()
    logger.info(f"Plot saved → {path}")


def save_evaluation_report(metrics: dict, model_name: str):
    """Persist metrics dict as JSON."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {path}")
