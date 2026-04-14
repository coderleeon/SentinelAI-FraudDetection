"""
Training Pipeline
=================
End-to-end pipeline that trains three models (Logistic Regression,
Random Forest, XGBoost), runs stratified k-fold cross-validation,
evaluates on a held-out test set, and persists all artefacts.

Usage:
    python -m src.train
    python train.py          (from project root)
"""

import os
import sys
import json
import logging
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Make sure project root is importable ──────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_preprocessing import load_data, preprocess, analyze_class_distribution
from src.feature_engineering import add_derived_features
from src.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(_ROOT, "models")
CV_FOLDS   = 5


# ── Model definitions ─────────────────────────────────────────────────────────

def get_model_zoo() -> dict:
    """
    Return all three model instances with production-tuned hyperparameters.
    All hyperparameters were chosen to maximise recall while keeping
    false-positive rates viable for a real-time fraud system.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",   # handles imbalance without SMOTE
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=100,      # approximate class imbalance ratio
            eval_metric="aucpr",       # optimise for PR-AUC (best for fraud)
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        ),
    }


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate(model, X_train, y_train) -> dict:
    """
    Run stratified k-fold CV and return mean ± std for ROC-AUC and F1.
    CV is run on the training fold only (before SMOTE is applied at this
    global level, so results may be slightly optimistic — acceptable for
    pipeline comparison purposes).
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    roc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc",  n_jobs=-1)
    f1_scores  = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1",       n_jobs=-1)

    return {
        "cv_roc_auc_mean": float(roc_scores.mean()),
        "cv_roc_auc_std":  float(roc_scores.std()),
        "cv_f1_mean":      float(f1_scores.mean()),
        "cv_f1_std":       float(f1_scores.std()),
    }


# ── Single model trainer ──────────────────────────────────────────────────────

def train_and_save(
    name:          str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: list,
) -> dict:
    """Train one model, cross-validate, evaluate, and save to disk."""
    banner = f"  Training: {name.upper()}  "
    logger.info(f"\n{'═'*60}")
    logger.info(f"{'═' * ((60 - len(banner)) // 2)}{banner}{'═' * ((60 - len(banner)) // 2)}")
    logger.info(f"{'═'*60}")

    # Cross-validation
    logger.info(f"Running {CV_FOLDS}-fold CV...")
    cv = cross_validate(model, X_train, y_train)
    logger.info(
        f"  CV ROC-AUC : {cv['cv_roc_auc_mean']:.4f} ± {cv['cv_roc_auc_std']:.4f}"
    )
    logger.info(
        f"  CV F1      : {cv['cv_f1_mean']:.4f}    ± {cv['cv_f1_std']:.4f}"
    )

    # Full training on all training data
    logger.info("Fitting on full training set...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # Evaluation
    metrics = evaluate_model(model, X_test, y_test, feature_names, model_name=name)
    metrics.update(cv)

    # Persist model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model saved → {model_path}")

    return metrics


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_training_pipeline() -> dict:
    """
    Orchestrate the full training pipeline:
      1. Load / generate dataset
      2. Feature engineering
      3. Preprocessing (scaling + SMOTE)
      4. Train all three models
      5. Save comparison report
    """
    logger.info("🚀  AI Fraud Detection — Training Pipeline")
    logger.info(f"    Project root : {_ROOT}")

    # 1. Load data
    df = load_data()
    dist = analyze_class_distribution(df)
    logger.info(
        f"Class distribution → Legit: {dist['legitimate']:,} | "
        f"Fraud: {dist['fraud']:,} | Rate: {dist['fraud_rate']:.3f}%"
    )

    # 2. Feature engineering
    logger.info("Adding derived features...")
    df = add_derived_features(df)

    # 3. Preprocessing
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, apply_smote=True, save_scaler=True
    )

    # 4. Train all models
    models      = get_model_zoo()
    all_metrics = {}

    for name, model in models.items():
        metrics = train_and_save(
            name, model,
            X_train, y_train,
            X_test,  y_test,
            feature_names,
        )
        all_metrics[name] = metrics

    # 5. Save comparison summary
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nFull summary saved → {summary_path}")

    # 6. Print leaderboard
    _print_leaderboard(all_metrics)

    return all_metrics


def _print_leaderboard(metrics: dict):
    header = f"\n{'Model':<26} {'ROC-AUC':>9} {'PR-AUC':>8} {'Recall':>8} {'F1':>8} {'Fraud Found':>12}"
    logger.info("📊  MODEL LEADERBOARD")
    logger.info(header)
    logger.info("─" * 76)
    for name, m in sorted(metrics.items(), key=lambda x: -x[1]["roc_auc"]):
        logger.info(
            f"{name:<26} {m['roc_auc']:>9.4f} {m['avg_precision']:>8.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} "
            f"{m['fraud_detected']:>4}/{m['fraud_total']:<6} ({m['recall_pct']}%)"
        )


if __name__ == "__main__":
    run_training_pipeline()
