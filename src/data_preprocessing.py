"""
Data Preprocessing Pipeline
============================
Handles data loading, cleaning, class-imbalance correction (SMOTE),
feature scaling, and train/test splitting.

All transformations are fit on training data only to prevent data leakage.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

logger = logging.getLogger(__name__)

# Default paths (resolved relative to project root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(_ROOT, "data", "creditcard.csv")
MODELS_DIR  = os.path.join(_ROOT, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the credit card dataset from CSV.
    Falls back to generating a synthetic dataset if the file doesn't exist.
    """
    if not os.path.exists(path):
        logger.warning(f"Dataset not found at '{path}'. Generating synthetic data...")
        from data.generate_sample_data import generate_creditcard_data
        df = generate_creditcard_data()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Synthetic dataset saved → {path}")
    else:
        df = pd.read_csv(path)

    logger.info(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── Class Distribution ────────────────────────────────────────────────────────

def analyze_class_distribution(df: pd.DataFrame) -> dict:
    """Return a summary of class balance in the dataset."""
    counts = df["Class"].value_counts()
    total  = len(df)
    return {
        "total":       total,
        "legitimate":  int(counts.get(0, 0)),
        "fraud":       int(counts.get(1, 0)),
        "fraud_rate":  float(counts.get(1, 0) / total * 100),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    test_size:    float = 0.20,
    random_state: int   = 42,
    apply_smote:  bool  = True,
    save_scaler:  bool  = True,
):
    """
    Full preprocessing pipeline:

    1. Drop exact duplicates
    2. Separate features from target
    3. Fit StandardScaler on training data
    4. Stratified train/test split
    5. SMOTE oversampling on training set (optional)
    6. Persist scaler and feature names for inference

    Args:
        df:           Raw DataFrame with a 'Class' column.
        test_size:    Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.
        apply_smote:  Apply SMOTE to balance training classes.
        save_scaler:  Persist scaler and feature metadata to disk.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    logger.info("─── Preprocessing pipeline start ─────────────────────────")

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    # 2. Features / target
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].astype(int)
    feature_names = X.columns.tolist()
    logger.info(f"Features: {len(feature_names)} | {feature_names[:5]} ...")

    # 3. Train/test split before fitting scaler (strict no-leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"Split → Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"Train fraud rate before SMOTE: {y_train.mean()*100:.3f}%")

    # 4. Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test  = pd.DataFrame(X_test_scaled,  columns=feature_names)

    # 5. Save artefacts
    if save_scaler:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        with open(FEATURE_NAMES_PATH, "w") as f:
            json.dump(feature_names, f)
        logger.info(f"Scaler saved → {SCALER_PATH}")
        logger.info(f"Feature names saved → {FEATURE_NAMES_PATH}")

    # 6. SMOTE on training set
    if apply_smote:
        logger.info("Applying SMOTE oversampling...")
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_arr, y_train_arr = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=feature_names)
        y_train = pd.Series(y_train_arr, name="Class")
        logger.info(
            f"Post-SMOTE → Train: {len(X_train):,} | "
            f"Fraud: {y_train.sum():,} | Legit: {(y_train==0).sum():,}"
        )

    logger.info("─── Preprocessing pipeline complete ──────────────────────")
    return X_train, X_test, y_train, y_test, feature_names


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    """Load the fitted scaler from disk."""
    return joblib.load(path)


def load_feature_names(path: str = FEATURE_NAMES_PATH) -> list:
    """Load the ordered feature name list from disk."""
    with open(path) as f:
        return json.load(f)
