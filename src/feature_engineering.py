"""
Feature Engineering
===================
Derives additional domain-specific features from raw transaction data.
These features supplement the V1-V28 PCA components with interpretable
business signals that improve model performance and explainability.
"""

import numpy as np
import pandas as pd


# Nighttime hours (UTC): highest fraud incidence window
_NIGHT_HOURS = set(range(23, 24)) | set(range(0, 6))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-derived features to a transaction DataFrame.

    New features:
    ┌──────────────────┬─────────────────────────────────────────────────────┐
    │ amount_log       │ log1p(Amount) — reduces right-skew                  │
    │ hour_of_day      │ hour derived from Time (seconds offset)             │
    │ is_night         │ 1 if hour in [23:00 – 05:59] UTC                    │
    │ amount_zscore    │ z-score of Amount (dataset-level)                   │
    │ v1_v3_product    │ V1 × V3 interaction (key fraud discriminators)      │
    │ v_primary_norm   │ L2 norm of V1–V14 (primary PCA variance)            │
    └──────────────────┴─────────────────────────────────────────────────────┘

    Args:
        df: DataFrame that must contain [Time, Amount, V1-V28].

    Returns:
        New DataFrame with additional feature columns appended.
    """
    df = df.copy()

    # 1. Log-transform of Amount (handles heavy right tail)
    df["amount_log"] = np.log1p(df["Amount"])

    # 2. Hour-of-day from Time (seconds since reference)
    df["hour_of_day"] = (df["Time"] // 3600) % 24

    # 3. Nighttime binary flag
    df["is_night"] = df["hour_of_day"].apply(lambda h: int(h in _NIGHT_HOURS))

    # 4. Amount z-score (detects unusual spend relative to dataset mean)
    mean_amt = df["Amount"].mean()
    std_amt  = df["Amount"].std() + 1e-9
    df["amount_zscore"] = (df["Amount"] - mean_amt) / std_amt

    # 5. V1 × V3 interaction term (both strongly discriminative in fraud)
    if "V1" in df.columns and "V3" in df.columns:
        df["v1_v3_product"] = df["V1"] * df["V3"]

    # 6. L2 norm of primary PCA components
    v_primary_cols = [f"V{i}" for i in range(1, 15) if f"V{i}" in df.columns]
    df["v_primary_norm"] = np.linalg.norm(df[v_primary_cols].values, axis=1)

    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excluding target 'Class')."""
    return [c for c in df.columns if c != "Class"]
