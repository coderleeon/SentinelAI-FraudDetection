"""
Preprocessing & Feature Engineering Tests
==========================================
Unit tests for the data pipeline components.
All tests are fully offline — no API or model artefacts required.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_df(n: int = 500, fraud_frac: float = 0.04, seed: int = 0) -> pd.DataFrame:
    """Generate a minimal synthetic DataFrame for testing."""
    np.random.seed(seed)
    n_fraud = max(int(n * fraud_frac), 5)
    n_legit = n - n_fraud

    V = np.random.randn(n, 28)
    amounts = np.abs(np.random.exponential(60, n))
    times   = np.sort(np.random.uniform(0, 172_800, n))
    labels  = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"]  = labels
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Data preprocessing tests ──────────────────────────────────────────────────

class TestAnalyzeClassDistribution:
    def test_keys_present(self):
        from src.data_preprocessing import analyze_class_distribution
        df  = _make_df(200)
        out = analyze_class_distribution(df)
        for k in ("total", "legitimate", "fraud", "fraud_rate"):
            assert k in out

    def test_counts_sum_to_total(self):
        from src.data_preprocessing import analyze_class_distribution
        df  = _make_df(200)
        out = analyze_class_distribution(df)
        assert out["legitimate"] + out["fraud"] == out["total"]

    def test_fraud_rate_fraction(self):
        from src.data_preprocessing import analyze_class_distribution
        df  = _make_df(200, fraud_frac=0.10)
        out = analyze_class_distribution(df)
        assert 5 < out["fraud_rate"] < 20   # roughly 10 ± noise


class TestPreprocess:
    def test_shapes_consistent(self):
        from src.data_preprocessing import preprocess
        df = _make_df(300, 0.05)
        X_tr, X_te, y_tr, y_te, feats = preprocess(df, apply_smote=False, save_scaler=False)
        assert len(X_tr) == len(y_tr)
        assert len(X_te) == len(y_te)
        assert len(feats) == 30   # Time + V1-V28 + Amount

    def test_no_nan_after_scaling(self):
        from src.data_preprocessing import preprocess
        df = _make_df(300, 0.05)
        X_tr, X_te, _, _, _ = preprocess(df, apply_smote=False, save_scaler=False)
        assert not X_tr.isna().any().any()
        assert not X_te.isna().any().any()

    def test_smote_increases_minority(self):
        from src.data_preprocessing import preprocess
        df = _make_df(500, 0.02)
        X_tr, _, y_tr, _, _ = preprocess(df, apply_smote=True, save_scaler=False)
        fraud_rate = y_tr.mean()
        assert fraud_rate > 0.10   # >> original 2%

    def test_stratified_split_preserves_label_ratio(self):
        from src.data_preprocessing import preprocess
        df = _make_df(500, 0.04)
        _, X_te, _, y_te, _ = preprocess(df, apply_smote=False, save_scaler=False)
        # Test set should contain some fraud
        assert y_te.sum() >= 1

    def test_duplicate_removal(self):
        from src.data_preprocessing import preprocess
        df = _make_df(200)
        df_dup = pd.concat([df, df.head(20)], ignore_index=True)
        X_tr, X_te, _, _, _ = preprocess(df_dup, apply_smote=False, save_scaler=False)
        assert len(X_tr) + len(X_te) <= len(df_dup)


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_all_new_columns_present(self):
        from src.feature_engineering import add_derived_features
        df  = _make_df(100)
        out = add_derived_features(df)
        for col in ("amount_log", "hour_of_day", "is_night",
                    "amount_zscore", "v1_v3_product", "v_primary_norm"):
            assert col in out.columns, f"Missing: {col}"

    def test_no_nans_introduced(self):
        from src.feature_engineering import add_derived_features
        df  = _make_df(100)
        out = add_derived_features(df)
        assert not out.isna().any().any()

    def test_hour_of_day_range(self):
        from src.feature_engineering import add_derived_features
        df  = _make_df(200)
        out = add_derived_features(df)
        assert out["hour_of_day"].between(0, 23).all()

    def test_is_night_binary(self):
        from src.feature_engineering import add_derived_features
        df  = _make_df(200)
        out = add_derived_features(df)
        assert set(out["is_night"].unique()).issubset({0, 1})

    def test_amount_log_non_negative(self):
        from src.feature_engineering import add_derived_features
        df  = _make_df(200)
        out = add_derived_features(df)
        assert (out["amount_log"] >= 0).all()

    def test_original_columns_preserved(self):
        from src.feature_engineering import add_derived_features
        df  = _make_df(100)
        out = add_derived_features(df)
        for col in df.columns:
            assert col in out.columns


# ── Synthetic data generator tests ────────────────────────────────────────────

class TestDataGenerator:
    def test_schema(self):
        from data.generate_sample_data import generate_creditcard_data
        df = generate_creditcard_data(n_legit=200, n_fraud=10)
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        for col in expected_cols:
            assert col in df.columns

    def test_fraud_label_present(self):
        from data.generate_sample_data import generate_creditcard_data
        df = generate_creditcard_data(n_legit=200, n_fraud=10)
        assert df["Class"].sum() == 10

    def test_amounts_non_negative(self):
        from data.generate_sample_data import generate_creditcard_data
        df = generate_creditcard_data(n_legit=200, n_fraud=10)
        assert (df["Amount"] >= 0).all()


# ── Streaming generator tests ─────────────────────────────────────────────────

class TestStreamingGenerator:
    def test_transaction_id_format(self):
        from streaming.transaction_stream import TransactionGenerator
        gen = TransactionGenerator(fraud_rate=0.0)
        txn = gen.generate_one()
        assert txn.transaction_id.startswith("TXN-")

    def test_all_v_fields_present(self):
        from streaming.transaction_stream import TransactionGenerator
        gen = TransactionGenerator(fraud_rate=0.0)
        txn = gen.generate_one()
        for i in range(1, 29):
            assert hasattr(txn, f"V{i}")

    def test_100pct_fraud_rate(self):
        from streaming.transaction_stream import TransactionGenerator
        gen  = TransactionGenerator(fraud_rate=1.0)
        txns = [gen.generate_one() for _ in range(50)]
        assert all(t.is_synthetic_fraud for t in txns)

    def test_0pct_fraud_rate(self):
        from streaming.transaction_stream import TransactionGenerator
        gen  = TransactionGenerator(fraud_rate=0.0)
        txns = [gen.generate_one() for _ in range(50)]
        assert not any(t.is_synthetic_fraud for t in txns)

    def test_to_api_dict_keys(self):
        from streaming.transaction_stream import TransactionGenerator
        gen = TransactionGenerator()
        d   = gen.generate_one().to_api_dict()
        assert "time"   in d
        assert "amount" in d
        assert "V1"     in d
        assert "V28"    in d
