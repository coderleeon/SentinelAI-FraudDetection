"""
API Integration Tests
======================
Tests the FastAPI endpoints using httpx.TestClient with mocked model loading
so that tests can run without pre-trained model artefacts.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Shared fixtures ───────────────────────────────────────────────────────────

VALID_TXN = {
    "time": 172_800,
    "amount": 250.00,
    "V1": -1.3598, "V2": -0.0728, "V3":  2.5363, "V4":  1.3782,
    "V5": -0.3383, "V6":  0.4624, "V7":  0.2396, "V8":  0.0987,
    "V9":  0.3638, "V10": 0.0908, "V11":-0.5517, "V12":-0.6178,
    "V13":-0.9914, "V14":-0.3111, "V15": 1.4681, "V16":-0.4704,
    "V17": 0.2079, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21":-0.0183, "V22": 0.2778, "V23":-0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26":-0.1891, "V27": 0.1336, "V28":-0.0211,
    "transaction_id": "TEST-001",
}

FRAUD_TXN = {
    **VALID_TXN,
    "V1": -8.5, "V3": -7.2, "V4": 5.8, "V14": -10.2, "V17": -6.3,
    "amount": 4_999.99,
    "transaction_id": "FRAUD-001",
}


@pytest.fixture(scope="module")
def client():
    """TestClient with model loading mocked out."""
    with (
        patch("api.predictor.FraudPredictor.load", return_value=True),
        patch("api.predictor.FraudPredictor.is_loaded", return_value=True),
        patch("api.predictor.FraudPredictor.predict", return_value=(0.12, None)),
        patch("api.predictor.FraudPredictor._ready", True, create=True),
    ):
        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/health").json()
        for field in ("status", "model_loaded", "version", "uptime_seconds"):
            assert field in data, f"Missing: {field}"

    def test_uptime_positive(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:

    def test_valid_transaction_200(self, client):
        with patch("api.predictor.predictor.is_loaded", return_value=True), \
             patch("api.predictor.predictor.predict",   return_value=(0.05, None)):
            r = client.post("/predict", json=VALID_TXN)
        assert r.status_code == 200

    def test_response_contains_required_fields(self, client):
        with patch("api.predictor.predictor.is_loaded", return_value=True), \
             patch("api.predictor.predictor.predict",   return_value=(0.05, None)):
            data = client.post("/predict", json=VALID_TXN).json()

        required = [
            "fraud_probability", "risk_score", "is_fraud",
            "risk_level", "confidence", "reasons", "model_used",
            "processing_time_ms",
        ]
        for f in required:
            assert f in data, f"Missing field: {f}"

    def test_fraud_probability_in_range(self, client):
        with patch("api.predictor.predictor.is_loaded", return_value=True), \
             patch("api.predictor.predictor.predict",   return_value=(0.73, None)):
            data = client.post("/predict", json=VALID_TXN).json()
        assert 0 <= data["fraud_probability"] <= 1
        assert 0 <= data["risk_score"]        <= 1

    def test_risk_level_valid_enum(self, client):
        with patch("api.predictor.predictor.is_loaded", return_value=True), \
             patch("api.predictor.predictor.predict",   return_value=(0.05, None)):
            data = client.post("/predict", json=VALID_TXN).json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_missing_v_fields_returns_422(self, client):
        incomplete = {"time": 1_000, "amount": 100.0}   # Missing V1-V28
        r = client.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_negative_amount_returns_422(self, client):
        bad = {**VALID_TXN, "amount": -99.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_high_amount_triggers_reason(self, client):
        """A $6,000 transaction should carry at least one rule reason."""
        txn = {**VALID_TXN, "amount": 6_000.0}
        with patch("api.predictor.predictor.is_loaded", return_value=True), \
             patch("api.predictor.predictor.predict",   return_value=(0.2, None)):
            data = client.post("/predict", json=txn).json()
        assert len(data["reasons"]) >= 1


# ── Models endpoint ───────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_returns_200_and_schema(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert "active_model" in data


# ── Business logic unit tests ─────────────────────────────────────────────────

class TestBusinessLogic:

    def _make_txn(self, **overrides):
        from api.schemas import TransactionInput
        base = dict(VALID_TXN)
        base.update(overrides)
        return TransactionInput(**base)

    def test_critical_amount_triggers(self):
        from api.business_logic import compute_rule_score
        txn   = self._make_txn(amount=6_000.0)
        score, reasons = compute_rule_score(txn)
        assert score >= 0.5
        assert any("critical" in r.lower() or "5,000" in r for r in reasons)

    def test_high_amount_triggers(self):
        from api.business_logic import compute_rule_score
        txn   = self._make_txn(amount=1_500.0)
        score, reasons = compute_rule_score(txn)
        assert score >= 0.25
        assert len(reasons) >= 1

    def test_micro_transaction_triggers(self):
        from api.business_logic import compute_rule_score
        txn   = self._make_txn(amount=0.50)
        score, reasons = compute_rule_score(txn)
        assert score > 0
        assert any("micro" in r.lower() for r in reasons)

    def test_hybrid_formula(self):
        """risk = 0.70*ml + 0.30*rules ≥ 0.70*ml for ml=0.8 → ≥ 0.56"""
        from api.business_logic import compute_hybrid_risk_score
        txn = self._make_txn()
        risk, rule, reasons, level = compute_hybrid_risk_score(0.8, txn)
        assert risk >= 0.56
        assert 0 <= risk <= 1

    def test_low_risk_classification(self):
        from api.business_logic import compute_hybrid_risk_score, RiskLevel
        txn = self._make_txn(amount=10.0)
        risk, _, _, level = compute_hybrid_risk_score(0.01, txn)
        assert level in (RiskLevel.LOW, RiskLevel.MEDIUM)
