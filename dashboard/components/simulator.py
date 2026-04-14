"""
Dashboard Transaction Simulator
================================
Drives the Streamlit dashboard by generating synthetic transactions,
submitting them to the FastAPI backend, collecting results, and
maintaining session history for chart rendering.

Falls back gracefully to offline rule-based scoring when the API is
unavailable — enabling full dashboard demo mode without a running backend.
"""

from __future__ import annotations

import logging
import sys
import os
import time
from typing import List, Optional

import httpx

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from streaming.transaction_stream import TransactionStream, StreamedTransaction

logger = logging.getLogger(__name__)


class DashboardSimulator:
    """
    Streamlit-side simulation controller.

    Responsibilities:
      1. Generate batches of synthetic transactions via TransactionStream.
      2. Submit each transaction to POST /predict (or use offline scoring).
      3. Accumulate a bounded history list for charting.
      4. Maintain a fraud alert log for the Alerts tab.
    """

    MAX_HISTORY = 500   # keep last N transactions in memory

    def __init__(
        self,
        api_url:    str   = "http://localhost:8000",
        tps:        float = 1.0,
        fraud_rate: float = 0.05,
    ):
        self.api_url = api_url.rstrip("/")
        self.stream  = TransactionStream(
            transactions_per_second=tps,
            fraud_rate=fraud_rate,
            use_kafka_queue=False,   # not needed for sync dashboard use
        )
        self.history:    List[dict] = []
        self.alert_log:  List[str]  = []
        self._api_alive: Optional[bool] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def run_batch(self, n: int = 5) -> List[dict]:
        """
        Generate n transactions, score them, and return result dicts.
        Results are also appended to self.history.
        """
        results = []
        batch = next(
            self.stream.stream_sync(max_transactions=n, batch_size=n),
            [],
        )

        for txn in batch:
            result = self._score(txn)
            if result is not None:
                self._record(result, txn)
                results.append(result)

        return results

    def predict_single(self, txn: StreamedTransaction) -> Optional[dict]:
        """Score one transaction (used by the Manual Test tab)."""
        result = self._score(txn)
        if result:
            self._record(result, txn)
        return result

    def get_stats(self) -> dict:
        """Return aggregate statistics for the current session."""
        h = self.history
        if not h:
            return {
                "total": 0, "fraud": 0, "legit": 0,
                "fraud_rate": 0.0, "avg_prob": 0.0,
                "avg_risk": 0.0, "total_amount": 0.0,
            }
        fraud_n = sum(1 for t in h if t.get("is_fraud"))
        return {
            "total":        len(h),
            "fraud":        fraud_n,
            "legit":        len(h) - fraud_n,
            "fraud_rate":   fraud_n / len(h) * 100,
            "avg_prob":     sum(t.get("fraud_probability", 0) for t in h) / len(h),
            "avg_risk":     sum(t.get("risk_score", 0)        for t in h) / len(h),
            "total_amount": sum(t.get("amount", 0)            for t in h),
        }

    def reset(self):
        """Clear session data."""
        self.history.clear()
        self.alert_log.clear()
        self._api_alive = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _score(self, txn: StreamedTransaction) -> Optional[dict]:
        """Try API → fallback to offline scoring."""
        result = self._call_api(txn)
        if result is None:
            result = self._offline_score(txn)
        return result

    def _call_api(self, txn: StreamedTransaction) -> Optional[dict]:
        """POST to /predict and return the parsed JSON response."""
        try:
            payload = txn.to_api_dict()
            with httpx.Client(timeout=4.0) as client:
                resp = client.post(f"{self.api_url}/predict", json=payload)
            if resp.status_code == 200:
                self._api_alive = True
                return resp.json()
            logger.warning("API %d: %s", resp.status_code, resp.text[:120])
            return None
        except httpx.ConnectError:
            if self._api_alive is not False:
                logger.info("API unreachable — switching to offline mode")
            self._api_alive = False
            return None
        except Exception as exc:
            logger.debug("API call error: %s", exc)
            return None

    def _offline_score(self, txn: StreamedTransaction) -> dict:
        """
        Lightweight rule-based scorer used when the API is unavailable.
        Provides a reasonable demo experience without requiring a backend.
        """
        import math

        amount_risk = min(txn.amount / 5_000.0, 1.0)
        pca_keys    = [1, 3, 4, 10, 12, 14, 17]
        pca_score   = min(
            sum(abs(getattr(txn, f"V{i}")) for i in pca_keys) / 35.0, 1.0
        )
        prob     = 0.4 * amount_risk + 0.6 * pca_score
        is_fraud = prob > 0.50 or txn.is_synthetic_fraud

        risk_level = (
            "CRITICAL" if prob > 0.85 else
            "HIGH"     if prob > 0.65 else
            "MEDIUM"   if prob > 0.35 else "LOW"
        )
        reasons = []
        if txn.amount >= 5_000:
            reasons.append(f"Offline: critical amount ${txn.amount:,.2f}")
        if pca_score > 0.5:
            reasons.append("Offline: anomalous PCA signature")

        return {
            "transaction_id":    txn.transaction_id,
            "fraud_probability": round(prob, 4),
            "risk_score":        round(prob, 4),
            "is_fraud":          is_fraud,
            "risk_level":        risk_level,
            "confidence":        round(abs(prob - 0.5) * 2, 4),
            "reasons":           reasons,
            "model_used":        "offline_rules",
            "processing_time_ms": 0.2,
            "shap_explanation":  None,
        }

    def _record(self, result: dict, txn: StreamedTransaction):
        """Append result to bounded history; fire alert if fraud."""
        result = dict(result)
        result.setdefault("amount",    txn.amount)
        result.setdefault("timestamp", time.time())
        result.setdefault("id",        txn.transaction_id)
        result["is_fraud"] = bool(result.get("is_fraud", False))

        self.history.append(result)
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

        if result["is_fraud"]:
            ts    = time.strftime("%H:%M:%S")
            level = result.get("risk_level", "?")
            score = result.get("risk_score", 0)
            alert = (
                f"🚨 {ts} | {txn.transaction_id} | "
                f"${txn.amount:,.2f} | Risk {score:.3f} ({level})"
            )
            self.alert_log.append(alert)
            if len(self.alert_log) > 100:
                self.alert_log.pop(0)
