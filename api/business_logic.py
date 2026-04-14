"""
Business Logic Layer — Hybrid ML + Rule-Based Risk Scoring
===========================================================
Augments the raw ML probability with domain-specific rules to produce a
hybrid risk score that reflects real-world fraud patterns more robustly
than a single model probability.

Formula
-------
    risk_score = (ML_WEIGHT × fraud_probability) + (RULE_WEIGHT × rule_score)

Triggered rules are surfaced in the API response `reasons` field, giving
operations teams immediate, human-readable context for each decision.
"""

from __future__ import annotations

import logging
from typing import Tuple, List

from api.schemas import TransactionInput, RiskLevel

logger = logging.getLogger(__name__)

# ── Scoring weights ────────────────────────────────────────────────────────────
ML_WEIGHT   = 0.70
RULE_WEIGHT = 0.30

# ── Configurable thresholds (can be loaded from env/config in production) ──────
AMOUNT_HIGH     = 1_000.0    # USD
AMOUNT_CRITICAL = 5_000.0    # USD
AMOUNT_MICRO    = 1.00       # USD — card-testing pattern
NIGHT_HOURS     = set(range(23, 24)) | set(range(0, 6))   # 11 pm – 5:59 am UTC
PCA_SIGMA_WARN  = 3.0        # Standard deviations from mean
PCA_WARN_COUNT  = 3          # Number of extreme PCA features to trigger warning
PCA_ALERT_COUNT = 5          # Number of extreme PCA features to trigger alert
DECISION_THRESHOLD = 0.50    # risk_score threshold for is_fraud classification


# ── Rule engine ───────────────────────────────────────────────────────────────

def compute_rule_score(txn: TransactionInput) -> Tuple[float, List[str]]:
    """
    Apply domain rules to a transaction and return (rule_score, reasons).

    rule_score is clamped to [0, 1]. Each triggered rule adds a partial score;
    multiple rules compound — a transaction can be flagged by ML and rules simultaneously.

    Rules applied:
      1. High / critical transaction amount
      2. Nighttime transaction (UTC)
      3. Multiple anomalous PCA components (>3σ)
      4. Micro-transaction (card-testing pattern)
    """
    score   = 0.0
    reasons: List[str] = []

    # ── Rule 1: Transaction amount ────────────────────────────────────────────
    if txn.amount >= AMOUNT_CRITICAL:
        score += 0.50
        reasons.append(
            f"Critical-value transaction: ${txn.amount:,.2f} exceeds ${AMOUNT_CRITICAL:,.0f} threshold"
        )
    elif txn.amount >= AMOUNT_HIGH:
        score += 0.25
        reasons.append(
            f"High-value transaction: ${txn.amount:,.2f} exceeds ${AMOUNT_HIGH:,.0f} threshold"
        )

    # ── Rule 2: Nighttime transaction ─────────────────────────────────────────
    hour = int((txn.time // 3600) % 24)
    if hour in NIGHT_HOURS:
        score += 0.15
        reasons.append(f"Off-hours transaction at {hour:02d}:00 UTC (elevated fraud window)")

    # ── Rule 3: Anomalous PCA component values ────────────────────────────────
    pca_vals    = [getattr(txn, f"V{i}") for i in range(1, 29)]
    extreme_cnt = sum(1 for v in pca_vals if abs(v) > PCA_SIGMA_WARN)

    if extreme_cnt >= PCA_ALERT_COUNT:
        score += 0.30
        reasons.append(
            f"Highly anomalous behavioural signature: {extreme_cnt} PCA features "
            f"exceed {PCA_SIGMA_WARN}σ (expected ≤ {PCA_WARN_COUNT})"
        )
    elif extreme_cnt >= PCA_WARN_COUNT:
        score += 0.10
        reasons.append(
            f"Unusual behavioural signature: {extreme_cnt} PCA features exceed {PCA_SIGMA_WARN}σ"
        )

    # ── Rule 4: Micro-transaction (card-testing) ──────────────────────────────
    if 0 < txn.amount <= AMOUNT_MICRO:
        score += 0.20
        reasons.append(
            f"Micro-transaction (${txn.amount:.2f}) — common card-testing precursor to fraud"
        )

    return min(score, 1.0), reasons


# ── Hybrid scorer ─────────────────────────────────────────────────────────────

def compute_hybrid_risk_score(
    ml_probability: float,
    transaction:    TransactionInput,
) -> Tuple[float, float, List[str], RiskLevel]:
    """
    Combine ML probability and rule score into a single risk score.

    Args:
        ml_probability: Raw model output probability of fraud [0, 1].
        transaction:    Validated transaction input.

    Returns:
        (risk_score, rule_score, reasons, risk_level)
    """
    rule_score, reasons = compute_rule_score(transaction)
    risk_score = min(
        (ML_WEIGHT * ml_probability) + (RULE_WEIGHT * rule_score),
        1.0,
    )

    risk_level = _classify_risk(risk_score)

    logger.debug(
        f"Hybrid score | ML={ml_probability:.4f}  Rules={rule_score:.4f}  "
        f"→ Risk={risk_score:.4f}  Level={risk_level}"
    )
    return risk_score, rule_score, reasons, risk_level


def _classify_risk(score: float) -> RiskLevel:
    if score >= 0.85:
        return RiskLevel.CRITICAL
    if score >= 0.65:
        return RiskLevel.HIGH
    if score >= 0.35:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
