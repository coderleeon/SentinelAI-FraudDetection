"""
API Input / Output Schemas
===========================
Pydantic v2 models for strict request validation and typed response serialisation.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Request ───────────────────────────────────────────────────────────────────

class TransactionInput(BaseModel):
    """
    Single credit card transaction submitted for fraud scoring.
    Fields mirror the Kaggle Credit Card Fraud Detection dataset schema.
    """

    # Time & amount
    time:   float = Field(..., ge=0,       description="Seconds elapsed since the first transaction in the rolling window")
    amount: float = Field(..., ge=0, le=1_000_000, description="Transaction amount in USD")

    # PCA components (V1-V28 are anonymised via PCA by the card issuer)
    V1:  float = Field(...)
    V2:  float = Field(...)
    V3:  float = Field(...)
    V4:  float = Field(...)
    V5:  float = Field(...)
    V6:  float = Field(...)
    V7:  float = Field(...)
    V8:  float = Field(...)
    V9:  float = Field(...)
    V10: float = Field(...)
    V11: float = Field(...)
    V12: float = Field(...)
    V13: float = Field(...)
    V14: float = Field(...)
    V15: float = Field(...)
    V16: float = Field(...)
    V17: float = Field(...)
    V18: float = Field(...)
    V19: float = Field(...)
    V20: float = Field(...)
    V21: float = Field(...)
    V22: float = Field(...)
    V23: float = Field(...)
    V24: float = Field(...)
    V25: float = Field(...)
    V26: float = Field(...)
    V27: float = Field(...)
    V28: float = Field(...)

    # Optional metadata (does not affect scoring)
    transaction_id:    Optional[str] = Field(None, description="Caller-supplied transaction identifier")
    merchant_category: Optional[str] = Field(None, description="Merchant category code (MCC)")

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "time": 172_800,
                "amount": 250.00,
                "V1": -1.3598,  "V2": -0.0728, "V3": 2.5363,  "V4": 1.3782,
                "V5": -0.3383,  "V6":  0.4624, "V7": 0.2396,  "V8": 0.0987,
                "V9":  0.3638, "V10":  0.0908, "V11": -0.5517, "V12": -0.6178,
                "V13": -0.9914, "V14": -0.3111, "V15": 1.4681, "V16": -0.4704,
                "V17": 0.2079,  "V18":  0.0258, "V19": 0.4040, "V20": 0.2514,
                "V21": -0.0183, "V22": 0.2778,  "V23": -0.1105, "V24": 0.0669,
                "V25": 0.1285,  "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
                "transaction_id": "TXN-20240101-001",
            }
        }
    }


# ── Response ──────────────────────────────────────────────────────────────────

class SHAPFeature(BaseModel):
    """Impact of one feature on a single prediction."""
    feature: str
    impact:  float   # SHAP value (positive → pushes toward fraud)
    value:   float   # Raw feature value after scaling


class SHAPExplanation(BaseModel):
    top_features: List[SHAPFeature]
    base_value:   float   # Model's expected output (log-odds)


class PredictionResponse(BaseModel):
    """
    Full fraud scoring response.

    The two key scores:
    • fraud_probability  — raw ML model output [0, 1]
    • risk_score         — hybrid score: 0.70 × ML + 0.30 × rule-based [0, 1]

    is_fraud is decided on risk_score ≥ 0.50 (configurable threshold).
    """
    transaction_id:    Optional[str]
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_score:        float = Field(..., ge=0, le=1)
    is_fraud:          bool
    risk_level:        RiskLevel
    confidence:        float = Field(..., ge=0, le=1)
    reasons:           List[str]
    shap_explanation:  Optional[SHAPExplanation] = None
    model_used:        str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status:         str           # "healthy" | "degraded"
    model_loaded:   bool
    active_model:   str
    version:        str
    uptime_seconds: float
