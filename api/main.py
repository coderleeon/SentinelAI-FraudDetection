"""
FastAPI Application — AI Fraud Detection System
================================================
Production-grade REST API for real-time credit card fraud detection.

Endpoints:
  POST /predict   → hybrid ML + rule-based fraud score with SHAP explanation
  GET  /health    → system liveness check
  GET  /models    → list available trained model artefacts

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  python run_api.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.schemas import (
    HealthResponse,
    PredictionResponse,
    SHAPExplanation,
    SHAPFeature,
    TransactionInput,
)
from api.predictor import predictor
from api.business_logic import compute_hybrid_risk_score, DECISION_THRESHOLD

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fraud_api")

_START_TIME     = time.time()
_PRIMARY_MODEL  = os.getenv("PRIMARY_MODEL", "xgboost")


# ── Alert logger ──────────────────────────────────────────────────────────────

_ALERT_LOG_PATH = os.path.join(_ROOT, "models", "fraud_alerts.log")

def _setup_alert_logger() -> logging.Logger:
    os.makedirs(os.path.dirname(_ALERT_LOG_PATH), exist_ok=True)
    al = logging.getLogger("fraud_alerts")
    if not al.handlers:
        fh = logging.FileHandler(_ALERT_LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        al.addHandler(fh)
    al.setLevel(logging.WARNING)
    return al

alert_logger = _setup_alert_logger()


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; release resources on shutdown."""
    logger.info("🚀 Starting AI Fraud Detection API …")

    loaded = predictor.load(model_name=_PRIMARY_MODEL)
    if not loaded:
        # Try fall-backs in priority order
        for fallback in ("random_forest", "logistic_regression"):
            logger.warning(f"Trying fallback model: {fallback}")
            if predictor.load(model_name=fallback):
                break
        else:
            logger.error(
                "❌ No trained model found.  Run: python train.py"
            )

    if predictor.is_loaded():
        logger.info(f"✅ API ready — active model: {getattr(predictor, 'model_name', 'unknown')}")
    yield
    logger.info("Shutting down …")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Fraud Detection API",
    description="""
## Real-Time Credit Card Fraud Detection

Uses an ensemble of **Logistic Regression**, **Random Forest**, and
**XGBoost** models trained on credit card transaction data.

### Scoring model
```
risk_score = 0.70 × ML_probability + 0.30 × rule_score
```
A transaction is flagged as fraudulent when `risk_score ≥ 0.50`.

### Key features
- Per-transaction **SHAP** feature importance
- **Hybrid** ML + rule-based scoring (amount, hour, PCA anomalies)
- Structured `reasons` field for operations team transparency
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="API health check",
)
async def health() -> HealthResponse:
    """Liveness probe — returns model status and uptime."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded() else "degraded",
        model_loaded=predictor.is_loaded(),
        active_model=getattr(predictor, "model_name", "none"),
        version="2.0.0",
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Score a single transaction for fraud",
    status_code=status.HTTP_200_OK,
)
async def predict(transaction: TransactionInput) -> PredictionResponse:
    """
    Submit a credit card transaction for real-time fraud scoring.

    Returns the ML fraud probability, hybrid risk score, risk level,
    human-readable rule triggers, and per-feature SHAP values.
    """
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Run: python train.py",
        )

    t0 = time.perf_counter()

    try:
        txn_dict = transaction.model_dump()

        # ML prediction
        fraud_prob, shap_raw = predictor.predict(txn_dict)

        # Hybrid risk scoring
        risk_score, rule_score, reasons, risk_level = compute_hybrid_risk_score(
            ml_probability=fraud_prob,
            transaction=transaction,
        )

        is_fraud   = risk_score >= DECISION_THRESHOLD
        confidence = abs(risk_score - 0.5) * 2.0   # distance from boundary → [0, 1]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Build SHAP response
        shap_explanation: SHAPExplanation | None = None
        if shap_raw:
            shap_explanation = SHAPExplanation(
                top_features=[SHAPFeature(**f) for f in shap_raw["top_features"]],
                base_value=shap_raw["base_value"],
            )

        response = PredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(fraud_prob, 6),
            risk_score=round(risk_score, 6),
            is_fraud=is_fraud,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            reasons=reasons,
            shap_explanation=shap_explanation,
            model_used=predictor.model_name,
            processing_time_ms=round(elapsed_ms, 2),
        )

        # ── Alert ──────────────────────────────────────────────────────────
        if is_fraud:
            _emit_alert(response, transaction)

        # ── Structured log ─────────────────────────────────────────────────
        logger.info(
            f"[PREDICT] TXN=%-24s  prob=%.4f  risk=%.4f  level=%-8s  "
            f"fraud=%-5s  ms=%.1f",
            (transaction.transaction_id or "N/A"),
            fraud_prob,
            risk_score,
            risk_level,
            str(is_fraud),
            elapsed_ms,
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        )


@app.get("/models", tags=["System"], summary="List available model artefacts")
async def list_models():
    """Return all .pkl model files found in the models/ directory."""
    models_dir = os.path.join(_ROOT, "models")
    items = []
    if os.path.exists(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname.endswith(".pkl") and fname != "scaler.pkl":
                name = fname[:-4]
                items.append({
                    "name":   name,
                    "active": name == getattr(predictor, "model_name", ""),
                    "size_mb": round(
                        os.path.getsize(os.path.join(models_dir, fname)) / 1e6, 2
                    ),
                })
    return {"active_model": getattr(predictor, "model_name", "none"), "models": items}


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": type(exc).__name__},
    )


# ── Alert helper ──────────────────────────────────────────────────────────────

def _emit_alert(response: PredictionResponse, txn: TransactionInput):
    """Log a structured fraud alert to console and file."""
    msg = (
        f"🚨 FRAUD ALERT | "
        f"TXN={response.transaction_id or 'N/A'} | "
        f"${txn.amount:,.2f} | "
        f"Risk={response.risk_score:.4f} ({response.risk_level}) | "
        f"Prob={response.fraud_probability:.4f} | "
        f"Triggers: {'; '.join(response.reasons) or 'ML model detection'}"
    )
    logger.warning(msg)
    alert_logger.warning(msg)
