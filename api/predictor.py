"""
Model Inference Engine
======================
Singleton that loads the trained model (and scaler) from disk at API startup,
builds SHAP explainers, and exposes a single `predict()` method used by the
FastAPI route handler.

Design decisions:
  • Singleton pattern avoids reloading the model on every request.
  • Feature names are loaded from models/feature_names.json so
    the predictor never hard-codes assumptions about the training schema.
  • SHAP explainer is initialised lazily — if it fails (e.g. model type not
    supported), predictions still succeed without SHAP.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import shap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logger = logging.getLogger(__name__)

MODELS_DIR         = os.path.join(_ROOT, "models")
SCALER_PATH        = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Night hours constant (must match feature_engineering.py)
_NIGHT_HOURS = set(range(23, 24)) | set(range(0, 6))


class FraudPredictor:
    """
    Thread-safe singleton for model loading and inference.
    Call `FraudPredictor().load()` once at application start.
    """

    _instance: Optional[FraudPredictor] = None

    def __new__(cls) -> FraudPredictor:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def load(self, model_name: str = "xgboost") -> bool:
        """
        Load model + scaler from disk and initialise the SHAP explainer.

        Returns True on success, False if any critical artefact is missing.
        """
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")

        if not os.path.exists(model_path):
            logger.error(
                f"Model not found: {model_path}\n"
                "→ Train models first:  python train.py"
            )
            return False

        try:
            self.model      = joblib.load(model_path)
            self.model_name = model_name
            logger.info(f"✅ Model loaded: {model_name}")

            # Scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("✅ Scaler loaded")
            else:
                self.scaler = None
                logger.warning("Scaler not found — inputs will not be standardised")

            # Feature names
            if os.path.exists(FEATURE_NAMES_PATH):
                with open(FEATURE_NAMES_PATH) as f:
                    self.feature_names: List[str] = json.load(f)
            else:
                # Fallback: base 30 features
                self.feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            logger.info(f"Feature schema: {len(self.feature_names)} features")

            # SHAP explainer
            self.explainer = self._build_explainer()

            self._ready = True
            return True

        except Exception as exc:
            logger.error(f"Failed to load model: {exc}", exc_info=True)
            return False

    def _build_explainer(self) -> Optional[object]:
        """Build an appropriate SHAP explainer for the loaded model type."""
        try:
            from sklearn.linear_model import LogisticRegression
            if isinstance(self.model, LogisticRegression):
                bg = pd.DataFrame(
                    np.zeros((1, len(self.feature_names))),
                    columns=self.feature_names,
                )
                explainer = shap.LinearExplainer(self.model, bg)
            else:
                explainer = shap.TreeExplainer(self.model)
            logger.info("✅ SHAP explainer ready")
            return explainer
        except Exception as exc:
            logger.warning(f"SHAP explainer init failed (predictions still work): {exc}")
            return None

    # ── Inference ─────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._ready

    def predict(self, txn_dict: dict) -> Tuple[float, Optional[dict]]:
        """
        Run fraud prediction and generate a SHAP explanation.

        Args:
            txn_dict: Raw dict from the API request (pre-validated by Pydantic).

        Returns:
            (fraud_probability, shap_explanation_dict | None)
        """
        if not self._ready:
            raise RuntimeError("Predictor not ready. Call FraudPredictor().load() first.")

        X = self._build_features(txn_dict)

        # Apply scaler
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X)
                X = pd.DataFrame(X_scaled, columns=self.feature_names)
            except Exception as exc:
                logger.warning(f"Scaling failed — using raw features: {exc}")

        prob = float(self.model.predict_proba(X)[0, 1])
        shap_data = self._explain(X)

        return prob, shap_data

    def _build_features(self, txn_dict: dict) -> pd.DataFrame:
        """
        Map API input fields to the exact feature vector the model expects.
        Derived features are recomputed from raw inputs at inference time.
        """
        amount   = txn_dict.get("amount", 0.0)
        time_val = txn_dict.get("time",   0.0)

        base: dict = {
            "Time":   time_val,
            "Amount": amount,
            **{f"V{i}": txn_dict.get(f"V{i}", 0.0) for i in range(1, 29)},
        }

        # Derived features (must match feature_engineering.py)
        hour = int((time_val // 3600) % 24)
        base["amount_log"]     = math.log1p(amount)
        base["hour_of_day"]    = float(hour)
        base["is_night"]       = float(hour in _NIGHT_HOURS)
        base["amount_zscore"]  = 0.0          # population stats not available at inference
        base["v1_v3_product"]  = base["V1"] * base["V3"]
        v_primary = [base[f"V{i}"] for i in range(1, 15)]
        base["v_primary_norm"] = float(np.linalg.norm(v_primary))

        # Select only the features the model was trained on (preserving order)
        row = {k: base.get(k, 0.0) for k in self.feature_names}
        return pd.DataFrame([row])

    def _explain(self, X: pd.DataFrame) -> Optional[dict]:
        """Generate SHAP values for a single prediction row."""
        if self.explainer is None:
            return None
        try:
            shap_vals = self.explainer.shap_values(X)

            # Tree models return a list [class0, class1]; take the fraud class
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
            else:
                sv = shap_vals[0]

            # Expected value
            ev = self.explainer.expected_value
            base_val = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

            impacts = [
                {"feature": name, "impact": float(val), "value": float(X[name].iloc[0])}
                for name, val in zip(self.feature_names, sv)
            ]
            impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)

            return {"top_features": impacts[:12], "base_value": base_val}

        except Exception as exc:
            logger.debug(f"SHAP explanation failed: {exc}")
            return None


# ── Module-level singleton ────────────────────────────────────────────────────
predictor = FraudPredictor()
