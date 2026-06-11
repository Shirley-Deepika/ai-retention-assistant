"""
backend/predict.py
------------------
Loads the trained model artifacts once at startup, then scores
new customer inputs on demand.

NOTE: SHAP is temporarily disabled until installed.
      Predictions and all other features work normally.
      Top features will show placeholder values until SHAP is enabled.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from backend.schemas import CustomerInput, SHAPFeature

# ─────────────────────────────────────────────
# 1. LOAD ARTIFACTS (once, at module import)
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR   = os.path.join(BASE_DIR, "ml")

try:
    MODEL        = joblib.load(os.path.join(ML_DIR, "model.pkl"))
    SCALER       = joblib.load(os.path.join(ML_DIR, "preprocessor.pkl"))
    FEATURE_COLS = joblib.load(os.path.join(ML_DIR, "feature_columns.pkl"))
    print(f"[predict] Loaded model with {len(FEATURE_COLS)} features")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model artifact not found: {e}\n"
        "Run `python ml/train.py` first to generate model.pkl, "
        "preprocessor.pkl, and feature_columns.pkl."
    )

# Try to load SHAP — silently skip if not installed
try:
    import shap
    EXPLAINER = shap.TreeExplainer(MODEL)
    SHAP_AVAILABLE = True
    print("[predict] SHAP loaded successfully")
except ImportError:
    EXPLAINER = None
    SHAP_AVAILABLE = False
    print("[predict] SHAP not installed — feature explanations will use XGBoost importance instead")

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ─────────────────────────────────────────────
# 2. INPUT → DATAFRAME
# ─────────────────────────────────────────────

def _input_to_dataframe(customer: CustomerInput) -> pd.DataFrame:
    raw = {
        "gender":             customer.gender,
        "SeniorCitizen":      customer.senior_citizen,
        "Partner":            customer.partner,
        "Dependents":         customer.dependents,
        "tenure":             customer.tenure,
        "PhoneService":       customer.phone_service,
        "MultipleLines":      customer.multiple_lines,
        "InternetService":    customer.internet_service,
        "Contract":           customer.contract,
        "PaperlessBilling":   customer.paperless_billing,
        "PaymentMethod":      customer.payment_method,
        "OnlineSecurity":     customer.online_security,
        "OnlineBackup":       customer.online_backup,
        "DeviceProtection":   customer.device_protection,
        "TechSupport":        customer.tech_support,
        "StreamingTV":        customer.streaming_tv,
        "StreamingMovies":    customer.streaming_movies,
        "MonthlyCharges":     customer.monthly_charges,
        "TotalCharges":       customer.total_charges,
    }

    df = pd.DataFrame([raw])
    df["gender"] = (df["gender"] == "Male").astype(int)

    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod", "InternetService"],
        drop_first=True
    )

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


# ─────────────────────────────────────────────
# 3. RISK TIER
# ─────────────────────────────────────────────

def _risk_tier(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (SHAP or XGBoost fallback)
# ─────────────────────────────────────────────

def _get_top_features(df_scaled: pd.DataFrame, n: int = 3) -> list[SHAPFeature]:
    """
    Uses SHAP if available, otherwise falls back to XGBoost feature importance.
    Both give per-prediction top features — SHAP is more accurate but optional.
    """
    if SHAP_AVAILABLE:
        shap_values = EXPLAINER.shap_values(df_scaled)
        feature_shap_pairs = list(zip(FEATURE_COLS, shap_values[0]))
        top = sorted(feature_shap_pairs, key=lambda x: abs(x[1]), reverse=True)[:n]
        return [
            SHAPFeature(feature=feat, shap_value=round(float(val), 4))
            for feat, val in top
        ]
    else:
        # Fallback: use XGBoost's built-in feature importance scores
        importance = MODEL.feature_importances_
        feature_importance_pairs = list(zip(FEATURE_COLS, importance))
        top = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)[:n]
        return [
            SHAPFeature(feature=feat, shap_value=round(float(val), 4))
            for feat, val in top
        ]


# ─────────────────────────────────────────────
# 5. MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────

def run_prediction(customer: CustomerInput) -> dict:
    # Convert input
    df = _input_to_dataframe(customer)

    # Scale numeric columns
    df_scaled = df.copy()
    cols_to_scale = [c for c in NUM_COLS if c in df_scaled.columns]
    df_scaled[cols_to_scale] = SCALER.transform(df_scaled[cols_to_scale])

    # Predict
    churn_probability = float(MODEL.predict_proba(df_scaled)[0][1])
    churn_prediction  = churn_probability >= 0.5
    risk_tier         = _risk_tier(churn_probability)

    # Top features
    top_features = _get_top_features(df_scaled)

    top_shap_json = json.dumps([
        {"feature": s.feature, "shap_value": s.shap_value}
        for s in top_features
    ])

    return {
        "churn_probability": round(churn_probability, 4),
        "churn_prediction":  churn_prediction,
        "risk_tier":         risk_tier,
        "top_shap_features": top_features,
        "top_shap_json":     top_shap_json,
    }
