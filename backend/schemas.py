"""
backend/schemas.py
------------------
Pydantic models define the shape of data coming IN to the API (requests)
and going OUT of the API (responses).

FastAPI uses these for:
  - Automatic request validation (wrong type = 422 error, not a silent bug)
  - Auto-generated /docs (Swagger UI) — great for your README demo
  - Response serialization

Two main request shapes:
  CustomerInput  - what the Streamlit dashboard sends to /predict
  RecommendInput - what /recommend receives (prediction result + customer data)
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────
# REQUEST: /predict
# ─────────────────────────────────────────────

class CustomerInput(BaseModel):
    """
    Exactly mirrors the feature columns the ML model was trained on.
    Field() lets us add descriptions that show up in /docs — useful for demos.
    """

    # Demographics
    gender:          str   = Field(..., description="Male or Female")
    senior_citizen:  int   = Field(..., ge=0, le=1, description="1 if senior citizen")
    partner:         int   = Field(..., ge=0, le=1, description="1 if has partner")
    dependents:      int   = Field(..., ge=0, le=1, description="1 if has dependents")

    # Account
    tenure:          float = Field(..., ge=0,  description="Months with company")
    phone_service:   int   = Field(..., ge=0, le=1)
    multiple_lines:  int   = Field(..., ge=0, le=1)
    internet_service: str  = Field(..., description="DSL / Fiber optic / No")
    contract:        str   = Field(..., description="Month-to-month / One year / Two year")
    paperless_billing: int = Field(..., ge=0, le=1)
    payment_method:  str   = Field(..., description="Electronic check / Mailed check / etc.")

    # Services
    online_security:   int = Field(..., ge=0, le=1)
    online_backup:     int = Field(..., ge=0, le=1)
    device_protection: int = Field(..., ge=0, le=1)
    tech_support:      int = Field(..., ge=0, le=1)
    streaming_tv:      int = Field(..., ge=0, le=1)
    streaming_movies:  int = Field(..., ge=0, le=1)

    # Charges
    monthly_charges: float = Field(..., ge=0, description="Monthly bill in USD")
    total_charges:   float = Field(..., ge=0, description="Total billed to date")

    @validator("internet_service")
    def validate_internet_service(cls, v):
        allowed = {"DSL", "Fiber optic", "No"}
        if v not in allowed:
            raise ValueError(f"internet_service must be one of {allowed}")
        return v

    @validator("contract")
    def validate_contract(cls, v):
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"contract must be one of {allowed}")
        return v

    class Config:
        # Allows FastAPI /docs to show example payloads
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "senior_citizen": 0,
                "partner": 1,
                "dependents": 0,
                "tenure": 12,
                "phone_service": 1,
                "multiple_lines": 0,
                "internet_service": "Fiber optic",
                "contract": "Month-to-month",
                "paperless_billing": 1,
                "payment_method": "Electronic check",
                "online_security": 0,
                "online_backup": 0,
                "device_protection": 0,
                "tech_support": 0,
                "streaming_tv": 1,
                "streaming_movies": 1,
                "monthly_charges": 85.0,
                "total_charges": 1020.0
            }
        }


# ─────────────────────────────────────────────
# RESPONSE: /predict
# ─────────────────────────────────────────────

class SHAPFeature(BaseModel):
    """One SHAP feature-value pair returned in the prediction response."""
    feature:    str
    shap_value: float


class PredictionResponse(BaseModel):
    """
    What /predict returns to the Streamlit dashboard.

    churn_probability  - the raw number (e.g. 0.73)
    churn_prediction   - boolean threshold result
    risk_tier          - Low / Medium / High (for colour coding in the dashboard)
    top_shap_features  - top 3 features explaining this prediction
    customer_id        - DB id so /recommend can fetch it later
    prediction_id      - DB id of the saved prediction row
    """
    customer_id:       int
    prediction_id:     int
    churn_probability: float
    churn_prediction:  bool
    risk_tier:         str
    top_shap_features: List[SHAPFeature]


# ─────────────────────────────────────────────
# REQUEST: /recommend
# ─────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """
    Streamlit calls /recommend after displaying the prediction.
    We pass the prediction_id so the backend can fetch everything
    it needs from the DB — keeps the request payload small.
    """
    prediction_id: int


# ─────────────────────────────────────────────
# RESPONSE: /recommend
# ─────────────────────────────────────────────

class RecommendResponse(BaseModel):
    prediction_id:  int
    recommendation: str


# ─────────────────────────────────────────────
# RESPONSE: /customers  (history list)
# ─────────────────────────────────────────────

class PredictionSummary(BaseModel):
    prediction_id:     int
    churn_probability: float
    risk_tier:         str
    created_at:        datetime

    class Config:
        from_attributes = True   # allows building from SQLAlchemy ORM objects


class CustomerHistory(BaseModel):
    """
    Returned by GET /customers — shows all customers and their latest prediction.
    Used to populate the history table in the Streamlit dashboard.
    """
    customer_id:     int
    tenure:          float
    contract:        str
    monthly_charges: float
    predictions:     List[PredictionSummary]

    class Config:
        from_attributes = True
