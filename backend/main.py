"""
backend/main.py
---------------
The FastAPI application. Three endpoints:

  POST /predict     - takes a customer profile, returns churn prediction + SHAP
  POST /recommend   - takes a prediction_id, returns LLM retention strategy
  GET  /customers   - returns all customers + their prediction history

Run locally:
  uvicorn backend.main:app --reload --port 8000

Then visit:
  http://localhost:8000/docs   ← interactive Swagger UI (great for demos)
  http://localhost:8000/redoc  ← alternative docs view
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database import init_db, get_db, Customer, Prediction
from backend.schemas import (
    CustomerInput,
    PredictionResponse,
    RecommendRequest,
    RecommendResponse,
    CustomerHistory,
)
from backend.predict import run_prediction
from backend.recommend import generate_recommendation


# ─────────────────────────────────────────────
# 1. LIFESPAN — runs on startup and shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs on startup.
    Code after `yield` runs on shutdown.

    We use this to create DB tables on first run.
    init_db() calls Base.metadata.create_all() — safe to call
    multiple times, it only creates tables that don't exist yet.
    """
    print("[startup] Initialising database tables...")
    init_db()
    print("[startup] Database ready.")
    yield
    print("[shutdown] Cleaning up...")


# ─────────────────────────────────────────────
# 2. APP INSTANCE
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI Customer Retention Assistant",
    description=(
        "Predicts customer churn using XGBoost + SHAP, "
        "then generates personalised retention strategies via Claude AI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# 3. CORS
# ─────────────────────────────────────────────

# CORS (Cross-Origin Resource Sharing) lets the Streamlit frontend
# make requests to this API even though they run on different ports.
# In production, replace "*" with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 4. HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """
    Simple endpoint to verify the API is running.
    Docker and Railway use this to check if the container is healthy.
    """
    return {"status": "ok", "message": "Retention API is running"}


# ─────────────────────────────────────────────
# 5. POST /predict
# ─────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer_input: CustomerInput, db: Session = Depends(get_db)):
    """
    Takes a customer profile and returns:
      - Churn probability (0.0 – 1.0)
      - Risk tier (Low / Medium / High)
      - Top 3 SHAP features explaining the prediction
      - customer_id and prediction_id for follow-up /recommend call

    Flow:
      1. Run ML inference (predict.py)
      2. Save customer profile to DB
      3. Save prediction result to DB
      4. Return structured response
    """

    # Step 1: Run inference
    result = run_prediction(customer_input)

    # Step 2: Save customer to DB
    # We always create a new customer row per prediction call.
    # In a real system you'd first check if the customer already exists.
    customer = Customer(
        gender             = customer_input.gender,
        senior_citizen     = customer_input.senior_citizen,
        partner            = customer_input.partner,
        dependents         = customer_input.dependents,
        tenure             = customer_input.tenure,
        phone_service      = customer_input.phone_service,
        multiple_lines     = customer_input.multiple_lines,
        internet_service   = customer_input.internet_service,
        contract           = customer_input.contract,
        paperless_billing  = customer_input.paperless_billing,
        payment_method     = customer_input.payment_method,
        online_security    = customer_input.online_security,
        online_backup      = customer_input.online_backup,
        device_protection  = customer_input.device_protection,
        tech_support       = customer_input.tech_support,
        streaming_tv       = customer_input.streaming_tv,
        streaming_movies   = customer_input.streaming_movies,
        monthly_charges    = customer_input.monthly_charges,
        total_charges      = customer_input.total_charges,
    )
    db.add(customer)
    db.flush()  # flush (not commit) to get the auto-generated customer.id

    # Step 3: Save prediction to DB
    prediction = Prediction(
        customer_id       = customer.id,
        churn_probability = result["churn_probability"],
        churn_prediction  = result["churn_prediction"],
        risk_tier         = result["risk_tier"],
        top_shap_features = result["top_shap_json"],
        recommendation    = None,   # filled in by /recommend
    )
    db.add(prediction)
    db.commit()
    db.refresh(customer)
    db.refresh(prediction)

    # Step 4: Return response
    return PredictionResponse(
        customer_id       = customer.id,
        prediction_id     = prediction.id,
        churn_probability = result["churn_probability"],
        churn_prediction  = result["churn_prediction"],
        risk_tier         = result["risk_tier"],
        top_shap_features = result["top_shap_features"],
    )


# ─────────────────────────────────────────────
# 6. POST /recommend
# ─────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
async def recommend(request: RecommendRequest, db: Session = Depends(get_db)):
    """
    Fetches a saved prediction by ID, then calls Claude API to generate
    a personalised retention strategy.

    This is a separate endpoint from /predict because:
      - LLM calls are slow (~2-3 seconds) — don't block the prediction result
      - Not every prediction needs a recommendation (e.g. Low risk customers)
      - Streamlit can show the prediction immediately, then stream the recommendation

    Note: this route is `async def` because generate_recommendation()
    uses httpx.AsyncClient. All other routes are sync def (simpler, fine
    for CPU-bound work like DB queries and ML inference).
    """

    # Fetch prediction row
    prediction = db.query(Prediction).filter(
        Prediction.id == request.prediction_id
    ).first()

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction {request.prediction_id} not found"
        )

    # Fetch the associated customer profile
    customer = db.query(Customer).filter(
        Customer.id == prediction.customer_id
    ).first()

    if not customer:
        raise HTTPException(
            status_code=404,
            detail=f"Customer for prediction {request.prediction_id} not found"
        )

    # If we already generated a recommendation for this prediction, return it.
    # Avoids making a redundant (and paid) LLM call.
    if prediction.recommendation:
        return RecommendResponse(
            prediction_id  = prediction.id,
            recommendation = prediction.recommendation,
        )

    # Parse the stored SHAP JSON string back to a list of dicts
    top_shap = json.loads(prediction.top_shap_features or "[]")

    # Build customer dict for the prompt
    customer_data = {
        "tenure":           customer.tenure,
        "contract":         customer.contract,
        "monthly_charges":  customer.monthly_charges,
        "internet_service": customer.internet_service,
        "payment_method":   customer.payment_method,
        "partner":          customer.partner,
        "dependents":       customer.dependents,
        "online_security":  customer.online_security,
        "tech_support":     customer.tech_support,
    }

    # Call Claude API
    recommendation = await generate_recommendation(
        customer_data      = customer_data,
        churn_probability  = prediction.churn_probability,
        risk_tier          = prediction.risk_tier,
        top_shap_features  = top_shap,
    )

    # Save recommendation back to the prediction row
    prediction.recommendation = recommendation
    db.commit()

    return RecommendResponse(
        prediction_id  = prediction.id,
        recommendation = recommendation,
    )


# ─────────────────────────────────────────────
# 7. GET /customers
# ─────────────────────────────────────────────

@app.get("/customers", response_model=list[CustomerHistory], tags=["History"])
def get_customers(
    skip:  int = 0,
    limit: int = 50,
    db:    Session = Depends(get_db)
):
    """
    Returns all customers and their prediction history.
    Used by the Streamlit dashboard to populate the history table.

    skip + limit implement simple pagination:
      GET /customers?skip=0&limit=50   ← first page
      GET /customers?skip=50&limit=50  ← second page
    """
    customers = (
        db.query(Customer)
        .order_by(Customer.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [CustomerHistory.from_orm_obj(c) for c in customers]


# ─────────────────────────────────────────────
# 8. GET /customers/{customer_id}
# ─────────────────────────────────────────────

@app.get("/customers/{customer_id}", response_model=CustomerHistory, tags=["History"])
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    """
    Returns a single customer and all their predictions.
    Useful for the detail view in the Streamlit dashboard.
    """
    customer = db.query(Customer).filter(Customer.id == customer_id).first()

    if not customer:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found"
        )

    return CustomerHistory.from_orm_obj(customer)
