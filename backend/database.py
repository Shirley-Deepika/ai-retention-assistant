"""
backend/database.py
-------------------
Database connection + ORM table definitions.

Two tables:
  customers   - raw customer profile data (one row per customer)
  predictions - model output + LLM recommendation (many per customer over time)

SQLAlchemy's ORM lets us write Python classes instead of raw SQL.
The engine connects to MySQL via the DATABASE_URL environment variable.
When running locally without Docker, it falls back to SQLite for easy dev.

MySQL driver: PyMySQL (pure Python, no C compiler needed).
Connection string format:
  mysql+pymysql://user:password@host:3306/dbname
"""

import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ─────────────────────────────────────────────
# 1. CONNECTION
# ─────────────────────────────────────────────

# DATABASE_URL is set in docker-compose.yml for production.
# Falls back to a local SQLite file so you can run without Docker during dev.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./retention.db"   # dev fallback — no MySQL needed locally
)

# connect_args is only needed for SQLite (disables same-thread check).
# For MySQL this dict is empty.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# pool_pre_ping=True tells SQLAlchemy to test the connection before using it.
# MySQL closes idle connections after wait_timeout (default 8 hours) —
# pool_pre_ping silently reconnects instead of crashing with "MySQL has gone away".
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

# SessionLocal is a factory — call SessionLocal() to get a db session.
# Each FastAPI request gets its own session (opened and closed per request).
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that all ORM models inherit from
Base = declarative_base()


# ─────────────────────────────────────────────
# 2. TABLE: customers
# ─────────────────────────────────────────────

class Customer(Base):
    """
    Stores the raw customer profile submitted via the /predict endpoint.

    All feature columns match exactly what the ML model was trained on.
    This gives us a complete audit trail — we always know what input
    produced what prediction.
    """
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)

    # ── Demographics ──────────────────────────
    gender          = Column(String(10))
    senior_citizen  = Column(Integer)          # 0 or 1
    partner         = Column(Integer)          # 0 or 1
    dependents      = Column(Integer)          # 0 or 1

    # ── Account ───────────────────────────────
    tenure          = Column(Float)            # months
    phone_service   = Column(Integer)
    multiple_lines  = Column(Integer)
    internet_service = Column(String(20))      # DSL / Fiber optic / No
    contract        = Column(String(30))       # Month-to-month / One year / Two year
    paperless_billing = Column(Integer)
    payment_method  = Column(String(40))

    # ── Services ──────────────────────────────
    online_security  = Column(Integer)
    online_backup    = Column(Integer)
    device_protection = Column(Integer)
    tech_support     = Column(Integer)
    streaming_tv     = Column(Integer)
    streaming_movies = Column(Integer)

    # ── Charges ───────────────────────────────
    monthly_charges = Column(Float)
    total_charges   = Column(Float)

    # ── Metadata ──────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow)

    # One customer can have many predictions over time
    predictions = relationship("Prediction", back_populates="customer")

    def __repr__(self):
        return f"<Customer id={self.id} tenure={self.tenure} contract={self.contract}>"


# ─────────────────────────────────────────────
# 3. TABLE: predictions
# ─────────────────────────────────────────────

class Prediction(Base):
    """
    Stores one model prediction + LLM recommendation per API call.

    churn_probability  - raw float from model.predict_proba() [0.0 – 1.0]
    churn_prediction   - binary threshold applied (default: 0.5)
    risk_tier          - human-readable label derived from probability:
                           < 0.3  → Low
                           0.3–0.6 → Medium
                           > 0.6  → High
    top_shap_features  - JSON string of the top 3 features driving this prediction.
                         Passed to Claude API so recommendations are grounded in data.
    recommendation     - LLM-generated retention strategy (free text, ~200 words)
    """
    __tablename__ = "predictions"

    id             = Column(Integer, primary_key=True, index=True)
    customer_id    = Column(Integer, ForeignKey("customers.id"), index=True)

    # ── Model output ──────────────────────────
    churn_probability = Column(Float, nullable=False)
    churn_prediction  = Column(Boolean, nullable=False)   # True = likely to churn
    risk_tier         = Column(String(10))                # Low / Medium / High

    # ── Explainability ────────────────────────
    # Stored as a JSON string, e.g.:
    # '[{"feature": "Contract_Two year", "shap_value": -0.82}, ...]'
    top_shap_features = Column(Text)

    # ── LLM output ────────────────────────────
    recommendation = Column(Text)

    # ── Metadata ──────────────────────────────
    created_at = Column(DateTime, default=datetime.utcnow)

    # Link back to the customer row
    customer = relationship("Customer", back_populates="predictions")

    def __repr__(self):
        return (
            f"<Prediction id={self.id} "
            f"customer_id={self.customer_id} "
            f"prob={self.churn_probability:.2f} "
            f"tier={self.risk_tier}>"
        )


# ─────────────────────────────────────────────
# 4. INIT TABLES
# ─────────────────────────────────────────────

def init_db():
    """
    Creates all tables if they don't already exist.
    Called once when FastAPI starts up (see main.py lifespan).
    Safe to call multiple times — won't overwrite existing data.
    """
    Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────
# 5. DEPENDENCY — used by FastAPI route functions
# ─────────────────────────────────────────────

def get_db():
    """
    FastAPI dependency that yields one database session per request.

    Usage in a route:
        @app.get("/customers")
        def get_customers(db: Session = Depends(get_db)):
            return db.query(Customer).all()

    The try/finally ensures the session is always closed even if
    the request raises an exception — prevents connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
