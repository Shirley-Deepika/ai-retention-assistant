"""
frontend/app.py
---------------
Streamlit dashboard for the AI Customer Retention Assistant.

Three sections:
  1. Sidebar form    — enter customer profile
  2. Prediction panel — churn probability, risk tier, SHAP explanation
  3. Recommendation  — LLM-generated retention strategy (on demand)
  4. History table   — all past predictions from the DB

Run locally:
  streamlit run frontend/app.py

Make sure the FastAPI backend is running first:
  uvicorn backend.main:app --reload --port 8000
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# API_URL points to the FastAPI backend.
# When running in Docker, this is the service name from docker-compose.yml.
# When running locally, it's localhost.
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Customer Retention Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_colour(tier: str) -> str:
    return {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}.get(tier, "#888")


def gauge_chart(probability: float, risk_tier: str):
    """
    Draws a simple semicircular gauge showing churn probability.
    Uses matplotlib — embedded directly in Streamlit via st.pyplot().
    """
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0e1117")   # match Streamlit dark background
    ax.set_facecolor("#0e1117")

    # Background arc (full semicircle)
    theta = np.linspace(np.pi, 0, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="#2a2a2a", linewidth=18,
            solid_capstyle="round")

    # Coloured fill arc (proportion = probability)
    fill_end = np.pi - (probability * np.pi)
    theta_fill = np.linspace(np.pi, fill_end, 100)
    colour = risk_colour(risk_tier)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=colour,
            linewidth=18, solid_capstyle="round")

    # Probability text in centre
    ax.text(0, -0.15, f"{probability * 100:.1f}%",
            ha="center", va="center", fontsize=22,
            fontweight="bold", color="white")
    ax.text(0, -0.45, f"{risk_tier} Risk",
            ha="center", va="center", fontsize=12, color=colour)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def shap_bar_chart(shap_features: list[dict]):
    """
    Horizontal bar chart of the top SHAP features.
    Red bars = features pushing toward churn.
    Green bars = features pushing away from churn.
    """
    # Clean up feature names for display
    def clean_name(name):
        name = name.replace("_", " ")
        replacements = {
            "PaymentMethod Electronic check": "Electronic Check",
            "PaymentMethod Credit card automatic": "Credit Card",
            "PaymentMethod Bank transfer automatic": "Bank Transfer",
            "PaymentMethod Mailed check": "Mailed Check",
            "InternetService Fiber optic": "Fiber Optic Internet",
            "InternetService DSL": "DSL Internet",
            "Contract One year": "1-Year Contract",
            "Contract Two year": "2-Year Contract",
            "MonthlyCharges": "Monthly Charges",
            "TotalCharges": "Total Charges",
        }
        return replacements.get(name, name)

    features = [clean_name(f["feature"]) for f in shap_features]
    values   = [f["shap_value"] for f in shap_features]
    colours  = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(features, values, color=colours, height=0.4)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("SHAP value", color="#aaa", fontsize=9)
    ax.tick_params(colors="#ccc", labelsize=10)
    ax.spines[:].set_visible(False)

    # Value labels outside the bars
    for bar, val in zip(bars, values):
        offset = 0.03 if val >= 0 else -0.03
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

    red_patch   = mpatches.Patch(color="#e74c3c", label="Increases churn risk")
    green_patch = mpatches.Patch(color="#2ecc71", label="Reduces churn risk")
    ax.legend(handles=[red_patch, green_patch], fontsize=8,
              facecolor="#1a1a1a", labelcolor="#ccc", loc="lower right")

    plt.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────

def call_predict(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Make sure the FastAPI backend is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Prediction failed: {e.response.text}")
        return None


def call_recommend(prediction_id: int) -> str | None:
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"prediction_id": prediction_id},
            timeout=30,   # LLM calls can take up to 10s
        )
        response.raise_for_status()
        return response.json().get("recommendation", "")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Recommendation failed: {e.response.text}")
        return None


def call_history() -> list[dict]:
    try:
        response = requests.get(f"{API_URL}/customers", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return []


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
# st.session_state persists values between Streamlit reruns.
# Without this, prediction results disappear when the user clicks
# the "Get Recommendation" button (which triggers a rerun).

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None


# ─────────────────────────────────────────────
# SIDEBAR — CUSTOMER INPUT FORM
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("Customer Profile")
    st.markdown("Enter the customer's details to predict churn risk.")
    st.divider()

    # Account — highest impact fields
    st.subheader("Account")
    tenure   = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract type", [
        "Month-to-month", "One year", "Two year"
    ])
    payment_method = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])

    st.divider()

    # Services — high impact
    st.subheader("Services")
    internet_service  = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online security", ["No", "Yes"])
    tech_support      = st.selectbox("Tech support", ["No", "Yes"])

    st.divider()

    # Charges — highest impact
    st.subheader("Charges")
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0,
                                       max_value=200.0, value=65.0, step=0.5)
    total_charges   = st.number_input("Total charges ($)", min_value=0.0,
                                       value=float(monthly_charges * tenure),
                                       step=10.0)

    st.divider()
    predict_btn = st.button("Predict Churn Risk", use_container_width=True,
                            type="primary")

    # Hidden defaults for low-impact fields
    gender            = "Male"
    senior_citizen    = "No"
    partner           = "No"
    dependents        = "No"
    phone_service     = "Yes"
    multiple_lines    = "No"
    online_backup     = "No"
    device_protection = "No"
    streaming_tv      = "No"
    streaming_movies  = "No"


# ─────────────────────────────────────────────
# BUILD PAYLOAD + CALL /predict
# ─────────────────────────────────────────────

def yn(val: str) -> int:
    """Convert 'Yes'/'No' selectbox value to 1/0."""
    return 1 if val == "Yes" else 0


if predict_btn:
    payload = {
        "gender":             gender,
        "senior_citizen":     yn(senior_citizen),
        "partner":            yn(partner),
        "dependents":         yn(dependents),
        "tenure":             tenure,
        "phone_service":      yn(phone_service),
        "multiple_lines":     yn(multiple_lines),
        "internet_service":   internet_service,
        "contract":           contract,
        "paperless_billing":  yn(paperless_billing),
        "payment_method":     payment_method,
        "online_security":    yn(online_security),
        "online_backup":      yn(online_backup),
        "device_protection":  yn(device_protection),
        "tech_support":       yn(tech_support),
        "streaming_tv":       yn(streaming_tv),
        "streaming_movies":   yn(streaming_movies),
        "monthly_charges":    monthly_charges,
        "total_charges":      total_charges,
    }

    with st.spinner("Running prediction..."):
        result = call_predict(payload)

    if result:
        st.session_state.prediction_result = result
        st.session_state.recommendation    = None   # clear old recommendation
        st.session_state.last_payload      = payload


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────

st.title("AI Customer Retention Assistant")
st.markdown(
    "Predicts churn risk using **XGBoost + SHAP**, then generates a "
    "personalised retention strategy via **Claude AI**."
)
st.divider()

result = st.session_state.prediction_result

if result is None:
    # Empty state — no prediction yet
    st.info("Fill in the customer profile in the sidebar and click **Predict Churn Risk**.")

    # Show a sample of the history table even before making a prediction
    st.subheader("Prediction History")
    history = call_history()
    if history:
        rows = []
        for c in history:
            for p in c.get("predictions", []):
                rows.append({
                    "Customer ID":   c["customer_id"],
                    "Tenure (mo)":   c["tenure"],
                    "Contract":      c["contract"],
                    "Monthly ($)":   c["monthly_charges"],
                    "Churn Prob":    f"{p['churn_probability'] * 100:.1f}%",
                    "Risk":          p["risk_tier"],
                    "Date":          p["created_at"][:10],
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No predictions yet. Make your first prediction using the sidebar.")

else:
    # ── PREDICTION RESULT PANEL ──────────────────
    prob      = result["churn_probability"]
    tier      = result["risk_tier"]
    pred_id   = result["prediction_id"]
    shap_feats = result["top_shap_features"]

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("Churn Risk Score")
        fig = gauge_chart(prob, tier)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Summary metrics
        colour = risk_colour(tier)
        st.markdown(
            f"""
            <div style="
                background: #1a1a1a;
                border-left: 4px solid {colour};
                padding: 12px 16px;
                border-radius: 4px;
                margin-top: 8px;
            ">
                <p style="color:#aaa; margin:0; font-size:13px;">Churn probability</p>
                <p style="color:white; margin:4px 0; font-size:24px; font-weight:bold;">
                    {prob * 100:.1f}%
                </p>
                <p style="color:#aaa; margin:0; font-size:13px;">Risk tier</p>
                <p style="color:{colour}; margin:4px 0; font-size:18px; font-weight:bold;">
                    {tier}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Why this prediction?")
        st.caption(
            "The chart below shows the top factors driving this prediction. "
            "Red bars increase churn risk; green bars reduce it."
        )
        fig2 = shap_bar_chart(shap_feats)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Feature breakdown in plain English
        st.markdown("**Top factors:**")
        for feat in shap_feats:
            direction = "↑ increases" if feat["shap_value"] > 0 else "↓ reduces"
            name = feat["feature"].replace("_", " ")
            st.markdown(
                f"- **{name}** — {direction} churn risk "
                f"(SHAP: `{feat['shap_value']:+.3f}`)"
            )

    st.divider()

    # ── RECOMMENDATION PANEL ─────────────────────
    st.subheader("AI Retention Strategy")

    if st.session_state.recommendation:
        # Already fetched — display it
        st.markdown(st.session_state.recommendation)
    else:
        st.caption(
            "Click below to generate a personalised retention strategy "
            "for this customer using Claude AI."
        )
        recommend_btn = st.button(
            "Generate Retention Strategy",
            type="primary",
            disabled=(tier == "Low"),  # skip LLM for low-risk customers
        )
        if tier == "Low":
            st.caption("Recommendation skipped for Low risk customers.")

        if recommend_btn:
            with st.spinner("Claude is writing a retention strategy..."):
                rec = call_recommend(pred_id)
            if rec:
                st.session_state.recommendation = rec
                st.rerun()   # rerender to show the recommendation cleanly

    st.divider()

    # ── HISTORY TABLE ─────────────────────────────
    st.subheader("Prediction History")
    history = call_history()
    if history:
        rows = []
        for c in history:
            for p in c.get("predictions", []):
                rows.append({
                    "Customer ID":  c["customer_id"],
                    "Tenure (mo)":  c["tenure"],
                    "Contract":     c["contract"],
                    "Monthly ($)":  c["monthly_charges"],
                    "Churn Prob":   f"{p['churn_probability'] * 100:.1f}%",
                    "Risk":         p["risk_tier"],
                    "Date":         p["created_at"][:10],
                })
        df = pd.DataFrame(rows)

        # Colour-code the Risk column
        def colour_risk(val):
            colours = {"Low": "#1a4a2e", "Medium": "#4a3a1a", "High": "#4a1a1a"}
            return f"background-color: {colours.get(val, '')}; color: white"

        st.dataframe(
            df.style.applymap(colour_risk, subset=["Risk"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No history yet.")
