"""
backend/recommend.py
--------------------
Calls the Claude API to generate a personalised retention strategy
based on the customer profile + SHAP explanation from the prediction.

This is the GenAI layer that makes this project stand out.
Instead of generic advice, the LLM reads the actual top SHAP features
for this customer and writes a targeted recommendation.

Example output:
  "This customer is at high churn risk primarily due to their
   Month-to-month contract and high monthly charges of $92.
   Recommended actions:
   1. Offer a 15% discount to upgrade to a 1-year contract...
   2. Bundle streaming services to increase perceived value..."
"""

import os
import json
import httpx

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Set ANTHROPIC_API_KEY in your .env file or docker-compose.yml environment block.
# Get your free key at: https://console.anthropic.com
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# We use claude-haiku-3-5 — fastest and cheapest model, perfect for this use case.
# For a project generating ~100 recommendations/day it costs almost nothing.
MODEL_NAME = "claude-haiku-3-5-20241022"


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def _build_prompt(
    customer_data: dict,
    churn_probability: float,
    risk_tier: str,
    top_shap_features: list[dict],
) -> str:
    """
    Builds the user message sent to Claude.

    The prompt is structured in three parts:
      1. Context: what this system does
      2. Customer data: profile + prediction result
      3. Task: what we want Claude to produce

    Including the SHAP features is the key differentiator.
    Without them, Claude gives generic advice.
    With them, it explains WHY the customer is at risk and
    gives targeted interventions for those specific factors.
    """

    # Format SHAP features into a readable bullet list
    shap_lines = "\n".join([
        f"  - {f['feature'].replace('_', ' ')}: SHAP value {f['shap_value']:+.3f} "
        f"({'increases' if f['shap_value'] > 0 else 'decreases'} churn risk)"
        for f in top_shap_features
    ])

    prompt = f"""You are a customer retention specialist at a telecom company.
You have access to a machine learning model's prediction and explanation for one customer.

CUSTOMER PROFILE:
- Tenure: {customer_data.get('tenure')} months
- Contract: {customer_data.get('contract')}
- Monthly charges: ${customer_data.get('monthly_charges')}
- Internet service: {customer_data.get('internet_service')}
- Payment method: {customer_data.get('payment_method')}
- Has partner: {'Yes' if customer_data.get('partner') else 'No'}
- Has dependents: {'Yes' if customer_data.get('dependents') else 'No'}
- Online security: {'Yes' if customer_data.get('online_security') else 'No'}
- Tech support: {'Yes' if customer_data.get('tech_support') else 'No'}

CHURN PREDICTION:
- Risk tier: {risk_tier}
- Churn probability: {churn_probability * 100:.1f}%

TOP FACTORS DRIVING THIS PREDICTION (from SHAP analysis):
{shap_lines}

YOUR TASK:
Write a concise, actionable retention strategy for this specific customer.
Structure your response as:

**Why they're at risk** (1-2 sentences referencing the SHAP factors above)

**Recommended actions** (3 specific, numbered steps a retention agent can take today)

**Talking points** (2 bullet points the agent can use when calling this customer)

Be specific to this customer's profile. Do not give generic advice.
Keep the total response under 250 words.
"""
    return prompt


# ─────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────

async def generate_recommendation(
    customer_data: dict,
    churn_probability: float,
    risk_tier: str,
    top_shap_features: list[dict],
) -> str:
    """
    Calls the Claude API and returns the recommendation as a string.

    Uses httpx.AsyncClient because FastAPI is async — using the
    synchronous requests library inside an async route would block
    the entire event loop.

    Error handling:
      - Missing API key: returns a clear instruction message
      - API error: returns a fallback message rather than crashing
    """

    if not ANTHROPIC_API_KEY:
        return (
            "Recommendation unavailable: ANTHROPIC_API_KEY not set. "
            "Add your key to the .env file and restart the server. "
            "Get a free key at https://console.anthropic.com"
        )

    prompt = _build_prompt(
        customer_data, churn_probability, risk_tier, top_shap_features
    )

    payload = {
        "model": MODEL_NAME,
        "max_tokens": 400,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    try:
        # timeout=30s — LLM calls can be slow; don't let them hang forever
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

        data = response.json()

        # Claude returns: {"content": [{"type": "text", "text": "..."}]}
        recommendation = data["content"][0]["text"].strip()
        return recommendation

    except httpx.HTTPStatusError as e:
        # e.g. 401 = wrong API key, 429 = rate limit
        return f"API error ({e.response.status_code}): {e.response.text}"

    except httpx.TimeoutException:
        return "Recommendation timed out. The LLM service took too long to respond."

    except Exception as e:
        return f"Unexpected error generating recommendation: {str(e)}"
