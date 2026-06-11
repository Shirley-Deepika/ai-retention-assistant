# AI-Powered Customer Retention Assistant

A full-stack ML application that predicts telecom customer churn and generates
personalised retention strategies using XGBoost, SHAP, FastAPI, and Claude AI.

---

## What it does

1. **Predicts churn risk** — XGBoost model trained on 7,000 telecom customers,
   achieving 88% accuracy and 0.85 ROC-AUC
2. **Explains predictions** — SHAP values show exactly which features drove
   each individual prediction (not just global importance)
3. **Generates retention strategies** — Claude AI reads the SHAP explanation
   and writes a targeted, actionable recommendation for the retention team
4. **Stores history** — every prediction is saved to PostgreSQL with full
   customer profile and model output

---

## Tech stack

| Layer | Technology |
|---|---|
| ML model | XGBoost + SHAP + imbalanced-learn (SMOTE) |
| Backend API | FastAPI + SQLAlchemy + PostgreSQL |
| GenAI | Anthropic Claude API (claude-haiku-3-5) |
| Frontend | Streamlit |
| Deployment | Docker + Docker Compose |

---

## Project structure

```
ai-retention-assistant/
│
├── data/
│   └── telecom_churn.csv          # Kaggle Telco dataset (not in repo)
│
├── ml/
│   ├── train.py                   # training script
│   ├── model.pkl                  # saved XGBoost model (generated)
│   ├── preprocessor.pkl           # saved StandardScaler (generated)
│   ├── feature_columns.pkl        # column order for inference (generated)
│   └── shap_summary.png           # global SHAP plot (generated)
│
├── backend/
│   ├── main.py                    # FastAPI routes
│   ├── predict.py                 # inference logic
│   ├── recommend.py               # Claude API integration
│   ├── database.py                # SQLAlchemy models + DB connection
│   └── schemas.py                 # Pydantic request/response schemas
│
├── frontend/
│   └── app.py                     # Streamlit dashboard
│
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/ai-retention-assistant.git
cd ai-retention-assistant
pip install -r requirements.txt
```

### 2. Get the dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
from Kaggle and place it at `data/telecom_churn.csv`.

### 3. Train the model

```bash
python ml/train.py
```

This saves `model.pkl`, `preprocessor.pkl`, `feature_columns.pkl`,
and `shap_summary.png` to the `ml/` directory.

Expected output:
```
[eval]  Accuracy : 0.8834
        ROC-AUC  : 0.8512
[done]  All artifacts saved to ml/
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
# Get a free key at https://console.anthropic.com
```

### 5. Run with Docker (recommended)

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| FastAPI health check | http://localhost:8000/health |

### 6. Run locally without Docker

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

> For local runs without PostgreSQL, the backend automatically
> falls back to a local SQLite database file (`retention.db`).

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict churn for a customer |
| `POST` | `/recommend` | Generate LLM retention strategy |
| `GET` | `/customers` | List all customers + predictions |
| `GET` | `/customers/{id}` | Single customer details |

Full interactive docs at `http://localhost:8000/docs`.

---

## Model performance

| Metric | Score |
|---|---|
| Accuracy | 88.3% |
| Precision | 84.1% |
| Recall | 79.2% |
| F1-Score | 81.6% |
| ROC-AUC | 0.851 |

SMOTE oversampling improved minority class (churn) recall from 54% to 79%.

Top 3 churn predictors (by mean SHAP value):
1. Contract type (Month-to-month vs longer)
2. Tenure (shorter tenure = higher risk)
3. Monthly charges (higher bill = higher risk)

---

## Key design decisions

**Why separate `/predict` and `/recommend` endpoints?**
Predictions are fast (<100ms). LLM calls take 2–5 seconds. Separating them
lets the UI show the prediction immediately while the user decides whether
they need a recommendation.

**Why SHAP over feature importance?**
Global feature importance tells you what the model cares about on average.
SHAP gives per-customer explanations — essential for actionable recommendations
that are specific to one person's situation, not generic advice.

**Why Claude Haiku?**
It's the fastest and most cost-efficient Claude model. For short structured
outputs like retention recommendations (~250 words), it performs as well as
larger models at a fraction of the cost.

---

## Author

Shirley Deepika  
B.E. in Artificial Intelligence & Data Science  
East Point College of Engineering & Technology, Bengaluru  
mshirleydeepika@gmail.com
