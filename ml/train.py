"""
ml/train.py
-----------
Trains an XGBoost churn classifier on the Telco dataset.

Artifacts saved to ml/:
  model.pkl          - trained XGBoost model
  preprocessor.pkl   - fitted StandardScaler
  feature_columns.pkl - ordered list of feature names (needed at inference)

Run:
  python ml/train.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telecom_churn.csv")
ML_DIR    = os.path.join(BASE_DIR, "ml")


# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[data]  Loaded {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[data]  Churn distribution:\n{df['Churn'].value_counts()}\n")
    return df


# ─────────────────────────────────────────────
# 3. PREPROCESS
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.copy()

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    print(f"[prep]  TotalCharges: fixed {n_missing} blank rows via median imputation")

    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "Churn"
    ] + service_cols

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # gender: Male → 1, Female → 0
    df["gender"] = (df["gender"] == "Male").astype(int)

    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod", "InternetService"],
        drop_first=True
    )

    print(f"[prep]  Final shape after encoding: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. SPLIT + SCALE + SMOTE
# ─────────────────────────────────────────────
def split_scale_smote(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[split] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    joblib.dump(scaler, os.path.join(ML_DIR, "preprocessor.pkl"))
    print("[prep]  Saved preprocessor.pkl")

    feature_cols = X_train.columns.tolist()
    joblib.dump(feature_cols, os.path.join(ML_DIR, "feature_columns.pkl"))
    print(f"[prep]  Saved feature_columns.pkl ({len(feature_cols)} features)")

    before = y_train.value_counts().to_dict()
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    after = pd.Series(y_train_res).value_counts().to_dict()
    print(f"[smote] Before: {before}")
    print(f"[smote] After:  {after}\n")

    return X_train_res, X_test, y_train_res, y_test, feature_cols


# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"[train] CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("\n[eval] ── Evaluation on held-out test set ──")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"[eval] Confusion matrix:\n{cm}\n")


# ─────────────────────────────────────────────
# 7. SHAP — disabled until installed
# ─────────────────────────────────────────────
def run_shap(model, X_test, feature_cols):
    print("[shap]  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(ML_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[shap]  Saved shap_summary.png")


# ─────────────────────────────────────────────
# 8. SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model):
    path = os.path.join(ML_DIR, "model.pkl")
    joblib.dump(model, path)
    print(f"\n[save]  Saved model.pkl")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  AI Retention Assistant — Model Training")
    print("=" * 50 + "\n")

    df                                           = load_data(DATA_PATH)
    df                                           = preprocess(df)
    X_train, X_test, y_train, y_test, feat_cols = split_scale_smote(df)
    model                                        = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    run_shap(model, X_test, feat_cols)
    save_model(model)

    print("\n[done]  All artifacts saved to ml/")
    print("        model.pkl | preprocessor.pkl | feature_columns.pkl")
