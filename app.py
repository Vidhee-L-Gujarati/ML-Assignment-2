"""
Streamlit App — Breast Cancer Classification Demo
Assignment 2 (Machine Learning, M.Tech AIML/DSE, BITS Pilani WILP)

Features:
  a. Upload test data (CSV)
  b. Select a trained model from a dropdown
  c. View evaluation metrics on the uploaded data
  d. View confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

TARGET_COL = "target"


@st.cache_resource
def load_model(model_key: str):
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_key])
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


st.title("🩺 Breast Cancer Classification — Model Demo")
st.caption(
    "Assignment 2 · Machine Learning · Dataset: Breast Cancer Wisconsin (Diagnostic) "
    "· 30 features, 569 instances, binary classification (0 = malignant, 1 = benign)"
)

# ---------------------------------------------------------------
# a. Upload test data
# ---------------------------------------------------------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must include the same 30 feature columns; "
    "a 'target' column is optional but enables metric evaluation)",
    type=["csv"],
)

st.sidebar.header("2. Choose Model")
model_choice = st.sidebar.selectbox("Select a classification model", list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("👈 Upload a test CSV file from the sidebar to get started. "
            "You can use the `test_data.csv` included in this repository.")
    st.stop()

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.subheader("Preview of Uploaded Data")
st.dataframe(data.head(), use_container_width=True)

has_labels = TARGET_COL in data.columns
feature_cols = [c for c in data.columns if c != TARGET_COL]

# ---------------------------------------------------------------
# Load model + scaler, predict
# ---------------------------------------------------------------
model = load_model(model_choice)
scaler = load_scaler()

try:
    X = data[feature_cols]
    X_scaled = scaler.transform(X)
except Exception as e:
    st.error(
        "Feature columns in the uploaded file don't match the model's expected "
        f"features. Details: {e}"
    )
    st.stop()

y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

st.subheader(f"Predictions — {model_choice}")
pred_display = data.copy()
pred_display["Predicted"] = y_pred
pred_display["Probability (class=1)"] = y_prob.round(4)
st.dataframe(pred_display.head(20), use_container_width=True)

# ---------------------------------------------------------------
# c. Evaluation metrics (only possible if ground-truth labels present)
# ---------------------------------------------------------------
st.subheader("Evaluation Metrics")

if has_labels:
    y_true = data[TARGET_COL]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metric_cols = st.columns(6)
    metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    metric_vals = [acc, auc, prec, rec, f1, mcc]
    for col, name, val in zip(metric_cols, metric_names, metric_vals):
        col.metric(name, f"{val:.4f}")

    # ---------------------------------------------------------------
    # d. Confusion matrix + classification report
    # ---------------------------------------------------------------
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant (0)", "Benign (1)"],
                yticklabels=["Malignant (0)", "Benign (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
else:
    st.warning(
        "No 'target' column found in the uploaded file, so evaluation metrics and the "
        "confusion matrix can't be computed — only predictions are shown above. "
        "Upload a file that includes the true 'target' column to see full evaluation."
    )

st.divider()
st.caption("Built for Assignment 2 — Machine Learning, M.Tech (AIML/DSE), BITS Pilani WILP")
