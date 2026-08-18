"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates each with Accuracy, AUC, Precision, Recall, F1 and MCC,
and saves:
  - trained models          -> model/*.pkl
  - the fitted scaler       -> model/scaler.pkl
  - the held-out test split -> test_data.csv (used by the Streamlit app)
  - a metrics summary       -> model/metrics_summary.csv

Dataset: Breast Cancer Wisconsin (Diagnostic)
  - Source: UCI ML Repository / built into scikit-learn (sklearn.datasets)
  - Instances: 569  (>= 500 required)
  - Features: 30    (>= 12 required)
  - Task: Binary classification (malignant vs benign)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame  # includes 'target' column (0 = malignant, 1 = benign)
feature_names = list(data.feature_names)

X = df[feature_names]
y = df["target"]

# ---------------------------------------------------------------
# 2. Train / test split (test set is exported for the Streamlit app)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Save the held-out test data (features + true label) as required test_data.csv
test_export = X_test.copy()
test_export["target"] = y_test.values
test_export.to_csv("../test_data.csv", index=False)
print("Saved test_data.csv with shape:", test_export.shape)

# ---------------------------------------------------------------
# 3. Scale features (helps LR / KNN especially)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ---------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": name, "Accuracy": acc, "AUC": auc,
        "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc,
    })

    # Save model to a filename-safe path
    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, fname)
    print(f"Trained + saved: {name} -> {fname}")

# ---------------------------------------------------------------
# 5. Save metrics summary
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).round(4)
results_df.to_csv("metrics_summary.csv", index=False)
print("\n=== Metrics Summary ===")
print(results_df.to_string(index=False))
