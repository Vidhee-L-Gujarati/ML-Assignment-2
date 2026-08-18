# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement
This project implements and compares six classification approaches (Logistic
Regression, Decision Tree, k-Nearest Neighbors, Naive Bayes, and Random Forest
as an ensemble model) to predict whether a breast tumor is **malignant** or
**benign** based on features computed from a digitized image of a fine needle
aspirate (FNA) of a breast mass. The trained models are exposed through an
interactive Streamlit web app that lets a user upload test data, pick a model,
and view its evaluation metrics, predictions, and confusion matrix.

## b. Dataset Description
- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also bundled with
  `scikit-learn` via `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — mean, standard error,
  and "worst" values of 10 real-valued measurements per cell nucleus (radius,
  texture, perimeter, area, smoothness, compactness, concavity, concave
  points, symmetry, fractal dimension)
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign
- **Train/test split:** 80% / 20%, stratified, `random_state=42`. The 20%
  held-out test split (114 rows, all 30 features + true `target` label) is
  exported as `test_data.csv` in this repository and is what the Streamlit
  app is designed to accept.

## c. GitHub Repository Link
> **TODO:** Replace with your actual repository URL after you push this code,
> e.g. `[https://github.com/<your-username>/ml-assignment-2-breast-cance](https://github.com/Vidhee-L-Gujarati/ML-Assignment-2)r`

## d. Models Used

### Comparison Table (metrics on the held-out test set, `test_data.csv`)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

*(Regenerated any time by running `model/train_models.py`; numbers are saved
to `model/metrics_summary.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset — the classes are close to linearly separable in the scaled feature space, so a simple linear decision boundary generalizes very well. Highest accuracy, F1, and MCC of all five models. |
| Decision Tree | Weakest performer. A single unpruned tree overfits the training data and is sensitive to small variations, which hurts generalization — lowest accuracy, AUC, and MCC of the five. |
| kNN | Very strong performer, achieving perfect recall (1.0) — it never misses a benign case in this test split. Distance-based classification works well once features are standardized, since the classes form fairly compact, well-separated clusters. |
| Naive Bayes | Reasonable but middling performance. Its assumption of conditional feature independence given the class is violated here (many of the 30 features are highly correlated, e.g. radius/perimeter/area), which caps its accuracy below the top models. |
| Random Forest (Ensemble) | Solid, well-balanced performance and the highest AUC among the tree-based/ensemble models — averaging many trees reduces the overfitting seen in the single Decision Tree, though it still trails Logistic Regression and kNN slightly on this particular split. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it has the best Accuracy, Precision, Recall, F1, and MCC, and is essentially tied for best AUC. It also has the advantage of being the simplest and most interpretable of the five models, which is valuable for a medical diagnosis use case. |

## Streamlit App Features
The deployed app (`app.py`) includes:
- **Dataset upload** — upload a CSV of test data (e.g. `test_data.csv`)
- **Model selection dropdown** — choose among all 5 trained models
- **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC
  computed live on the uploaded data (when it includes the `target` column)
- **Confusion matrix & classification report** — visual heatmap + full
  per-class precision/recall/F1 breakdown

## Repository Structure
```
project-folder/
│-- app.py                  # Streamlit application
│-- requirements.txt        # Python dependencies
│-- README.md                # This file
│-- test_data.csv           # Held-out test split (features + true labels)
│-- model/
│   │-- train_models.py     # Trains all 5 models, computes metrics, saves everything
│   │-- metrics_summary.csv # Generated metrics table
│   │-- scaler.pkl          # Fitted StandardScaler
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # (optional) retrain models
streamlit run app.py
```

## How This Was Deployed
1. Pushed this folder to a public GitHub repository.
2. Signed in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Clicked **New App** → selected this repository → branch `main` → file `app.py`.
4. Clicked **Deploy**.

> **Live Streamlit App Link:** *ml-assignment-2-ilbijdb46kxj9bcrdpjzmj.streamlit.app*
