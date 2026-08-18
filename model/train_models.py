"""
train_models.py
----------------
Trains 5 classification models on a binary classification dataset,
computes evaluation metrics, and saves:
  - trained models (model/*.pkl)
  - a fitted StandardScaler (model/scaler.pkl)
  - the held-out test split as test_data.csv (features + true label column)
  - a metrics summary as model/metrics_summary.csv (used to fill the README table)

NOTE FOR THE STUDENT:
This script currently uses the sklearn "Breast Cancer Wisconsin" dataset
(bundled with scikit-learn, so it runs with no internet access) purely as a
DEMO / SCAFFOLD so you can see the full pipeline working end-to-end.

For your actual submission you must replace `load_dataset()` below with
YOUR OWN dataset chosen from Kaggle/UCI (>=12 features, >=500 instances),
per the assignment's anti-plagiarism rules. Everything else
(model training, metrics, saving, README table) will keep working as long
as your dataset ends up as a pandas DataFrame `X` (features) and a Series
`y` (0/1 or multi-class labels).
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

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
    confusion_matrix, classification_report
)

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_dataset():
    """
    Loads the Telco Customer Churn dataset (Kaggle:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

    7,043 customers x 20 features (after dropping the ID column),
    binary target: Churn (Yes/No).
    """
    csv_path = HERE.parent / "raw_data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)

    # Drop the non-predictive identifier column
    df = df.drop(columns=["customerID"])

    # TotalCharges is read as text because a few new customers (tenure=0)
    # have a blank value instead of 0 - coerce to numeric and fill with 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Target: Yes -> 1, No -> 0
    y = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["Churn"])

    # One-hot encode all remaining categorical (object) columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    X = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return X, y


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    X, y = load_dataset()
    assert X.shape[1] >= 12, "Dataset must have at least 12 features"
    assert X.shape[0] >= 500, "Dataset must have at least 500 instances"

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (helps Logistic Regression / kNN); tree-based &
    # Naive Bayes models are trained on the same scaled data here for
    # a single consistent pipeline (common & acceptable approach).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = []
    model_dir = HERE
    model_dir.mkdir(exist_ok=True)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(model, model_dir / fname)

        print(f"\n=== {name} ===")
        print(metrics)
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save scaler + feature names (needed by the Streamlit app)
    joblib.dump(scaler, model_dir / "scaler.pkl")
    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Save metrics summary (used to build README comparison table)
    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics_df.to_csv(model_dir / "metrics_summary.csv", index=False)
    print("\n\nMETRICS SUMMARY\n", metrics_df.round(4).to_string(index=False))

    # Save the held-out test split as test_data.csv (features + true label)
    # This is what gets uploaded to the assignment repo AND to the Streamlit app.
    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"\nSaved test_data.csv with {len(test_df)} rows -> {ROOT / 'test_data.csv'}")


if __name__ == "__main__":
    main()
