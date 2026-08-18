"""
app.py
------
Streamlit app for the ML Assignment 2 demo.

Features (per assignment requirements):
  a. Dataset (test data CSV) upload option
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix + classification report

Run locally with:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

st.set_page_config(page_title="ML Assignment 2 - Classification Demo", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return scaler, feature_names, models


def main():
    st.title("🔬 Classification Models Demo")
    st.caption(
        "Upload test data, pick a model, and view evaluation metrics, "
        "the confusion matrix, and the classification report."
    )

    scaler, feature_names, models = load_artifacts()

    # ---- (a) Dataset upload ----
    st.sidebar.header("1. Upload Test Data (CSV)")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv (must include a 'target' column with true labels)",
        type=["csv"],
    )

    # ---- (b) Model selection dropdown ----
    st.sidebar.header("2. Select Model")
    model_choice = st.sidebar.selectbox("Choose a classification model", list(models.keys()))

    if uploaded_file is None:
        st.info("👈 Upload a CSV file (e.g. `test_data.csv` from this repo) to get started.")
        st.write("Expected columns:", feature_names + ["target"])
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        return

    has_labels = "target" in df.columns

    X = df[feature_names]
    X_scaled = scaler.transform(X)

    model = models[model_choice]
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    st.subheader(f"Predictions — {model_choice}")
    result_df = df.copy()
    result_df["prediction"] = y_pred
    result_df["probability_class_1"] = y_proba.round(4)
    st.dataframe(result_df.head(20))

    if has_labels:
        y_true = df["target"]

        # ---- (c) Evaluation metrics ----
        st.subheader("Evaluation Metrics")
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")

        # ---- (d) Confusion matrix / classification report ----
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3))
    else:
        st.warning(
            "No 'target' column found in the uploaded file — showing predictions only. "
            "Upload a CSV with a 'target' column to see evaluation metrics."
        )


if __name__ == "__main__":
    main()
