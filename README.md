# ML Assignment 2 — Classification Models with Streamlit Deployment

## a. Problem Statement

Predict whether a telecom customer will **churn** (cancel their
subscription) based on their account and service usage details, using
multiple classical ML classifiers, and expose the results through an
interactive Streamlit web application.

## b. Dataset Description

- **Name:** Telco Customer Churn
- **Source:** Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Goal:** Study the behavior of telecom customers to identify which
  ones are most likely to stop using the service ("churn"), first through
  exploratory analysis and then through predictive classification models.
- **Size:** 7,043 rows (one row per customer), 21 raw columns (20 features
  + the `Churn` target).

Each customer record captures four broad groups of information:

| Group | Columns |
|---|---|
| **Churn label (target)** | `Churn` — whether the customer left in the last month |
| **Services subscribed** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Account information** | `tenure` (months as a customer), `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges` |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |

`customerID` is a unique identifier only and carries no predictive signal,
so it is dropped before modeling.

- **After preprocessing:** `customerID` dropped; `TotalCharges`
  converted from text to numeric (a handful of zero-tenure customers had
  a blank value here, filled with 0); all categorical columns one-hot
  encoded → **30 model input features** (satisfies the ≥12-feature
  requirement comfortably).
- **Target:** Binary — `Churn` (`Yes` → 1, `No` → 0)
- **Class balance:** 5,174 non-churn (73.5%) vs 1,869 churn (26.5%) —
  moderately imbalanced, which is common in real-world churn data and is
  reflected in the precision/recall trade-offs seen in the results below.
- **Train/Test split:** 80% / 20%, stratified, `random_state=42`
  (test set = 1,409 rows, saved as `test_data.csv`)

## c. GitHub Repository Link

https://github.com/2025DA04285/ml-assignment-2.git 

Repository structure:
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- raw_data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│-- model/
    │-- train_models.py
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest_ensemble.pkl
    │-- scaler.pkl
    │-- feature_names.json
    │-- metrics_summary.csv
```

## d. Models Used

All 5 models were trained on the same dataset and the same train/test
split, using `StandardScaler`-transformed, one-hot-encoded features.

### Comparison Table

| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression         | 0.8070   | 0.8418 | 0.6584    | 0.5668 | 0.6092 | 0.4843 |
| Decision Tree                | 0.7253   | 0.6460 | 0.4825    | 0.4786 | 0.4805 | 0.2939 |
| kNN                          | 0.7473   | 0.7716 | 0.5253    | 0.5000 | 0.5123 | 0.3422 |
| Naive Bayes                  | 0.6558   | 0.8092 | 0.4269    | 0.8663 | 0.5719 | 0.3951 |
| Random Forest (Ensemble)     | 0.7899   | 0.8259 | 0.6291    | 0.5080 | 0.5621 | 0.4302 |

### Observations

| ML Model Name              | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Best all-round performer — highest Accuracy, F1, and MCC. The relationship between features (contract type, tenure, monthly charges) and churn is largely monotonic/linear, which a linear model captures well. |
| Decision Tree                | Weakest model — a single unpruned tree overfits the training split, giving the lowest AUC and MCC; it memorizes noise rather than the general churn pattern. |
| kNN                           | Middling performance; distance-based similarity is diluted by the large number of one-hot-encoded binary features, making "nearest neighbors" less meaningful in this high-dimensional sparse space. |
| Naive Bayes                   | Lowest accuracy but by far the highest recall (0.87) — it aggressively flags customers as churn-risk, catching most true churners at the cost of many false alarms (low precision). Useful if the business cost of missing a churner is high. |
| Random Forest (Ensemble)      | Best AUC and second-best Accuracy/MCC — averaging many trees fixes the single Decision Tree's overfitting and gives the most balanced precision/recall trade-off among the tree-based methods. |
| **Overall Winner for your dataset?** | **Logistic Regression** for overall balanced accuracy/F1/MCC. If catching as many churners as possible matters more than precision, **Naive Bayes** is the better business choice despite its lower accuracy. |

## Streamlit App Features

The deployed app (`app.py`) provides:
- CSV upload of test data
- A model-selection dropdown (all 5 trained models)
- Evaluation metrics displayed for the uploaded test data
- A confusion matrix heatmap and full classification report

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates models + test_data.csv
streamlit run app.py
```

## Live App Link

`<PASTE YOUR STREAMLIT COMMUNITY CLOUD APP LINK HERE>`
