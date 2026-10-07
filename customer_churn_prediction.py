# Customer Churn Prediction
# Codec Technologies - Data Analytics Internship

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("CUSTOMER CHURN PREDICTION")
print("=" * 70)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# Data cleaning
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop_duplicates()

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])

# -----------------------------
# Exploratory analysis
# -----------------------------
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).round(4))

print("\nChurn rate by Contract:")
print(df.groupby("Contract")["Churn"].mean().sort_values(ascending=False).round(4))

print("\nChurn rate by InternetService:")
print(
    df.groupby("InternetService")["Churn"]
    .mean()
    .sort_values(ascending=False)
    .round(4)
)

plt.figure(figsize=(7, 5))
df["Churn"].value_counts().rename(index={0: "Stayed", 1: "Churned"}).plot(
    kind="bar"
)
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# -----------------------------
# Train/test split
# -----------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ]
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
    ),
}

results = []
fitted_models = {}

for name, model in models.items():
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        }
    )

    fitted_models[name] = pipe

results_df = (
    pd.DataFrame(results)
    .sort_values("ROC-AUC", ascending=False)
    .reset_index(drop=True)
)

print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

# -----------------------------
# Best model evaluation
# -----------------------------
best_name = results_df.loc[0, "Model"]
best_model = fitted_models[best_name]

best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBest model:", best_name)
print("\nClassification report:")
print(classification_report(y_test, best_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, best_pred))

ConfusionMatrixDisplay.from_predictions(y_test, best_pred)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()

# -----------------------------
# ROC curves
# -----------------------------
plt.figure(figsize=(8, 6))

for name, pipe in fitted_models.items():
    prob = pipe.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(
        y_test,
        prob,
        name=name,
    )

plt.title("ROC Curves - Customer Churn Models")
plt.tight_layout()
plt.show()

# -----------------------------
# Risk scoring
# -----------------------------
scored = X_test.copy()
scored["ActualChurn"] = y_test.values
scored["ChurnProbability"] = best_prob

scored["RiskLevel"] = pd.cut(
    scored["ChurnProbability"],
    bins=[-0.01, 0.35, 0.60, 1.00],
    labels=["Low", "Medium", "High"],
)

print("\nTop 20 high-risk customers:")
print(
    scored.sort_values("ChurnProbability", ascending=False)
    [["ChurnProbability", "RiskLevel", "ActualChurn"]]
    .head(20)
)

print("\nDone. Results saved to model_comparison.csv")
