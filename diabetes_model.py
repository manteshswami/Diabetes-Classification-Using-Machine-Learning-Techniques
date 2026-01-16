
"""
Diabetes Classification using Gradient Boosting
Author: Mantesh Swami
"""

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# =========================
# Load Dataset
# =========================
df = pd.read_csv("dataset/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# =========================
# Preprocessing
# =========================
missing_value_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
numerical_cols = X.columns.tolist()
remaining_cols = [c for c in numerical_cols if c not in missing_value_cols]

impute_then_scale = Pipeline(
    steps=[
        ("imputer", SimpleImputer(missing_values=0, strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

scale_remaining = Pipeline(
    steps=[("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("impute_then_scale", impute_then_scale, missing_value_cols),
        ("scale_remaining", scale_remaining, remaining_cols),
    ]
)

# =========================
# Model & Pipeline
# =========================
model = GradientBoostingClassifier(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=150,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# =========================
# Train Model
# =========================
pipeline.fit(X_train, y_train)

# =========================
# Predictions
# =========================
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

y_test_prob = pipeline.predict_proba(X_test)[:, 1]

# =========================
# Evaluation Function
# =========================
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }

train_metrics = evaluate_model(y_train, y_train_pred)
test_metrics = evaluate_model(y_test, y_test_pred)

print("Training Metrics:\n", train_metrics["classification_report"])
print("Testing Metrics:\n", test_metrics["classification_report"])

# =========================
# Confusion Matrix (Test)
# =========================
plt.figure(figsize=(6, 4))
sns.heatmap(
    test_metrics["confusion_matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.show()

# =========================
# ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
auc_score = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Gradient Boosting (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# Save Model
# =========================
joblib.dump(pipeline, "diabetes_prediction_model.pkl")
print("Model saved as diabetes_prediction_model.pkl")
