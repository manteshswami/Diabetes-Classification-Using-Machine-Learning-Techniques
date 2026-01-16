#!/usr/bin/env python
# coding: utf-8

"""
Diabetes Classification Script

This script trains a Gradient Boosting Classifier to predict diabetes outcomes.
It includes data loading, preprocessing (imputation/scaling), model training, 
evaluation, and artifact saving.

Usage:
    python diabetes_classification.py
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = 'dataset/diabetes.csv'
MODEL_SAVE_PATH = 'diabetes_model_with_threshold.pkl'
TEST_SIZE = 0.1
RANDOM_STATE = 42
CUSTOM_THRESHOLD = 0.32

def load_data(path):
    """Loads the dataset from the specified path."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        sys.exit(1)
    
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Data Shape: {df.shape}")
    return df

def perform_eda_checks(df):
    """Performs basic exploratory data checks."""
    print("\n--- EDA Summary ---")
    print(df.isnull().sum())
    
    # Check for zero values which indicate missing data in medical context
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_with_zeros:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"Column '{col}' has {zero_count} zero values (treated as missing).")

    # Correlation Matrix Plot
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='RdBu', fmt='.2f', vmin=-1, vmax=1, 
                center=0, linewidths=0.5, cbar_kws={"shrink": .75})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("Saved correlation_matrix.png")
    plt.close()

def create_preprocessing_pipeline(X):
    """Creates the preprocessing pipeline with imputation and scaling."""
    missing_value_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    remaining_cols = [c for c in numerical_cols if c not in missing_value_cols]

    transformers = []

    # Pipeline for columns with missing values (0s)
    if missing_value_cols:
        impute_then_scale = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=0, strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('impute_then_scale', impute_then_scale, missing_value_cols))

    # Pipeline for remaining numerical columns
    if remaining_cols:
        scale_remaining = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('scale_remaining', scale_remaining, remaining_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """Computes various classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred)
    }
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    return metrics

def plot_confusion_matrix(cm, title, filename):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def plot_roc_curve(model, X_test, y_test, filename='roc_curve.png'):
    """Plots and saves the ROC curve."""
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f'Gradient Boosting (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def train_and_optimize(X_train, y_train, preprocessor):
    """Defines pipeline, performs RandomizedSearch, and returns best model."""
    
    # Base Model
    gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', gbc)
    ])

    # Hyperparameter Grid
    param_distributions = {
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__learning_rate': [0.001, 0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 9]
    }

    print("\nStarting Hyperparameter Optimization...")
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions, 
        cv=5, 
        scoring='accuracy', 
        n_iter=10, 
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    print(f"Best Params: {search.best_params_}")
    print(f"Best CV Score: {search.best_score_:.4f}")
    
    return search.best_estimator_

def main():
    # 1. Load Data
    df = load_data(DATA_PATH)

    # 2. Split Features and Target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 3. EDA (Optional - generates images)
    perform_eda_checks(df)

    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

    # 5. Build Preprocessor
    preprocessor = create_preprocessing_pipeline(X)

    # 6. Train and Optimize
    # Note: We use the best estimator found via RandomizedSearchCV
    best_pipeline = train_and_optimize(X_train, y_train, preprocessor)

    # 7. Evaluate on Test Data (Default Threshold 0.5)
    print("\n--- Evaluation (Default Threshold 0.5) ---")
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
    metrics = compute_classification_metrics(y_test, y_test_pred, y_test_prob)
    print(metrics['classification_report'])
    plot_confusion_matrix(metrics['confusion_matrix'], 'Confusion Matrix - Default', 'cm_default.png')

    # 8. ROC Curve
    plot_roc_curve(best_pipeline, X_test, y_test)

    # 9. Custom Threshold Evaluation
    print(f"\n--- Evaluation (Custom Threshold {CUSTOM_THRESHOLD}) ---")
    y_test_pred_custom = (y_test_prob >= CUSTOM_THRESHOLD).astype(int)
    
    metrics_custom = compute_classification_metrics(y_test, y_test_pred_custom)
    print(metrics_custom['classification_report'])
    plot_confusion_matrix(metrics_custom['confusion_matrix'], 
                         f'Confusion Matrix - Threshold {CUSTOM_THRESHOLD}', 
                         'cm_custom_threshold.png')

    # 10. Save Model
    # Saving the entire pipeline (preprocessor + classifier)
    bundle = {
        "pipeline": best_pipeline,
        "threshold": CUSTOM_THRESHOLD,
        "features": list(X.columns)
    }
    joblib.dump(bundle, MODEL_SAVE_PATH)
    print(f"\nModel bundle saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()