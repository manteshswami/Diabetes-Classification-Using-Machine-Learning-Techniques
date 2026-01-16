# 🩺 Diabetes Classification Using Machine Learning
---
## 📌 Project Overview

This project focuses on predicting whether a patient has diabetes based on diagnostic health measurements. A machine learning classification pipeline was built using proper preprocessing, model training, evaluation, and validation techniques to ensure robust generalization and avoid overfitting.

The final model was selected after careful experimentation and hyperparameter tuning, prioritizing model stability and real-world applicability over artificially high accuracy.

## 🎯 Objective

- Build a reliable machine learning model to classify diabetes outcomes

- Handle missing values appropriately in medical data

- Reduce overfitting through preprocessing, model tuning, and validation

- Evaluate the model using multiple performance metrics

## 📂 Dataset

**Source**: `Pima Indians Diabetes Dataset`

**Target Variable**: Outcome

`1 → Patient has diabetes`

`0 → Patient does not have diabetes`

**Key Features**

- Pregnancies

- Glucose

- Blood Pressure

- Skin Thickness

- Insulin

- BMI

- Diabetes Pedigree Function

- Age

`⚠️ In medical datasets, zero values in some features represent missing data, not valid measurements.`

## ⚙️ Methodology
### 1️⃣ Data Preprocessing

- Identified invalid zero values in medical features

- Replaced missing values using median imputation

- Applied standard scaling to numerical features

- Implemented preprocessing using Scikit-learn Pipelines and ColumnTransformer

### 2️⃣ Model Selection

*The following considerations guided model choice:*

- Ability to capture non-linear relationships

- Resistance to overfitting

- Stable performance on unseen data

✅ **Final Model**: `Gradient Boosting Classifier`

Key hyperparameters were carefully chosen to control model complexity.

### 3️⃣ Model Training

- Data split using stratified train–test split

- Entire workflow implemented as a single pipeline

- Ensured no data leakage between training and testing

### 4️⃣ Model Evaluation

*The model was evaluated using:*

- Accuracy

- Precision

- Recall

- Confusion Matrix

- ROC–AUC Curve

Evaluation was performed on unseen test data to assess generalization.

**📊 Results**
| Metric                | Training | Testing |
|-----------------------|----------|---------|
| Accuracy              | ~80%     | ~76%    |
| Precision (Diabetes)  | ~84%     | ~71%    |
| Recall (Diabetes)     | ~55%     | ~56%    |
| ROC–AUC               | —        | ~0.72   |


📌 `The small train–test gap (~4%) indicates that the model generalizes well and does not overfit.`

### 🧠 Key Insights

- Perfect training accuracy is not desirable in healthcare ML problems

- Slightly lower but stable test accuracy indicates better real-world performance

- Recall is especially important in medical prediction tasks

- Further accuracy gains were intentionally avoided to prevent overfitting

### 🧪 Why Accuracy Was Finalized

- After hyperparameter tuning and validation:

- Accuracy improvements became marginal

- Risk of overfitting increased with further tuning

- Model stability was prioritized

### 📌 Final decision: Model performance was frozen at a well-generalized state.

#### 💾 Model Saving

*The trained pipeline was saved using joblib:*

`diabetes_prediction_model.pkl`

**This allows easy reuse for:**

- Deployment
- Inference
- Future integration

#### 📁 Project Structure
<pre>
📦 **diabetes-classification**
 ┣ 📂 dataset
 ┃ ┗ 📄 diabetes.csv
 ┣ 📄 auc.png
 ┣ 📄 diabetes_model.py
 ┣ 📄 diabetes_prediction_ml.ipynb
 ┣ 📄 diabetes_prediction_model.pkl
 ┣ 📄 README.md
</pre>

### 🚀 Future Improvements

- Improve recall for diabetic patients using class weighting

- Add feature engineering for enhanced interpretability

- Deploy the model using Streamlit or Flask

- Perform external dataset validation

### 🛠️ Tech Stack

- Python

- NumPy, Pandas

- Scikit-learn

- Matplotlib, Seaborn

- Joblib
---
