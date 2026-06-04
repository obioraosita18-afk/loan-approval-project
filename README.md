# 🏦 Loan Approval Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview

An end-to-end machine learning project that predicts whether a loan application will be approved or rejected based on applicant financial and demographic information. The project follows an industry-standard ML workflow from data preprocessing to model deployment preparation.

---

## 🎯 Project Objectives

- Develop a classification model to predict loan approval status
- Perform data cleaning, preprocessing, and feature engineering
- Conduct exploratory data analysis (EDA) to uncover patterns
- Train and compare multiple ML algorithms
- Evaluate models using comprehensive metrics
- Identify key features influencing loan decisions
- Prepare a trained model for deployment via FastAPI

---

## 📁 Project Structure

```
loan-approval-project/
├── notebooks/
│   └── Loan_Approval.ipynb       # Main analysis notebook
├── data/
│   └── (place your dataset here)
├── models/
│   └── (saved .pkl model files)
├── src/
│   └── (source scripts)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Graduate or not |
| Self_Employed | Employment status |
| ApplicantIncome | Applicant salary |
| CoapplicantIncome | Co-applicant salary |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment duration |
| Credit_History | Past credit behaviour |
| Property_Area | Urban / Semiurban / Rural |
| **Loan_Status** | **Target variable** |

---

## 🔧 ML Pipeline

1. **Data Loading & Exploration** — shape, dtypes, missing values, duplicates
2. **EDA** — class distribution, income distribution, credit history analysis, correlation heatmap
3. **Data Cleaning** — mode imputation for categorical, median for numerical
4. **Encoding** — Label Encoding for all categorical variables
5. **Outlier Handling** — log transformation on ApplicantIncome and LoanAmount
6. **Feature Engineering** — Total_Income, Total_Income_Log
7. **Model Training** — Logistic Regression, Decision Tree, Random Forest, XGBoost
8. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
9. **Feature Importance** — Random Forest feature importances
10. **Model Saving** — joblib export of model and scaler

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble — best overall performance |
| XGBoost | Gradient boosting — strong competitor |

> ✅ **Random Forest / XGBoost** selected as final production candidate based on ROC-AUC and F1-Score.

---

## 🚀 Deployment

A **FastAPI** prediction endpoint is planned to serve real-time loan approval predictions using the saved `loan_approval_model.pkl` and `scaler.pkl`.

---

## 📓 View Full Notebook

👉 [View on nbviewer](https://nbviewer.org/github/obioraosita18-afk/loan-approval-project/blob/main/notebooks/Loan_Approval.ipynb)

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## 📬 Author

**Obiora Osita**
GitHub: [@obioraosita18-afk](https://github.com/obioraosita18-afk)
