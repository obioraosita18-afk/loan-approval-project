#  Loan Approval Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![API](https://img.shields.io/badge/API-Live-green)

##  Live Demo

| Resource | Link |
|---|---|
| ** Live API** | [https://loan-approval-api-0clc.onrender.com](https://loan-approval-api-0clc.onrender.com) |
| ** Interactive API Docs (Swagger UI)** | [https://loan-approval-api-0clc.onrender.com/docs](https://loan-approval-api-0clc.onrender.com/docs) |
| ** Full Notebook Viewer** | [View on nbviewer](https://nbviewer.org/github/obioraosita18-afk/loan-approval-project/blob/main/notebooks/Loan_Approval.ipynb) |

>  The API is hosted on a free instance — it may take **30–50 seconds** to wake up on first request.

---

##  Project Overview

An end-to-end machine learning project that predicts whether a loan application will be **Approved ✅** or **Rejected ❌** based on applicant financial and demographic information.

---

##  Test the API

### Option 1 — Swagger UI (No coding needed)
1. Visit → [https://loan-approval-api-0clc.onrender.com/docs](https://loan-approval-api-0clc.onrender.com/docs)
2. Click **POST /predict** → **Try it out** → **Execute**
3. See the prediction result instantly!

### Option 2 — Sample Request (curl)
```bash
curl -X POST "https://loan-approval-api-0clc.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "1",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
  }'
```

### Sample Response
```json
{
  "prediction": "Approved",
  "prediction_label": 1,
  "probability_approved": 0.87,
  "probability_rejected": 0.13
}
```

---

##  Dataset Features

| Feature | Description |
|---|---|
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Graduate or Not Graduate |
| Self_Employed | Employment status |
| ApplicantIncome | Monthly applicant income |
| CoapplicantIncome | Monthly co-applicant income |
| LoanAmount | Requested loan amount (thousands) |
| Loan_Amount_Term | Repayment duration (months) |
| Credit_History | Past credit behaviour (1=good, 0=bad) |
| Property_Area | Urban / Semiurban / Rural |
| **Loan_Status** | **Target variable (Y/N)** |

---

## 🔧 ML Pipeline

1. **Data Loading & Exploration** — shape, dtypes, missing values
2. **EDA** — class distribution, income distribution, correlation heatmap
3. **Data Cleaning** — mode imputation for categorical, median for numerical
4. **Encoding** — Label Encoding for all categorical variables
5. **Feature Engineering** — Log transforms, Total_Income
6. **Model Training** — Logistic Regression, Decision Tree, Random Forest, XGBoost
7. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
8. **Model Saving** — joblib export
9. **API Deployment** — FastAPI on Render

---

##  Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble — best overall performance |
| XGBoost | Gradient boosting — strong competitor |

---

##  Project Structure

```
loan-approval-project/
├── notebooks/
│   └── Loan_Approval.ipynb
├── data/
├── models/
├── src/
├── requirements.txt
└── README.md
```

---

##  Installation

```bash
git clone https://github.com/obioraosita18-afk/loan-approval-project.git
cd loan-approval-project
pip install -r requirements.txt
```

---

##  Author

**Obiora Osita**
GitHub: [@obioraosita18-afk](https://github.com/obioraosita18-afk)
