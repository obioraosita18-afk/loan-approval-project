"""
Loan Approval Prediction API
Author: Obiora Osita
Description: FastAPI endpoint that serves real-time loan approval predictions
             using a trained Random Forest / XGBoost model.
"""

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🏦 Loan Approval Prediction API",
    description=(
        "Predicts whether a loan application will be **Approved (Y)** or "
        "**Rejected (N)** based on applicant information.\n\n"
        "Built with FastAPI + Scikit-Learn / XGBoost."
    ),
    version="1.0.0",
)

# ── Load model & scaler ───────────────────────────────────────────────────────
try:
    model  = joblib.load("loan_approval_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    raise RuntimeError(
        "Model files not found. Make sure 'loan_approval_model.pkl' and "
        "'scaler.pkl' are in the same directory as main.py."
    ) from e


# ── Request schema ─────────────────────────────────────────────────────────────
class LoanApplication(BaseModel):
    """
    Raw applicant data — same fields as the original dataset.
    Categorical fields use human-readable string values;
    the API encodes them internally before prediction.
    """

    Gender: Literal["Male", "Female"] = Field(
        ..., example="Male", description="Applicant gender"
    )
    Married: Literal["Yes", "No"] = Field(
        ..., example="Yes", description="Marital status"
    )
    Dependents: Literal["0", "1", "2", "3+"] = Field(
        ..., example="1", description="Number of dependents"
    )
    Education: Literal["Graduate", "Not Graduate"] = Field(
        ..., example="Graduate", description="Education level"
    )
    Self_Employed: Literal["Yes", "No"] = Field(
        ..., example="No", description="Self-employed status"
    )
    ApplicantIncome: float = Field(
        ..., example=5000, description="Monthly applicant income"
    )
    CoapplicantIncome: float = Field(
        ..., example=1500, description="Monthly co-applicant income"
    )
    LoanAmount: float = Field(
        ..., example=120, description="Loan amount requested (in thousands)"
    )
    Loan_Amount_Term: float = Field(
        ..., example=360, description="Loan repayment term in months"
    )
    Credit_History: float = Field(
        ..., example=1.0, description="Credit history (1 = good, 0 = bad)"
    )
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = Field(
        ..., example="Urban", description="Property location"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
                "Property_Area": "Urban",
            }
        }


# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Approved or Rejected")
    prediction_label: int = Field(..., description="1 = Approved, 0 = Rejected")
    probability_approved: float = Field(..., description="Probability of approval (0–1)")
    probability_rejected: float = Field(..., description="Probability of rejection (0–1)")


# ── Encoding helpers ───────────────────────────────────────────────────────────
GENDER_MAP        = {"Female": 0, "Male": 1}
MARRIED_MAP       = {"No": 0, "Yes": 1}
DEPENDENTS_MAP    = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP     = {"Graduate": 0, "Not Graduate": 1}
SELF_EMP_MAP      = {"No": 0, "Yes": 1}
PROPERTY_MAP      = {"Rural": 0, "Semiurban": 1, "Urban": 2}


def encode_and_engineer(app_data: LoanApplication) -> np.ndarray:
    """
    Mirrors the exact preprocessing pipeline used during training:
      1. Label-encode categorical columns
      2. Compute log-transformed income & loan amount features
      3. Compute Total_Income and Total_Income_Log
      4. Assemble the 16-feature vector (same order as X in the notebook)
      5. Scale with the saved StandardScaler
    """

    # --- raw values ---
    gender         = GENDER_MAP[app_data.Gender]
    married        = MARRIED_MAP[app_data.Married]
    dependents     = DEPENDENTS_MAP[app_data.Dependents]
    education      = EDUCATION_MAP[app_data.Education]
    self_employed  = SELF_EMP_MAP[app_data.Self_Employed]
    applicant_inc  = app_data.ApplicantIncome
    coapplicant_inc = app_data.CoapplicantIncome
    loan_amount    = app_data.LoanAmount
    loan_term      = app_data.Loan_Amount_Term
    credit_history = app_data.Credit_History
    property_area  = PROPERTY_MAP[app_data.Property_Area]

    # --- engineered features ---
    applicant_inc_log  = np.log(applicant_inc + 1)
    loan_amount_log    = np.log(loan_amount + 1)
    total_income       = applicant_inc + coapplicant_inc
    total_income_log   = np.log(total_income + 1)

    # --- feature vector (matches X.columns order from notebook) ---
    # Loan_ID is dropped at prediction time (not a real feature).
    # Order:
    # Loan_ID(dropped) | Gender | Married | Dependents | Education |
    # Self_Employed | ApplicantIncome | CoapplicantIncome | LoanAmount |
    # Loan_Amount_Term | Credit_History | Property_Area |
    # ApplicantIncome_Log | LoanAmount_Log | Total_Income | Total_Income_Log
    features = np.array([[
        0,                  # Loan_ID placeholder (ignored by model)
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_inc,
        coapplicant_inc,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        applicant_inc_log,
        loan_amount_log,
        total_income,
        total_income_log,
    ]])

    # --- scale ---
    features_scaled = scaler.transform(features)
    return features_scaled


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "🏦 Loan Approval Prediction API is running!",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    """
    Submit a loan application and receive an approval prediction.

    - **prediction**: `Approved` or `Rejected`
    - **probability_approved**: confidence score (0–1)
    """
    try:
        features = encode_and_engineer(application)
        pred_label = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]  # [P(reject), P(approve)]

        return PredictionResponse(
            prediction="Approved" if pred_label == 1 else "Rejected",
            prediction_label=pred_label,
            probability_approved=round(float(proba[1]), 4),
            probability_rejected=round(float(proba[0]), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
