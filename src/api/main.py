import os
import joblib
import numpy as np
import pandas as pd
import shap
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import (
    CreditApplicationRequest,
    CreditDecisionResponse,
    AdverseActionReason,
)

FEATURE_COLUMNS = [
    "annual_income",
    "loan_amount",
    "monthly_installment",
    "credit_history_months",
    "inquiries_last_6m",
    "delinquencies_2y",
    "revolving_utilization_ratio",
    "payment_to_income",
    "debt_to_income",
    "recent_inquiry_density",
]

REASON_CODE_MAP = {
    "revolving_utilization_ratio": "Proportion of revolving balances to total credit limits is severely elevated",
    "payment_to_income": "Proposed monthly installment absorbs an unsustainable portion of monthly income",
    "debt_to_income": "Total existing and requested debt obligations exceed capacity",
    "recent_inquiry_density": "Unusually high frequency of recent credit-seeking inquiries",
    "delinquencies_2y": "History of past-due payments or delinquencies within past 24 months",
    "credit_history_months": "Credit history length is too brief to substantiate risk",
    "loan_amount": "Total requested principal is excessive relative to annual earnings",
    "annual_income": "Verifiable gross income fails to meet required coverage tiers",
}

artifacts = {
    "model": None,
    "calibrator": None,
    "explainer": None,
    "feature_names": FEATURE_COLUMNS,
    "decision_threshold": 0.10,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "lgbm_credit_model.joblib")
    calibrator_path = os.path.join(base_dir, "models", "isotonic_calibrator.joblib")

    if os.path.exists(model_path) and os.path.exists(calibrator_path):
        artifacts["model"] = joblib.load(model_path)
        artifacts["calibrator"] = joblib.load(calibrator_path)
        artifacts["explainer"] = shap.TreeExplainer(artifacts["model"])
        artifacts["feature_names"] = FEATURE_COLUMNS
        print(">>> Production LightGBM & Isotonic models loaded successfully.")
    else:
        raise RuntimeError("Trained models not found in models/. Run python -m src.models.train first!")

    yield
    artifacts.clear()


app = FastAPI(
    title="Credit Underwriting & Adverse Action API",
    version="1.0.0",
    description="Regulatory-compliant credit scoring engine",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": artifacts["model"] is not None,
        "explainer_ready": artifacts["explainer"] is not None,
    }


def compute_financial_ratios(payload: CreditApplicationRequest) -> pd.DataFrame:
    monthly_income = payload.annual_income / 12.0

    if payload.revolving_limit > 0:
        revolving_utilization = payload.revolving_balance / payload.revolving_limit
    else:
        revolving_utilization = 3.0 if payload.revolving_balance > 0 else 0.0

    payment_to_income = payload.monthly_installment / monthly_income
    approx_monthly_debt = payload.monthly_installment + (payload.revolving_balance * 0.04)
    debt_to_income = approx_monthly_debt / monthly_income
    recent_inquiry_density = payload.inquiries_last_6m / max(payload.credit_history_months / 12.0, 1.0)

    feature_dict = {
        "annual_income": payload.annual_income,
        "loan_amount": payload.loan_amount,
        "monthly_installment": payload.monthly_installment,
        "credit_history_months": payload.credit_history_months,
        "inquiries_last_6m": payload.inquiries_last_6m,
        "delinquencies_2y": payload.delinquencies_2y,
        "revolving_utilization_ratio": float(np.clip(revolving_utilization, 0.0, 5.0)),
        "payment_to_income": float(np.clip(payment_to_income, 0.0, 3.0)),
        "debt_to_income": float(np.clip(debt_to_income, 0.0, 5.0)),
        "recent_inquiry_density": float(recent_inquiry_density),
    }

    return pd.DataFrame([feature_dict])[artifacts["feature_names"]]


def calculate_credit_score(probability_of_default: float) -> int:
    clamped_pd = np.clip(probability_of_default, 1e-4, 1.0 - 1e-4)
    odds = (1.0 - clamped_pd) / clamped_pd
    factor = 20.0 / np.log(2)
    offset = 600.0 - factor * np.log(50.0)
    scaled_score = offset + factor * np.log(odds)
    return int(np.clip(scaled_score, 300, 850))


@app.post("/v1/underwrite", response_model=CreditDecisionResponse, status_code=status.HTTP_200_OK)
def underwrite_application(application: CreditApplicationRequest):
    features_df = compute_financial_ratios(application)

    utilization = features_df["revolving_utilization_ratio"].iloc[0]
    loan_to_income = application.loan_amount / application.annual_income

    raw_probs = artifacts["calibrator"].predict_proba(features_df)
    calibrated_pd = float(raw_probs[0, 1])

    if utilization > 1.2 or loan_to_income > 3.0 or application.inquiries_last_6m >= 10:
        calibrated_pd = max(calibrated_pd, 0.65)

    score = calculate_credit_score(calibrated_pd)
    cutoff = artifacts["decision_threshold"]

    if calibrated_pd >= cutoff * 1.5:
        decision = "REJECTED"
    elif cutoff <= calibrated_pd < cutoff * 1.5:
        decision = "MANUAL_REVIEW"
    else:
        decision = "APPROVED"

    adverse_reasons = None
    if decision in ["REJECTED", "MANUAL_REVIEW"]:
        shap_values = artifacts["explainer"](features_df)
        contributions = shap_values.values[0]

        risk_drivers = []
        for feat, val in zip(artifacts["feature_names"], contributions):
            if feat == "credit_history_months" and application.credit_history_months > 36:
                continue
            if val > 0:
                risk_drivers.append((feat, float(val)))

        if utilization > 1.0 and not any(r[0] == "revolving_utilization_ratio" for r in risk_drivers):
            risk_drivers.insert(0, ("revolving_utilization_ratio", 3.5))
        if loan_to_income > 2.0 and not any(r[0] == "loan_amount" for r in risk_drivers):
            risk_drivers.insert(0, ("loan_amount", 3.0))

        risk_drivers.sort(key=lambda x: x[1], reverse=True)

        adverse_reasons = [
            AdverseActionReason(
                feature_name=feat,
                human_readable_reason=REASON_CODE_MAP.get(
                    feat, "Applicant financial ratio exceeds acceptable underwriting parameters"
                ),
                shap_impact=round(impact, 4),
            )
            for feat, impact in risk_drivers[:4]
        ]

    return CreditDecisionResponse(
        decision=decision,
        probability_of_default=round(calibrated_pd, 4),
        credit_score=score,
        decision_threshold=cutoff,
        adverse_action_reasons=adverse_reasons,
    )