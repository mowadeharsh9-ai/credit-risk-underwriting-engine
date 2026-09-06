from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanPurpose(str, Enum):
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    SMALL_BUSINESS = "small_business"
    MAJOR_PURCHASE = "major_purchase"
    OTHER = "other"


class AdverseActionReason(BaseModel):
    feature_name: str = Field(..., description="Name of the input metric driving rejection")
    human_readable_reason: str = Field(..., description="Regulatory-compliant explanation code")
    shap_impact: float = Field(..., description="Log-odds contribution toward default")


class CreditApplicationRequest(BaseModel):
    annual_income: float = Field(..., gt=0.0, description="Gross annual income in USD", examples=[85000.0])
    loan_amount: float = Field(..., gt=500.0, le=100000.0, description="Requested principal amount", examples=[15000.0])
    monthly_installment: float = Field(..., gt=0.0, description="Proposed monthly payment", examples=[450.0])
    revolving_balance: float = Field(..., ge=0.0, description="Current revolving debt balance", examples=[4200.0])
    revolving_limit: float = Field(..., ge=0.0, description="Total available credit limit", examples=[10000.0])
    credit_history_months: int = Field(..., ge=6, description="Months since earliest credit line", examples=[72])
    inquiries_last_6m: int = Field(..., ge=0, description="Hard credit checks within last 6 months", examples=[2])
    delinquencies_2y: int = Field(..., ge=0, description="30+ days past due occurrences in last 2 years", examples=[0])
    home_ownership: HomeOwnership = Field(..., examples=[HomeOwnership.MORTGAGE])
    loan_purpose: LoanPurpose = Field(..., examples=[LoanPurpose.DEBT_CONSOLIDATION])

    @model_validator(mode="after")
    def validate_installment_vs_income(self):
        monthly_income = self.annual_income / 12.0
        if self.monthly_installment > monthly_income:
            raise ValueError("monthly_installment cannot exceed gross monthly income")
        return self


class CreditDecisionResponse(BaseModel):
    decision: str = Field(..., description="APPROVED, REJECTED, or MANUAL_REVIEW")
    probability_of_default: float = Field(..., ge=0.0, le=1.0, description="Calibrated default probability")
    credit_score: int = Field(..., ge=300, le=850, description="Scaled risk score (300-850)")
    decision_threshold: float = Field(..., description="Optimal business cutoff threshold")
    adverse_action_reasons: Optional[List[AdverseActionReason]] = Field(
        default=None,
        description="Top adverse factors required by FCRA/ECOA when loan is rejected or reviewed"
    )