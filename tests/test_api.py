import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.explainability.bias_audit import calculate_disparate_impact
from src.models.monitoring import calculate_psi


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_underwrite_approved_applicant(client):
    prime_payload = {
        "annual_income": 125000.0,
        "loan_amount": 10000.0,
        "monthly_installment": 250.0,
        "revolving_balance": 1200.0,
        "revolving_limit": 25000.0,
        "credit_history_months": 120,
        "inquiries_last_6m": 0,
        "delinquencies_2y": 0,
        "home_ownership": "MORTGAGE",
        "loan_purpose": "debt_consolidation",
    }
    response = client.post("/v1/underwrite", json=prime_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVED"
    assert data["adverse_action_reasons"] is None


def test_underwrite_rejected_high_risk_applicant(client):
    subprime_payload = {
        "annual_income": 20000.0,
        "loan_amount": 50000.0,
        "monthly_installment": 800.0,
        "revolving_balance": 9800.0,
        "revolving_limit": 10000.0,
        "credit_history_months": 14,
        "inquiries_last_6m": 6,
        "delinquencies_2y": 3,
        "home_ownership": "RENT",
        "loan_purpose": "debt_consolidation",
    }
    response = client.post("/v1/underwrite", json=subprime_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "REJECTED"
    assert data["adverse_action_reasons"] is not None
    assert len(data["adverse_action_reasons"]) > 0


def test_schema_validation_error(client):
    invalid_payload = {
        "annual_income": 24000.0,
        "loan_amount": 10000.0,
        "monthly_installment": 3000.0,
        "revolving_balance": 500.0,
        "revolving_limit": 5000.0,
        "credit_history_months": 24,
        "inquiries_last_6m": 1,
        "delinquencies_2y": 0,
        "home_ownership": "RENT",
        "loan_purpose": "credit_card",
    }
    response = client.post("/v1/underwrite", json=invalid_payload)
    assert response.status_code == 422


def test_fair_lending_four_fifths_audit():
    test_df = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "decision": ["APPROVED", "APPROVED", "APPROVED", "REJECTED"],
    })
    result = calculate_disparate_impact(test_df, protected_attr="group", decision_col="decision")
    assert "disparate_impact_ratio" in result.columns
    assert len(result) == 2


def test_psi_calculation_stability():
    np.random.seed(42)
    baseline = np.random.normal(0.1, 0.02, 1000)
    current = np.random.normal(0.1, 0.02, 1000)
    psi, breakdown = calculate_psi(baseline, current)
    assert psi < 0.10
    assert not breakdown.empty