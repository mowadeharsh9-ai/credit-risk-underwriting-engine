# 💳 Production Credit Risk Scoring & Adverse Action Engine

An end-to-end, regulatory-compliant credit underwriting platform built with LightGBM, Isotonic Probability Calibration, and Explainable AI (SHAP). The engine evaluates borrower default risk, applies institutional policy knockouts, and automatically generates Equal Credit Opportunity Act (ECOA) / Fair Credit Reporting Act (FCRA) compliant Adverse Action notices.

---

## 🏗️ System Architecture

* **Predictive Core:** LightGBM classifier optimized for credit default detection on tabular financial data.
* **Calibration Layer:** Cross-validated Isotonic Regression to map raw classifier scores into true, reliable default probabilities ($\text{PD}$).
* **Policy Guardrails:** Hard-rule knockout thresholds (debt-to-income, credit utilization, and inquiry frequency) to enforce institutional risk limits.
* **Explainable AI (XAI):** TreeSHAP integration to dynamically extract local feature attributions, translating mathematical contributions into human-readable Adverse Action reason codes.
* **Microservice API:** Asynchronous FastAPI backend enforcing strict data contracts via Pydantic v2.
* **Underwriter Dashboard:** Interactive Streamlit web interface for loan officers with real-time risk assessment and visual attribution graphs.

---

## 📊 Core Risk Metrics Engineered

* **Payment-to-Income (PTI):** $\frac{\text{Monthly Installment}}{\text{Monthly Gross Income}}$
* **Revolving Credit Utilization:** $\frac{\text{Revolving Balance}}{\text{Total Revolving Limit}}$
* **Debt-to-Income (DTI):** Approximated total monthly debt obligations relative to verifiable gross income.
* **Recent Inquiry Density:** Frequency of recent hard credit checks normalized against the total credit history length.

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/mowadeharsh9-ai/credit-risk-underwriting-engine.git](https://github.com/mowadeharsh9-ai/credit-risk-underwriting-engine.git)
cd credit-risk-underwriting-engine