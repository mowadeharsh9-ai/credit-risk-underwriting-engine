import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Dynamic routing: checks Streamlit Cloud Secrets, environment variables, or defaults to localhost
if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
else:
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="Credit Underwriting & Risk Engine",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Institutional Credit Underwriting & Model Governance")

tabs = st.tabs(["🚀 Real-Time Underwriting", "⚖️ Fair Lending Audit", "📈 Population Drift (PSI)"])

# -------------------------------------------------------------
# TAB 1: REAL-TIME UNDERWRITING
# -------------------------------------------------------------
with tabs[0]:
    st.markdown("Automated credit risk assessment with Fair Lending Adverse Action explanations (SHAP).")

    with st.sidebar:
        st.header("Borrower Financial Profile")
        annual_income = st.number_input("Annual Gross Income ($)", min_value=5000.0, max_value=1000000.0, value=75000.0, step=1000.0)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=250000.0, value=15000.0, step=500.0)
        monthly_installment = st.number_input("Proposed Monthly Installment ($)", min_value=50.0, max_value=10000.0, value=350.0, step=25.0)

        st.header("Credit File Metrics")
        revolving_balance = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=500000.0, value=4500.0, step=100.0)
        revolving_limit = st.number_input("Revolving Credit Limit ($)", min_value=100.0, max_value=500000.0, value=20000.0, step=500.0)
        credit_history_months = st.slider("Credit History (Months)", min_value=6, max_value=480, value=72)
        inquiries_last_6m = st.slider("Hard Inquiries (Last 6 Months)", min_value=0, max_value=20, value=1)
        delinquencies_2y = st.slider("Past-Due Incidents (Last 2 Years)", min_value=0, max_value=10, value=0)

        st.header("Loan Attributes")
        home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        loan_purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "other"]
        )

    col1, col2 = st.columns([1.1, 1.2])

    with col1:
        st.subheader("Decision & Scorecard")
        submit_btn = st.button("Run Underwriting Evaluation", type="primary", use_container_width=True)

        if submit_btn:
            payload = {
                "annual_income": float(annual_income),
                "loan_amount": float(loan_amount),
                "monthly_installment": float(monthly_installment),
                "revolving_balance": float(revolving_balance),
                "revolving_limit": float(revolving_limit),
                "credit_history_months": int(credit_history_months),
                "inquiries_last_6m": int(inquiries_last_6m),
                "delinquencies_2y": int(delinquencies_2y),
                "home_ownership": home_ownership,
                "loan_purpose": loan_purpose,
            }

            try:
                response = requests.post(f"{API_URL}/v1/underwrite", json=payload, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    decision = data["decision"]
                    score = data["credit_score"]
                    pd_val = data["probability_of_default"]

                    if decision == "APPROVED":
                        st.success(f"### Decision: {decision}\nCredit Score: **{score}** (PD: {pd_val:.2%})")
                    elif decision == "MANUAL_REVIEW":
                        st.warning(f"### Decision: {decision}\nCredit Score: **{score}** (PD: {pd_val:.2%})")
                    else:
                        st.error(f"### Decision: {decision}\nCredit Score: **{score}** (PD: {pd_val:.2%})")

                    m1, m2 = st.columns(2)
                    m1.metric("Est. Probability of Default", f"{pd_val:.2%}")
                    m2.metric("Business Cutoff Threshold", f"{data['decision_threshold']:.2%}")

                    if data.get("adverse_action_reasons"):
                        st.subheader("Regulatory Adverse Action Notice")
                        st.markdown("Primary drivers pushing the risk profile above the acceptable threshold:")
                        for idx, reason in enumerate(data["adverse_action_reasons"], 1):
                            st.write(f"**{idx}. {reason['human_readable_reason']}** (Feature: `{reason['feature_name']}`)")

                        with col2:
                            st.subheader("Local Interpretability Analysis (SHAP)")
                            reasons = data["adverse_action_reasons"]
                            features = [r["feature_name"] for r in reasons][::-1]
                            impacts = [r["shap_impact"] for r in reasons][::-1]

                            fig, ax = plt.subplots(figsize=(6, 3.5))
                            bars = ax.barh(features, impacts, color="#e0564c")
                            ax.set_xlabel("SHAP Impact (Log-Odds toward Default)")
                            ax.set_title("Top Risk Contributors for Applicant")
                            ax.bar_label(bars, fmt="%+.2f", padding=3)
                            ax.grid(axis='x', linestyle='--', alpha=0.5)
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.error(f"Validation error: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Failed to connect to API backend: {e}")

# -------------------------------------------------------------
# TAB 2: FAIR LENDING BIAS AUDIT
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Fair Lending Disparate Impact Audit (CFPB / ECOA Reg B)")
    st.markdown("Auditing underwriting outcomes across protected classes to prevent algorithmic bias using the **Four-Fifths (80%) Rule**.")

    from src.explainability.bias_audit import calculate_disparate_impact

    np.random.seed(42)
    audit_data = pd.DataFrame({
        "age_bracket": np.random.choice(["18-25", "26-40", "41-60", "60+"], size=1200, p=[0.2, 0.4, 0.3, 0.1]),
        "underwriting_decision": np.random.choice(["APPROVED", "REJECTED"], size=1200, p=[0.72, 0.28])
    })

    audit_summary = calculate_disparate_impact(audit_data, protected_attr="age_bracket", decision_col="underwriting_decision")
    st.dataframe(audit_summary, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: POPULATION DRIFT (PSI)
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Production Score Drift Monitoring (PSI)")
    st.markdown("Quantifying distributional drift between training baseline and incoming live applicant default scores.")

    from src.models.monitoring import calculate_psi, evaluate_psi_status

    np.random.seed(42)
    baseline_scores = np.random.beta(2, 10, size=3000)
    live_scores = np.random.beta(2.8, 8.5, size=800)

    psi_val, breakdown_df = calculate_psi(baseline_scores, live_scores)

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Calculated Portfolio PSI", f"{psi_val:.4f}")
    col_m2.info(f"**Governance Status:** {evaluate_psi_status(psi_val)}")

    st.write("#### Quantile Decile Shift Breakdown")
    st.dataframe(breakdown_df, use_container_width=True)