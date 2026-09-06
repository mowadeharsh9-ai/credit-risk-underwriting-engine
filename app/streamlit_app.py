import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credit Underwriting Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Real-Time Credit Risk & Underwriting Engine")
st.markdown("Automated credit risk assessment with Fair Lending Adverse Action explanations (SHAP).")

# API Configuration
API_URL = "http://127.0.0.1:8000/v1/underwrite"

# Sidebar - Applicant Inputs
st.sidebar.header("Borrower Financial Profile")

annual_income = st.sidebar.number_input("Annual Gross Income ($)", min_value=1000.0, max_value=1000000.0, value=45000.0, step=1000.0)
loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=500.0)
monthly_installment = st.sidebar.number_input("Proposed Monthly Installment ($)", min_value=50.0, max_value=10000.0, value=450.0, step=25.0)

st.sidebar.subheader("Credit File Metrics")
revolving_balance = st.sidebar.number_input("Revolving Balance ($)", min_value=0.0, max_value=500000.0, value=4500.0, step=500.0)
revolving_limit = st.sidebar.number_input("Revolving Credit Limit ($)", min_value=100.0, max_value=500000.0, value=8000.0, step=500.0)
credit_history_months = st.sidebar.slider("Credit History (Months)", min_value=6, max_value=480, value=36)
inquiries_last_6m = st.sidebar.slider("Hard Inquiries (Last 6 Months)", min_value=0, max_value=20, value=3)
delinquencies_2y = st.sidebar.slider("Delinquencies (Last 2 Years)", min_value=0, max_value=15, value=1)

home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["debt_consolidation", "credit_card", "home_improvement", "small_business", "major_purchase", "other"]
)

# Main Screen Layout
col1, col2 = st.columns([1, 1.2])

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
    "loan_purpose": loan_purpose
}

with col1:
    st.subheader("Decision & Scorecard")
    if st.button("Run Underwriting Evaluation", type="primary", use_container_width=True):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                decision = result["decision"]
                pd_val = result["probability_of_default"]
                score = result["credit_score"]
                threshold = result["decision_threshold"]

                # Decision Badge
                if decision == "APPROVED":
                    st.success(f"### Decision: APPROVED\nCredit Score: **{score}** (PD: {pd_val:.2%})")
                elif decision == "MANUAL_REVIEW":
                    st.warning(f"### Decision: MANUAL REVIEW\nCredit Score: **{score}** (PD: {pd_val:.2%})")
                else:
                    st.error(f"### Decision: REJECTED\nCredit Score: **{score}** (PD: {pd_val:.2%})")

                # Metric Cards
                m1, m2 = st.columns(2)
                m1.metric("Est. Probability of Default", f"{pd_val:.2%}")
                m2.metric("Business Cutoff Threshold", f"{threshold:.2%}")

                # Adverse Action Notices if applicable
                if result.get("adverse_action_reasons"):
                    st.subheader("Regulatory Adverse Action Notice")
                    st.caption("Primary drivers pushing the risk profile above the acceptable threshold:")
                    reasons = result["adverse_action_reasons"]
                    for idx, r in enumerate(reasons, 1):
                        st.markdown(f"**{idx}. {r['human_readable_reason']}** (Feature: `{r['feature_name']}`)")

                st.session_state["last_result"] = result
            else:
                st.error(f"Underwriting service returned status code {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not reach FastAPI. Make sure `uvicorn src.api.main:app --reload` is running in another terminal.")

with col2:
    st.subheader("Local Interpretability Analysis (SHAP)")
    if "last_result" in st.session_state and st.session_state["last_result"].get("adverse_action_reasons"):
        reasons = st.session_state["last_result"]["adverse_action_reasons"]
        df_reasons = pd.DataFrame(reasons)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.barh(df_reasons["feature_name"], df_reasons["shap_impact"], color="#d9534f")
        ax.set_xlabel("SHAP Impact (Log-Odds toward Default)")
        ax.set_title("Top Risk Contributors for Applicant")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"+{width:.2f}",
                    ha='left', va='center', fontsize=9)

        st.pyplot(fig)
    else:
        st.info("Run an underwriting evaluation that results in a rejection or review to visualize SHAP factor attribution.")