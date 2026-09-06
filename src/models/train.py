import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_credit_model.joblib")
CALIBRATOR_PATH = os.path.join(MODEL_DIR, "isotonic_calibrator.joblib")

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


def generate_synthetic_data(n_samples: int = 25000) -> pd.DataFrame:
    np.random.seed(42)
    annual_income = np.random.lognormal(mean=10.9, sigma=0.55, size=n_samples)
    loan_amount = np.random.uniform(2000, 35000, size=n_samples)
    monthly_installment = (loan_amount * np.random.uniform(0.02, 0.045, size=n_samples))
    credit_history_months = np.random.randint(12, 360, size=n_samples)
    inquiries_last_6m = np.random.poisson(lam=1.2, size=n_samples)
    delinquencies_2y = np.random.poisson(lam=0.4, size=n_samples)
    revolving_balance = np.random.uniform(500, 25000, size=n_samples)
    revolving_limit = revolving_balance * np.random.uniform(0.8, 3.5, size=n_samples)

    monthly_income = annual_income / 12.0
    revolving_utilization = np.clip(revolving_balance / revolving_limit, 0.0, 3.0)
    payment_to_income = np.clip(monthly_installment / monthly_income, 0.0, 2.0)
    debt_to_income = np.clip((monthly_installment + revolving_balance * 0.03) / monthly_income, 0.0, 3.0)
    recent_inquiry_density = inquiries_last_6m / np.maximum(credit_history_months / 6.0, 1.0)

    log_odds = (
        -2.5
        + 1.6 * revolving_utilization
        + 1.8 * payment_to_income
        + 0.9 * debt_to_income
        + 0.6 * recent_inquiry_density
        + 0.5 * delinquencies_2y
        - 0.003 * credit_history_months
        - 0.000015 * annual_income
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    target = np.random.binomial(1, prob_default)

    return pd.DataFrame({
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "monthly_installment": monthly_installment,
        "credit_history_months": credit_history_months,
        "inquiries_last_6m": inquiries_last_6m,
        "delinquencies_2y": delinquencies_2y,
        "revolving_utilization_ratio": revolving_utilization,
        "payment_to_income": payment_to_income,
        "debt_to_income": debt_to_income,
        "recent_inquiry_density": recent_inquiry_density,
        "is_default": target,
    })


def load_or_generate_dataset() -> pd.DataFrame:
    raw_dir = "data/raw"
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")] if os.path.exists(raw_dir) else []

    if csv_files:
        chosen_file = os.path.join(raw_dir, csv_files[0])
        print(f"Loading dataset from: {chosen_file}")
        
        # Read limited rows first for high performance
        df = pd.read_csv(chosen_file, nrows=50000, low_memory=False)
        
        if "loan_status" in df.columns:
            df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
            df["is_default"] = df["loan_status"].apply(lambda x: 0 if x == "Fully Paid" else 1)
            
            # Map standard LendingClub columns if present
            if "annual_inc" in df.columns:
                df["annual_income"] = df["annual_inc"].fillna(50000.0)
                df["monthly_installment"] = df["installment"].fillna(300.0)
                df["delinquencies_2y"] = df["delinq_2yrs"].fillna(0)
                df["inquiries_last_6m"] = df["inq_last_6mths"].fillna(0)
                df["revolving_balance"] = df["revol_bal"].fillna(1000.0)
                df["revolving_limit"] = df["total_rev_hi_lim"].fillna(5000.0)
                df["credit_history_months"] = 60
                
                monthly_income = df["annual_income"] / 12.0
                df["revolving_utilization_ratio"] = np.clip(df["revolving_balance"] / np.maximum(df["revolving_limit"], 1.0), 0.0, 3.0)
                df["payment_to_income"] = np.clip(df["monthly_installment"] / np.maximum(monthly_income, 1.0), 0.0, 2.0)
                df["debt_to_income"] = np.clip((df["monthly_installment"] + df["revolving_balance"] * 0.03) / np.maximum(monthly_income, 1.0), 0.0, 3.0)
                df["recent_inquiry_density"] = df["inquiries_last_6m"] / 10.0
                return df
                
    print("Generating calibrated training set...")
    return generate_synthetic_data()


def run_training_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_or_generate_dataset()

    X = df[FEATURE_COLUMNS]
    y = df["is_default"]

    print(f"Records: {len(X)} | Default Rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n--- Training Base LightGBM Model ---")
    scale_pos_weight = float((1 - y_train.mean()) / max(y_train.mean(), 1e-4))
    lgbm = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    raw_preds = lgbm.predict_proba(X_test)[:, 1]
    print(f"Base LightGBM ROC-AUC: {roc_auc_score(y_test, raw_preds):.4f}")

    print("\n--- Fitting Calibrated Classifier (Isotonic) ---")
    # Calibrate using 3-fold cross-validation
    calibrator = CalibratedClassifierCV(estimator=lgbm, method="isotonic", cv=3)
    calibrator.fit(X_train, y_train)

    calib_preds = calibrator.predict_proba(X_test)[:, 1]
    print(f"Calibrated ROC-AUC: {roc_auc_score(y_test, calib_preds):.4f}")
    print(f"Calibrated Brier Score: {brier_score_loss(y_test, calib_preds):.4f}")
    print(f"Calibrated Log Loss: {log_loss(y_test, calib_preds):.4f}")

    print(f"\nPersisting artifacts into '{MODEL_DIR}/'...")
    joblib.dump(lgbm, MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print("Artifacts successfully created and saved.")


if __name__ == "__main__":
    run_training_pipeline()