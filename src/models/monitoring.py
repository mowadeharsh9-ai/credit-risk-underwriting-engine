import numpy as np
import pandas as pd
from typing import Tuple, Dict


def calculate_psi(
    expected: np.ndarray, 
    actual: np.ndarray, 
    num_buckets: int = 10
) -> Tuple[float, pd.DataFrame]:
    """
    Calculates the Population Stability Index (PSI) between a baseline (training)
    distribution and production incoming traffic.
    """
    # Define bucket boundaries based on expected distribution quantiles
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bucket_bounds)
    actual_counts, _ = np.histogram(actual, bins=bucket_bounds)

    # Convert to fractions and apply smoothing to prevent division by zero
    expected_pct = np.maximum(expected_counts / len(expected), 1e-4)
    actual_pct = np.maximum(actual_counts / len(actual), 1e-4)

    # PSI calculation per bucket
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_values))

    breakdown = pd.DataFrame({
        "Bucket": range(1, num_buckets + 1),
        "Expected_%": np.round(expected_pct * 100, 2),
        "Actual_%": np.round(actual_pct * 100, 2),
        "Bucket_PSI": np.round(psi_values, 4)
    })

    return round(total_psi, 4), breakdown


def evaluate_psi_status(psi_value: float) -> str:
    if psi_value < 0.10:
        return "STABLE: No significant distribution change"
    elif psi_value < 0.25:
        return "WARNING: Moderate distribution shift detected"
    else:
        return "CRITICAL: Severe drift. Retraining/Recalibration required"


if __name__ == "__main__":
    np.random.seed(42)
    # Baseline expected default probabilities from training
    baseline_pd = np.random.beta(a=2, b=10, size=5000)

    # Scenario A: Stable production traffic
    stable_production_pd = np.random.beta(a=2, b=10, size=1500)
    psi_stable, _ = calculate_psi(baseline_pd, stable_production_pd)
    print(f"\nScenario A (Stable Traffic) PSI: {psi_stable:.4f} -> {evaluate_psi_status(psi_stable)}")

    # Scenario B: Distressed economic shift (credit deterioration)
    distressed_production_pd = np.random.beta(a=3, b=7, size=1500)
    psi_drifted, breakdown = calculate_psi(baseline_pd, distressed_production_pd)
    print(f"\nScenario B (Economic Shock) PSI: {psi_drifted:.4f} -> {evaluate_psi_status(psi_drifted)}")
    print("\nBucket Drift Breakdown:")
    print(breakdown.to_string(index=False))