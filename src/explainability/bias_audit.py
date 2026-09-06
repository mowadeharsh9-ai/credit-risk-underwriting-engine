import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_disparate_impact(
    df: pd.DataFrame, 
    protected_attr: str, 
    decision_col: str, 
    favorable_outcome: str = "APPROVED"
) -> pd.DataFrame:
    """
    Computes Disparate Impact Ratio (DIR) across demographic slices.
    Applies the EEOC / CFPB Four-Fifths (80%) Rule benchmark.
    """
    groups = df[protected_attr].unique()
    summary = []

    # Calculate baseline approval rate per demographic slice
    group_stats = {}
    for group in groups:
        group_df = df[df[protected_attr] == group]
        total = len(group_df)
        approved = len(group_df[group_df[decision_col] == favorable_outcome])
        approval_rate = approved / total if total > 0 else 0.0
        group_stats[group] = {
            "total_applicants": total,
            "approved": approved,
            "approval_rate": approval_rate
        }

    # Reference rate is the highest approval rate among all groups
    max_rate = max(s["approval_rate"] for s in group_stats.values()) if group_stats else 1.0

    for group, stats in group_stats.items():
        dir_ratio = stats["approval_rate"] / max_rate if max_rate > 0 else 1.0
        # 80% Rule violation if ratio < 0.80
        violates_four_fifths = dir_ratio < 0.80

        summary.append({
            "demographic_group": str(group),
            "total_applicants": stats["total_applicants"],
            "approvals": stats["approved"],
            "approval_rate": round(stats["approval_rate"] * 100, 2),
            "disparate_impact_ratio": round(dir_ratio, 3),
            "four_fifths_compliance": "PASS" if not violates_four_fifths else "FAIL (Adverse Impact)"
        })

    return pd.DataFrame(summary).sort_values(by="disparate_impact_ratio", ascending=False)


if __name__ == "__main__":
    # Generate synthetic validation batch with demographic proxy for testing
    np.random.seed(42)
    sample_size = 1000

    age_brackets = np.random.choice(["18-25", "26-40", "41-60", "60+"], size=sample_size, p=[0.2, 0.4, 0.3, 0.1])
    decisions = []

    # Simulate realistic underwriting pass/fail rates across age
    for age in age_brackets:
        prob_pass = 0.55 if age == "18-25" else (0.75 if age in ["26-40", "41-60"] else 0.70)
        decisions.append("APPROVED" if np.random.rand() < prob_pass else "REJECTED")

    audit_df = pd.DataFrame({
        "age_bracket": age_brackets,
        "underwriting_decision": decisions
    })

    results = calculate_disparate_impact(audit_df, protected_attr="age_bracket", decision_col="underwriting_decision")
    print("\n=== FAIR LENDING COMPLIANCE AUDIT: AGE BRACKETS ===")
    print(results.to_string(index=False))