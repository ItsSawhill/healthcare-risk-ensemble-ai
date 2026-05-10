from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from src.data import FEATURE_COLUMNS


def apply_outlier_rules(data: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    flagged = data.copy()
    flagged["rule_high_claim_cost"] = (flagged["total_claim_cost_6m"] > thresholds["claim_cost_p97"]).astype(int)
    flagged["rule_high_ed_visits"] = (flagged["emergency_visit_count"] >= thresholds["ed_visits_p98"]).astype(int)
    flagged["rule_low_adherence_high_util"] = (
        (flagged["medication_adherence_score"].fillna(thresholds["adherence_median"]) < 0.35)
        & (flagged["total_claim_cost_6m"] > thresholds["claim_cost_p90"])
        & (flagged["prior_claim_count"] > thresholds["prior_claims_p90"])
    ).astype(int)
    return flagged


def score_outliers(member_df: pd.DataFrame, outlier_artifacts: dict[str, object]) -> pd.DataFrame:
    imputer = outlier_artifacts["imputer"]
    iso = outlier_artifacts["model"]
    thresholds = outlier_artifacts["thresholds"]

    scored = apply_outlier_rules(member_df, thresholds)
    matrix = imputer.transform(scored[FEATURE_COLUMNS])
    scored["iforest_outlier_flag"] = (iso.predict(matrix) == -1).astype(int)
    scored["outlier_flag"] = (
        scored[
            [
                "iforest_outlier_flag",
                "rule_high_claim_cost",
                "rule_high_ed_visits",
                "rule_low_adherence_high_util",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)
    return scored


def detect_outliers(
    data: pd.DataFrame,
    metrics_dir: str | Path,
    random_state: int = 42,
) -> dict[str, object]:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(data[FEATURE_COLUMNS])
    iso = IsolationForest(contamination=0.06, random_state=random_state)
    iso_flag = iso.fit_predict(matrix)
    thresholds = {
        "claim_cost_p97": float(data["total_claim_cost_6m"].quantile(0.97)),
        "claim_cost_p90": float(data["total_claim_cost_6m"].quantile(0.90)),
        "ed_visits_p98": float(data["emergency_visit_count"].quantile(0.98)),
        "prior_claims_p90": float(data["prior_claim_count"].quantile(0.90)),
        "adherence_median": float(data["medication_adherence_score"].median()),
    }

    flagged = data.copy()
    flagged["iforest_outlier_flag"] = (iso_flag == -1).astype(int)
    flagged = apply_outlier_rules(flagged, thresholds)
    flagged["outlier_flag"] = (
        flagged[
            [
                "iforest_outlier_flag",
                "rule_high_claim_cost",
                "rule_high_ed_visits",
                "rule_low_adherence_high_util",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    summary = pd.DataFrame(
        [
            {"metric": "member_count", "value": len(flagged)},
            {"metric": "isolation_forest_outliers", "value": int(flagged["iforest_outlier_flag"].sum())},
            {"metric": "rule_high_claim_cost", "value": int(flagged["rule_high_claim_cost"].sum())},
            {"metric": "rule_high_ed_visits", "value": int(flagged["rule_high_ed_visits"].sum())},
            {"metric": "rule_low_adherence_high_util", "value": int(flagged["rule_low_adherence_high_util"].sum())},
            {"metric": "total_outlier_flagged", "value": int(flagged["outlier_flag"].sum())},
            {"metric": "outlier_rate", "value": round(float(flagged["outlier_flag"].mean()), 4)},
        ]
    )
    summary.to_csv(metrics_dir / "outlier_summary.csv", index=False)
    return {"data": flagged, "summary": summary, "model": iso, "imputer": imputer, "thresholds": thresholds}
