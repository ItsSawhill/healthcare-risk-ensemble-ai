from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS


def get_global_feature_importance(model) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)

    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values}).sort_values(
        "importance", ascending=False
    )


def compute_local_driver_scores(
    row: pd.Series,
    reference_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> list[dict[str, Any]]:
    reference = reference_df[FEATURE_COLUMNS].median(numeric_only=True)
    importance_map = dict(zip(feature_importance["feature"], feature_importance["importance"]))
    contributions = []

    for feature in FEATURE_COLUMNS:
        row_value = row[feature]
        if pd.isna(row_value):
            row_value = reference[feature]
        ref_value = reference[feature]
        direction = 1.0
        if feature in {"medication_adherence_score", "preventive_visit_count"}:
            direction = -1.0
        delta = float(row_value - ref_value) * direction
        score = abs(delta) * float(importance_map.get(feature, 0.0))
        contributions.append(
            {
                "feature": feature,
                "value": float(row_value),
                "reference": float(ref_value),
                "score": score,
                "is_higher_risk": delta > 0,
            }
        )

    return sorted(contributions, key=lambda item: item["score"], reverse=True)


def explain_member(
    row: pd.Series,
    route_row: pd.Series,
    reference_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict[str, Any]:
    drivers = compute_local_driver_scores(row, reference_df, feature_importance)
    top_drivers = []
    for driver in drivers[:3]:
        descriptor = "above" if driver["is_higher_risk"] else "below"
        top_drivers.append(
            f"{driver['feature']} is {descriptor} the portfolio median"
        )

    return {
        "member_id": int(row["member_id"]),
        "model_used": route_row["model_used"],
        "risk_score": round(float(route_row["risk_score"]), 3),
        "confidence_level": route_row["confidence_level"],
        "cluster_id": int(route_row["cluster_id"]),
        "outlier_flag": bool(route_row["outlier_flag"]),
        "human_review_flag": bool(route_row["human_review_flag"]),
        "top_risk_drivers": top_drivers,
    }
