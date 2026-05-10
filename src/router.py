from __future__ import annotations

import pandas as pd

from src.data import FEATURE_COLUMNS


def classify_confidence(score: float) -> str:
    distance = abs(score - 0.5)
    if distance < 0.1:
        return "low"
    if distance < 0.22:
        return "medium"
    return "high"


def route_members(
    scored_df: pd.DataFrame,
    global_model,
    segment_models: dict[int, object],
    segment_decisions: dict[int, dict[str, float | bool]],
) -> pd.DataFrame:
    routed = scored_df.copy()
    model_used = []
    risk_scores = []
    confidence_levels = []
    human_review_flags = []

    global_scores = global_model.predict_proba(routed[FEATURE_COLUMNS])[:, 1]

    for position, (_, row) in enumerate(routed.iterrows()):
        cluster_id = int(row["cluster_id"])
        outlier_flag = bool(row["outlier_flag"])
        segment_decision = segment_decisions.get(cluster_id, {"recommended_segment_model": False})

        selected_model_name = "global_model"
        score = float(global_scores[position])

        if segment_decision.get("recommended_segment_model") and cluster_id in segment_models:
            score = float(segment_models[cluster_id].predict_proba(pd.DataFrame([row[FEATURE_COLUMNS]]))[:, 1][0])
            selected_model_name = f"segment_model_cluster_{cluster_id}"

        confidence = classify_confidence(score)
        human_review = outlier_flag or confidence == "low"

        model_used.append(selected_model_name)
        risk_scores.append(score)
        confidence_levels.append(confidence)
        human_review_flags.append(int(human_review))

    routed["model_used"] = model_used
    routed["risk_score"] = risk_scores
    routed["confidence_level"] = confidence_levels
    routed["human_review_flag"] = human_review_flags
    return routed
