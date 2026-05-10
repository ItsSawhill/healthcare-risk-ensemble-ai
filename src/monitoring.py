from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_monitoring_summary(
    data: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    segment_comparison: pd.DataFrame,
    routed_df: pd.DataFrame,
    metrics_dir: str | Path,
) -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_rates = data.isna().mean().sort_values(ascending=False)
    for feature, rate in missing_rates.items():
        rows.append({"check": f"missing_rate_{feature}", "value": round(float(rate), 4), "status": "warn" if rate > 0.02 else "ok"})

    for cluster_id, cluster_size in data["cluster_id"].value_counts(normalize=True).sort_index().items():
        rows.append(
            {
                "check": f"cluster_share_{cluster_id}",
                "value": round(float(cluster_size), 4),
                "status": "warn" if cluster_size < 0.08 else "ok",
            }
        )

    rows.append(
        {
            "check": "outlier_rate",
            "value": round(float(data["outlier_flag"].mean()), 4),
            "status": "warn" if data["outlier_flag"].mean() > 0.12 else "ok",
        }
    )

    rows.append(
        {
            "check": "human_review_rate",
            "value": round(float(routed_df["human_review_flag"].mean()), 4),
            "status": "warn" if routed_df["human_review_flag"].mean() > 0.2 else "ok",
        }
    )

    prediction_gap = abs(routed_df["risk_score"] - routed_df["global_risk_score"]).mean()
    rows.append(
        {
            "check": "avg_global_vs_routed_prediction_gap",
            "value": round(float(prediction_gap), 4),
            "status": "warn" if prediction_gap > 0.08 else "ok",
        }
    )

    for _, row in segment_comparison.dropna(subset=["global_f1", "segment_f1"]).iterrows():
        status = "warn" if row["segment_f1"] + 0.02 < row["global_f1"] else "ok"
        rows.append(
            {
                "check": f"segment_performance_cluster_{int(row['cluster_id'])}",
                "value": round(float(row["segment_f1"] - row["global_f1"]), 4),
                "status": status,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(metrics_dir / "monitoring_summary.csv", index=False)
    return summary
