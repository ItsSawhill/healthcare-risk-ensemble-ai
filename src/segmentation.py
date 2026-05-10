from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS


def derive_cluster_labels(cluster_profiles: pd.DataFrame) -> dict[int, str]:
    labels: dict[int, str] = {}
    for _, row in cluster_profiles.iterrows():
        cluster_id = int(row["cluster_id"])
        if row["avg_claim_cost"] > 22000 or row["positive_rate"] > 0.85:
            labels[cluster_id] = "anomalous_high_risk"
        elif row["avg_ed_visits"] > 2.5 and row["avg_claim_cost"] > 10000:
            labels[cluster_id] = "acute_high_utilization"
        elif row["avg_adherence"] > 0.7 and row["positive_rate"] < 0.1:
            labels[cluster_id] = "low_risk_preventive"
        else:
            labels[cluster_id] = "chronic_care"
    return labels


def assign_cluster(member_df: pd.DataFrame, segmentation_artifacts: dict[str, object]) -> pd.Series:
    prep = segmentation_artifacts["preprocessor"]
    centroids = segmentation_artifacts["cluster_centroids"]
    transformed = prep.transform(member_df[FEATURE_COLUMNS])
    distances = ((transformed[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return pd.Series(np.argmin(distances, axis=1), index=member_df.index)


def run_segmentation(
    data: pd.DataFrame,
    metrics_dir: str | Path,
    figures_dir: str | Path,
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict[str, object]:
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    segment_features = FEATURE_COLUMNS
    prep = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformed = prep.fit_transform(data[segment_features])
    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(transformed)

    segmented = data.copy()
    segmented["cluster_id"] = clusters
    cluster_centroids = np.vstack(
        [transformed[segmented["cluster_id"] == cluster_id].mean(axis=0) for cluster_id in sorted(segmented["cluster_id"].unique())]
    )

    cluster_profiles = (
        segmented.groupby("cluster_id")[FEATURE_COLUMNS + ["risk_label"]]
        .mean(numeric_only=True)
        .round(3)
        .rename(columns={"risk_label": "avg_risk_label"})
        .reset_index()
    )
    cluster_profiles["member_count"] = segmented.groupby("cluster_id").size().values
    cluster_profiles.to_csv(metrics_dir / "cluster_profiles.csv", index=False)

    cluster_summary = (
        segmented.groupby("cluster_id")
        .agg(
            member_count=("risk_label", "size"),
            positive_rate=("risk_label", "mean"),
            avg_claim_cost=("total_claim_cost_6m", "mean"),
            avg_ed_visits=("emergency_visit_count", "mean"),
            avg_adherence=("medication_adherence_score", "mean"),
        )
        .round(3)
        .reset_index()
    )
    cluster_labels = derive_cluster_labels(cluster_summary)
    cluster_profiles["segment_label"] = cluster_profiles["cluster_id"].map(cluster_labels)
    cluster_profiles.to_csv(metrics_dir / "cluster_profiles.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(cluster_summary["cluster_id"].astype(str), cluster_summary["member_count"], color="#4C78A8")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Member Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_distribution.png", dpi=160)
    plt.close()

    return {
        "data": segmented,
        "preprocessor": prep,
        "model": model,
        "cluster_centroids": cluster_centroids,
        "cluster_profiles": cluster_profiles,
        "cluster_summary": cluster_summary,
        "cluster_labels": cluster_labels,
    }
