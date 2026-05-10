from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.assistant import generate_analyst_summary
from src.data import FEATURE_COLUMNS, generate_synthetic_healthcare_data
from src.explain import explain_member, get_global_feature_importance
from src.monitoring import build_monitoring_summary
from src.outliers import detect_outliers, score_outliers
from src.router import route_members
from src.segmentation import assign_cluster, run_segmentation
from src.train_global_model import train_global_models
from src.train_segment_models import train_segment_models


def render_feature_importance(feature_importance: pd.DataFrame, figure_path: Path) -> Path:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.6))
    ordered = feature_importance.sort_values("importance")
    plt.barh(ordered["feature"], ordered["importance"], color="#F58518")
    plt.title("Global Model Feature Importance")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return figure_path


def predict_member(
    member_df: pd.DataFrame,
    results: dict[str, object],
) -> dict[str, object]:
    scored_member = member_df.copy()
    scored_member["cluster_id"] = assign_cluster(scored_member, results["segmentation"])
    scored_member["segment_label"] = scored_member["cluster_id"].map(results["segmentation"]["cluster_labels"])
    scored_member = score_outliers(scored_member, results["outlier_results"])
    scored_member = route_members(
        scored_df=scored_member,
        global_model=results["global_training"]["best_model"],
        segment_models=results["segment_training"]["segment_models"],
        segment_decisions=results["segment_training"]["segment_decisions"],
    )

    explained = explain_member(
        scored_member.iloc[0],
        scored_member.iloc[0],
        results["global_training"]["train_df"],
        results["feature_importance"],
    )
    analyst_summary = generate_analyst_summary(explained)

    return {
        "member_df": scored_member,
        "explanation": explained,
        "assistant_summary": analyst_summary,
    }


def run_demo_pipeline(base_dir: str | Path) -> dict[str, object]:
    base_dir = Path(base_dir)
    metrics_dir = base_dir / "outputs" / "metrics"
    figures_dir = base_dir / "outputs" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_healthcare_data(
        n_samples=2400,
        seed=42,
        output_path=metrics_dir / "synthetic_member_data.csv",
    )
    data = data.reset_index(drop=True)
    data["member_id"] = range(100000, 100000 + len(data))

    segmentation = run_segmentation(data, metrics_dir, figures_dir)
    segmented = segmentation["data"].copy()

    global_training = train_global_models(segmented, metrics_dir)
    train_df = global_training["train_df"].copy()
    test_df = global_training["test_df"].copy()

    segment_training = train_segment_models(
        train_df=train_df,
        test_df=test_df,
        global_model=global_training["best_model"],
        metrics_dir=metrics_dir,
    )

    outlier_results = detect_outliers(segmented, metrics_dir)
    outlier_df = outlier_results["data"].copy()

    train_df = train_df.merge(outlier_df[["member_id", "outlier_flag"]], on="member_id", how="left")

    routed = route_members(
        scored_df=global_training["test_results"].merge(
            outlier_df[["member_id", "cluster_id", "outlier_flag"]],
            on=["member_id", "cluster_id"],
            how="left",
        ),
        global_model=global_training["best_model"],
        segment_models=segment_training["segment_models"],
        segment_decisions=segment_training["segment_decisions"],
    )

    feature_importance = get_global_feature_importance(global_training["best_model"])
    feature_importance.to_csv(metrics_dir / "global_feature_importance.csv", index=False)
    render_feature_importance(feature_importance, figures_dir / "global_feature_importance.png")

    monitoring_summary = build_monitoring_summary(
        data=outlier_df,
        cluster_profiles=segmentation["cluster_profiles"],
        segment_comparison=segment_training["comparison_df"],
        routed_df=routed,
        metrics_dir=metrics_dir,
    )

    sample_member_id = int(
        routed.sort_values(["human_review_flag", "risk_score"], ascending=[False, False]).iloc[0]["member_id"]
    )
    sample_row = routed[routed["member_id"] == sample_member_id].iloc[0]
    explanation = explain_member(sample_row, sample_row, train_df, feature_importance)
    assistant_summary = generate_analyst_summary(explanation)

    routed_output = routed.copy()
    routed_output["top_risk_drivers"] = [
        ", ".join(explain_member(row, row, train_df, feature_importance)["top_risk_drivers"])
        for _, row in routed.iterrows()
    ]
    routed_output.to_csv(metrics_dir / "routed_member_predictions.csv", index=False)

    return {
        "data": data,
        "segmentation": segmentation,
        "global_training": global_training,
        "segment_training": segment_training,
        "outlier_results": outlier_results,
        "routed": routed_output,
        "feature_importance": feature_importance,
        "monitoring_summary": monitoring_summary,
        "sample_member_id": sample_member_id,
        "sample_explanation": explanation,
        "assistant_summary": assistant_summary,
        "metrics_dir": metrics_dir,
        "figures_dir": figures_dir,
    }
