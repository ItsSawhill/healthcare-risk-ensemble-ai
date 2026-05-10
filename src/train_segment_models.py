from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data import FEATURE_COLUMNS
from src.train_global_model import evaluate_binary_classifier


def train_segment_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    global_model,
    metrics_dir: str | Path,
    random_state: int = 42,
) -> dict[str, object]:
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    segment_models: dict[int, Pipeline] = {}
    segment_decisions: dict[int, dict[str, float | bool]] = {}

    for cluster_id in sorted(train_df["cluster_id"].unique()):
        train_seg = train_df[train_df["cluster_id"] == cluster_id]
        test_seg = test_df[test_df["cluster_id"] == cluster_id]

        if len(train_seg) < 120 or len(test_seg) < 40 or test_seg["risk_label"].nunique() < 2:
            comparison_rows.append(
                {
                    "cluster_id": cluster_id,
                    "train_count": len(train_seg),
                    "test_count": len(test_seg),
                    "global_roc_auc": None,
                    "global_f1": None,
                    "segment_roc_auc": None,
                    "segment_f1": None,
                    "recommended_segment_model": False,
                    "note": "insufficient_segment_data",
                }
            )
            segment_decisions[cluster_id] = {"recommended_segment_model": False, "delta_f1": 0.0}
            continue

        X_test = test_seg[FEATURE_COLUMNS]
        y_test = test_seg["risk_label"]
        global_scores = global_model.predict_proba(X_test)[:, 1]
        global_preds = (global_scores >= 0.5).astype(int)
        global_metrics = evaluate_binary_classifier(y_test, global_scores, global_preds)

        segment_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=6,
                        min_samples_leaf=6,
                        random_state=random_state + int(cluster_id),
                    ),
                ),
            ]
        )
        segment_model.fit(train_seg[FEATURE_COLUMNS], train_seg["risk_label"])
        segment_scores = segment_model.predict_proba(X_test)[:, 1]
        segment_preds = (segment_scores >= 0.5).astype(int)
        segment_metrics = evaluate_binary_classifier(y_test, segment_scores, segment_preds)

        use_segment_model = (
            segment_metrics["f1"] >= global_metrics["f1"] + 0.02
            or segment_metrics["roc_auc"] >= global_metrics["roc_auc"] + 0.015
        )

        comparison_rows.append(
            {
                "cluster_id": cluster_id,
                "train_count": len(train_seg),
                "test_count": len(test_seg),
                "global_roc_auc": round(global_metrics["roc_auc"], 4),
                "global_f1": round(global_metrics["f1"], 4),
                "segment_roc_auc": round(segment_metrics["roc_auc"], 4),
                "segment_f1": round(segment_metrics["f1"], 4),
                "recommended_segment_model": use_segment_model,
                "note": "segment_model_available",
            }
        )
        segment_models[int(cluster_id)] = segment_model
        segment_decisions[int(cluster_id)] = {
            "recommended_segment_model": bool(use_segment_model),
            "delta_f1": float(segment_metrics["f1"] - global_metrics["f1"]),
            "delta_roc_auc": float(segment_metrics["roc_auc"] - global_metrics["roc_auc"]),
        }

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(metrics_dir / "segment_model_comparison.csv", index=False)
    return {
        "comparison_df": comparison_df,
        "segment_models": segment_models,
        "segment_decisions": segment_decisions,
    }
