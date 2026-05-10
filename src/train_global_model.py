from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS


def build_candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    tree_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", ColumnTransformer([("num", numeric_pipe, FEATURE_COLUMNS)])),
                ("model", LogisticRegression(max_iter=400, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", ColumnTransformer([("num", tree_pipe, FEATURE_COLUMNS)])),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=7,
                        min_samples_leaf=8,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("prep", ColumnTransformer([("num", tree_pipe, FEATURE_COLUMNS)])),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def evaluate_binary_classifier(y_true: pd.Series, scores: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_global_models(
    data: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        data,
        test_size=0.3,
        stratify=data["risk_label"],
        random_state=random_state,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["risk_label"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["risk_label"]

    model_candidates = build_candidate_models(random_state=random_state)
    metrics_rows = []
    fitted_models: dict[str, Pipeline] = {}
    test_predictions: dict[str, np.ndarray] = {}
    test_labels: dict[str, np.ndarray] = {}

    for name, model in model_candidates.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        scores = model.predict_proba(X_test)[:, 1]
        preds = (scores >= 0.5).astype(int)
        metrics = evaluate_binary_classifier(y_test, scores, preds)
        metrics_rows.append({"model_name": name, **metrics})
        test_predictions[name] = scores
        test_labels[name] = preds

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "global_model_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model_name"]
    best_model = fitted_models[best_model_name]
    best_scores = test_predictions[best_model_name]
    best_preds = test_labels[best_model_name]

    test_results = test_df.copy()
    test_results["global_risk_score"] = best_scores
    test_results["global_prediction"] = best_preds

    return {
        "train_df": train_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "metrics_df": metrics_df,
        "fitted_models": fitted_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "test_results": test_results.reset_index(drop=True),
    }
