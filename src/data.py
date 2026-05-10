from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "chronic_condition_count",
    "prior_claim_count",
    "total_claim_cost_6m",
    "emergency_visit_count",
    "medication_adherence_score",
    "missed_appointment_count",
    "preventive_visit_count",
]

SEGMENT_LABELS = {
    0: "chronic_care",
    1: "acute_high_utilization",
    2: "low_risk_preventive",
    3: "anomalous_members",
}


def generate_synthetic_healthcare_data(
    n_samples: int = 2400,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Create synthetic member-level healthcare data for demo purposes only.
    No PHI is used and no values map to real patients.
    """
    rng = np.random.default_rng(seed)
    segment_mix = np.array([0.34, 0.27, 0.31, 0.08])
    hidden_segment = rng.choice(list(SEGMENT_LABELS.keys()), size=n_samples, p=segment_mix)
    rows: list[dict[str, float | int | str]] = []

    for segment_id in hidden_segment:
        if segment_id == 0:
            age = rng.normal(68, 8)
            chronic = rng.poisson(3.6)
            prior_claims = rng.poisson(7.2)
            claim_cost = rng.normal(9000, 2800)
            ed_visits = rng.poisson(1.8)
            adherence = rng.beta(3.5, 2.0)
            missed = rng.poisson(1.5)
            preventive = rng.poisson(1.3) + 1
        elif segment_id == 1:
            age = rng.normal(47, 13)
            chronic = rng.poisson(1.6)
            prior_claims = rng.poisson(10.5)
            claim_cost = rng.normal(15000, 4500)
            ed_visits = rng.poisson(3.7)
            adherence = rng.beta(2.2, 2.5)
            missed = rng.poisson(2.7)
            preventive = rng.poisson(0.6)
        elif segment_id == 2:
            age = rng.normal(39, 11)
            chronic = rng.poisson(0.7)
            prior_claims = rng.poisson(2.3)
            claim_cost = rng.normal(2600, 1100)
            ed_visits = rng.poisson(0.4)
            adherence = rng.beta(5.0, 1.6)
            missed = rng.poisson(0.5)
            preventive = rng.poisson(2.0) + 1
        else:
            age = rng.normal(58, 16)
            chronic = rng.poisson(2.8)
            prior_claims = rng.poisson(14.0)
            claim_cost = rng.normal(27000, 9000)
            ed_visits = rng.poisson(5.2)
            adherence = rng.beta(1.2, 4.2)
            missed = rng.poisson(4.0)
            preventive = rng.poisson(0.5)

        age = int(np.clip(np.round(age), 18, 90))
        chronic = int(np.clip(chronic, 0, 8))
        prior_claims = int(np.clip(prior_claims, 0, 40))
        claim_cost = float(np.clip(claim_cost, 150, 60000))
        ed_visits = int(np.clip(ed_visits, 0, 15))
        adherence = float(np.clip(adherence, 0.05, 0.99))
        missed = int(np.clip(missed, 0, 12))
        preventive = int(np.clip(preventive, 0, 8))

        risk_signal = (
            -4.6
            + 0.03 * age
            + 0.42 * chronic
            + 0.09 * prior_claims
            + 0.00007 * claim_cost
            + 0.38 * ed_visits
            - 2.6 * adherence
            + 0.28 * missed
            - 0.24 * preventive
        )

        if segment_id == 0:
            risk_signal += 0.45 * (chronic >= 4) + 0.2 * (claim_cost > 10000)
        elif segment_id == 1:
            risk_signal += 0.4 * (ed_visits >= 4) + 0.25 * (prior_claims >= 12)
        elif segment_id == 2:
            risk_signal -= 0.55 * (preventive >= 2) - 0.15 * (adherence > 0.85)
        else:
            risk_signal += 0.95 + 0.4 * (adherence < 0.35 and claim_cost > 20000)

        probability = 1.0 / (1.0 + np.exp(-risk_signal))
        risk_label = int(rng.random() < probability)

        rows.append(
            {
                "age": age,
                "chronic_condition_count": chronic,
                "prior_claim_count": prior_claims,
                "total_claim_cost_6m": round(claim_cost, 2),
                "emergency_visit_count": ed_visits,
                "medication_adherence_score": round(adherence, 3),
                "missed_appointment_count": missed,
                "preventive_visit_count": preventive,
                "risk_label": risk_label,
                "hidden_segment": SEGMENT_LABELS[segment_id],
            }
        )

    frame = pd.DataFrame(rows)

    # Inject a small amount of missingness to support data quality checks.
    for column, frac in {
        "medication_adherence_score": 0.03,
        "missed_appointment_count": 0.02,
        "preventive_visit_count": 0.015,
    }.items():
        missing_index = frame.sample(frac=frac, random_state=seed + len(column)).index
        frame.loc[missing_index, column] = np.nan

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    return frame
