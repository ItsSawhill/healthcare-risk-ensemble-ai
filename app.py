from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.demo_pipeline import predict_member, run_demo_pipeline


BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def load_demo_results() -> dict[str, object]:
    return run_demo_pipeline(BASE_DIR)


def risk_category(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.65:
        return "Medium"
    return "High"


def format_model_label(model_used: str) -> str:
    if model_used == "global_model":
        return "Global model"
    return f"Segment-specific model ({model_used.split('_')[-1]})"


def render_app_chrome() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #F8FAFC;
            color: #0F172A;
        }

        .block-container {
            max-width: 1160px;
            padding-top: 1.8rem;
            padding-bottom: 3.2rem;
        }

        h1, h2, h3, h4, p, li, label, div {
            color: #0F172A;
        }

        [data-testid="stMarkdownContainer"] p {
            color: #475569;
            line-height: 1.6;
        }

        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetricValue"] {
            color: #0F172A;
            font-weight: 800;
        }

        div[data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 700;
        }

        .hero-shell {
            background: #FFFFFF;
            border: 1px solid #DDE7EE;
            border-radius: 24px;
            padding: 1.8rem 1.9rem 1.6rem 1.9rem;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: #0F766E;
            font-weight: 800;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.08;
            font-weight: 800;
            margin: 0.45rem 0 0.7rem 0;
            color: #0F172A;
        }

        .hero-copy {
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.55;
            color: #475569;
            margin-bottom: 1.1rem;
        }

        .hero-tags {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .hero-tag {
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            background: #F0FDFA;
            border: 1px solid #99F6E4;
            color: #115E59;
            font-size: 0.86rem;
            font-weight: 700;
        }

        .section-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            padding: 1.2rem 1.2rem 1rem 1.2rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            margin: 1rem 0 1.2rem 0;
        }

        .panel-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.35rem;
        }

        .panel-copy {
            color: #475569;
            font-size: 0.98rem;
            margin-bottom: 0.9rem;
            max-width: 860px;
        }

        .result-band {
            background: #F8FAFC;
            border: 1px solid #DCE7EF;
            border-radius: 18px;
            padding: 1rem 1rem 0.35rem 1rem;
            margin-top: 0.9rem;
            margin-bottom: 1rem;
        }

        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin: 0.5rem 0 0.9rem 0;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.48rem 0.8rem;
            font-size: 0.86rem;
            font-weight: 700;
        }

        .pill-teal {
            background: #CCFBF1;
            border: 1px solid #5EEAD4;
            color: #115E59;
        }

        .pill-blue {
            background: #DBEAFE;
            border: 1px solid #93C5FD;
            color: #1D4ED8;
        }

        .pill-red {
            background: #FEE2E2;
            border: 1px solid #FCA5A5;
            color: #B91C1C;
        }

        .pill-green {
            background: #DCFCE7;
            border: 1px solid #86EFAC;
            color: #166534;
        }

        .route-panel {
            border: 1px solid #BFDBFE;
            background: #EFF6FF;
            color: #1E3A8A;
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.9rem;
            font-size: 0.96rem;
            line-height: 1.55;
        }

        .review-panel {
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.95rem;
            font-weight: 700;
            line-height: 1.5;
        }

        .review-panel-warning {
            background: #FFF7ED;
            border: 1px solid #FCD34D;
            color: #9A3412;
        }

        .review-panel-success {
            background: #F0FDF4;
            border: 1px solid #86EFAC;
            color: #166534;
        }

        .drivers-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.95rem;
        }

        .drivers-title {
            color: #0F172A;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .drivers-list {
            margin: 0;
            padding-left: 1.1rem;
            color: #334155;
        }

        .drivers-list li {
            margin-bottom: 0.35rem;
            line-height: 1.5;
        }

        .summary-card {
            border: 1px solid #C7D2FE;
            background: #F8FAFF;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-top: 0.5rem;
        }

        .summary-title {
            color: #1E3A8A;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }

        .summary-copy {
            color: #1E293B;
            font-size: 1rem;
            line-height: 1.7;
            margin: 0;
        }

        div[data-testid="stExpander"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stExpander"] summary {
            color: #0F172A;
            font-weight: 800;
        }

        div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stExpander"] label {
            color: #475569;
        }

        .stButton > button {
            background: #0F766E;
            color: #FFFFFF;
            border: 0;
            border-radius: 14px;
            font-weight: 800;
            padding: 0.72rem 1.2rem;
            box-shadow: 0 10px 22px rgba(15, 118, 110, 0.18);
        }

        .stButton > button:hover {
            background: #0B5F59;
            color: #FFFFFF;
        }

        div[data-testid="stSelectbox"], div[data-testid="stNumberInput"], div[data-baseweb="slider"] {
            background: transparent;
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            border-radius: 12px !important;
            border-color: #CBD5E1 !important;
        }

        label[data-testid="stWidgetLabel"] p {
            color: #0F172A !important;
            font-weight: 700;
        }

        .subtle-caption {
            color: #64748B;
            font-size: 0.86rem;
            margin-top: -0.35rem;
            margin-bottom: 0.75rem;
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Enterprise AI Decision Support Demo</div>
            <div class="hero-title">Healthcare Risk Ensemble AI</div>
            <div class="hero-copy">
                Synthetic healthcare analytics workflow demonstrating segmentation, explainability,
                and human-in-the-loop AI support.
            </div>
            <div class="hero-tags">
                <span class="hero-tag">Synthetic data only</span>
                <span class="hero-tag">Human-in-the-loop review</span>
                <span class="hero-tag">Segment-aware model routing</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Healthcare Risk Ensemble AI", layout="wide")
    render_app_chrome()
    results = load_demo_results()

    render_header()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">What This Demo Shows</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">This app demonstrates a practical enterprise AI pattern: score a synthetic member, '
        'route the case through a global or segment-specific model, flag unusual cases for review, and explain the result clearly.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    data = results["data"]
    metrics_df = results["global_training"]["metrics_df"]
    cluster_profiles = results["segmentation"]["cluster_profiles"]
    comparison_df = results["segment_training"]["comparison_df"]
    outlier_summary = results["outlier_results"]["summary"]
    routed = results["routed"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Synthetic Members", f"{len(data):,}")
    col2.metric("Best Global Model", results["global_training"]["best_model_name"])
    col3.metric("Human Review Rate", f"{routed['human_review_flag'].mean():.1%}")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Interactive Member Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Enter a synthetic member profile or use a preset scenario to show how the app scores, routes, flags, and explains the case.</div>',
        unsafe_allow_html=True,
    )

    preset_options = {
        "Balanced Example": {
            "age": 57,
            "chronic_condition_count": 2,
            "prior_claim_count": 8,
            "total_claim_cost_6m": 9500.0,
            "emergency_visit_count": 2,
            "medication_adherence_score": 0.62,
            "missed_appointment_count": 1,
            "preventive_visit_count": 1,
        },
        "High-Risk Review Case": {
            "age": 74,
            "chronic_condition_count": 5,
            "prior_claim_count": 16,
            "total_claim_cost_6m": 24500.0,
            "emergency_visit_count": 4,
            "medication_adherence_score": 0.31,
            "missed_appointment_count": 3,
            "preventive_visit_count": 0,
        },
        "Low-Risk Preventive Case": {
            "age": 36,
            "chronic_condition_count": 0,
            "prior_claim_count": 1,
            "total_claim_cost_6m": 900.0,
            "emergency_visit_count": 0,
            "medication_adherence_score": 0.91,
            "missed_appointment_count": 0,
            "preventive_visit_count": 3,
        },
    }
    selected_preset = st.selectbox("Quick Demo Scenario", list(preset_options.keys()), index=1)
    preset = preset_options[selected_preset]

    input_col1, input_col2 = st.columns(2)
    with input_col1:
        age = st.slider("Age", min_value=18, max_value=90, value=preset["age"], help="Synthetic member age used for demo scoring.")
        chronic_condition_count = st.slider(
            "Chronic Condition Count",
            min_value=0,
            max_value=8,
            value=preset["chronic_condition_count"],
            help="Number of chronic conditions in the synthetic member profile.",
        )
        prior_claim_count = st.slider(
            "Prior Claim Count",
            min_value=0,
            max_value=40,
            value=preset["prior_claim_count"],
            help="Count of prior claims in the recent historical period.",
        )
        total_claim_cost_6m = st.number_input(
            "Total Claim Cost (6m)",
            min_value=0.0,
            max_value=60000.0,
            value=float(preset["total_claim_cost_6m"]),
            step=250.0,
            help="Synthetic total claim cost over the last 6 months.",
        )
    with input_col2:
        emergency_visit_count = st.slider(
            "Emergency Visit Count",
            min_value=0,
            max_value=15,
            value=preset["emergency_visit_count"],
            help="Emergency department visit count for the synthetic member.",
        )
        medication_adherence_score = st.slider(
            "Medication Adherence Score",
            min_value=0.05,
            max_value=0.99,
            value=float(preset["medication_adherence_score"]),
            step=0.01,
            help="Synthetic adherence score from 0 to 1, where lower values may increase risk.",
        )
        missed_appointment_count = st.slider(
            "Missed Appointment Count",
            min_value=0,
            max_value=12,
            value=preset["missed_appointment_count"],
            help="Missed appointment count in the synthetic profile.",
        )
        preventive_visit_count = st.slider(
            "Preventive Visit Count",
            min_value=0,
            max_value=8,
            value=preset["preventive_visit_count"],
            help="Preventive care visit count; higher values can indicate stronger preventive engagement.",
        )

    if st.button("Run Risk Assessment", type="primary"):
        member_df = pd.DataFrame(
            [
                {
                    "member_id": 999999,
                    "age": age,
                    "chronic_condition_count": chronic_condition_count,
                    "prior_claim_count": prior_claim_count,
                    "total_claim_cost_6m": total_claim_cost_6m,
                    "emergency_visit_count": emergency_visit_count,
                    "medication_adherence_score": medication_adherence_score,
                    "missed_appointment_count": missed_appointment_count,
                    "preventive_visit_count": preventive_visit_count,
                }
            ]
        )
        prediction = predict_member(member_df, results)
        member_result = prediction["member_df"].iloc[0]
        explanation = prediction["explanation"]
        analyst_summary = prediction["assistant_summary"]

        st.markdown('<div class="result-band">', unsafe_allow_html=True)
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        result_col1.metric("Risk Score", f"{member_result['risk_score']:.1%}")
        result_col2.metric("Risk Category", risk_category(float(member_result["risk_score"])))
        result_col3.metric("Model Route", format_model_label(str(member_result["model_used"])))
        result_col4.metric("Human Review", "Yes" if bool(member_result["human_review_flag"]) else "No")
        st.markdown("</div>", unsafe_allow_html=True)

        outlier_pill = "pill-red" if bool(member_result["outlier_flag"]) else "pill-green"
        confidence_pill = "pill-blue" if member_result["confidence_level"] != "high" else "pill-green"
        st.markdown(
            f"""
            <div class="pill-row">
                <span class="pill pill-teal">Segment: Cluster {int(member_result['cluster_id'])} ({member_result['segment_label']})</span>
                <span class="pill {outlier_pill}">Outlier Flag: {'Yes' if bool(member_result['outlier_flag']) else 'No'}</span>
                <span class="pill {confidence_pill}">Confidence: {member_result['confidence_level'].title()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="route-panel">
                <strong>Route Explanation</strong><br/>
                The app first assigns the member to a behavioral segment, then uses either the global model
                or a segment-specific model depending on segment performance and review rules.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if bool(member_result["human_review_flag"]):
            st.markdown(
                """
                <div class="review-panel review-panel-warning">
                    Human review recommended because the case is unusual, high risk, or low confidence.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="review-panel review-panel-success">
                    No human review trigger was raised for this synthetic case.
                </div>
                """,
                unsafe_allow_html=True,
            )

        drivers_html = "".join(f"<li>{driver}</li>" for driver in explanation["top_risk_drivers"])
        st.markdown(
            f"""
            <div class="drivers-card">
                <div class="drivers-title">Top Risk Drivers</div>
                <ul class="drivers-list">{drivers_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-title">AI Analyst Summary</div>
                <p class="summary-copy">{analyst_summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Technical Details", expanded=False):
        st.markdown('<div class="subtle-caption">Supporting metrics for deeper discussion if requested.</div>', unsafe_allow_html=True)
        st.subheader("Global Model Metrics")
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Cluster / Segment Profiles")
        st.dataframe(cluster_profiles, use_container_width=True)
        st.image(str(results["figures_dir"] / "cluster_distribution.png"), caption="Cluster distribution")

        st.subheader("Segment Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("Outlier Summary")
        st.dataframe(outlier_summary, use_container_width=True)

        st.subheader("Monitoring Panel")
        st.dataframe(results["monitoring_summary"], use_container_width=True)
        st.image(str(results["figures_dir"] / "global_feature_importance.png"), caption="Global feature importance")


if __name__ == "__main__":
    main()
