from __future__ import annotations


def generate_analyst_summary(explanation: dict) -> str:
    review_text = (
        "Because the case is unusual or uncertain, analyst review is recommended before any operational action."
        if explanation["human_review_flag"]
        else "The case follows a standard scoring path and can be handled through the normal analytics workflow."
    )
    outlier_text = "An outlier pattern was detected." if explanation["outlier_flag"] else "No material outlier pattern was detected."
    driver_text = ", ".join(explanation["top_risk_drivers"])
    return (
        f"Member is routed to the {explanation['model_used']} path. "
        f"Risk score is {explanation['risk_score']:.3f} with {explanation['confidence_level']} confidence. "
        f"Key drivers are {driver_text}. {outlier_text} {review_text} "
        "This summary is for analyst decision support only and is not clinical guidance."
    )
