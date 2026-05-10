# Healthcare Risk Ensemble AI

## Enterprise AI Decision Support Demo for Healthcare Analytics

## Overview
Healthcare Risk Ensemble AI is a lightweight portfolio project that demonstrates a practical enterprise AI workflow using synthetic healthcare member data. The demo compares a global risk model with segment-specific models, detects unusual cases, routes low-confidence or outlier cases to human review, and explains predictions in analyst-friendly language.

The project is designed to show how healthcare analytics systems can balance prediction, explainability, escalation logic, and monitoring rather than relying on a single model score alone.

## Key Features
- Synthetic healthcare member generation with multiple hidden utilization patterns
- Global vs segment-specific risk modeling
- Outlier detection for unusual utilization or cost behavior
- Model routing logic with human review escalation
- Simple explainability for top risk drivers
- Analyst-friendly AI summaries for review support
- Monitoring checks for missingness, outlier rates, and segment performance
- Interactive Streamlit interface for single-member assessment

## Why This Matters
In real enterprise healthcare analytics, one global model may not behave equally well across all member populations. Chronic-care members, acute high-utilization members, preventive-care members, and anomalous cases can follow different risk patterns.

This demo highlights three practical system-design ideas:
- A single model can be strong overall while still needing segment-aware evaluation
- Enterprise AI systems need escalation paths for unusual or low-confidence cases
- Explainability, monitoring, and review workflows matter alongside predictive accuracy

## Demo Workflow
1. The user enters synthetic member data in the Streamlit app.
2. The system assigns the member to a behavioral segment.
3. The router chooses the global model or a segment-specific model.
4. Outlier checks evaluate unusual utilization or cost patterns.
5. The app returns a risk score, route explanation, and analyst summary.
6. Uncertain or unusual cases are escalated for human review support.

## Tech Stack
- Python
- pandas
- numpy
- scikit-learn
- Streamlit
- matplotlib
- IsolationForest
- Agglomerative clustering

## Example Use Case
An analyst enters a synthetic member profile with high claim cost, multiple chronic conditions, low medication adherence, and elevated emergency utilization. The app assigns the member to a high-risk segment, evaluates outlier conditions, routes the case through the most appropriate model path, and recommends human review with a clear explanation of the main risk drivers.

## Screenshots
The repository is kept lightweight, so generated app screenshots are not committed by default. Launch the Streamlit app locally to view the full interactive workflow.

## How to Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If you want to avoid matplotlib cache warnings in some local environments:
```bash
MPLCONFIGDIR=/tmp/mpl streamlit run app.py --server.fileWatcherType none
```

## Project Structure
```text
healthcare-risk-ensemble-ai/
├── .streamlit/
│   └── config.toml
├── outputs/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── assistant.py
│   ├── data.py
│   ├── demo_pipeline.py
│   ├── explain.py
│   ├── monitoring.py
│   ├── outliers.py
│   ├── router.py
│   ├── segmentation.py
│   ├── train_global_model.py
│   └── train_segment_models.py
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Limitations
- Uses synthetic data only
- Not clinical guidance
- Not production-ready
- Educational and portfolio-oriented demonstration
- Clustering and routing thresholds are intentionally simple

## Future Improvements
- Drift monitoring across repeated synthetic data refreshes
- Probability calibration and threshold tuning
- Configurable routing rules from the UI
- Fairness checks across synthetic population slices
- API packaging for deployment and integration demos

## 60-Second Explanation
This project demonstrates how I think about enterprise AI in healthcare analytics. Instead of assuming one model works for every member, I built a synthetic healthcare risk workflow that compares a global model with segment-specific models, detects outliers, and routes uncertain cases to human review. The Streamlit app lets a user enter synthetic member data, see the risk score, understand which model path was used, and read an analyst-friendly explanation. The goal is to show practical system design around heterogeneity, explainability, monitoring, and decision support.
