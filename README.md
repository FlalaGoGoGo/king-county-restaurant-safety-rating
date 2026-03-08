# King County Restaurant Safety Rating

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://king-county-restaurant-safety-rating.streamlit.app/)
[![Open Data](https://img.shields.io/badge/Open%20Data-King%20County-0d8a98)](https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/f29f-zza5)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-0f2d36?logo=github)](https://github.com/FlalaGoGoGo/king-county-restaurant-safety-rating)

An end-to-end data science workflow built on King County restaurant inspection data.

This project serves two purposes at the same time:

1. A public-facing restaurant safety dashboard for consumers, restaurant owners, and local stakeholders.
2. An `MSIS 522` homework deliverable that covers descriptive analytics, predictive modeling, explainability, and Streamlit deployment.

The core owner-facing prediction question is:

`Can a restaurant owner use the latest inspection profile to estimate whether the next inspection will be high risk, and which controllable signals should be fixed first to reduce that risk?`

## Quick Links

- Live Streamlit app: [king-county-restaurant-safety-rating.streamlit.app](https://king-county-restaurant-safety-rating.streamlit.app/)
- Public repository: [FlalaGoGoGo/king-county-restaurant-safety-rating](https://github.com/FlalaGoGoGo/king-county-restaurant-safety-rating)
- King County open dataset: [Food Establishment Inspection Data](https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/f29f-zza5)

## Open Data Source

- King County Food Establishment Inspection Data: [data.kingcounty.gov](https://data.kingcounty.gov/Health-Wellness/Food-Establishment-Inspection-Data/f29f-zza5)
- API endpoint: [query.json](https://data.kingcounty.gov/api/v3/views/f29f-zza5/query.json)

## App Structure

The dashboard is organized into two tracks:

- Public release tabs:
  - `Project Overview`
  - `Restaurant Search`
  - `Historical Insights`
- Assignment tabs:
  - `Executive Summary`
  - `Descriptive Analytics`
  - `Model Performance`
  - `Explainability & Interactive Prediction`

The Streamlit app is the primary interactive deliverable. The HTML export mirrors the public and homework content in a static format.

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Build the static HTML dashboard:

```bash
bash scripts/build_html_dashboard.sh
```

## Model Training

Train and save the predictive models:

```bash
python3 scripts/train_predict_models.py \
  --root . \
  --output-dir models/hw1_predict \
  --max-rows 12000 \
  --cv-folds 5 \
  --test-size 0.3 \
  --shap-sample-size 300
```

Saved artifacts include:

- Logistic Regression baseline
- Decision Tree with cross-validation
- Random Forest with cross-validation
- XGBoost
- MLP
- ROC curves
- SHAP summary / bar plots
- MLP tuning bonus outputs

The app loads pre-trained models from `models/hw1_predict/` and does not retrain on page load.

## Repository Layout

```text
app/
  __init__.py
  dashboard_app.py
streamlit_app.py
scripts/
  train_predict_models.py
  export_html_dashboard.py
  build_html_dashboard.sh
models/
  hw1_predict/
deploy_bundle/
images/
.streamlit/
docs/
outputs/
  analysis/
  dashboard/proposals/
```

## Why Raw Data Is Excluded

The public GitHub package excludes raw and intermediate data under `Data/` and excludes the generated `outputs/dashboard/index.html` artifact because it is too large for a lightweight public repository. For deployment, the repo includes a compressed `deploy_bundle/` with only the files required to run the Streamlit app on Streamlit Community Cloud. The project keeps:

- code
- saved models
- selected documentation
- compact analysis outputs
- UI proposal files

This keeps the repo reproducible without pushing bulky local artifacts.

## Deployment

- Streamlit Community Cloud entry file: `streamlit_app.py`
- Streamlit app link: [king-county-restaurant-safety-rating.streamlit.app](https://king-county-restaurant-safety-rating.streamlit.app/)
- HTML export path: `outputs/dashboard/index.html`

## Reproducibility Notes

- `requirements.txt` is included for environment setup.
- `.streamlit/config.toml` disables usage telemetry.
- Models are small enough for standard Git; Git LFS is not required for this repo.
