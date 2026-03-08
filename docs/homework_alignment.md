# MSIS 522 Homework Alignment

This document maps the current project deliverable to the `MSIS 522 Homework 1: The Complete Data Science Workflow` rubric.

## Part 1: Descriptive Analytics

### 1.1 Dataset Introduction

- The app explains that the project uses King County restaurant inspection data.
- The Executive Summary states the prediction target: `target_next_high_risk`.
- The dataset purpose, source tables, and feature types are described in the Executive Summary.
- The business lens is fixed to the restaurant owner perspective.

### 1.2 Target Distribution

- The `Descriptive Analytics` tab shows the class distribution of `target_next_high_risk`.
- The commentary explicitly calls out class imbalance and explains why F1 / recall / ROC-AUC are emphasized.

### 1.3 Feature Distributions and Relationships

The `Descriptive Analytics` tab includes commentary for each plot:

- inspection score vs. future high-risk target
- red points vs. future high-risk target
- violation count vs. future high-risk target
- inspection result vs. future high-risk rate
- city-level future high-risk pattern

Each plot includes interpretation text and a separate takeaway statement.

### 1.4 Correlation Heatmap

- The `Descriptive Analytics` tab includes a correlation heatmap over the core numeric modeling features and the target.
- The commentary explains the strongest relationship pattern and why no single variable is sufficient.

## Part 2: Predictive Analytics

### 2.2 Logistic Regression Baseline

- Logistic Regression is trained and saved.
- Test metrics are surfaced in the `Model Performance` tab.

### 2.3 Decision Tree with CV

- Decision Tree artifacts, metrics, and best hyperparameters are surfaced in the app.
- The saved tree visualization is displayed in `Model Performance`.

### 2.4 Random Forest with CV

- Random Forest metrics, best hyperparameters, ROC curve, and feature importance are surfaced in the app.

### 2.5 Boosted Trees

- XGBoost metrics, best hyperparameters, ROC curve, and feature importance are surfaced in the app.

### 2.6 Neural Network (MLP)

- MLP metrics are surfaced in the app.
- The training history plot is surfaced in `Model Performance`.

### 2.7 Model Comparison Summary

- The app includes a model comparison table.
- The app includes an F1 comparison bar chart.
- The app includes a comparison narrative discussing best model and trade-offs.

## Part 3: Explainability

- The app includes SHAP summary and bar plots.
- The app includes a custom-input SHAP waterfall in Streamlit.
- The explainability commentary connects the model output back to owner-facing action prioritization.

## Part 4: Streamlit Deployment

### Tab 1: Executive Summary

- dataset description
- target definition
- why the problem matters
- approach summary
- key findings
- grouped field dictionary

### Tab 2: Descriptive Analytics

- target distribution
- multiple feature relationship plots
- correlation heatmap
- commentary and takeaways for each plot

### Tab 3: Model Performance

- comparison table
- bar chart
- ROC curves
- best hyperparameters
- saved model-specific artifacts

### Tab 4: Explainability & Interactive Prediction

- SHAP summary
- SHAP bar plot
- interactive model selection
- editable feature inputs in Streamlit
- predicted probability and class
- custom-input SHAP waterfall

## Bonus

- MLP hyperparameter tuning results are included.
- The tuning results table and tuning plot are surfaced in the `Model Performance` tab.

