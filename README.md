# StudyPlan AI
An explainable ML web app that predicts student exam performance and generates personalized study improvement plans.

**Live Demo:** https://studyplan.streamlit.app

## Overview
StudyPlan AI addresses a core gap in academic planning: traditional study tools give generic advice that ignores individual behavioral patterns. This system predicts a student's expected exam score from 14+ lifestyle and academic factors, explains *why* that score was predicted, and generates actionable improvement plans tailored to that student.

## Features
- **Exam Score Prediction** — Gradient Boosting Regressor trained on 6,607 student records
- **SHAP Explainability** — Global feature importance + per-student waterfall explanations
- **DiCE Counterfactuals** — 3 personalized improvement plans (study-focused, wellness-focused, balanced)
- **Grade Band Classification** — A/B/C/D prediction with 94% accuracy on test set
- **Bias Analysis** — Subgroup MAE evaluation across vulnerable demographics

## Model Performance
| Metric | Test Set |
|--------|----------|
| MAE | 0.71 |
| RMSE | 1.98 |
| R² | 0.74 |
| Weighted F1 | 0.93 |

## Tech Stack
- **ML:** scikit-learn (Gradient Boosting Regressor), SHAP, DiCE
- **Frontend/Deployment:** Streamlit, Streamlit Cloud
- **Data:** Python, Pandas, NumPy

## How It Works
1. User inputs academic and lifestyle factors via sliders and dropdowns
2. Preprocessing pipeline applies feature engineering (6 composite features), encoding, and scaling
3. Trained GBR model predicts exam score in real time
4. SHAP layer explains which factors drove the prediction
5. DiCE generates minimum-change improvement scenarios to reach a target score