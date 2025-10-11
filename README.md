# Wearable Health Fusion

## Project Overview

This project aims to develop a **personalized health monitoring system** using wearable device data.  
We integrate multimodal data (physiological, activity, lifestyle) to create **feature-rich datasets** for predictive modeling and anomaly detection.

The project is structured in multiple phases:

1. **Phase 0 — Setup & Exploration** (Completed)  
2. **Phase 1 — Feature Engineering & Labeling** (Completed)  
3. **Phase 2 — Modeling & Prediction** (Upcoming)  
4. **Phase 3 — Privacy-preserving Federated Learning & Edge Deployment** (Upcoming)

---

## Dataset

We use the **Smart Healthcare – DailyLife Dataset (Wearable Device)** from Kaggle:  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/mdimammahdi/smart-healthcare-dailylife-dataset-wearable-device?resource=download)

### Columns in the cleaned dataset:

| Column                 | Description                                     |
|------------------------|-------------------------------------------------|
| user_id                | Unique identifier for each individual          |
| day                    | Day of measurement (1–7)                        |
| Gender                 | Male / Female                                   |
| Age (years)            | Age of the individual                           |
| Height (meter)         | Height in meters                                |
| Weight (kg)            | Weight in kilograms                             |
| BMI                    | Body Mass Index                                 |
| steps                  | Total steps taken per day                        |
| distance_km            | Distance covered per day (km)                  |
| heart_rate             | Average heart rate (BPM)                        |
| spO2                   | Blood oxygen level (%)                           |
| sleep_min              | Total sleep duration (minutes)                  |
| screen_min             | Screen time (minutes)                           |
| earphone_min           | Earphone usage per day (minutes)                |
| systolic_bp            | Systolic blood pressure                          |
| diastolic_bp           | Diastolic blood pressure                         |
| gender                 | Categorical gender column                        |
| activity_ratio         | Steps / distance ratio                           |
| sleep_efficiency       | Sleep duration / recommended sleep (480 min)   |

---

## Folder Structure
'''
Wearable-Data-Fusion/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📁 data/
│   ├── raw/
│   │   └── Smart Healthcare - Daily Lifestyle Dataset (Wearable device) (1).csv
│   ├── cleaned/
│   │   └── canonical_dataset.csv
│   ├── processed/
│   │   ├── feature_dataset.csv
│   │   └── feature_dataset_with_cluster.csv
│   
├── 📁 src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── shap_explainability.py
│   └── utils.py
│
├── 📁 outputs/
│   ├── correlation_heatmap.png
│   ├── shap_rf_summary.png
│   ├── shap_rf_detailed.png
│   ├── shap_xgb_summary.png
│   ├── shap_xgb_detailed.png
│   ├── shap_lgbm_summary.png
│   ├── shap_lgbm_detailed.png
│   ├── shap_ensemble_summary.png
│   ├── shap_ensemble_detailed.png
│   └── model_comparison.png
│
└── 📁 models/
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    ├── lightgbm_model.pkl
    └── ensemble_model.pkl

'''
---

## Phase 0 — Setup & Exploration

- Inspected dataset for **columns, units, missing values**  
- Converted categorical fields (`Gender`) to `category`  
- Created **canonical dataset** (`canonical_dataset.csv`)  
- Split blood pressure into systolic/diastolic columns  
- Documented **summary statistics per user**  
- Plotted **distributions, correlations, and outliers**  

---

## Phase 1 — Feature Engineering & Labeling

- Time-domain features per user/day:  
  - Δsteps, Δsleep, Δheart_rate (day-to-day differences)  
  - Rolling averages (3–5 day trends)  
- Activity ratio and sleep efficiency (already calculated)  
- Cross-feature analysis: correlations between activity & heart rate, sleep & screen time  
- Labeling for anomaly detection based on **heuristic thresholds**  
  - Steps < 2000 or > 20,000 → abnormal  
  - Sleep < 300 min or > 600 min → abnormal  
  - Heart rate / SpO₂ outside normal range → abnormal
  -  Applied **K-Means clustering** and rule-based labeling for health risk levels 

### **Phase 2 – Predictive Modeling & Explainability**
- Trained multiple ML models:
  - Random Forest
  - XGBoost
  - LightGBM  
  - Ensemble (weighted average)
- Compared model performance using accuracy, precision, and recall metrics  
- Used **SHAP explainability** to interpret feature importance and risk contributions  
- Visualized feature influence for each model (summary and detailed plots)

---
