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


