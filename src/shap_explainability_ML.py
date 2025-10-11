# src/shap_explainability.py
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

MODELS = Path("models")
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

def run():
    data = joblib.load('data/processed/test_split.pkl')
    X_test = data['X_test']

    # Load models
    rf = joblib.load(MODELS / "random_forest_model.pkl")
    xgbm = joblib.load(MODELS / "xgboost_model.pkl")
    lgbm = joblib.load(MODELS / "lightgbm_model.pkl")

    sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

    # Random Forest SHAP
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(sample)
    plt.figure(figsize=(10,6))
    if isinstance(shap_values_rf, list):
        shap.summary_plot(shap_values_rf[1], sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values_rf, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_rf_summary.png", dpi=200)
    plt.close()

    # XGBoost SHAP
    explainer_xgb = shap.TreeExplainer(xgbm)
    shap_values_xgb = explainer_xgb.shap_values(sample)
    plt.figure(figsize=(10,6))
    if isinstance(shap_values_xgb, list):
        shap.summary_plot(shap_values_xgb[1], sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values_xgb, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_xgb_summary.png", dpi=200)
    plt.close()

    # LightGBM SHAP
    explainer_lgbm = shap.TreeExplainer(lgbm)
    shap_values_lgbm = explainer_lgbm.shap_values(sample)
    plt.figure(figsize=(10,6))
    if isinstance(shap_values_lgbm, list):
        shap.summary_plot(shap_values_lgbm[1], sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values_lgbm, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_lgbm_summary.png", dpi=200)
    plt.close()

    print("Saved SHAP summaries to outputs/")

if __name__ == "__main__":
    run()
