# ======================================
# 9. REGRESSION ANALYSIS (PHASE 2)
# ======================================

print("\n" + "="*60)
print("REGRESSION ANALYSIS - HEALTH SCORE PREDICTION")
print("="*60)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# --- Load feature dataset ---
df_reg = pd.read_csv('/content/feature_dataset_with_cluster.csv')

# One-hot encode categorical columns
categorical_features = ['gender']
df_reg = pd.get_dummies(df_reg, columns=categorical_features, drop_first=True)

# --- Define target and features ---
target = 'Health_Score'
exclude_cols = ['user_id', 'day', 'Anomaly', 'Risk_Level', 'Risk_Level_Cluster',
                'Age (years)', 'Height (meter)', 'Weight (kg)', 'BMI', target]
features = [col for col in df_reg.columns if col not in exclude_cols]

X_reg = df_reg[features]
y_reg = df_reg[target]

# --- Train-test split ---
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# --- Helper function for metrics ---
def regression_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {model_name} Regression Results ===")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

# ======================================
# Train 4 Regression Models
# ======================================

results = []

# 1️⃣ Random Forest Regressor
print("\n[1/4] Training Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train_r, y_train_r)
y_pred_rf_r = rf_reg.predict(X_test_r)
results.append(regression_metrics("Random Forest", y_test_r, y_pred_rf_r))

# 2️⃣ XGBoost Regressor
print("\n[2/4] Training XGBoost Regressor...")
xgb_reg = XGBRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=6,
    random_state=42, objective='reg:squarederror'
)
xgb_reg.fit(X_train_r, y_train_r)
y_pred_xgb_r = xgb_reg.predict(X_test_r)
results.append(regression_metrics("XGBoost", y_test_r, y_pred_xgb_r))

# 3️⃣ LightGBM Regressor
print("\n[3/4] Training LightGBM Regressor...")
lgbm_reg = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
lgbm_reg.fit(X_train_r, y_train_r)
y_pred_lgbm_r = lgbm_reg.predict(X_test_r)
results.append(regression_metrics("LightGBM", y_test_r, y_pred_lgbm_r))

# 4️⃣ Ensemble (Averaging)
print("\n[4/4] Building Ensemble Regressor (Average of 3 Models)...")
y_pred_ensemble_r = (y_pred_rf_r + y_pred_xgb_r + y_pred_lgbm_r) / 3
results.append(regression_metrics("Ensemble (Average)", y_test_r, y_pred_ensemble_r))

# ======================================
# PERFORMANCE SUMMARY
# ======================================

print("\n" + "="*60)
print("REGRESSION MODEL PERFORMANCE SUMMARY")
print("="*60)

reg_results_df = pd.DataFrame(results)
print(reg_results_df.to_string(index=False))

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot MAE, RMSE, R2 side-by-side
sns.barplot(x="Model", y="MAE", data=reg_results_df, ax=axes[0], palette="coolwarm")
axes[0].set_title("Mean Absolute Error (MAE)", fontweight='bold')
axes[0].grid(axis="y", alpha=0.3)

sns.barplot(x="Model", y="RMSE", data=reg_results_df, ax=axes[1], palette="coolwarm")
axes[1].set_title("Root Mean Squared Error (RMSE)", fontweight='bold')
axes[1].grid(axis="y", alpha=0.3)

sns.barplot(x="Model", y="R2", data=reg_results_df, ax=axes[2], palette="coolwarm")
axes[2].set_title("R² Score", fontweight='bold')
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("regression_performance.png", dpi=300, bbox_inches='tight')
plt.show()

# ======================================
#  Predicted vs Actual Plot (Best Model)
# ======================================

best_model_name = reg_results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
best_pred = {
    "Random Forest": y_pred_rf_r,
    "XGBoost": y_pred_xgb_r,
    "LightGBM": y_pred_lgbm_r,
    "Ensemble (Average)": y_pred_ensemble_r
}[best_model_name]

plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, best_pred, alpha=0.6, color='teal')
plt.xlabel("Actual Health Score")
plt.ylabel("Predicted Health Score")
plt.title(f"{best_model_name} - Predicted vs Actual", fontweight='bold')
plt.plot([0, 1], [0, 1], 'r--', lw=2)
plt.tight_layout()
plt.savefig("regression_pred_vs_actual.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Regression phase completed successfully!")
print("✅ Regression plots saved: regression_performance.png, regression_pred_vs_actual.png")
