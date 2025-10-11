# src/model_training.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
from src.utils import risk_percentile_factory, save

PROCESSED = Path("data/processed")
MODELS = Path("models")
OUT = Path("outputs")
MODELS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

def run():
    df = pd.read_csv(PROCESSED / "feature_dataset.csv")

    # compute percentile bounds
    steps_low, steps_high = df["steps"].quantile([0.05, 0.95]) if "steps" in df else (0,1)
    sleep_low, sleep_high = df["sleep_min"].quantile([0.05, 0.95]) if "sleep_min" in df else (0,1)
    hr_low, hr_high = df["heart_rate"].quantile([0.05, 0.95]) if "heart_rate" in df else (0,1)
    spo2_low = df["spO2"].quantile(0.05) if "spO2" in df else 0

    bounds = {
        'steps_low': steps_low, 'steps_high': steps_high,
        'sleep_low': sleep_low, 'sleep_high': sleep_high,
        'hr_low': hr_low, 'hr_high': hr_high,
        'spo2_low': spo2_low
    }
    df['Risk_Level'] = df.apply(risk_percentile_factory(bounds), axis=1)

    # KMeans clustering-based risk
    cluster_features = [c for c in ['steps','sleep_min','heart_rate','spO2'] if c in df.columns]
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_cluster)
    df['Risk_Level_Cluster'] = kmeans.predict(X_cluster)

    # map clusters by average health score to ordered risk levels
    cluster_order = df.groupby('Risk_Level_Cluster')['Health_Score'].mean().sort_values().index
    cluster_map = {cluster_order[0]: 2, cluster_order[1]: 1, cluster_order[2]: 0}
    df['Risk_Level_Cluster'] = df['Risk_Level_Cluster'].map(cluster_map)

    # encode gender and prepare features
    df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)
    exclude_features = ['user_id','day','Anomaly','Risk_Level','Risk_Level_Cluster',
                        'Age (years)','Height (meter)','Weight (kg)','BMI']
    all_features = [c for c in df_encoded.columns if c not in exclude_features]
    X = df_encoded[all_features]
    y = df_encoded['Risk_Level_Cluster']

    # train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Save test split for SHAP step later
    save({
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test
    }, 'data/processed/test_split.pkl')

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    print("RandomForest acc:", accuracy_score(y_test,y_rf))
    joblib.dump(rf, MODELS / "random_forest_model.pkl")

    # 2. XGBoost
    xgbm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             objective='multi:softmax', num_class=3, random_state=42,
                             eval_metric='mlogloss', use_label_encoder=False)
    xgbm.fit(X_train, y_train)
    y_xgb = xgbm.predict(X_test)
    print("XGBoost acc:", accuracy_score(y_test,y_xgb))
    joblib.dump(xgbm, MODELS / "xgboost_model.pkl")

    # 3. LightGBM
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    lgbm.fit(X_train, y_train)
    y_lgbm = lgbm.predict(X_test)
    print("LightGBM acc:", accuracy_score(y_test,y_lgbm))
    joblib.dump(lgbm, MODELS / "lightgbm_model.pkl")

    # 4. Ensemble
    ensemble = VotingClassifier([
        ('rf', rf), ('xgb', xgbm), ('lgbm', lgbm)
    ], voting='hard')
    ensemble.fit(X_train, y_train)
    y_e = ensemble.predict(X_test)
    print("Ensemble acc:", accuracy_score(y_test,y_e))
    joblib.dump(ensemble, MODELS / "ensemble_model.pkl")

    # Save a basic performance CSV
    perf = {
        'Model': ['RandomForest','XGBoost','LightGBM','Ensemble'],
        'Accuracy': [accuracy_score(y_test,y_rf), accuracy_score(y_test,y_xgb),
                     accuracy_score(y_test,y_lgbm), accuracy_score(y_test,y_e)],
        'F1': [f1_score(y_test,y_rf, average='weighted'),
               f1_score(y_test,y_xgb, average='weighted'),
               f1_score(y_test,y_lgbm, average='weighted'),
               f1_score(y_test,y_e, average='weighted')]
    }
    pd.DataFrame(perf).to_csv(OUT / "models_performance.csv", index=False)
    print("Saved models to", MODELS)
    print("Saved model metrics to", OUT / "models_performance.csv")

if __name__ == "__main__":
    run()
