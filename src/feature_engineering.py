# src/feature_engineering.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from src.utils import anomaly_label

CLEANED = Path("data/cleaned")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def run():
    df = pd.read_csv(CLEANED / "canonical_dataset.csv")
    df = df.sort_values(["user_id", "day"]).reset_index(drop=True)

    delta_cols = ["steps", "distance_km", "heart_rate", "spO2",
                  "sleep_min", "screen_min", "earphone_min"]

    for col in delta_cols:
        if col in df.columns:
            df[f"{col}_delta"] = df.groupby("user_id")[col].diff().fillna(0)

    window = 3
    for col in delta_cols:
        if col in df.columns:
            df[f"{col}_roll_mean"] = (
                df.groupby("user_id")[col].rolling(window).mean().reset_index(0,drop=True)
            )
            df[f"{col}_roll_std"] = (
                df.groupby("user_id")[col].rolling(window).std().reset_index(0,drop=True)
            )

    df.fillna(0, inplace=True)

    df["Anomaly"] = df.apply(anomaly_label, axis=1)

    norm_cols = [c for c in ["steps", "distance_km", "sleep_min", "heart_rate", "spO2"] if c in df.columns]
    scaler = MinMaxScaler()
    if norm_cols:
        df[norm_cols] = scaler.fit_transform(df[norm_cols])
        df["Health_Score"] = df[norm_cols].mean(axis=1)
    else:
        df["Health_Score"] = 0.0

    out_path = PROCESSED / "feature_dataset.csv"
    df.to_csv(out_path, index=False)
    print("Saved feature-engineered dataset to", out_path)
    return out_path

if __name__ == "__main__":
    run()
