# src/data_preprocessing.py
import pandas as pd
import numpy as np
import os
from pathlib import Path

RAW = Path("data/raw")
CLEANED = Path("data/cleaned")
CLEANED.mkdir(parents=True, exist_ok=True)

def run(input_csv=None):
    if input_csv is None:
        input_csv = RAW / "Smart Healthcare - Daily Lifestyle Dataset (Wearable device) (1).csv"
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Put your raw CSV at {input_csv}")

    df = pd.read_csv(input_csv)
    df = df.rename(columns={
        "ID": "user_id",
        "Day": "day",
        "Step Count": "steps",
        "Distance Travel (Km)": "distance_km",
        "Heart Rate (BPM)": "heart_rate",
        "Blood Oxygen Level": "spO2",
        "Sleep Duration (minutes)": "sleep_min",
        "Screen Time (minute)": "screen_min",
        "Earphone Time (minute)": "earphone_min",
        "Gender": "gender"
    })

    # Blood pressure -> systolic/diastolic
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].astype(str).str.split("/", expand=True)
        df["systolic_bp"] = pd.to_numeric(bp_split[0], errors='coerce')
        df["diastolic_bp"] = pd.to_numeric(bp_split[1], errors='coerce')
        df.drop(columns=["Blood Pressure"], inplace=True)

    df["gender"] = df.get("gender", pd.Series()).astype("category")
    df["day"] = pd.to_numeric(df.get("day", pd.Series()), errors="coerce")

    numeric_cols = ["Age (years)", "Height (meter)", "Weight (kg)", "BMI",
                    "steps", "distance_km", "heart_rate", "spO2",
                    "sleep_min", "screen_min", "earphone_min",
                    "systolic_bp", "diastolic_bp"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    if "gender" in df.columns:
        df["gender"] = df["gender"].cat.add_categories("Unknown").fillna("Unknown")

    # engineered features
    if "distance_km" in df.columns and "steps" in df.columns:
        df["activity_ratio"] = df["steps"] / df["distance_km"].replace(0, 1)
    if "sleep_min" in df.columns:
        df["sleep_efficiency"] = df["sleep_min"] / (24*60)

    out_path = CLEANED / "canonical_dataset.csv"
    df.to_csv(out_path, index=False)
    print("Saved canonical dataset to", out_path)
    return out_path

if __name__ == "__main__":
    run()
