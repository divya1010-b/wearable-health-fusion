# src/utils.py
import numpy as np
import joblib
from pathlib import Path

def anomaly_label(row):
    # same rules you used earlier
    if (row.get("steps", 0) < 2000 or row.get("steps", 0) > 20000):
        return 1
    if (row.get("sleep_min", 0) < 300 or row.get("sleep_min", 0) > 600):
        return 1
    if (row.get("heart_rate", 0) < 50 or row.get("heart_rate", 0) > 120):
        return 1
    if (row.get("spO2", 100) < 92):
        return 1
    return 0

def risk_percentile_factory(bounds):
    # bounds = dict with keys: steps_low, steps_high, sleep_low, sleep_high, hr_low, hr_high, spo2_low
    def risk_percentile(row):
        abnormal_count = 0
        if not (bounds['steps_low'] <= row["steps"] <= bounds['steps_high']):
            abnormal_count += 1
        if not (bounds['sleep_low'] <= row["sleep_min"] <= bounds['sleep_high']):
            abnormal_count += 1
        if not (bounds['hr_low'] <= row["heart_rate"] <= bounds['hr_high']):
            abnormal_count += 1
        if row["spO2"] < bounds['spo2_low']:
            abnormal_count += 1
        if abnormal_count == 0:
            return 0
        elif abnormal_count == 1:
            return 1
        else:
            return 2
    return risk_percentile

def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load(path):
    return joblib.load(path)
