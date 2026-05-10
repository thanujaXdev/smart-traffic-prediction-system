# predict.py
# ─────────────────────────────────────────────
# Loads saved model and makes predictions.
# Used by app.py — no training happens here.
# ─────────────────────────────────────────────

import joblib
import pandas as pd
from config import MODEL_PATH, WEATHER_MAP, WEEKEND_DAYS, TRAFFIC_LOW, TRAFFIC_MEDIUM


def load_model(path: str = MODEL_PATH) -> tuple:
    """Load saved pipeline + metadata."""
    payload = joblib.load(path)
    return payload["pipeline"], payload["feature_names"], payload["metrics"], payload["best_model"]


def predict_traffic(
    pipeline,
    feature_names: list,
    hour: int,
    day: str,
    weather: str,
    vehicle_count: int = 200,
    holiday: int = 0
) -> float:
    """
    Predict traffic volume from user inputs.

    Parameters
    ----------
    pipeline       : trained sklearn Pipeline
    feature_names  : column names used during training
    hour           : 0–23
    day            : e.g. 'Monday'
    weather        : 'Clear', 'Cloudy', or 'Rainy'
    vehicle_count  : approximate vehicles on road
    holiday        : 1 if public holiday, else 0

    Returns
    -------
    float: predicted traffic volume
    """
    # Validate
    if not (0 <= hour <= 23):
        raise ValueError("Hour must be 0–23")
    if weather not in WEATHER_MAP:
        raise ValueError(f"Weather must be one of {list(WEATHER_MAP.keys())}")

    # Build zero row
    row = {col: 0 for col in feature_names}

    # Fill known numeric features
    row["Time"]             = hour
    row["Vehicle_Count"]    = vehicle_count
    row["Holiday"]          = holiday
    row["Weather_Severity"] = WEATHER_MAP[weather]
    row["Weekend"]          = 1 if day in WEEKEND_DAYS else 0
    row["Rush_Hour"]        = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0

    # One-hot Day
    day_col = f"Day_{day}"
    if day_col in feature_names:
        row[day_col] = 1

    # One-hot Weather
    weather_col = f"Weather_{weather}"
    if weather_col in feature_names:
        row[weather_col] = 1

    input_df = pd.DataFrame([row])
    prediction = pipeline.predict(input_df)[0]
    return round(float(prediction), 1)


def get_traffic_level(volume: float) -> tuple:
    """
    Returns (level, color, emoji) based on predicted volume.
    """
    if volume < TRAFFIC_LOW:
        return "Low Traffic", "green", "✅"
    elif volume < TRAFFIC_MEDIUM:
        return "Medium Traffic", "orange", "⚠️"
    else:
        return "High Traffic", "red", "🚨"
