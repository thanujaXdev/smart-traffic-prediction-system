# data_preprocessing.py
# ─────────────────────────────────────────────
# Handles all data loading and feature engineering.
# Clean separation from model logic.
# ─────────────────────────────────────────────

import pandas as pd
import os
from config import DATA_PATH, TARGET_COL, WEATHER_MAP, WEEKEND_DAYS


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load CSV with basic validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Dataset not found at '{path}'.\n"
            f"Place traffic_data.csv in the project folder."
        )
    df = pd.read_csv(path)
    required = {TARGET_COL, "Day", "Weather", "Time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing columns in CSV: {missing}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering steps:
    1. Weather_Severity  — ordinal encoding (Clear=0, Cloudy=1, Rainy=2)
    2. Weekend           — binary flag
    3. Rush_Hour         — peak traffic hours (7-9 AM, 5-7 PM)
    4. One-hot encode Day and Weather
    """
    df = df.copy()

    # Ordinal weather severity
    df["Weather_Severity"] = df["Weather"].map(WEATHER_MAP).fillna(0).astype(int)

    # Weekend flag
    df["Weekend"] = df["Day"].apply(lambda x: 1 if x in WEEKEND_DAYS else 0)

    # Rush hour flag — new feature
    df["Rush_Hour"] = df["Time"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )

    # One-hot encode Day and Weather
    df = pd.get_dummies(df, columns=["Day", "Weather"], drop_first=True)

    return df


def get_features_and_target(df: pd.DataFrame):
    """Split into X (features) and y (target)."""
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return X, y
