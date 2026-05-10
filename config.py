# config.py
# ─────────────────────────────────────────────
# Central configuration for the entire project.
# Change values here — no need to touch other files.
# ─────────────────────────────────────────────

# Data
DATA_PATH = "traffic_data.csv"
TARGET_COL = "Traffic_Volume"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Weather mapping (ordinal severity)
WEATHER_MAP = {"Clear": 0, "Cloudy": 1, "Rainy": 2}

# Days considered weekend
WEEKEND_DAYS = ["Saturday", "Sunday"]

# Traffic thresholds for alert levels
TRAFFIC_LOW    = 300
TRAFFIC_MEDIUM = 600

# Model save path
MODEL_PATH = "saved_model.pkl"

# Cross-validation folds
CV_FOLDS = 5
