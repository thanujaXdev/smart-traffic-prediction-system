# app.py
# ─────────────────────────────────────────────
# Professional Streamlit UI for Traffic Prediction.
# Run: streamlit run app.py
# ─────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from predict import load_model, predict_traffic, get_traffic_level
from data_preprocessing import load_data, engineer_features, get_features_and_target
from config import MODEL_PATH

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Traffic Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Data ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading model...")
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Please run `python model_training.py` first.")
        st.stop()
    return load_model()


@st.cache_data(show_spinner="📂 Loading data...")
def get_data():
    df = load_data()
    df_eng = engineer_features(df)
    X, y = get_features_and_target(df_eng)
    return df, X, y


pipeline, feature_names, metrics, best_model_name = get_model()
raw_df, X, y = get_data()


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🚦 Smart Traffic Prediction System</div>', unsafe_allow_html=True)
st.markdown("---")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=80)
    st.title("⚙️ Prediction Controls")

    day = st.selectbox(
        "📅 Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    hour = st.slider("⏰ Hour of Day", 0, 23, 8,
                     help="Rush hours: 7–9 AM and 5–7 PM")
    weather = st.selectbox("🌤️ Weather", ["Clear", "Cloudy", "Rainy"])
    vehicle_count = st.number_input("🚗 Estimated Vehicles on Road", 50, 1000, 200, step=10)
    holiday = st.checkbox("🎉 Public Holiday")

    st.markdown("---")
    st.caption(f"🤖 Active Model: **{best_model_name}**")
    st.caption(f"📊 Test MAE: **{metrics['MAE']:.2f}** | R²: **{metrics['R2']:.3f}**")


# ── Main Tabs ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "📂 Dataset"])


# ════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════

with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.subheader("🔮 Traffic Prediction")

        # Run prediction
        volume = predict_traffic(
            pipeline, feature_names,
            hour=hour, day=day, weather=weather,
            vehicle_count=vehicle_count, holiday=int(holiday)
        )
        level, color, emoji = get_traffic_level(volume)

        # Result box
        color_map = {"green": "#d4edda", "orange": "#fff3cd", "red": "#f8d7da"}
        border_map = {"green": "#28a745", "orange": "#ffc107", "red": "#dc3545"}

        st.markdown(f"""
        <div class="prediction-box" style="
            background:{color_map[color]};
            border: 3px solid {border_map[color]};">
            {emoji} {level}<br>
            <span style="font-size:2.5rem; color:{border_map[color]}">
                {int(volume)} vehicles/hr
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Input summary
        st.markdown("#### 📋 Your Input Summary")
        rush = "Yes ⚡" if (7 <= hour <= 9) or (17 <= hour <= 19) else "No"
        weekend = "Yes 🏖️" if day in ["Saturday", "Sunday"] else "No"

        summary_df = pd.DataFrame({
            "Parameter": ["Day", "Hour", "Weather", "Vehicles", "Holiday", "Rush Hour", "Weekend"],
            "Value":     [day, f"{hour}:00", weather, vehicle_count, "Yes" if holiday else "No", rush, weekend]
        })
        st.table(summary_df)

    # Hourly trend chart
    st.markdown("---")
    st.subheader("📈 Predicted Traffic Throughout the Day")

    hours = list(range(24))
    hourly_preds = [
        predict_traffic(pipeline, feature_names, h, day, weather, vehicle_count, int(holiday))
        for h in hours
    ]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(hours, hourly_preds, alpha=0.3, color="#667eea")
    ax.plot(hours, hourly_preds, color="#764ba2", linewidth=2.5, marker="o", markersize=4)
    ax.axvline(hour, color="red", linestyle="--", linewidth=1.5, label=f"Selected: {hour}:00")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h}:00" for h in hours], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Traffic Volume")
    ax.set_title(f"Traffic Forecast — {day}, {weather} weather")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)


# ════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════

with tab2:
    st.subheader("📊 Model Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", best_model_name)
    col2.metric("Test MAE", f"{metrics['MAE']:.2f}")
    col3.metric("R² Score", f"{metrics['R2']:.3f}")

    st.markdown("---")

    # Feature importance
    st.subheader("🔍 Feature Importance (Random Forest)")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from config import TEST_SIZE, RANDOM_STATE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_temp.fit(X_train, y_train)

    importances = pd.Series(
        rf_temp.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax2)
    ax2.set_title("Feature Importance — Random Forest")
    ax2.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig2)

    # Actual vs Predicted
    st.subheader("📉 Actual vs Predicted (Test Set)")
    y_pred = pipeline.predict(X_test)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(y_test.values, label="Actual", color="black", marker="o", linewidth=2)
    ax3.plot(y_pred, label=f"{best_model_name} (Predicted)",
             color="#764ba2", linestyle="--", marker="x", linewidth=2)
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Traffic Volume")
    ax3.set_title("Actual vs Predicted Traffic Volume")
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)


# ════════════════════════════════════════════════
# TAB 3 — DATASET
# ════════════════════════════════════════════════

with tab3:
    st.subheader("📂 Raw Dataset")
    st.dataframe(raw_df, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(raw_df))
    col2.metric("Features", raw_df.shape[1] - 1)
    col3.metric("Avg Traffic", f"{raw_df['Traffic_Volume'].mean():.0f}")
    col4.metric("Max Traffic", raw_df['Traffic_Volume'].max())

    st.markdown("---")
    st.subheader("📊 Traffic Distribution by Day")

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_avg = raw_df.groupby("Day")["Traffic_Volume"].mean().reindex(day_order)
    sns.barplot(x=day_avg.index, y=day_avg.values, palette="coolwarm", ax=ax4)
    ax4.set_title("Average Traffic Volume by Day")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Average Traffic Volume")
    plt.tight_layout()
    st.pyplot(fig4)
