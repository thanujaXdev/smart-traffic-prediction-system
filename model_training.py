# model_training.py
# ─────────────────────────────────────────────
# Trains models, evaluates them, saves the best one.
# Run this once: python model_training.py
# ─────────────────────────────────────────────

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_data, engineer_features, get_features_and_target
from config import TEST_SIZE, RANDOM_STATE, CV_FOLDS, MODEL_PATH


# ── Model definitions ──────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Decision Tree": Pipeline([
            ("model", DecisionTreeRegressor(
                random_state=RANDOM_STATE,
                max_depth=8,
                min_samples_leaf=3
            ))
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=12,
                min_samples_leaf=2
            ))
        ]),
        "Gradient Boosting": Pipeline([          # NEW: stronger model
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=RANDOM_STATE
            ))
        ]),
    }


# ── Training & Evaluation ──────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    models = get_models()
    results = {}

    print("\n📊 Model Evaluation")
    print("=" * 60)

    for name, pipeline in models.items():
        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            scoring="neg_mean_absolute_error", cv=CV_FOLDS
        )
        cv_mae = -cv_scores.mean()

        # Fit & predict
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        results[name] = {
            "pipeline":    pipeline,
            "predictions": preds,
            "MAE":         mae,
            "RMSE":        rmse,
            "R2":          r2,
            "CV_MAE":      cv_mae,
        }

        print(f"\n🔹 {name}")
        print(f"   CV MAE  : {cv_mae:.2f}")
        print(f"   MAE     : {mae:.2f}")
        print(f"   RMSE    : {rmse:.2f}")
        print(f"   R²      : {r2:.4f}")

    return results


# ── Save Best Model ────────────────────────────────────────────────────────────

def save_best_model(results: dict, feature_names: list) -> str:
    best_name = min(results, key=lambda k: results[k]["MAE"])
    payload = {
        "pipeline":      results[best_name]["pipeline"],
        "feature_names": feature_names,
        "best_model":    best_name,
        "metrics": {
            "MAE":  results[best_name]["MAE"],
            "RMSE": results[best_name]["RMSE"],
            "R2":   results[best_name]["R2"],
        }
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n💾 Saved best model: {best_name} → {MODEL_PATH}")
    return best_name


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_all(results: dict, y_test: pd.Series, feature_names: list):
    # 1. Predictions vs Actual
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values, label="Actual", linewidth=2, color="black", marker="o")
    for name, res in results.items():
        ax.plot(res["predictions"], linestyle="--", marker="x", label=name)
    ax.set_title("Traffic Volume: Actual vs Predicted")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Traffic Volume")
    ax.legend()
    plt.tight_layout()
    plt.savefig("predictions_plot.png", dpi=150)
    plt.show()

    # 2. Model Comparison
    names = list(results.keys())
    maes  = [r["MAE"]  for r in results.values()]
    rmses = [r["RMSE"] for r in results.values()]
    r2s   = [r["R2"]   for r in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title, color in zip(
        axes,
        [maes, rmses, r2s],
        ["MAE (lower=better)", "RMSE (lower=better)", "R² (higher=better)"],
        ["#4C72B0", "#DD8452", "#55A868"]
    ):
        ax.bar(names, vals, color=color)
        ax.set_title(title)
        ax.set_xticklabels(names, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()

    # 3. Feature Importance (Random Forest)
    rf_pipeline = results["Random Forest"]["pipeline"]
    rf_model = rf_pipeline.named_steps["model"]
    importances = pd.Series(
        rf_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Top Feature Importances - Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    print("📈 Plots saved: predictions_plot.png, model_comparison.png, feature_importance.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_best_model(results, X.columns.tolist())
    plot_all(results, y_test, X.columns.tolist())
