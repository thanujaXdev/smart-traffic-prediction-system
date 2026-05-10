# 🚦 Smart Traffic Prediction System

A job-level ML project that predicts traffic volume using multiple regression models,
served through a professional Streamlit dashboard.

---

## 📁 Project Structure

```
traffic_prediction_project/
│
├── traffic_data.csv          # Dataset
├── requirements.txt          # Dependencies
├── config.py                 # Central configuration
├── data_preprocessing.py     # Feature engineering pipeline
├── model_training.py         # Train, evaluate & save models
├── predict.py                # Prediction logic (used by app)
└── app.py                    # Streamlit web application
```

---

## 🚀 How to Run (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the models
```bash
python model_training.py
```
This will:
- Load and preprocess `traffic_data.csv`
- Train 4 models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
- Print MAE, RMSE, R² for each model
- Save the best model to `saved_model.pkl`
- Generate and save 3 plots

### Step 3 — Launch the web app
```bash
streamlit run app.py
```

---

## 🧠 Models Used

| Model               | Notes                              |
|---------------------|------------------------------------|
| Linear Regression   | Baseline model                     |
| Decision Tree       | Interpretable, handles non-linearity |
| Random Forest       | Ensemble, robust to overfitting    |
| Gradient Boosting   | Often best performance             |

---

## 📊 Features Engineered

| Feature           | Description                          |
|-------------------|--------------------------------------|
| Weather_Severity  | Ordinal: Clear=0, Cloudy=1, Rainy=2  |
| Weekend           | 1 if Saturday/Sunday                 |
| Rush_Hour         | 1 if 7–9 AM or 5–7 PM               |
| Day_*             | One-hot encoded day columns          |
| Weather_*         | One-hot encoded weather columns      |

---

## 💼 Interview Talking Points

- **Pipelines**: Models wrapped in `sklearn.Pipeline` to prevent data leakage
- **Cross-Validation**: 5-fold CV for reliable evaluation on small datasets
- **Multiple Metrics**: MAE, RMSE, and R² — not just one metric
- **Model Persistence**: Best model saved with `joblib` for deployment
- **Separation of Concerns**: Data, training, prediction, and UI are all separate modules
- **Gradient Boosting**: Added as a stronger alternative to Random Forest
- **Rush Hour Feature**: Domain knowledge encoded as a new feature
