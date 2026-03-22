from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import sys
import os


# 🔥 Fix module path (VERY IMPORTANT)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.sentiment import get_sentiment_score

app = FastAPI()

# ======================
# 🔹 Load Models (once)
# ======================
xgb_model = joblib.load("models/xgboost_model.pkl")
lstm_prediction = 0  # temporary placeholder
scaler = joblib.load("models/lstm_scaler.pkl")

# ======================
# 🔹 Features
# ======================
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
]

# ======================
# 🔹 Home Endpoint
# ======================
@app.get("/")
def home():
    return {"message": "Tesla AI MLOps API is running 🚀"}


# ======================
# 🔹 Prediction Endpoint
# ======================
@app.post("/predict")
def predict(news: str):

    # Load latest data
    df = pd.read_csv("data/tesla_features.csv", index_col=0)

    # ======================
    # XGBoost Prediction
    # ======================
    x_input = df[FEATURES].iloc[-1:].values
    xgb_pred = xgb_model.predict(x_input)[0]

    # ======================
    # LSTM Prediction
    # ======================
    close_data = df[['Close']].values
    scaled_data = scaler.transform(close_data)

    window_size = 60
    last_sequence = scaled_data[-window_size:]
    last_sequence = np.reshape(last_sequence, (1, window_size, 1))

    lstm_pred = lstm_model.predict(last_sequence)[0][0]
    lstm_pred = scaler.inverse_transform([[lstm_pred]])[0][0]

    # ======================
    # Sentiment
    # ======================
    sentiment_score = get_sentiment_score(news)

    # ======================
    # Ensemble
    # ======================
    final_prediction = (
        0.4 * xgb_pred +
        0.4 * lstm_pred +
        0.2 * sentiment_score
    )

    return {
        "xgboost_prediction": float(xgb_pred),
        "lstm_prediction": float(lstm_pred),
        "sentiment_score": float(sentiment_score),
        "final_prediction": float(final_prediction)
    }