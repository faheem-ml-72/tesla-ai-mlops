from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.sentiment import get_sentiment_score

app = FastAPI()

# ======================
# 🔹 Lazy Load Model
# ======================
model = None

def get_model():
    global model
    if model is None:
        model = joblib.load("models/xgboost_model.pkl")
    return model

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

    # Load model (lazy)
    model = get_model()

    # Load latest data
    df = pd.read_csv("data/tesla_features.csv", index_col=0)

    # ======================
    # XGBoost Prediction
    # ======================
    x_input = df[FEATURES].iloc[-1:].values
    xgb_pred = model.predict(x_input)[0]

    # ======================
    # Sentiment
    # ======================
    sentiment_score = get_sentiment_score(news)

    # ======================
    # Ensemble
    # ======================
    final_prediction = (
        0.7 * xgb_pred +
        0.3 * sentiment_score
    )

    return {
        "xgboost_prediction": float(xgb_pred),
        "lstm_prediction": 0.0,  # placeholder
        "sentiment_score": float(sentiment_score),
        "final_prediction": float(final_prediction)
    }