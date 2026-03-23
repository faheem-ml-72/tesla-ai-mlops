from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
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
    # Gaussian Process
    # ======================
    close_prices = df['Close'].values[-100:]

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices

    gp = GaussianProcessRegressor()
    gp.fit(X, y)

    future_x = np.array([[len(close_prices)]])
    gp_pred, gp_std = gp.predict(future_x, return_std=True)

    gp_pred = gp_pred[0]
    gp_uncertainty = gp_std[0]

    # ======================
    # LSTM (temporary placeholder)
    # ======================
    lstm_pred = 0.0

    # ======================
    # Sentiment
    # ======================
    sentiment_score = get_sentiment_score(news)

    # ======================
    # Ensemble
    # ======================
    final_prediction = (
        0.5 * xgb_pred +
        0.3 * sentiment_score +
        0.2 * gp_pred
    )

    return {
        "xgboost_prediction": float(xgb_pred),
        "lstm_prediction": float(lstm_pred),
        "sentiment_score": float(sentiment_score),
        "gp_prediction": float(gp_pred),
        "gp_uncertainty": float(gp_uncertainty),
        "final_prediction": float(final_prediction)
    }