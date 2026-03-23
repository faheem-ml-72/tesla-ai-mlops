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

    # Load model
    model = get_model()

    # Load data
    df = pd.read_csv("data/tesla_features.csv", index_col=0)

    # ======================
    # XGBoost Prediction
    # ======================
    x_input = df[FEATURES].iloc[-1:].values
    xgb_pred = model.predict(x_input)[0]

    # ======================
    # 📈 Future Prediction (7 days using GP)
    # ======================
    close_prices = df['Close'].values[-100:]

    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = close_prices

    gp = GaussianProcessRegressor()
    gp.fit(X, y)

    future_steps = 7
    future_x = np.arange(len(close_prices), len(close_prices) + future_steps).reshape(-1, 1)

    gp_preds, gp_std = gp.predict(future_x, return_std=True)

    future_prices = gp_preds.tolist()
    uncertainty = gp_std.tolist()

    # ======================
    # Single GP Prediction (next step)
    # ======================
    gp_pred = gp_preds[0]

    # ======================
    # LSTM (placeholder)
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

    # ======================
    # Response
    # ======================
    return {
        "xgboost_prediction": float(xgb_pred),
        "sentiment_score": float(sentiment_score),
        "future_prices": future_prices,
        "uncertainty": uncertainty,
        "final_prediction": float(final_prediction)
    }