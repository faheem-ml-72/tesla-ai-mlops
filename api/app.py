from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import joblib
import sys
import os
from tensorflow.keras.models import load_model

# ======================
# 🔧 Fix path (CRITICAL FIX)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from utils.sentiment import get_sentiment_score

app = FastAPI()

# ======================
# 🔹 Lazy Load Models
# ======================
xgb_model = None
lstm_model = None


def get_xgb_model():
    global xgb_model
    if xgb_model is None:
        xgb_model = joblib.load(os.path.join(BASE_DIR, "..", "models", "latest_model.pkl"))
    return xgb_model


def get_lstm_model():
    global lstm_model
    if lstm_model is None:
        lstm_model = load_model(os.path.join(BASE_DIR, "..", "models", "lstm_model.h5"))
    return lstm_model


# ======================
# 🔹 Normalize Function
# ======================
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val + 1e-8)


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
    try:
        # ======================
        # 📥 Load Models
        # ======================
        xgb_model = get_xgb_model()
        lstm_model = get_lstm_model()

        # ======================
        # 📊 Load Data
        # ======================
        df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "tesla_features.csv"), index_col=0)

        if df.empty:
            raise HTTPException(status_code=400, detail="Data file is empty")

        # ======================
        # 📊 XGBoost
        # ======================
        x_input = df[FEATURES].iloc[-1:].values
        xgb_pred = float(xgb_model.predict(x_input)[0])

        # ======================
        # 🤖 LSTM
        # ======================
        lstm_input = df[FEATURES].tail(30).values

        if lstm_input.shape[0] < 30:
            raise HTTPException(status_code=400, detail="Not enough data for LSTM")

        lstm_input = lstm_input.reshape(1, 30, len(FEATURES))
        lstm_pred = float(lstm_model.predict(lstm_input, verbose=0)[0][0])

        # ======================
        # 📈 Gaussian Process
        # ======================
        close_prices = df['Close'].values[-100:]

        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        gp = GaussianProcessRegressor()
        gp.fit(X, y)

        future_x = np.arange(len(close_prices), len(close_prices) + 7).reshape(-1, 1)
        gp_preds, gp_std = gp.predict(future_x, return_std=True)

        gp_mean = float(np.mean(gp_preds))
        gp_uncertainty = float(np.mean(gp_std))

        # ======================
        # 🧠 Sentiment
        # ======================
        sentiment_score = float(get_sentiment_score(news))
        sentiment_norm = (sentiment_score + 1) / 2

        # ======================
        # 🔄 Normalize Predictions
        # ======================
        recent_prices = df['Close'].values[-100:]
        min_p, max_p = np.min(recent_prices), np.max(recent_prices)

        xgb_norm = normalize(xgb_pred, min_p, max_p)
        lstm_norm = normalize(lstm_pred, min_p, max_p)
        gp_norm = normalize(gp_mean, min_p, max_p)

        # ======================
        # 🧠 Dynamic Weights
        # ======================
        w_xgb = 0.35
        w_lstm = 0.35
        w_sentiment = 0.1
        w_gp = 0.2 if gp_uncertainty < 5 else 0.1

        total = w_xgb + w_lstm + w_sentiment + w_gp
        w_xgb /= total
        w_lstm /= total
        w_sentiment /= total
        w_gp /= total

        # ======================
        # 🚀 Final Ensemble
        # ======================
        final_norm = (
            w_xgb * xgb_norm +
            w_lstm * lstm_norm +
            w_sentiment * sentiment_norm +
            w_gp * gp_norm
        )

        final_prediction = final_norm * (max_p - min_p) + min_p

        # ======================
        # 📊 Direction + Confidence
        # ======================
        last_price = df['Close'].iloc[-1]
        direction = "UP" if final_prediction > last_price else "DOWN"
        confidence = 1 / (1 + gp_uncertainty)

        # ======================
        # 📤 Response
        # ======================
        return {
            "xgboost_prediction": xgb_pred,
            "lstm_prediction": lstm_pred,
            "sentiment_score": sentiment_score,
            "future_prices": gp_preds.tolist(),
            "uncertainty": gp_std.tolist(),
            "final_prediction": float(final_prediction),
            "confidence": float(confidence),
            "direction": direction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("🚀 API Loaded Successfully")