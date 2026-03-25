from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import sys
import os
from tensorflow.keras.models import load_model

# ======================
# 🔧 Base Path Fix
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from utils.sentiment import get_sentiment_score
from utils.drift import detect_drift

app = FastAPI(title="Tesla AI MLOps API 🚀")

xgb_model = None
lstm_model = None


def get_xgb_model():
    global xgb_model
    if xgb_model is None:
        path = os.path.join(BASE_DIR, "..", "models", "latest_model.pkl")
        xgb_model = joblib.load(path)
    return xgb_model


def get_lstm_model():
    global lstm_model
    if lstm_model is None:
        path = os.path.join(BASE_DIR, "..", "models", "lstm_model.h5")
        try:
            lstm_model = load_model(path, compile=False)
        except:
            lstm_model = None
    return lstm_model


def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.5
    return (value - min_val) / (max_val - min_val)


FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
]


@app.get("/")
def home():
    return {"message": "Tesla AI MLOps API is running 🚀"}


@app.post("/predict")
def predict(news: str):
    try:
        xgb = get_xgb_model()
        lstm = get_lstm_model()

        # ======================
        # 📊 Load Data
        # ======================
        data_path = os.path.join(BASE_DIR, "..", "data", "tesla_features.csv")
        df = pd.read_csv(data_path, index_col=0)

        # 🔥 FIX: ensure all features exist
        for col in FEATURES:
            if col not in df.columns:
                raise Exception(f"Missing column: {col}")

        # ======================
        # 📉 Drift
        # ======================
        try:
            if len(df) >= 200:
                drift_result = detect_drift(
                    df['Close'].values[-200:-100],
                    df['Close'].values[-100:]
                )
            else:
                drift_result = {"drift_detected": False, "drift_score": 0.0}
        except:
            drift_result = {"drift_detected": False, "drift_score": 0.0}

        # ======================
        # 📊 XGBoost
        # ======================
        x_input = df[FEATURES].iloc[-1:].values
        xgb_pred = float(xgb.predict(x_input)[0])

        # ======================
        # 🤖 LSTM
        # ======================
        try:
            if lstm:
                lstm_input = df[FEATURES].tail(30).values.reshape(1, 30, len(FEATURES))
                lstm_pred = float(lstm.predict(lstm_input, verbose=0)[0][0])
            else:
                lstm_pred = xgb_pred
        except:
            lstm_pred = xgb_pred

        # ======================
        # ⚡ GP Approx
        # ======================
        close_prices = df['Close'].values[-100:]
        gp_preds = close_prices[-7:]
        gp_std = np.std(close_prices[-30:]) * np.ones(7)

        gp_mean = float(np.mean(gp_preds))
        gp_uncertainty = float(np.mean(gp_std))

        # ======================
        # 🧠 Sentiment
        # ======================
        try:
            sentiment_score = float(get_sentiment_score(news))
        except:
            sentiment_score = 0.0

        sentiment_norm = (sentiment_score + 1) / 2

        # ======================
        # 🔄 Normalize
        # ======================
        min_p, max_p = np.min(close_prices), np.max(close_prices)

        xgb_norm = normalize(xgb_pred, min_p, max_p)
        lstm_norm = normalize(lstm_pred, min_p, max_p)
        gp_norm = normalize(gp_mean, min_p, max_p)

        # ======================
        # 🚀 Ensemble
        # ======================
        final_norm = (
            0.4 * xgb_norm +
            0.3 * lstm_norm +
            0.15 * sentiment_norm +
            0.15 * gp_norm
        )

        final_prediction = float(final_norm * (max_p - min_p) + min_p)

        last_price = df['Close'].iloc[-1]
        direction = "UP" if final_prediction > last_price else "DOWN"
        confidence = float(1 / (1 + gp_uncertainty))

        return {
            "final_prediction": final_prediction,
            "confidence": confidence,
            "direction": direction,
            "drift_detected": drift_result["drift_detected"],
            "drift_score": drift_result["drift_score"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}