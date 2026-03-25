from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
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
from utils.drift import detect_drift   # ✅ NEW

app = FastAPI(title="Tesla AI MLOps API 🚀")

# ======================
# 🔹 Lazy Load Models
# ======================
xgb_model = None
lstm_model = None


def get_xgb_model():
    global xgb_model
    if xgb_model is None:
        path = os.path.join(BASE_DIR, "..", "models", "latest_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError("XGBoost model not found")
        xgb_model = joblib.load(path)
    return xgb_model


def get_lstm_model():
    global lstm_model
    if lstm_model is None:
        path = os.path.join(BASE_DIR, "..", "models", "lstm_model.h5")
        if not os.path.exists(path):
            raise FileNotFoundError("LSTM model not found")
        lstm_model = load_model(path)
    return lstm_model


# ======================
# 🔹 Normalize Function
# ======================
def normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.5
    return (value - min_val) / (max_val - min_val)


# ======================
# 🔹 Features
# ======================
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
]


# ======================
# 🔹 Health Check
# ======================
@app.get("/health")
def health():
    return {"status": "ok"}


# ======================
# 🔹 Home
# ======================
@app.get("/")
def home():
    return {"message": "Tesla AI MLOps API is running 🚀"}


# ======================
# 🔹 Prediction
# ======================
@app.post("/predict")
def predict(news: str):
    try:
        # ======================
        # 📥 Load Models
        # ======================
        xgb = get_xgb_model()
        lstm = get_lstm_model()

        # ======================
        # 📊 Load Data
        # ======================
        data_path = os.path.join(BASE_DIR, "..", "data", "tesla_features.csv")
        if not os.path.exists(data_path):
            raise HTTPException(status_code=500, detail="Data file not found")

        df = pd.read_csv(data_path, index_col=0)

        if df.empty or len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough data")

        # ======================
        # 📉 Drift Detection (NEW)
        # ======================
        if len(df) < 200:
            drift_result = {
                "drift_detected": False,
                "drift_score": 0.0
            }
        else:
            historical = df['Close'].values[-200:-100]
            recent = df['Close'].values[-100:]
            drift_result = detect_drift(historical, recent)

        # ======================
        # 📊 XGBoost
        # ======================
        x_input = df[FEATURES].iloc[-1:].values
        xgb_pred = float(xgb.predict(x_input)[0])

        # ======================
        # 🤖 LSTM
        # ======================
        lstm_input = df[FEATURES].tail(30).values.reshape(1, 30, len(FEATURES))
        lstm_pred = float(lstm.predict(lstm_input, verbose=0)[0][0])

        # ======================
        # 📈 Gaussian Process
        # ======================
        close_prices = df['Close'].values[-100:]

        X = np.arange(len(close_prices)).reshape(-1, 1)
        gp = GaussianProcessRegressor()
        gp.fit(X, close_prices)

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
        # 🔄 Normalize
        # ======================
        recent_prices = df['Close'].values[-100:]
        min_p, max_p = np.min(recent_prices), np.max(recent_prices)

        xgb_norm = normalize(xgb_pred, min_p, max_p)
        lstm_norm = normalize(lstm_pred, min_p, max_p)
        gp_norm = normalize(gp_mean, min_p, max_p)

        # ======================
        # 🧠 Dynamic Weights
        # ======================
        w_xgb, w_lstm, w_sentiment = 0.35, 0.35, 0.1
        w_gp = 0.2 if gp_uncertainty < 5 else 0.1

        total = w_xgb + w_lstm + w_sentiment + w_gp
        w_xgb /= total
        w_lstm /= total
        w_sentiment /= total
        w_gp /= total

        # ======================
        # 🚀 Ensemble
        # ======================
        final_norm = (
            w_xgb * xgb_norm +
            w_lstm * lstm_norm +
            w_sentiment * sentiment_norm +
            w_gp * gp_norm
        )

        final_prediction = float(final_norm * (max_p - min_p) + min_p)

        # ======================
        # 📊 Direction & Confidence
        # ======================
        last_price = df['Close'].iloc[-1]
        direction = "UP" if final_prediction > last_price else "DOWN"
        confidence = float(1 / (1 + gp_uncertainty))

        # ======================
        # 📤 Response
        # ======================
        return {
            "xgboost_prediction": xgb_pred,
            "lstm_prediction": lstm_pred,
            "sentiment_score": sentiment_score,
            "future_prices": gp_preds.tolist(),
            "uncertainty": gp_std.tolist(),
            "final_prediction": final_prediction,
            "confidence": confidence,
            "direction": direction,
            "drift_detected": drift_result["drift_detected"],   # ✅ NEW
            "drift_score": drift_result["drift_score"]          # ✅ NEW
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


print("🚀 API Loaded Successfully")