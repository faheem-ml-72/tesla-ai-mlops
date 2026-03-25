from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import joblib
import sys
import os
from tensorflow.keras.models import load_model

# ======================
# 🔧 Fix path
# ======================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), "..")))

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
        xgb_model = joblib.load("models/latest_model.pkl")
    return xgb_model


def get_lstm_model():
    global lstm_model
    if lstm_model is None:
        lstm_model = load_model("models/lstm_model.h5")
    return lstm_model


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
        df = pd.read_csv("data/tesla_features.csv", index_col=0)

        if df.empty:
            raise HTTPException(status_code=400, detail="Data file is empty")

        # ======================
        # 📊 XGBoost Prediction
        # ======================
        x_input = df[FEATURES].iloc[-1:].values
        xgb_pred = xgb_model.predict(x_input)[0]

        # ======================
        # 🤖 LSTM Prediction
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

        future_steps = 7
        future_x = np.arange(len(close_prices), len(close_prices) + future_steps).reshape(-1, 1)

        gp_preds, gp_std = gp.predict(future_x, return_std=True)

        # ======================
        # 🧠 Sentiment
        # ======================
        sentiment_score = get_sentiment_score(news)

        # ======================
        # 🚀 Ensemble Prediction
        # ======================
        final_prediction = (
            0.4 * xgb_pred +
            0.3 * lstm_pred +
            0.15 * sentiment_score +
            0.15 * np.mean(gp_preds)
        )

        # ======================
        # 📤 Response
        # ======================
        return {
            "xgboost_prediction": float(xgb_pred),
            "lstm_prediction": float(lstm_pred),
            "sentiment_score": float(sentiment_score),
            "future_prices": gp_preds.tolist(),
            "uncertainty": gp_std.tolist(),
            "final_prediction": float(final_prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================
# 🔥 Debug (optional)
# ======================
print("🚀 API Loaded Successfully")