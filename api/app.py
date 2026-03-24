from fastapi import FastAPI
from pydantic import BaseModel
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
# 🔹 Request Schema (IMPORTANT FIX)
# ======================
class PredictRequest(BaseModel):
    news: str
    days: int

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
# 🔹 Prediction Endpoint (FIXED)
# ======================
class PredictRequest(BaseModel):
    news: str
    days: int

@app.post("/predict")
def predict(data: PredictRequest):

        # Load model
        model = get_model()

        # Load data
        df = pd.read_csv("data/tesla_features.csv", index_col=0)

        if df.empty:
            return {"error": "Data not available"}

        # ======================
        # 📊 XGBoost Prediction
        # ======================
        x_input = df[FEATURES].iloc[-1:].values
        xgb_pred = model.predict(x_input)[0]

        # ======================
        # 📈 Gaussian Process
        # ======================
        close_prices = df['Close'].dropna().values[-100:]

        if len(close_prices) < 10:
            return {"error": "Not enough data for prediction"}

        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        gp = GaussianProcessRegressor()
        gp.fit(X, y)

        future_steps = days  # ✅ dynamic from frontend
        future_x = np.arange(len(close_prices), len(close_prices) + future_steps).reshape(-1, 1)

        gp_preds, gp_std = gp.predict(future_x, return_std=True)

        future_prices = gp_preds.tolist()
        uncertainty = gp_std.tolist()

        # ======================
        # 🧠 Sentiment
        # ======================
        sentiment_score = get_sentiment_score(news)

        # ======================
        # 🚀 Final Prediction
        # ======================
        final_prediction = (
            0.6 * xgb_pred +
            0.2 * sentiment_score +
            0.2 * np.mean(gp_preds)
        )

        # ======================
        # 📤 Response
        # ======================
        return {
            "xgboost_prediction": float(xgb_pred),
            "sentiment_score": float(sentiment_score),
            "future_prices": future_prices,
            "uncertainty": uncertainty,
            "final_prediction": float(final_prediction)
        }

    except Exception as e:
        return {"error": str(e)}