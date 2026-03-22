import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ======================
# 🔹 Load Models
# ======================
xgb_model = joblib.load("models/xgboost_model.pkl")
lstm_model = load_model("models/lstm_model.h5", compile=False)
scaler = joblib.load("models/lstm_scaler.pkl")

# ======================
# 🔹 Load Data
# ======================
df = pd.read_csv("data/tesla_features.csv", index_col=0)

# ======================
# 🔹 Define Features (IMPORTANT)
# ======================
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
]

# ======================
# 🔹 XGBoost Prediction
# ======================
x_input = df[FEATURES].iloc[-1:].values
xgb_pred = xgb_model.predict(x_input)[0]

# ======================
# 🔹 LSTM Prediction
# ======================
close_data = df[['Close']].values
scaled_data = scaler.transform(close_data)

window_size = 60
last_sequence = scaled_data[-window_size:]

last_sequence = np.reshape(last_sequence, (1, window_size, 1))
lstm_pred = lstm_model.predict(last_sequence)[0][0]

# Inverse scaling
lstm_pred = scaler.inverse_transform([[lstm_pred]])[0][0]

# ======================
# 🔹 Sentiment Score (placeholder for now)
# ======================
# Later we will connect FinBERT here
from utils.sentiment import get_sentiment_score

# Example news (later we will automate this)
news = "Tesla stock surges after strong earnings report"

sentiment_score = get_sentiment_score(news) 

# ======================
# 🔹 Final Ensemble
# ======================
final_prediction = (
    0.4 * xgb_pred +
    0.4 * lstm_pred +
    0.2 * sentiment_score
)

# ======================
# 🔹 Output
# ======================
print("XGBoost Prediction:", xgb_pred)
print("LSTM Prediction:", lstm_pred)
print("Sentiment Score:", sentiment_score)
print("Final Ensemble Prediction:", final_prediction)
print("X input:", x_input)
print("XGB prediction:", xgb_pred)