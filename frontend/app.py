import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ======================
# 🎨 PROFESSIONAL UI STYLE
# ======================
st.set_page_config(page_title="Tesla AI Predictor", layout="wide")

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* 🔥 FIX TEXT VISIBILITY */
body, .stMarkdown, .stText, .stMetric {
    color: #ffffff !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px;
    font-weight: bold;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #cfd8dc !important;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
    color: white;
}

/* Headers */
h1, h2, h3 {
    color: white !important;
}

/* Input box */
input {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# ======================
# 🚀 TITLE
# ======================
st.markdown('<div class="title">🚀 Tesla AI Stock Predictor</div>', unsafe_allow_html=True)

# ======================
# 📈 STOCK DATA
# ======================
data_chart = yf.download("TSLA", period="1mo")

st.markdown("## 📈 Tesla Stock Price")
st.line_chart(data_chart["Close"])

# ======================
# 📊 KPI CARDS
# ======================
col1, col2, col3 = st.columns(3)

close_series = data_chart["Close"].dropna()
volume_series = data_chart["Volume"].dropna()

latest_price = float(close_series.iloc[-1])
latest_change = float(close_series.pct_change().iloc[-1]) * 100
latest_volume = int(volume_series.iloc[-1])

col1.metric("💰 Price", round(latest_price, 2))
col2.metric("📈 Change %", round(latest_change, 2))
col3.metric("📊 Volume", latest_volume)

# ======================
# 🧠 INPUT
# ======================
st.markdown("## 🧠 AI Prediction")
news = st.text_input("Enter Tesla news")

# ======================
# ⚙️ CONTROLS
# ======================
col1, col2 = st.columns(2)

with col1:
    days = st.slider("Days to Predict", 1, 30, 7)

with col2:
    selected_date = st.date_input("Start Date")

# ======================
# 🔮 PREDICT BUTTON
# ======================
if st.button("Predict 🚀"):

    if not news:
        st.warning("Enter news first")
        st.stop()

    API_URL = "https://tesla-ai-mlops.onrender.com/predict"

    response = requests.post(API_URL, params={"news": news})

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    data = response.json()

    # ======================
    # 📊 TESLA FORECAST TABLE (NEW)
    # ======================
    st.markdown("## 📊 Tesla Forecast Table")

    future = data["future_prices"]
    uncertainty = data["uncertainty"]

    future_dates = pd.date_range(
        start=pd.Timestamp.today(),
        periods=len(future)
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": np.round(future, 2),
        "Uncertainty": np.round(uncertainty, 4)
    })

    st.dataframe(forecast_df, use_container_width=True)

    # ======================
    # 📊 RESULTS
    # ======================
    st.markdown("## 📊 Results")

    col1, col2, col3 = st.columns(3)

    trend = "📉 DOWN" if data["xgboost_prediction"] == 0 else "📈 UP"

    col1.metric("Trend", trend)
    col2.metric("Sentiment", round(data["sentiment_score"], 3))
    col3.metric("Prediction", round(data["final_prediction"], 3))

    # ======================
    # 🎯 SIGNAL
    # ======================
    pred = data["final_prediction"]

    if pred > 0.6:
        st.markdown("### 🟢 STRONG BUY")
    elif pred > 0.4:
        st.markdown("### 🟡 HOLD")
    else:
        st.markdown("### 🔴 SELL")

    # ======================
    # 📊 CONFIDENCE
    # ======================
    confidence = abs(pred) / 10
    confidence = min(confidence, 1.0)

    st.progress(confidence)
    st.write(f"Confidence: {round(confidence, 2)}")

    # ======================
    # 🧠 MODEL INFO
    # ======================
    st.caption("Model: XGBoost + Sentiment + Gaussian Process")