import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ======================
# 🎨 UI CONFIG
# ======================
st.set_page_config(page_title="Tesla AI Predictor", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
body, .stMarkdown, .stText, .stMetric {
    color: #ffffff !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px;
    font-weight: bold;
}
[data-testid="stMetricLabel"] {
    color: #cfd8dc !important;
}
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: white;
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
# 📊 KPI
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
# 🔮 PREDICT
# ======================
if st.button("Predict 🚀"):

    if not news:
        st.warning("Enter news first")
        st.stop()

    API_URL = "https://tesla-ai-mlops.onrender.com/predict"

    try:
        response = requests.post(API_URL, params={"news": news})
        data = response.json()
    except:
        st.error("API connection failed")
        st.stop()

    if "error" in data:
        st.error(data["error"])
        st.stop()

    # ======================
    # 📊 RESULTS
    # ======================
    st.markdown("## 📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("📈 Prediction", round(data["final_prediction"], 2))
    col2.metric("🎯 Confidence", round(data["confidence"], 3))
    col3.metric("📊 Direction", data["direction"])

    # ======================
    # 🎯 SIGNAL
    # ======================
    if data["direction"] == "UP":
        st.success("🟢 Market looks bullish")
    else:
        st.error("🔴 Market looks bearish")

    # ======================
    # 📊 FORECAST GRAPH (NEW 🔥)
    # ======================
    st.markdown("## 📈 Tesla Forecast")

    future_prices = [data["final_prediction"]] * 7
    uncertainty = [1 - data["confidence"]] * 7

    future_dates = pd.date_range(
        start=pd.Timestamp.today(),
        periods=7
    )

    fig = go.Figure()

    # Prediction line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines+markers',
        name='Prediction'
    ))

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=np.array(future_prices) + np.array(uncertainty),
        mode='lines',
        name='Upper Bound',
        line=dict(dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=np.array(future_prices) - np.array(uncertainty),
        mode='lines',
        name='Lower Bound',
        line=dict(dash='dash'),
        fill='tonexty'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 🧠 MODEL HEALTH
    # ======================
    st.markdown("## 🧠 Model Health")

    drift = data["drift_detected"]
    drift_score = data["drift_score"]

    st.write("Drift Detected:", drift)
    st.write("Drift Score:", round(drift_score, 4))

    if drift:
        st.warning("⚠️ Model drift detected — retraining recommended")
    else:
        st.success("✅ Model stable")

    # ======================
    # 📌 FOOTER
    # ======================
    st.caption("Model: XGBoost + LSTM + Sentiment + Drift Detection + Forecast")