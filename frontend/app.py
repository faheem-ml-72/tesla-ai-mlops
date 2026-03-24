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
    color: white;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
}

/* Card */
.card {
    background-color: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

/* Section spacing */
.section {
    margin-top: 30px;
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
    # 📈 CLEAN FORECAST GRAPH
    # ======================
    st.markdown("## 📊 Tesla Forecast")

    past = data_chart["Close"].tail(30)
    future = data["future_prices"]

    future_dates = pd.date_range(
        start=past.index[-1],
        periods=len(future)+1
    )[1:]

    future_series = pd.Series(future, index=future_dates)

    uncertainty = np.std(future)

    fig = go.Figure()

    # Past
    fig.add_trace(go.Scatter(
        x=past.index,
        y=past,
        name="Past Price",
        line=dict(color="#00d4ff", width=3)
    ))

    # Future
    fig.add_trace(go.Scatter(
        x=future_series.index,
        y=future_series,
        name="Prediction",
        line=dict(color="#ff9800", width=3, dash="dash")
    ))

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=future_series.index,
        y=future_series + uncertainty,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_series.index,
        y=future_series - uncertainty,
        fill='tonexty',
        name="Uncertainty",
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(width=0)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # 📊 RESULTS (CLEAN)
    # ======================
    st.markdown("## 📊 Results")

    col1, col2, col3 = st.columns(3)

    trend = "UP 📈" if data["xgboost_prediction"] == 1 else "DOWN 📉"
    col1.metric("Trend", trend)

    col2.metric("Sentiment", round(data["sentiment_score"], 3))
    col3.metric("Prediction", round(data["final_prediction"], 3))

    # ======================
    # 🎯 SIGNAL
    # ======================
    pred = data["final_prediction"]

    if pred > 0.6:
        st.success("🟢 STRONG BUY")
    elif pred > 0.4:
        st.warning("🟡 HOLD")
    else:
        st.error("🔴 SELL")

    # ======================
    # 📊 CONFIDENCE (FIXED)
    # ======================
    confidence = abs(pred) / 10
    confidence = min(confidence, 1.0)

    st.progress(confidence)
    st.write(f"Confidence: {round(confidence, 2)}")

    # ======================
    # 🧠 MODEL INFO
    # ======================
    st.caption("Model: XGBoost + Sentiment + Gaussian Process")