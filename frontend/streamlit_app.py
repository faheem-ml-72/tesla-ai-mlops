import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np

# ======================
# 🎨 UI STYLE
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
    margin-bottom: 10px;
    color: white;
}
h1, h2, h3 {
    color: white !important;
}
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
# 🔮 PREDICT
# ======================
if st.button("Predict 🚀"):

    if not news:
        st.warning("Enter news first")
        st.stop()

    API_URL = "https://tesla-ai-mlops.onrender.com/predict"

    with st.spinner("🤖 AI is analyzing..."):
        response = requests.post(API_URL, params={"news": news})

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    data = response.json()

    # ======================
    # 📊 FORECAST TABLE
    # ======================
    st.markdown("## 📊 Tesla Forecast")

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
    # 📊 MODEL OUTPUT
    # ======================
    st.markdown("## 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("📈 Direction", data["direction"])
    col2.metric("🧠 Sentiment", round(data["sentiment_score"], 3))
    col3.metric("💰 Predicted Price", round(data["final_prediction"], 2))

    # ======================
    # 📊 MODEL INTELLIGENCE (NEW 🔥)
    # ======================
    st.markdown("## 📊 Model Intelligence")

    col1, col2 = st.columns(2)

    col1.metric("Confidence", round(data["confidence"], 3))
    col2.metric("Drift Score", round(data["drift_score"], 4))

    # Drift Alert
    if data["drift_detected"]:
        st.warning("⚠️ Market drift detected! Model retraining may be triggered.")
    else:
        st.success("✅ No drift detected")

    # ======================
    # 🎯 SIGNAL
    # ======================
    if data["direction"] == "UP":
        st.markdown("### 🟢 BUY SIGNAL")
    else:
        st.markdown("### 🔴 SELL SIGNAL")

    # ======================
    # 📊 CONFIDENCE BAR
    # ======================
    st.progress(min(data["confidence"], 1.0))
    st.write(f"Confidence Score: {round(data['confidence'], 2)}")

    # ======================
    # 🧠 MODEL INFO
    # ======================
    st.caption("Model: Ensemble (XGBoost + LSTM + Sentiment + Drift-aware system)")