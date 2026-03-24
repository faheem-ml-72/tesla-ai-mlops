import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ======================
# 🎨 Styling
# ======================
st.markdown("""
<style>
body { background-color: #0e1117; }
.main { background-color: #0e1117; }
.title {
    font-size: 40px;
    font-weight: bold;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ======================
# 🚀 Title
# ======================
st.markdown('<div class="title">🚀 Tesla Stock AI Dashboard</div>', unsafe_allow_html=True)
st.write("")

# ======================
# 📈 Stock Data
# ======================
try:
    data_chart = yf.download("TSLA", period="1mo")

    if data_chart is None or data_chart.empty or "Close" not in data_chart.columns:
        st.error("❌ Failed to load Tesla stock data.")
        st.stop()

except Exception as e:
    st.error(f"❌ Error loading stock data: {e}")
    st.stop()

st.markdown("## 📈 Tesla Stock Price")
st.line_chart(data_chart["Close"])

# ======================
# 📊 KPIs
# ======================
col1, col2, col3 = st.columns(3)

try:
    close_series = data_chart["Close"].dropna()
    volume_series = data_chart["Volume"].dropna()

    latest_price = float(close_series.iloc[-1]) if len(close_series) > 0 else 0
    latest_change = float(close_series.pct_change().dropna().iloc[-1]) * 100 if len(close_series) > 1 else 0
    latest_volume = int(volume_series.iloc[-1]) if len(volume_series) > 0 else 0

    col1.metric("💰 Price", round(latest_price, 2))
    col2.metric("📈 Daily Change %", round(latest_change, 2))
    col3.metric("📊 Volume", latest_volume)

except:
    st.warning("⚠️ Unable to calculate metrics")

# ======================
# 🧠 Input
# ======================
st.markdown("## 🧠 AI Prediction")
news = st.text_input("Enter Tesla news:")

# ======================
# ⚙️ Controls
# ======================
st.markdown("### ⚙️ Prediction Controls")

col1, col2 = st.columns(2)

with col1:
    days = st.slider("📅 Days to Predict", 1, 30, 7)

with col2:
    selected_date = st.date_input("📆 Select Start Date")

# ======================
# 🔮 Prediction
# ======================
if st.button("Predict"):

    if not news:
        st.warning("⚠️ Enter news text")

    else:
        try:
            API_URL = "https://tesla-ai-mlops.onrender.com/predict"

            response = requests.post(
                API_URL,
                params={"news": news}
            )

            if response.status_code != 200:
                st.error(f"❌ API failed: {response.text}")
                st.stop()

            data = response.json()

            # ======================
            # 📈 Graph
            # ======================
            if "future_prices" in data and len(data["future_prices"]) > 0:

                past_df = data_chart["Close"].tail(30)
                future_prices = data["future_prices"]

                future_dates = pd.date_range(
                    start=data_chart.index[-1],
                    periods=len(future_prices) + 1
                )[1:]

                future_df = pd.Series(future_prices, index=future_dates)

                upper = future_df + np.std(future_prices)
                lower = future_df - np.std(future_prices)

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=past_df.index,
                    y=past_df,
                    mode='lines',
                    name='Past Price',
                    line=dict(color='cyan', width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=future_df.index,
                    y=future_df,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='orange', width=3, dash='dash')
                ))

                fig.add_trace(go.Scatter(
                    x=future_df.index,
                    y=upper,
                    line=dict(width=0),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter(
                    x=future_df.index,
                    y=lower,
                    fill='tonexty',
                    line=dict(width=0),
                    name='Uncertainty',
                    fillcolor='rgba(255,165,0,0.2)'
                ))

                fig.update_layout(
                    template="plotly_dark",
                    height=500,
                    title="📈 Tesla Forecast"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("⚠️ No prediction data returned")

            # ======================
            # 📊 Results
            # ======================
            st.markdown("## 📊 Results")

            direction = "📈 UP" if data.get("xgboost_prediction", 0) == 1 else "📉 DOWN"
            st.metric("📊 XGBoost Trend", direction)

            st.write("💬 Sentiment:", round(data.get("sentiment_score", 0), 3))
            st.write("🚀 Final Prediction:", round(data.get("final_prediction", 0), 3))

            # ======================
            # 🎯 Signal
            # ======================
            pred_val = data.get("final_prediction", 0)

            if pred_val > 0.6:
                st.success("🟢 BUY")
            elif pred_val > 0.4:
                st.info("🟡 HOLD")
            else:
                st.error("🔴 SELL")

            # ======================
            # 📊 Confidence
            # ======================
            if "future_prices" in data and len(data["future_prices"]) > 0:
                confidence = 1 - np.std(data["future_prices"]) / np.mean(data["future_prices"])
                confidence = max(0, min(1, confidence))
            else:
                confidence = 0.5

            st.progress(confidence)
            st.write(f"Confidence: {round(confidence, 2)}")

        except Exception as e:
            st.error(f"🚨 {e}")