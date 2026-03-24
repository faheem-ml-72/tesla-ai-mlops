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

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin: 10px 0;
}

.green { color: #00ff88; font-weight: bold; }
.red { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ======================
# 🚀 Title
# ======================
st.markdown('<div class="title">🚀 Tesla Stock AI Dashboard</div>', unsafe_allow_html=True)
st.write("")

# ======================
# 📈 Stock Chart (SAFE)
# ======================
data_chart = yf.download("TSLA", period="1mo")

if data_chart.empty:
    st.error("❌ Failed to load Tesla stock data. Try refreshing.")
    st.stop()

st.markdown("## 📈 Tesla Stock Price")
st.line_chart(data_chart["Close"])

# ======================
# 📊 KPIs (SAFE)
# ======================
col1, col2, col3 = st.columns(3)

try:
    col1.metric("💰 Price", round(data_chart["Close"].iloc[-1], 2))
    col2.metric("📈 Daily Change %", round(data_chart["Close"].pct_change().iloc[-1]*100, 2))
    col3.metric("📊 Volume", int(data_chart["Volume"].iloc[-1]))
except:
    st.warning("⚠️ Unable to calculate metrics")

# ======================
# 🔍 Input
# ======================
st.markdown("## 🧠 AI Prediction")
news = st.text_input("Enter Tesla news:")

# ======================
# 🎛️ Controls
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
            url = "https://tesla-ai-mlops.onrender.com/predict"

            response = requests.post(
                url,
                json={
                    "news": news,
                    "days": days
                },
                timeout=30
            )

            if response.status_code != 200:
                st.error(f"❌ API failed: {response.text}")
                st.stop()

            data = response.json()

            # ======================
            # 📈 Graph
            # ======================
            if "future_prices" in data:

                past_df = data_chart["Close"].tail(30)

                future_prices = data["future_prices"]

                future_dates = pd.date_range(
                    start=data_chart.index[-1],
                    periods=len(future_prices)+1
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
                st.warning("⚠️ No future prediction returned")

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
            if data.get("final_prediction", 0) > 0.6:
                st.success("🟢 BUY")
            elif data.get("final_prediction", 0) > 0.4:
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