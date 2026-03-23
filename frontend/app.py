import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np

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
    box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
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
# 📈 Live Stock Chart
# ======================
data_chart = yf.download("TSLA", period="1mo")

st.markdown("## 📈 Tesla Stock Price")
st.line_chart(data_chart["Close"])

# KPI
col1, col2, col3 = st.columns(3)
col1.metric("💰 Price", round(data_chart["Close"].iloc[-1], 2))
col2.metric("📈 Daily Change %", round(data_chart["Close"].pct_change().iloc[-1]*100, 2))
col3.metric("📊 Volume", int(data_chart["Volume"].iloc[-1]))

# ======================
# 🔍 Input
# ======================
st.markdown("## 🧠 AI Prediction")
news = st.text_input("Enter Tesla news:")

# ======================
# 🔮 Prediction
# ======================
if st.button("Predict"):

    if news:
        try:
            url = "https://tesla-ai-mlops.onrender.com/predict"

            response = requests.post(url, params={"news": news})
            data = response.json()

            # ======================
            # 📈 Combined Forecast Graph
            # ======================
            past_df = data_chart["Close"].tail(30)

            future_prices = data["future_prices"]
            future_dates = pd.date_range(start=data_chart.index[-1], periods=8)[1:]

            future_df = pd.Series(future_prices, index=future_dates)

            combined = pd.concat([past_df, future_df])

            st.markdown("## 📈 Price Forecast (Past + Future)")
            st.line_chart(combined)

            # ======================
            # 📊 Results
            # ======================
            st.markdown("## 📊 Results")

            direction = "📈 UP" if data["xgboost_prediction"] == 1 else "📉 DOWN"
            st.metric("📊 XGBoost Trend", direction)

            st.write("💬 Sentiment Score:", data["sentiment_score"])
            st.write("🚀 Final Prediction:", data["final_prediction"])

            # ======================
            # 🧠 AI Signal
            # ======================
            st.markdown("## 🧠 AI Signal")

            if data["final_prediction"] > 0.6:
                st.success("🟢 STRONG BUY")
            elif data["final_prediction"] > 0.4:
                st.info("🟡 HOLD")
            else:
                st.error("🔴 SELL")

            # ======================
            # 🎯 Confidence
            # ======================
            st.markdown("## 🎯 AI Confidence")

            confidence = 1 - np.std(data["future_prices"]) / 100
            confidence = max(0, min(1, confidence))

            st.progress(confidence)
            st.write(f"Confidence Score: {round(confidence, 2)}")

            # ======================
            # 🧾 Cards UI
            # ======================
            st.markdown("## 📊 Analysis Cards")

            st.markdown(f"""
            <div class="card">
            📊 <b>XGBoost Trend:</b> 
            <span class="{ 'green' if data['xgboost_prediction']==1 else 'red' }">
            { 'UP 📈' if data['xgboost_prediction']==1 else 'DOWN 📉' }
            </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
            💬 <b>Sentiment Score:</b> {data['sentiment_score']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
            🚀 <b>Final Prediction:</b> {data['final_prediction']}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"API Error: {e}")

    else:
        st.warning("Please enter news text")