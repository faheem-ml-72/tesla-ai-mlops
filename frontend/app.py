import streamlit as st
import requests
import yfinance as yf

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

            st.markdown("## 📊 Results")

            # 📊 XGBoost
            direction = "📈 UP" if data["xgboost_prediction"] == 1 else "📉 DOWN"
            st.metric("📊 XGBoost Trend", direction)

            # 💬 Sentiment
            st.write("💬 Sentiment Score:", data["sentiment_score"])

            # 📈 GP
            st.write("📈 GP Prediction:", data["gp_prediction"])
            st.write("⚠️ Uncertainty:", data["gp_uncertainty"])

            # 🔥 Sentiment Interpretation
            if data["sentiment_score"] > 0.7:
                st.success("📈 Positive market sentiment")
            elif data["sentiment_score"] < 0.3:
                st.error("📉 Negative market sentiment")
            else:
                st.info("⚖️ Neutral sentiment")

            # 🚀 Final Prediction
            st.write("🚀 Final Prediction:", data["final_prediction"])

            if data["final_prediction"] > 0.6:
                st.success("📈 Strong Buy Signal")
            elif data["final_prediction"] > 0.4:
                st.info("⚖️ Neutral Trend")
            else:
                st.error("📉 Bearish Trend")

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
            📈 <b>GP Prediction:</b> {data['gp_prediction']} <br>
            ⚠️ <b>Uncertainty:</b> {data['gp_uncertainty']}
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