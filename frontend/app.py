import streamlit as st
import requests

st.title("🚀 Tesla Stock Prediction AI")

# Input
news = st.text_input("Enter Tesla news:")

if st.button("Predict"):
    if news:
        url = "http://127.0.0.1:8000/predict"
        
        params = {"news": news}
        
        response = requests.post(url, params=params)
        data = response.json()

        st.subheader("Results:")

        # 📊 XGBoost Trend
        direction = "📈 UP" if data["xgboost_prediction"] == 1 else "📉 DOWN"
        st.metric("📊 XGBoost Trend", direction)

        # 🧠 LSTM Prediction
        st.write("🧠 LSTM:", data["lstm_prediction"])

        # 💬 Sentiment Score
        st.write("💬 Sentiment Score:", data["sentiment_score"])

        # 🔥 Sentiment Interpretation
        if data["sentiment_score"] > 0.7:
            st.success("📈 Positive market sentiment")
        elif data["sentiment_score"] < 0.3:
            st.error("📉 Negative market sentiment")
        else:
            st.info("⚖️ Neutral sentiment")

        # 🚀 Final Prediction
        st.write("🚀 Final Prediction:", data["final_prediction"])

    else:
        st.warning("Please enter news text")