import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def retrain_model():
    print("🚀 Retraining started...")

    data_path = os.path.join(BASE_DIR, "..", "data", "tesla_features.csv")
    model_path = os.path.join(BASE_DIR, "..", "models", "latest_model.pkl")

    df = pd.read_csv(data_path)

    FEATURES = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
    ]

    X = df[FEATURES]
    y = df['Close']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, model_path)

    print("✅ Model retrained and saved!")


if __name__ == "__main__":
    retrain_model()