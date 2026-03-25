import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def retrain_model():
    print("🚀 Retraining started...")

    # ======================
    # 📊 Load Data
    # ======================
    data_path = os.path.join(BASE_DIR, "..", "data", "tesla_features.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError("❌ Data file not found")

    df = pd.read_csv(data_path)

    if df.empty or len(df) < 50:
        raise ValueError("❌ Not enough data for training")

    # ======================
    # 🔹 Features
    # ======================
    FEATURES = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
    ]

    X = df[FEATURES]
    y = df['Close']

    # ======================
    # 🤖 Train Model
    # ======================
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # ======================
    # 💾 Save Model (Versioning)
    # ======================
    models_dir = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    versioned_path = os.path.join(models_dir, f"model_{timestamp}.pkl")
    latest_path = os.path.join(models_dir, "latest_model.pkl")

    joblib.dump(model, versioned_path)
    joblib.dump(model, latest_path)

    # ======================
    # 📝 Logging
    # ======================
    log_path = os.path.join(BASE_DIR, "..", "retrain_log.txt")
    with open(log_path, "a") as f:
        f.write(f"Retrained at: {timestamp}\n")

    print("✅ Model retrained and saved!")
    print(f"📁 Versioned model: {versioned_path}")
    print(f"📁 Latest model updated: {latest_path}")


if __name__ == "__main__":
    retrain_model()