import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# ======================
# 🔹 TRAIN FUNCTION
# ======================
def train_lstm():

    print("🚀 Training LSTM Model...")

    # Load data
    df = pd.read_csv("data/tesla_features.csv", index_col=0)

    # Use Close price
    data = df[['Close']].values

    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Save scaler
    joblib.dump(scaler, "models/lstm_scaler.pkl")

    # ======================
    # Create sequences
    # ======================
    X, y = [], []
    window_size = 60

    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    # Reshape (VERY IMPORTANT)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ======================
    # Train/Test Split
    # ======================
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ======================
    # Build Model
    # ======================
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # ======================
    # Train
    # ======================
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # ======================
    # Evaluate
    # ======================
    loss = model.evaluate(X_test, y_test)
    print(f"📊 Test Loss: {loss}")

    # ======================
    # Save Model
    # ======================
    model.save("models/lstm_model.h5")

    print("✅ LSTM model saved successfully!")

# ======================
# 🔹 RUN
# ======================
if __name__ == "__main__":
    train_lstm()