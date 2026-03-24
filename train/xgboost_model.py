import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import datetime
import os

# =========================
# 📥 Load Data
# =========================
df = pd.read_csv("data/tesla_features.csv")

# =========================
# 🎯 Create Target
# =========================
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop NaN rows
df.dropna(inplace=True)

# =========================
# 🧠 Features & Labels
# =========================
FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'EMA_10', 'EMA_50', 'RSI', 'MACD', 'Signal_Line'
]

X = df[FEATURES]
y = df['Target']

# =========================
# ⚠️ Safety Check
# =========================
if X.empty or y.empty:
    raise ValueError("❌ Dataset is empty after preprocessing")

# =========================
# ✂️ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# 🚀 Train XGBoost
# =========================
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# =========================
# 📊 Evaluate
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.4f}")

# =========================
# 💾 Model Versioning
# =========================
os.makedirs("models", exist_ok=True)

version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = f"models/xgboost_model_{version}.pkl"

joblib.dump(model, model_path)

# Save latest pointer
joblib.dump(model, "models/latest_model.pkl")

print(f"✅ Versioned model saved: {model_path}")
print("✅ Latest model updated: models/latest_model.pkl")