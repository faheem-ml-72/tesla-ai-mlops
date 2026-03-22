import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv("data/tesla_features.csv")

# =========================
# 🎯 Create Target
# =========================
# Predict: Will price go UP tomorrow?
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop last row (NaN target)
df.dropna(inplace=True)

# =========================
# 🧠 Features & Labels
# =========================
FEATURES = ['Open','High','Low','Close','Volume','EMA_10','EMA_50','RSI','MACD','Signal_Line']

X = df[FEATURES]
y = df['Target']

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

print(f"Accuracy: {accuracy:.2f}")

# =========================
# 💾 Save Model
# =========================
joblib.dump(model, "models/xgboost_model.pkl")

print("Model saved successfully ✅")