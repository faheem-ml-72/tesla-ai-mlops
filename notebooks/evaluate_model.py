import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/tesla_features.csv")

# Create target
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

FEATURES = ['Open','High','Low','Close','Volume','EMA_10','EMA_50','RSI','MACD','Signal_Line']

X = df[FEATURES]
y = df['Target']

# Load model
model = joblib.load("models/xgboost_model.pkl")

# Predict
y_pred = model.predict(X)

# Accuracy
acc = accuracy_score(y, y_pred)

print("Model Accuracy:", acc)