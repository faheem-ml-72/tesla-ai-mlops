import pandas as pd

# Load data
df = pd.read_csv("data/tesla_stock.csv")

# Remove unwanted rows
df = df[df.iloc[:, 0] != 'Ticker']

# Rename first column to Date
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Drop invalid rows
df.dropna(subset=['Date'], inplace=True)

# Set index
df.set_index('Date', inplace=True)

# 🔥 Convert numeric values
df = df.apply(pd.to_numeric, errors='ignore')

# 🔥 Sort by date
df.sort_index(inplace=True)

# =========================
# EMA
# =========================
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

# =========================
# RSI
# =========================
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# =========================
# MACD
# =========================
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()

df['MACD'] = ema12 - ema26
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Drop NaN
df.dropna(inplace=True)

# Save
df.to_csv("data/tesla_features.csv")

print("Feature engineering completed ✅")
print(df.head())
print("Columns:", df.columns)