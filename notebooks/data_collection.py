import yfinance as yf
import pandas as pd

# Download Tesla stock data
tesla = yf.download("TSLA", start="2018-01-01", end="2024-12-31")

# Save to CSV
tesla.to_csv("data/tesla_stock.csv")

print("Data saved successfully!")
print(tesla.head())