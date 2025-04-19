import yfinance as yf

# Fetch historical data for Apple
data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
print(data.head())
