import yfinance as yf
import pandas as pd

# US Tech Stiocks
tech_tickers = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Class A)
    "GOOG",   # Alphabet (Class C)
    "AMZN",   # Amazon
    "META",   # Meta Platforms
    "TSLA",   # Tesla
    "NVDA",   # Nvidia
    "ORCL",   # Oracle
    "INTC",   # Intel
    "CRM",    # Salesforce
    "ADBE",   # Adobe
    "CSCO",   # Cisco
    "AMD",    # AMD
    "IBM",    # IBM
]


# Fetch data
data = yf.download(tech_tickers, start="2024-04-01", end="2025-04-01", group_by='ticker', auto_adjust=True)

# Get closing prices
closing_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tech_tickers})
print(closing_prices.head())

closing_prices.to_csv("us_tech_stock_prices.csv")
