import pandas as pd
import yfinance as yf

# Load tickers from CSV file (columns = tickers)
df = pd.read_csv("us_tech_stock_prices.csv")
tickers = df.columns.tolist()

# Rule-based fundamental analyzer
def analyze_company(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        analysis = []

        analysis.append(f"📌 {info.get('shortName', ticker_symbol)} ({ticker_symbol})")

        pe = info.get("trailingPE")
        if pe:
            analysis.append(f"PE Ratio: {pe}")
            if pe < 20:
                analysis.append("✅ Valuation is moderate (P/E < 20)")
            elif pe > 40:
                analysis.append("⚠️ High valuation, may be overvalued")

        debt_to_eq = info.get("debtToEquity")
        if debt_to_eq:
            analysis.append(f"Debt/Equity: {debt_to_eq}")
            if debt_to_eq < 0.5:
                analysis.append("✅ Low debt level, financially healthy")
            else:
                analysis.append("⚠️ High debt level, risky balance sheet")

        margin = info.get("profitMargins")
        if margin:
            analysis.append(f"Profit Margin: {margin}")
            if margin > 0.15:
                analysis.append("✅ Strong profitability")
            else:
                analysis.append("⚠️ Weak profitability")

        roe = info.get("returnOnEquity")
        if roe:
            analysis.append(f"ROE: {roe}")
            if roe > 15:
                analysis.append("✅ Good return on equity")
            else:
                analysis.append("⚠️ Low return on equity")

        return "\n".join(analysis)

    except Exception as e:
        return f"❌ Error fetching data for {ticker_symbol}: {e}"

# Run the analyzer
for ticker in tickers:
    print("=" * 60)
    print(analyze_company(ticker))
    print("=" * 60)
