import pandas as pd
import requests

# Read tickers from CSV columns
df = pd.read_csv("us_tech_stock_prices.csv")
tickers = df.columns.tolist()

# Set API key
PPLX_API_KEY = "your_api_key_here"
HEADERS = {
    "Authorization": f"Bearer {PPLX_API_KEY}",
    "Content-Type": "application/json"
}

def analyze_with_perplexity(ticker):
    query = f"""
    Analyze the financial fundamentals of the company {ticker} based on its latest earnings report, valuation (PE, PB), profit margins, ROE, debt, and revenue growth.
    Provide investor-friendly insights. If possible, compare with industry benchmarks.
    """
    
    payload = {
        "model": "pplx-7b-online",  # Or pplx-70b-online if available
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=HEADERS, json=payload)
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error for {ticker}: {e}"

# Run it for all tickers
for ticker in tickers:
    print("=" * 60)
    print(f"📈 LLM Analysis for {ticker}")
    print(analyze_with_perplexity(ticker))
    print("=" * 60)
