import os
import yfinance as yf
from datetime import datetime

def download_stock_data(symbols):
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        stock = yf.Ticker(symbol)
        data = stock.history(period="5y")
        
        # Convert timezone-aware timestamps to timezone-naive
        data.index = data.index.tz_localize(None)
        
        # Save to data directory
        data.to_excel(f"data/{symbol}_Stock_Price_Prediction.xlsx")
        print(f"Data saved for {symbol}")


if __name__ == "__main__":
    symbols = [
        "TSLA",  # Tesla
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "AMZN",  # Amazon
        "GOOGL",  # Google
        "META",  # Meta
        "NVDA",  # NVIDIA
        "JPM",  # JPMorgan Chase
        "V",  # Visa
        "WMT",  # Walmart
    ]

    download_stock_data(symbols)
