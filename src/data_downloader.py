import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_stock_data(symbols, start_date='2018-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        # Convert timezone-aware timestamps to timezone-naive
        data.index = data.index.tz_localize(None)
        
        data.to_excel(f'{symbol}_Stock_Price_Prediction.xlsx')
        print(f"Data saved for {symbol}")

if __name__ == "__main__":
    symbols = [
        'TSLA',  # Tesla
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'AMZN',  # Amazon
        'GOOGL', # Google
        'META',  # Meta
        'NVDA',  # NVIDIA
        'JPM',   # JPMorgan Chase
        'V',     # Visa
        'WMT'    # Walmart
    ]
    
    download_stock_data(symbols)