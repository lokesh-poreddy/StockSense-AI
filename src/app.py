from flask import Flask, render_template, send_from_directory, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

app = Flask(__name__)

def get_real_time_price(symbol):
    stock = yf.Ticker(symbol)
    return stock.info.get('regularMarketPrice', 0)

@app.route('/')
def home():
    summary_df = pd.read_excel('results/summary_report.xlsx')
    summary_data = summary_df.to_dict('records')
    image_files = [f for f in os.listdir('results') if f.endswith('_analysis.png')]
    
    # Add real-time prices
    for row in summary_data:
        row['Current_Price'] = get_real_time_price(row['Symbol'])
    
    return render_template('index.html', summary_data=summary_data, image_files=image_files)

@app.route('/stock_data/<symbol>')
def stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period='1y')
    return jsonify({
        'dates': hist.index.strftime('%Y-%m-%d').tolist(),
        'prices': hist['Close'].tolist(),
        'volume': hist['Volume'].tolist()
    })

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)