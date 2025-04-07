# StockSense AI

## Project Description
StockSense AI is an advanced stock market prediction system that leverages LSTM (Long Short-Term Memory) neural networks to analyze and predict stock prices for major technology and financial companies. The system processes historical data from 2018 onwards and provides real-time market insights through an interactive web dashboard.

### Core Functionality
- Historical data analysis from 2018 to present
- Real-time stock price monitoring for 10 major companies
- LSTM-based price prediction model
- Interactive visualization dashboard
- Performance metrics calculation
- Volatility and returns analysis

### Target Companies
- Technology: TSLA, AAPL, MSFT, AMZN, GOOGL, META, NVDA
- Financial: JPM, V
- Retail: WMT

## Development Workflow

### 1. Data Pipeline
- Automated data collection using Yahoo Finance API
- Data preprocessing and normalization
- Feature engineering for LSTM model
- Train-test data splitting

### 2. Model Training
- LSTM neural network implementation
- Hyperparameter optimization
- Model validation and testing
- Performance metrics calculation
    - RMSE (Root Mean Square Error)
    - R² Score
    - Volatility Analysis
    - Annual Returns

### 3. Web Interface
- Real-time price updates
- Interactive stock charts
- Performance comparison dashboard
- Historical trend visualization

### 4. Continuous Integration/Deployment
- Automated testing with pytest
- Code quality checks (flake8, black, isort)
- Automated documentation generation
- Deployment pipeline for web interface

### 5. Documentation
- API documentation using pdoc3
- Usage guides and examples
- Model architecture details
- Performance metrics explanation

## Project Structure
- Python
- PyTorch (LSTM)
- Flask
- Pandas & NumPy
- Plotly & Matplotlib
- Yahoo Finance API

## Installation
1. Clone the repository
```bash
git clone https://github.com/lokesh-poreddy/StockSense-AI.git
cd StockSense-AI