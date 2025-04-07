import matplotlib.pyplot as plt
import numpy as np  # Add this import
import pandas as pd
import torch
import torch.nn as nn

from analysis import calculate_metrics, calculate_returns, calculate_volatility
from data_processor import DataProcessor
from model import LSTMModel


def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            if len(batch_X.shape) == 2:
                batch_X = batch_X.unsqueeze(-1)
            outputs = model(batch_X)
            # Split long line
            loss = criterion(
                outputs, 
                batch_y.unsqueeze(1)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}"
            )


def main():
    # Define stock symbols
    symbols = [
        "TSLA",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "NVDA",
        "JPM",
        "V",
        "WMT",
    ]
    stock_files = {
        symbol: f"{symbol}_Stock_Price_Prediction.xlsx" for symbol in symbols
    }

    results = {}

    # Initialize data processor with multiple stocks
    data_processor = DataProcessor(stock_files)

    # Load and prepare data
    datasets, scalers = data_processor.prepare_data()  # Remove unused 'data' variable

    # Create results directory if it doesn't exist
    import os

    os.makedirs("results", exist_ok=True)

    # Process each stock
    for symbol in symbols:
        print(f"\nProcessing {symbol} stock data...")

        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"Training model for {symbol}...")
        train_model(model, datasets[symbol]["train"], criterion, optimizer)

        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in datasets[symbol]["test"]:
                if len(batch_X.shape) == 2:
                    batch_X = batch_X.unsqueeze(-1)
                outputs = model(batch_X)
                predictions.extend(outputs.numpy().flatten())
                actuals.extend(batch_y.numpy().flatten())

        # Calculate metrics
        metrics = calculate_metrics(actuals, predictions)
        returns = calculate_returns(actuals)
        volatility = calculate_volatility(returns)

        results[symbol] = {
            "metrics": metrics,
            "volatility": volatility,
            "returns": np.mean(returns) * 252,
        }

        # Create visualization
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(actuals, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title(f"{symbol} Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.hist(returns, bins=50, density=True, alpha=0.7)
        plt.title(f"{symbol} Returns Distribution")
        plt.xlabel("Returns")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(f"results/{symbol}_analysis.png")
        plt.close()

    # Generate summary report
    print("\nSummary Report:")
    summary_data = []
    for symbol, result in results.items():
        summary_data.append(
            {
                "Symbol": symbol,
                "RMSE": result["metrics"]["RMSE"],
                "R2": result["metrics"]["R2"],
                "Volatility": result["volatility"],
                "Ann. Returns": result["returns"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel("results/summary_report.xlsx", index=False)
    print("\nAnalysis complete! Check the 'results' directory for detailed outputs.")


if __name__ == "__main__":
    main()
