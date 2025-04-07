"""
StockSense AI - Data Processing Module
Copyright (c) 2023 Lokesh Poreddy
Licensed under the MIT License

This module handles data processing for stock price prediction, including:
- Loading and preprocessing stock data
- Scaling and sequence preparation
- Dataset creation for PyTorch
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class StockDataset(Dataset):
    """Custom PyTorch Dataset for stock price data.

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Target values
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (features, target) pair
        """
        return self.X[idx], self.y[idx]


class DataProcessor:
    """Handles data processing for stock price prediction.

    Args:
        file_paths (dict or str): Paths to stock price data files
    """

    def __init__(self, file_paths):
        self.file_paths = (
            file_paths if isinstance(file_paths, dict) else {"Default": file_paths}
        )
        self.scalers = {}

    def load_data(self):
        """Load stock price data from Excel files.

        Returns:
            dict: Dictionary containing DataFrames for each stock
        """
        self.data = {}
        for company, file_path in self.file_paths.items():
            self.data[company] = pd.read_excel(file_path)
        return self.data

    def prepare_data(self, sequence_length=60, batch_size=32):
        """Prepare data for model training.

        Args:
            sequence_length (int): Length of input sequences
            batch_size (int): Size of batches for training

        Returns:
            tuple: (datasets, scalers) pair containing prepared data
        """
        datasets = {}
        for company, df in self.data.items():
            scaler = MinMaxScaler()
            self.scalers[company] = scaler
            scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

            X = []
            y = []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i - sequence_length : i, 0])
                y.append(scaled_data[i, 0])

            X = np.array(X)
            y = np.array(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            train_dataset = StockDataset(X_train, y_train)
            test_dataset = StockDataset(X_test, y_test)

            datasets[company] = {
                "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            }

        return datasets, self.scalers
