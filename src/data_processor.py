"""
StockSense AI - Data Processing Module
Copyright (c) 2023 Lokesh Poreddy
Licensed under the MIT License
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataProcessor:
    def __init__(self, file_paths):
        self.file_paths = (
            file_paths if isinstance(file_paths, dict) else {"Default": file_paths}
        )
        self.scalers = {}

    def load_data(self):
        self.data = {}
        for company, file_path in self.file_paths.items():
            self.data[company] = pd.read_excel(file_path)
        return self.data

    def prepare_data(self, sequence_length=60, batch_size=32):
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
