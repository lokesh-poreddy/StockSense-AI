import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def calculate_returns(prices):
    return np.diff(prices) / prices[:-1]


def calculate_volatility(returns):
    return np.std(returns) * np.sqrt(252)  # Annualized volatility
