import pandas as pd
from config import TRAIN_YEARS, DATA_PROCESSED

def compute_monthly_returns(price_data):
    monthly_prices = price_data.resample("ME").last()
    monthly_returns = monthly_prices.pct_change().dropna()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    monthly_returns.to_csv(DATA_PROCESSED / "monthly_returns.csv")

    return monthly_returns


def train_test_split(returns):
    split_index = int(len(returns) * TRAIN_YEARS / 5)

    train = returns.iloc[:split_index]
    test = returns.iloc[split_index:]

    train.to_csv(DATA_PROCESSED / "monthly_returns_train.csv")
    test.to_csv(DATA_PROCESSED / "monthly_returns_test.csv")

    return train, test