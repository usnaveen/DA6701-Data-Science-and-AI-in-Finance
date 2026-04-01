import numpy as np
import pandas as pd

def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    active = port_returns - bench_returns
    return active.std() * np.sqrt(252)

def information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    active = port_returns - bench_returns
    if active.std() == 0:
        return np.nan
    return (active.mean() / active.std()) * np.sqrt(252)

def portfolio_returns(weights: dict, returns_df: pd.DataFrame) -> pd.Series:
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    return returns_df[tickers] @ w

def evaluate(weights: dict, R_val, b_val, R_hold, b_hold, label="") -> dict:
    p_val = portfolio_returns(weights, R_val)
    p_hold = portfolio_returns(weights, R_hold)
    return {
        "label": label,
        "k": len(weights),
        "TE_val": tracking_error(p_val, b_val),
        "IR_val": information_ratio(p_val, b_val),
        "TE_hold": tracking_error(p_hold, b_hold),
        "IR_hold": information_ratio(p_hold, b_hold),
    }
