"""eval.py — Shared evaluation harness for Assignment IV.

Provides tracking error, information ratio, portfolio return construction,
and a unified evaluate() function used by all method modules.
"""

import numpy as np
import pandas as pd


def tracking_error(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Annualised tracking error (std of active returns × √252)."""
    active = port_returns - bench_returns
    return float(active.std() * np.sqrt(252))


def information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """Annualised Information Ratio = mean(active) / std(active) × √252."""
    active = port_returns - bench_returns
    if active.std() == 0:
        return np.nan
    return float((active.mean() / active.std()) * np.sqrt(252))


def portfolio_returns(weights: dict, returns_df: pd.DataFrame) -> pd.Series:
    """Compute portfolio return series from a weight dictionary.

    Parameters
    ----------
    weights : dict
        {ticker: float} — must sum to 1 (normalised internally just in case).
    returns_df : pd.DataFrame
        T x N daily return matrix.
    """
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers], dtype=float)
    w = w / w.sum()
    return returns_df[tickers] @ w


def evaluate(
    weights: dict,
    R_val: pd.DataFrame,
    b_val: pd.Series,
    R_hold: pd.DataFrame,
    b_hold: pd.Series,
    label: str = "",
) -> dict:
    """Full evaluation on both validation and holdout sets.

    Returns a flat dict with keys: label, k, TE_val, IR_val, TE_hold, IR_hold.
    """
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


def load_splits(data_dir: str = "data") -> tuple:
    """Load pre-built train / val / test splits from the data/ folder.

    Returns
    -------
    R_train, b_train, R_val, b_val, R_hold, b_hold
    """
    from pathlib import Path

    base = Path(data_dir)
    train = pd.read_parquet(base / "train_returns.parquet")
    val = pd.read_parquet(base / "val_returns.parquet")
    test = pd.read_parquet(base / "test_returns.parquet")

    R_train = train.drop(columns="^GSPC")
    b_train = train["^GSPC"]
    R_val = val.drop(columns="^GSPC")
    b_val = val["^GSPC"]
    R_hold = test.drop(columns="^GSPC")
    b_hold = test["^GSPC"]

    return R_train, b_train, R_val, b_val, R_hold, b_hold
