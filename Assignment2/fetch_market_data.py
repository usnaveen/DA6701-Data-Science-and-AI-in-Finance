"""
fetch_market_data.py
====================
Fetches daily adjusted OHLCV data for the 6-stock Indian equity universe
from Yahoo Finance using the yfinance library.

NSE tickers on Yahoo Finance use the suffix '.NS'
e.g., RELIANCE.NS, HDFCBANK.NS, INFY.NS, M&M.NS, BHARTIARTL.NS, HUL.NS

Usage:
    python fetch_market_data.py
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS_NSE = {
    "RELIANCE":   "RELIANCE.NS",
    "HDFCBANK":   "HDFCBANK.NS",
    "INFY":       "INFY.NS",
    "MM":         "M&M.NS",      # M&M requires special handling
    "BHARTIARTL": "BHARTIARTL.NS",
    "HUL":        "HINDUNILVR.NS",  # HUL listed as HINDUNILVR on NSE
}

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

# Train/test split: Oct 2025 onwards is forward-test period
FORWARD_TEST_START = "2025-10-01"

RAW_DIR       = Path("data/raw/yahoo")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Fetch & Save ─────────────────────────────────────────────────────────────

def fetch_single(name: str, ticker: str) -> pd.DataFrame:
    """Download OHLCV for one ticker, return cleaned DataFrame."""
    print(f"  Fetching {name} ({ticker}) ...")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusts for splits & dividends
        progress=False,
    )

    if df.empty:
        print(f"    WARNING: No data returned for {ticker}")
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df["Ticker"] = name

    # Compute log-returns on adjusted close
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Simple daily return
    df["Return"] = df["Close"].pct_change()

    df.to_csv(RAW_DIR / f"{name}.csv")
    print(f"    Saved {len(df)} rows  →  {RAW_DIR / name}.csv")
    return df


def build_panel(ticker_dfs: dict) -> pd.DataFrame:
    """Stack all ticker DataFrames into a long-form panel."""
    frames = []
    for name, df in ticker_dfs.items():
        if not df.empty:
            frames.append(df)

    panel = pd.concat(frames, axis=0).reset_index()
    panel.sort_values(["Ticker", "Date"], inplace=True)
    panel.to_csv(PROCESSED_DIR / "panel_daily.csv", index=False)
    print(f"\nPanel saved: {len(panel)} rows  →  {PROCESSED_DIR}/panel_daily.csv")
    return panel


def split_forward_test(panel: pd.DataFrame):
    """Separate out Oct-Dec 2025 as the forward-test set."""
    panel["Date"] = pd.to_datetime(panel["Date"])
    train = panel[panel["Date"] < FORWARD_TEST_START]
    fwdtest = panel[panel["Date"] >= FORWARD_TEST_START]

    train.to_csv(PROCESSED_DIR / "panel_train.csv", index=False)
    fwdtest.to_csv(PROCESSED_DIR / "panel_forward_test.csv", index=False)
    print(f"Train rows:        {len(train)}")
    print(f"Forward-test rows: {len(fwdtest)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1 – Market Data Ingestion")
    print("=" * 60)

    ticker_dfs = {}
    for name, yf_ticker in TICKERS_NSE.items():
        ticker_dfs[name] = fetch_single(name, yf_ticker)

    panel = build_panel(ticker_dfs)
    split_forward_test(panel)

    print("\nDone.")


if __name__ == "__main__":
    main()
