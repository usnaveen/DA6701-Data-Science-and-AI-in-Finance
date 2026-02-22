"""
build_features.py
=================
Merges market data, macro indicators, fundamentals, and sentiment into a
single feature matrix ready for model training.

Technical features computed:
  • Log returns (1d, 5d, 10d, 21d)
  • Rolling volatility (5d, 21d)
  • RSI (14-period)
  • MACD and signal line
  • Bollinger Band width
  • Volume z-score (21d)
  • 52-week high/low ratio

All features are:
  1. Lagged appropriately (market features use close price at T → feature at T)
  2. Target: forward 1-day log return (log(Close_{T+1}/Close_T))
  3. Robust-scaled (median / IQR) to handle financial outliers

Output
------
  data/processed/features_train.parquet
  data/processed/features_forward_test.parquet

Usage:
    python build_features.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS = ["RELIANCE", "HDFCBANK", "INFY", "MM", "BHARTIARTL", "HUL"]
FORWARD_TEST_START = "2025-10-01"
FUNDAMENTAL_FEATURES = [
    "PE",
    "Debt_Equity",
    "ROE",
    "EPS",
    "Revenue",
    "EBITDA_Margin",
    "Promoter_Holding",
]

PROCESSED_DIR = Path("data/processed")


# ─── Technical Indicator Library ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line


def compute_bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / (sma + 1e-10)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: Open, High, Low, Close, Volume.
    Returns df with additional feature columns.
    """
    close = df["Close"]
    vol   = df["Volume"]

    # Returns
    df["log_ret_1d"]  = np.log(close / close.shift(1))
    df["log_ret_5d"]  = np.log(close / close.shift(5))
    df["log_ret_10d"] = np.log(close / close.shift(10))
    df["log_ret_21d"] = np.log(close / close.shift(21))

    # Rolling volatility
    df["vol_5d"]  = df["log_ret_1d"].rolling(5).std()
    df["vol_21d"] = df["log_ret_1d"].rolling(21).std()

    # RSI
    df["rsi_14"] = compute_rsi(close, 14)

    # MACD
    df["macd"], df["macd_signal"] = compute_macd(close)
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Band width (proxy for volatility regime)
    df["bb_width"] = compute_bollinger_width(close)

    # Volume z-score (21d)
    df["vol_zscore"] = (vol - vol.rolling(21).mean()) / (vol.rolling(21).std() + 1e-10)

    # Price position: distance from 52-week high/low
    df["hi_252"]    = close.rolling(252).max()
    df["lo_252"]    = close.rolling(252).min()
    df["hi_ratio"]  = close / df["hi_252"]
    df["lo_ratio"]  = close / df["lo_252"]

    # Price momentum (simple momentum, normalized)
    df["mom_21d"]  = close.pct_change(21)
    df["mom_63d"]  = close.pct_change(63)

    # Average True Range (ATR) — volatility proxy
    hl   = df["High"] - df["Low"]
    hpc  = (df["High"] - close.shift(1)).abs()
    lpc  = (df["Low"]  - close.shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_norm"] = df["atr_14"] / (close + 1e-10)   # normalize by price

    # Target: next-day log return (shift -1 so it aligns with T features)
    df["target"] = df["log_ret_1d"].shift(-1)

    # Drop construction helpers
    df.drop(columns=["hi_252", "lo_252", "atr_14"], inplace=True, errors="ignore")
    return df


# ─── Merge All Data Sources ───────────────────────────────────────────────────

def load_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "panel_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"Market data not found: {path}. Run fetch_market_data.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def load_macro() -> pd.DataFrame:
    path = PROCESSED_DIR / "macro_daily.csv"
    if not path.exists():
        print("  WARNING: macro_daily.csv not found. Skipping macro features.")
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def load_fundamentals() -> pd.DataFrame:
    path = PROCESSED_DIR / "fundamentals_daily.csv"
    if not path.exists():
        print("  WARNING: fundamentals_daily.csv not found. Skipping fundamental features.")
        return pd.DataFrame()
    fundamentals = pd.read_csv(path, parse_dates=["Date"])
    expected = ["Date", "Ticker"] + FUNDAMENTAL_FEATURES

    # Backward compatibility: previous fundamentals format was wide.
    if "Ticker" not in fundamentals.columns:
        print("  WARNING: fundamentals_daily.csv is in old wide format. Skipping fundamentals.")
        return pd.DataFrame()

    for col in FUNDAMENTAL_FEATURES:
        if col not in fundamentals.columns:
            fundamentals[col] = np.nan

    fundamentals = fundamentals[expected].copy()
    fundamentals.sort_values(["Date", "Ticker"], inplace=True)
    return fundamentals


def load_sentiment() -> pd.DataFrame:
    path = PROCESSED_DIR / "news_sentiment.csv"
    if not path.exists():
        print("  WARNING: news_sentiment.csv not found. Skipping sentiment features.")
        return pd.DataFrame()
    sentiment = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    # Backward compatibility with historical extract naming.
    if "M&M_sentiment" in sentiment.columns and "MM_sentiment" not in sentiment.columns:
        sentiment = sentiment.rename(columns={"M&M_sentiment": "MM_sentiment"})
    elif "M&M_sentiment" in sentiment.columns and "MM_sentiment" in sentiment.columns:
        sentiment["MM_sentiment"] = sentiment["MM_sentiment"].fillna(sentiment["M&M_sentiment"])
        sentiment = sentiment.drop(columns=["M&M_sentiment"])

    return sentiment


# ─── Per-Ticker Feature Engineering ──────────────────────────────────────────

def build_ticker_features(
    ticker_name: str,
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    fundamentals: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature set for one ticker."""
    df = panel[panel["Ticker"] == ticker_name].copy()
    df = df.set_index("Date").sort_index()

    # Technical indicators
    df = add_technical_features(df)
    df["ticker"] = ticker_name

    # Merge macro (already lagged in fetch_macro_data.py)
    if not macro.empty:
        df = df.join(macro, how="left")

    # Merge fundamentals (already lagged 45 days in fetch_fundamentals.py)
    if not fundamentals.empty:
        fund_ticker = fundamentals[fundamentals["Ticker"] == ticker_name].copy()
        if not fund_ticker.empty:
            fund_ticker = fund_ticker.set_index("Date")[FUNDAMENTAL_FEATURES]
            df = df.join(fund_ticker, how="left")

    # Merge sentiment (already lagged 1 day in fetch_sentiment.py)
    if not sentiment.empty:
        sent_col = f"{ticker_name}_sentiment"
        if sent_col in sentiment.columns:
            df = df.join(sentiment[[sent_col]], how="left")

    return df


# ─── Robust Scaling ───────────────────────────────────────────────────────────

EXCLUDE_FROM_SCALING = ["ticker", "target", "Date", "Ticker"]

def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """
    Fit RobustScaler on train, transform both train and test.
    Returns scaled DataFrames and the fitted scaler.
    """
    feature_cols = [c for c in train_df.columns if c not in EXCLUDE_FROM_SCALING]

    scaler = RobustScaler()
    train_scaled = train_df.copy()
    test_scaled  = test_df.copy()

    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols].fillna(0))
    test_scaled[feature_cols]  = scaler.transform(test_df[feature_cols].fillna(0))

    return train_scaled, test_scaled, scaler


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 5 – Feature Engineering")
    print("=" * 60)

    panel        = load_panel()
    macro        = load_macro()
    fundamentals = load_fundamentals()
    sentiment    = load_sentiment()

    all_frames = []
    for ticker in TICKERS:
        print(f"\nBuilding features for {ticker} ...")
        if ticker not in panel["Ticker"].values:
            print(f"  WARNING: {ticker} not in panel. Skipping.")
            continue

        feat_df = build_ticker_features(ticker, panel, macro, fundamentals, sentiment)
        all_frames.append(feat_df)
        print(f"  Shape: {feat_df.shape}")

    if not all_frames:
        print("ERROR: No feature frames built.")
        return

    full_df = pd.concat(all_frames, axis=0).sort_index()
    print(f"\nFull feature matrix: {full_df.shape}")

    # Drop rows with NaN targets (e.g., last row per ticker)
    full_df = full_df.dropna(subset=["target"])

    # Train / Forward-test split
    train_df = full_df[full_df.index < FORWARD_TEST_START]
    test_df  = full_df[full_df.index >= FORWARD_TEST_START]

    # Scale
    train_scaled, test_scaled, _ = scale_features(train_df, test_df)

    # Save
    train_scaled.to_parquet(PROCESSED_DIR / "features_train.parquet")
    test_scaled.to_parquet(PROCESSED_DIR / "features_forward_test.parquet")

    print(f"\nTrain:        {train_scaled.shape}")
    print(f"Forward-test: {test_scaled.shape}")
    print(f"Saved → {PROCESSED_DIR}/features_train.parquet")
    print(f"Saved → {PROCESSED_DIR}/features_forward_test.parquet")
    print("\nDone.")


if __name__ == "__main__":
    main()
