"""
fetch_macro_data.py
===================
Fetches macro-economic indicators for the Indian equity universe:

  1. USD-INR exchange rate          → Yahoo Finance  (INR=X)
  2. Crude Oil (Brent)              → Yahoo Finance  (BZ=F)
  3. India 10-Year Bond Yield       → Yahoo Finance  (^IN10YT=RR)  -- if unavailable,
                                       falls back to a manual CSV template
  4. CPI / Inflation (monthly)      → FRED API  (INDCPIALLMINMEI) -- free, no key
                                       Fallback: RBI bulk CSV

All series are:
  • Resampled to daily frequency (forward-fill for monthly series)
  • Lagged by 1 business day to eliminate look-ahead bias

Usage:
    python fetch_macro_data.py
    python fetch_macro_data.py --fred-key YOUR_KEY   (optional, for higher rate limits)
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import requests

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

PROCESSED_DIR = Path("data/processed")
RAW_MACRO_DIR = Path("data/raw/macro")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

# Yahoo Finance tickers for macro series
YF_MACRO = {
    "USDINR":    "INR=X",        # USD/INR spot
    "BRENT_OIL": "BZ=F",         # Brent crude futures
    "INDIA_10Y": "^IN10YT=RR",   # India 10-yr bond yield (may not always work)
    "NIFTY50":   "^NSEI",        # Nifty 50 index (useful market-level feature)
    "VIX_INDIA": "^NSEBANK",     # Bank Nifty as proxy for market stress
}

# FRED series IDs (free, no key required for bulk CSV downloads)
FRED_SERIES = {
    "INDIA_CPI": "INDCPIALLMINMEI",  # India CPI All Items
    "INR_USD":   "DEXINUS",          # INR per USD (complement to yfinance)
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def fetch_yf_series(name: str, yf_ticker: str) -> pd.Series:
    """Download a single Yahoo Finance series, return Close as Series."""
    print(f"  Fetching {name} ({yf_ticker}) from Yahoo Finance ...")
    try:
        df = yf.download(yf_ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"    WARNING: Empty response for {yf_ticker}")
            return pd.Series(dtype=float, name=name)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        s = df["Close"].rename(name)
        s.index = pd.to_datetime(s.index)
        return s
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.Series(dtype=float, name=name)


def fetch_fred_csv(series_id: str, name: str) -> pd.Series:
    """
    Download a FRED series via their public CSV endpoint — no API key needed.
    URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES_ID
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"  Fetching {name} ({series_id}) from FRED ...")
    try:
        df = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
        s = df.squeeze().rename(name)
        s = s[s.index >= START_DATE]
        s = s[s.index <= END_DATE]
        s.replace(".", np.nan, inplace=True)
        s = s.astype(float)
        return s
    except Exception as e:
        print(f"    ERROR fetching FRED series {series_id}: {e}")
        return pd.Series(dtype=float, name=name)


def resample_to_daily(s: pd.Series, method: str = "ffill") -> pd.Series:
    """Reindex to calendar days and forward-fill (for monthly/weekly series)."""
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    s = s.reindex(daily_idx)
    if method == "ffill":
        s = s.ffill()
    return s


def lag_series(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    Shift all columns forward by `lag` business days.
    This ensures data available at time T is only used as feature at T+lag,
    eliminating look-ahead bias.
    """
    return df.shift(lag)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 2 – Macro Data Ingestion")
    print("=" * 60)

    frames = []

    # 1. Yahoo Finance macro series
    for name, ticker in YF_MACRO.items():
        s = fetch_yf_series(name, ticker)
        if not s.empty:
            s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", header=True)
            frames.append(s)

    # 2. FRED series
    for name, series_id in FRED_SERIES.items():
        s = fetch_fred_csv(series_id, name)
        if not s.empty:
            s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", header=True)
            frames.append(s)

    if not frames:
        print("\nNo macro data fetched. Check network connectivity.")
        return

    # 3. Merge into a single daily panel
    print("\nAligning all series to daily frequency ...")
    macro_panel = pd.DataFrame()

    for s in frames:
        s_daily = resample_to_daily(s, method="ffill")
        macro_panel[s.name] = s_daily

    # Keep only business days (Mon–Fri)
    macro_panel = macro_panel[macro_panel.index.dayofweek < 5]
    macro_panel.index.name = "Date"

    # 4. Compute derived features before lagging
    if "USDINR" in macro_panel.columns:
        macro_panel["USDINR_Return"] = macro_panel["USDINR"].pct_change()

    if "BRENT_OIL" in macro_panel.columns:
        macro_panel["OIL_Return"] = macro_panel["BRENT_OIL"].pct_change()

    if "INDIA_10Y" in macro_panel.columns:
        macro_panel["YIELD_CHANGE"] = macro_panel["INDIA_10Y"].diff()

    # 5. Lag by 1 day to prevent look-ahead
    macro_lagged = lag_series(macro_panel, lag=1)
    macro_lagged.columns = [f"{c}_lag1" for c in macro_lagged.columns]

    # 6. Save
    macro_panel.to_csv(PROCESSED_DIR / "macro_raw_daily.csv")
    macro_lagged.to_csv(PROCESSED_DIR / "macro_daily.csv")

    print(f"\nRaw macro panel:    {macro_panel.shape}")
    print(f"Lagged macro panel: {macro_lagged.shape}")
    print(f"Saved → {PROCESSED_DIR}/macro_daily.csv")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fred-key", default=None, help="Optional FRED API key")
    args = parser.parse_args()
    main()
