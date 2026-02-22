# """
# fetch_macro_data.py
# ===================
# Fetches macro-economic indicators for the Indian equity universe:

#   1. USD-INR exchange rate          → Yahoo Finance  (INR=X)
#   2. Crude Oil (Brent)              → Yahoo Finance  (BZ=F)
#   3. India 10-Year Bond Yield       → Yahoo Finance  (^IN10YT=RR)  -- if unavailable,
#                                        falls back to a manual CSV template
#   4. CPI / Inflation (monthly)      → FRED API  (INDCPIALLMINMEI) -- free, no key
#                                        Fallback: RBI bulk CSV

# All series are:
#   • Resampled to daily frequency (forward-fill for monthly series)
#   • Lagged by 1 business day to eliminate look-ahead bias

# Usage:
#     python fetch_macro_data.py
#     python fetch_macro_data.py --fred-key YOUR_KEY   (optional, for higher rate limits)
# """

# import argparse
# import os
# import warnings
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from pathlib import Path
# import requests

# warnings.filterwarnings("ignore")

# # ─── Configuration ────────────────────────────────────────────────────────────

# START_DATE = "2020-01-01"
# END_DATE   = "2025-12-31"

# PROCESSED_DIR = Path("data/processed")
# RAW_MACRO_DIR = Path("data/raw/macro")
# PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
# RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

# # Yahoo Finance tickers for macro series
# YF_MACRO = {
#     "USDINR":    "INR=X",        # USD/INR spot
#     "BRENT_OIL": "BZ=F",         # Brent crude futures
#     # "INDIA_10Y": "^IN10YT=RR",   # India 10-yr bond yield (may not always work)
#     "INDIA_10Y": "^INDIAVIX",   # India 10-yr bond yield (may not always work)
#     "NIFTY50":   "^NSEI",        # Nifty 50 index (useful market-level feature)
#     "VIX_INDIA": "^NSEBANK",     # Bank Nifty as proxy for market stress
# }

# # FRED series IDs (free, no key required for bulk CSV downloads)
# FRED_SERIES = {
#     "INDIA_CPI": "INDCPIALLMINMEI",  # India CPI All Items
#     "INR_USD":   "DEXINUS",          # INR per USD (complement to yfinance)
# }

# # ─── Helpers ──────────────────────────────────────────────────────────────────

# def fetch_yf_series(name: str, yf_ticker: str) -> pd.Series:
#     """Download a single Yahoo Finance series, return Close as Series."""
#     print(f"  Fetching {name} ({yf_ticker}) from Yahoo Finance ...")
#     try:
#         df = yf.download(yf_ticker, start=START_DATE, end=END_DATE,
#                          auto_adjust=True, progress=False)
#         if df.empty:
#             print(f"    WARNING: Empty response for {yf_ticker}")
#             return pd.Series(dtype=float, name=name)

#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = df.columns.get_level_values(0)

#         s = df["Close"].rename(name)
#         s.index = pd.to_datetime(s.index)
#         return s
#     except Exception as e:
#         print(f"    ERROR: {e}")
#         return pd.Series(dtype=float, name=name)


# def fetch_fred_csv(series_id: str, name: str) -> pd.Series:
#     url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
#     print(f"  Fetching {name} ({series_id}) from FRED ...")
#     try:
#         df = pd.read_csv(url, header=0)
#         df.columns = df.columns.str.strip()
#         date_col = df.columns[0]
#         val_col  = df.columns[1]
#         df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
#         df = df.dropna(subset=[date_col]).set_index(date_col)
#         s = df[val_col].replace(".", np.nan).astype(float).rename(name)
#         s.index = pd.to_datetime(s.index)
#         s = s[(s.index >= START_DATE) & (s.index <= END_DATE)]
#         return s
#     except Exception as e:
#         print(f"    ERROR fetching FRED series {series_id}: {e}")
#         return pd.Series(dtype=float, name=name)


# def resample_to_daily(s: pd.Series, method: str = "ffill") -> pd.Series:
#     """Reindex to calendar days and forward-fill (for monthly/weekly series)."""
#     daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
#     s = s.reindex(daily_idx)
#     if method == "ffill":
#         s = s.ffill()
#     return s


# def lag_series(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
#     """
#     Shift all columns forward by `lag` business days.
#     This ensures data available at time T is only used as feature at T+lag,
#     eliminating look-ahead bias.
#     """
#     return df.shift(lag)


# # ─── Main ─────────────────────────────────────────────────────────────────────

# def main():
#     print("=" * 60)
#     print("Step 2 – Macro Data Ingestion")
#     print("=" * 60)

#     frames = []

#     # 1. Yahoo Finance macro series
#     for name, ticker in YF_MACRO.items():
#         s = fetch_yf_series(name, ticker)
#         if not s.empty:
#             s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", header=True)
#             frames.append(s)

#     # 2. FRED series
#     for name, series_id in FRED_SERIES.items():
#         s = fetch_fred_csv(series_id, name)
#         if not s.empty:
#             s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", header=True)
#             frames.append(s)

#     if not frames:
#         print("\nNo macro data fetched. Check network connectivity.")
#         return

#     # 3. Merge into a single daily panel
#     print("\nAligning all series to daily frequency ...")
#     macro_panel = pd.DataFrame()

#     for s in frames:
#         s_daily = resample_to_daily(s, method="ffill")
#         macro_panel[s.name] = s_daily

#     # Keep only business days (Mon–Fri)
#     macro_panel = macro_panel[macro_panel.index.dayofweek < 5]
#     macro_panel.index.name = "Date"

#     # 4. Compute derived features before lagging
#     if "USDINR" in macro_panel.columns:
#         macro_panel["USDINR_Return"] = macro_panel["USDINR"].pct_change()

#     if "BRENT_OIL" in macro_panel.columns:
#         macro_panel["OIL_Return"] = macro_panel["BRENT_OIL"].pct_change()

#     if "INDIA_10Y" in macro_panel.columns:
#         macro_panel["YIELD_CHANGE"] = macro_panel["INDIA_10Y"].diff()

#     # 5. Lag by 1 day to prevent look-ahead
#     macro_lagged = lag_series(macro_panel, lag=1)
#     macro_lagged.columns = [f"{c}_lag1" for c in macro_lagged.columns]

#     # 6. Save
#     macro_panel.to_csv(PROCESSED_DIR / "macro_raw_daily.csv")
#     macro_lagged.to_csv(PROCESSED_DIR / "macro_daily.csv")

#     print(f"\nRaw macro panel:    {macro_panel.shape}")
#     print(f"Lagged macro panel: {macro_lagged.shape}")
#     print(f"Saved → {PROCESSED_DIR}/macro_daily.csv")
#     print("\nDone.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--fred-key", default=None, help="Optional FRED API key")
#     args = parser.parse_args()
#     main()

"""
fetch_macro_data.py
===================
Fetches macro-economic indicators for Indian equities.

Sources:
  - Yahoo Finance (FX, Oil, Yields, Index, VIX)
  - FRED (CPI)

All series:
  • Resampled to BUSINESS-DAY frequency
  • Log-return transformed where appropriate
  • Rolling features added (5d, 21d, vol, z-score)
  • Lagged by 1 business day (no look-ahead bias)
  • Deterministic column ordering (Git-safe)
"""

# (Your entire commented section remains unchanged above this point)

"""
fetch_macro_data.py
===================
Fetches macro-economic indicators for Indian equities.

Sources:
  - Yahoo Finance (FX, Oil, Yields, Index, VIX)
  - FRED (CPI)

All series:
  • Resampled to BUSINESS-DAY frequency
  • Log-return transformed where appropriate
  • Rolling features added (5d, 21d, vol, z-score)
  • Lagged by 1 business day (no look-ahead bias)
  • Deterministic column ordering (Git-safe)
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

PROCESSED_DIR = Path("data/processed")
RAW_MACRO_DIR = Path("data/raw/macro")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

YF_MACRO = {
    "USDINR": "INR=X",
    "BRENT_OIL": "BZ=F",
    "INDIA_10Y": "^IN10YT=RR",
    "NIFTY50": "^NSEI",
    "VIX_INDIA": "^INDIAVIX",
}

FRED_SERIES = {
    "INDIA_CPI": "INDCPIALLMINMEI"
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fetch_yf_series(name, ticker):
    print(f"Fetching {name} ({ticker})...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        print(f"WARNING: {ticker} returned empty.")
        return pd.Series(dtype=float)

    # Handle MultiIndex columns (new yfinance behavior)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        print(f"WARNING: Close column missing for {ticker}")
        return pd.Series(dtype=float)

    s = df["Close"]
    s.name = name
    s.index = pd.to_datetime(s.index)

    return s


def fetch_fred_series(series_id, name):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Fetching {name} from FRED...")

    df = pd.read_csv(url)
    df.columns = ["Date", name]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna().set_index("Date")

    s = df[name].replace(".", np.nan).astype(float)
    s = s[(s.index >= START_DATE) & (s.index <= END_DATE)]
    return s


def business_daily_index():
    return pd.date_range(start=START_DATE, end=END_DATE, freq="B")


def log_return(series):
    return np.log(series / series.shift(1))


def rolling_zscore(series, window=252):
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return (series - mean) / std


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Macro Data Ingestion (Stable Version)")
    print("=" * 60)

    frames = []

    # Fetch Yahoo Finance data
    for name, ticker in YF_MACRO.items():
        s = fetch_yf_series(name, ticker)
        if not s.empty:
            s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", float_format="%.6f")
            frames.append(s)

    # Fetch FRED CPI
    for name, series_id in FRED_SERIES.items():
        s = fetch_fred_series(series_id, name)
        if not s.empty:
            s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", float_format="%.6f")
            frames.append(s)

    if not frames:
        print("No macro data fetched.")
        return

    # Align to business-day index
    idx = business_daily_index()
    macro = pd.DataFrame(index=idx)

    for s in frames:
        s = s.reindex(idx).ffill()
        macro[s.name] = s

    # ─────────────────────────────────────────
    # Feature Engineering
    # ─────────────────────────────────────────

    # USDINR features
    if "USDINR" in macro:
        macro["USDINR_ret_1d"]  = log_return(macro["USDINR"])
        macro["USDINR_ret_5d"]  = np.log(macro["USDINR"] / macro["USDINR"].shift(5))
        macro["USDINR_vol_21d"] = macro["USDINR_ret_1d"].rolling(21).std()
        macro["USDINR_z_252d"]  = rolling_zscore(macro["USDINR"])

    # Oil features
    if "BRENT_OIL" in macro:
        macro["OIL_ret_1d"]  = log_return(macro["BRENT_OIL"])
        macro["OIL_ret_5d"]  = np.log(macro["BRENT_OIL"] / macro["BRENT_OIL"].shift(5))
        macro["OIL_vol_21d"] = macro["OIL_ret_1d"].rolling(21).std()
        macro["OIL_z_252d"]  = rolling_zscore(macro["BRENT_OIL"])

    # Yield features
    if "INDIA_10Y" in macro:
        macro["YIELD_change_1d"] = macro["INDIA_10Y"].diff()
        macro["YIELD_vol_21d"]   = macro["YIELD_change_1d"].rolling(21).std()

    # CPI feature
    if "INDIA_CPI" in macro:
        macro["CPI_inflation"] = log_return(macro["INDIA_CPI"])

    macro = macro.sort_index()
    macro_raw = macro.copy()

    # Lag by 1 business day (critical for no look-ahead bias)
    macro_lagged = macro.shift(1)
    macro_lagged.columns = [f"{c}_lag1" for c in macro_lagged.columns]

    # Deterministic column ordering
    macro_raw = macro_raw.reindex(sorted(macro_raw.columns), axis=1)
    macro_lagged = macro_lagged.reindex(sorted(macro_lagged.columns), axis=1)

    # Save
    macro_raw.to_csv(
        PROCESSED_DIR / "macro_raw_daily.csv",
        float_format="%.6f"
    )

    macro_lagged.to_csv(
        PROCESSED_DIR / "macro_daily.csv",
        float_format="%.6f"
    )

    print(f"Raw macro shape:    {macro_raw.shape}")
    print(f"Lagged macro shape: {macro_lagged.shape}")
    print("Saved successfully.")


if __name__ == "__main__":
    main()