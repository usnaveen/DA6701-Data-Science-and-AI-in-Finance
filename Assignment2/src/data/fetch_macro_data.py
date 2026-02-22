"""
fetch_macro_data.py
===================
Fetches macro-economic indicators for Indian equities.

Sources:
  - Yahoo Finance (FX, Oil, Index, VIX)
  - FRED (CPI, India 10Y Bond Yield)

All series:
  • Resampled to BUSINESS-DAY frequency
  • Log-return transformed where appropriate
  • Rolling features added (5d, 21d, vol, z-score)
  • Lagged by 1 business day (no look-ahead bias)
  • Deterministic column ordering (Git-safe)
"""

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
    "USDINR":    "INR=X",
    "BRENT_OIL": "BZ=F",
    "NIFTY50":   "^NSEI",
    "VIX_INDIA": "^INDIAVIX",
}

FRED_SERIES = {
    "INDIA_CPI": "INDCPIALLMINMEI",   # monthly CPI, forward-filled to daily
    "INDIA_10Y": "INDIRLTLT01STM",    # India long-term interest rate (monthly)
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fetch_yf_series(name: str, ticker: str) -> pd.Series:
    print(f"Fetching {name} ({ticker})...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        print(f"  WARNING: {ticker} returned empty.")
        return pd.Series(dtype=float, name=name)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        print(f"  WARNING: Close column missing for {ticker}")
        return pd.Series(dtype=float, name=name)

    s = df["Close"].rename(name)
    s.index = pd.to_datetime(s.index)
    return s


def fetch_fred_series(series_id: str, name: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"Fetching {name} ({series_id}) from FRED...")

    try:
        df = pd.read_csv(url)
        df.columns = ["Date", name]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        s = df[name].replace(".", np.nan).astype(float)
        s = s[(s.index >= START_DATE) & (s.index <= END_DATE)]
        return s
    except Exception as e:
        print(f"  ERROR fetching FRED {series_id}: {e}")
        return pd.Series(dtype=float, name=name)


def load_cached_raw_series(name: str) -> pd.Series:
    """
    Load a previously cached raw macro file when online fetch fails.
    Expects file at data/raw/macro/{name}_raw.csv.
    """
    path = RAW_MACRO_DIR / f"{name}_raw.csv"
    if not path.exists():
        return pd.Series(dtype=float, name=name)

    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.Series(dtype=float, name=name)

        date_col = df.columns[0]
        val_col = df.columns[1] if len(df.columns) > 1 else name
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        s = pd.to_numeric(df[val_col], errors="coerce").rename(name)
        print(f"  Loaded cached raw series for {name}: {path}")
        return s
    except Exception as e:
        print(f"  WARNING: Failed reading cached raw series {path}: {e}")
        return pd.Series(dtype=float, name=name)


def business_daily_index() -> pd.DatetimeIndex:
    return pd.date_range(start=START_DATE, end=END_DATE, freq="B")


def log_return(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return (series - mean) / (std + 1e-10)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Macro Data Ingestion")
    print("=" * 60)

    frames = []

    # Yahoo Finance series
    for name, ticker in YF_MACRO.items():
        s = fetch_yf_series(name, ticker)
        if s.empty:
            s = load_cached_raw_series(name)
        if not s.empty:
            s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", float_format="%.6f")
            frames.append(s)

    # FRED series (CPI + 10Y yield)
    for name, series_id in FRED_SERIES.items():
        s = fetch_fred_series(series_id, name)
        if s.empty:
            s = load_cached_raw_series(name)
        if not s.empty:
            s.to_csv(RAW_MACRO_DIR / f"{name}_raw.csv", float_format="%.6f")
            frames.append(s)

    if not frames:
        print("No macro data fetched.")
        return

    # Align everything to business-day index (forward-fill monthly series)
    idx = business_daily_index()
    macro = pd.DataFrame(index=idx)
    for s in frames:
        macro[s.name] = s.reindex(idx).ffill()

    # ─────────────────────────────────────────
    # Derived Features
    # ─────────────────────────────────────────

    if "USDINR" in macro:
        macro["USDINR_ret_1d"]  = log_return(macro["USDINR"])
        macro["USDINR_ret_5d"]  = np.log(macro["USDINR"] / macro["USDINR"].shift(5))
        macro["USDINR_vol_21d"] = macro["USDINR_ret_1d"].rolling(21).std()
        macro["USDINR_z_252d"]  = rolling_zscore(macro["USDINR"])

    if "BRENT_OIL" in macro:
        macro["OIL_ret_1d"]  = log_return(macro["BRENT_OIL"])
        macro["OIL_ret_5d"]  = np.log(macro["BRENT_OIL"] / macro["BRENT_OIL"].shift(5))
        macro["OIL_vol_21d"] = macro["OIL_ret_1d"].rolling(21).std()
        macro["OIL_z_252d"]  = rolling_zscore(macro["BRENT_OIL"])

    if "INDIA_10Y" in macro:
        macro["YIELD_change_1d"] = macro["INDIA_10Y"].diff()
        macro["YIELD_vol_21d"]   = macro["YIELD_change_1d"].rolling(21).std()

    if "INDIA_CPI" in macro:
        macro["CPI_inflation"] = log_return(macro["INDIA_CPI"])

    if "NIFTY50" in macro:
        macro["NIFTY_ret_1d"]  = log_return(macro["NIFTY50"])
        macro["NIFTY_vol_21d"] = macro["NIFTY_ret_1d"].rolling(21).std()

    if "VIX_INDIA" in macro:
        macro["VIX_change_1d"] = macro["VIX_INDIA"].diff()
        macro["VIX_z_252d"]    = rolling_zscore(macro["VIX_INDIA"])

    macro = macro.sort_index()
    macro.index.name = "Date"
    macro_raw = macro.reindex(sorted(macro.columns), axis=1)

    # Lag by 1 business day — eliminates look-ahead bias
    macro_lagged = macro.shift(1)
    macro_lagged.index.name = "Date"
    macro_lagged.columns = [f"{c}_lag1" for c in macro_lagged.columns]
    macro_lagged = macro_lagged.reindex(sorted(macro_lagged.columns), axis=1)

    # Save
    macro_raw.to_csv(PROCESSED_DIR / "macro_raw_daily.csv", float_format="%.6f")
    macro_lagged.to_csv(PROCESSED_DIR / "macro_daily.csv", float_format="%.6f")

    print(f"\nRaw macro shape:    {macro_raw.shape}")
    print(f"Lagged macro shape: {macro_lagged.shape}")
    print(f"Columns: {list(macro_raw.columns)}")
    print("Saved successfully.")


if __name__ == "__main__":
    main()
