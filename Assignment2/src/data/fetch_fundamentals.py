"""
fetch_fundamentals.py
=====================
Scrapes quarterly fundamental metrics for the 6 Indian stocks from Screener.in.

Screener.in is free to use without login for most data.
The exported CSV link pattern:  https://www.screener.in/company/{TICKER}/consolidated/

Metrics extracted:
  • P/E ratio
  • Debt / Equity
  • Return on Equity (ROE)
  • EPS (Earnings per Share)
  • Revenue / Sales
  • EBITDA Margin
  • Promoter Holding %

All quarterly values are forward-filled to daily frequency and lagged by one
quarter (i.e., we only use Q1 data starting from the Q1 announcement date + 45 days
to approximate reporting delay and avoid look-ahead bias).

Usage:
    python fetch_fundamentals.py

Notes:
  - If scraping fails (rate limiting, CAPTCHA), manually download the CSV from
    Screener.in: Company Page → "Export to Excel" → parse the sheets.
  - Alternative free source: financialmodelingprep.com (free tier: 250 calls/day)
    Set ENABLE_FMP=True below and add your free API key.
"""

import time
import warnings
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from io import StringIO

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

SCREENER_TICKERS = {
    "RELIANCE":   "RELIANCE",
    "HDFCBANK":   "HDFCBANK",
    "INFY":       "INFY",
    "MM":         "M&M",
    "BHARTIARTL": "BHARTIARTL",
    "HUL":        "HINDUNILVR",
}

# Optional: FinancialModelingPrep free API (250 calls/day)
ENABLE_FMP   = False
FMP_API_KEY  = "YOUR_FREE_FMP_KEY"   # register free at financialmodelingprep.com
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

RAW_FUND_DIR  = Path("data/raw/fundamentals")
PROCESSED_DIR = Path("data/processed")
RAW_FUND_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ─── Screener.in Scraper ──────────────────────────────────────────────────────

def scrape_screener(ticker: str) -> pd.DataFrame:
    """
    Attempt to fetch the consolidated quarterly P&L table from Screener.in.
    Returns a DataFrame indexed by quarter-end date.
    """
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    print(f"  Scraping {ticker} from Screener.in ...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        # Parse all HTML tables; Screener has Quarterly Results table
        tables = pd.read_html(StringIO(resp.text))

        # The first large table is typically the Quarterly Results
        qdf = None
        for t in tables:
            if t.shape[1] > 4 and t.shape[0] > 3:
                qdf = t
                break

        if qdf is None:
            print(f"    WARNING: No suitable table found for {ticker}")
            return pd.DataFrame()

        qdf = qdf.set_index(qdf.columns[0]).T
        qdf.index = pd.to_datetime(qdf.index, errors="coerce")
        qdf = qdf[~qdf.index.isna()]
        qdf = qdf.apply(pd.to_numeric, errors="coerce")

        # Save raw
        qdf.to_csv(RAW_FUND_DIR / f"{ticker}_quarterly_raw.csv")
        return qdf

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"    Rate limited. Sleeping 30 s ...")
            time.sleep(30)
        else:
            print(f"    HTTP error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


# ─── FMP API (alternative) ────────────────────────────────────────────────────

def fetch_fmp_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Fetch quarterly income statement & ratios from FinancialModelingPrep (free tier).
    NSE tickers on FMP: RELIANCE.NS, HDFCBANK.NS, etc.
    """
    fmp_ticker = f"{ticker}.NS"
    frames = []

    endpoints = {
        "ratios":    f"{FMP_BASE_URL}/ratios/{fmp_ticker}?period=quarter&limit=24&apikey={FMP_API_KEY}",
        "income":    f"{FMP_BASE_URL}/income-statement/{fmp_ticker}?period=quarter&limit=24&apikey={FMP_API_KEY}",
    }

    for ep_name, url in endpoints.items():
        try:
            resp = requests.get(url, timeout=20)
            data = resp.json()
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                frames.append(df)
        except Exception as e:
            print(f"    FMP {ep_name} error for {ticker}: {e}")

    if frames:
        combined = pd.concat(frames, axis=1)
        combined.to_csv(RAW_FUND_DIR / f"{ticker}_fmp_raw.csv")
        return combined
    return pd.DataFrame()


# ─── Alignment to Daily ───────────────────────────────────────────────────────

def align_to_daily(
    quarterly_df: pd.DataFrame,
    ticker_name: str,
    reporting_lag_days: int = 45,
) -> pd.DataFrame:
    """
    Forward-fill quarterly fundamentals onto a daily business-day index.

    Applies a `reporting_lag_days` offset to each quarterly date to approximate
    when the data actually becomes public (e.g., Q1 April results → available ~May 15).
    This is critical to avoid look-ahead bias.

    Parameters
    ----------
    quarterly_df     : DataFrame with quarterly dates as index
    ticker_name      : Used to prefix columns
    reporting_lag_days: Days after quarter-end before data is treated as available
    """
    if quarterly_df.empty:
        return pd.DataFrame()

    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")

    # Shift each observation forward by the reporting lag
    quarterly_df.index = quarterly_df.index + pd.Timedelta(days=reporting_lag_days)

    # Reindex and forward-fill
    daily_df = quarterly_df.reindex(daily_idx, method="ffill")
    daily_df.index.name = "Date"

    # Prefix columns with ticker
    daily_df.columns = [f"{ticker_name}_{c}" for c in daily_df.columns]
    return daily_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 3 – Fundamental Data Ingestion")
    print("=" * 60)

    all_daily = []

    for name, screener_ticker in SCREENER_TICKERS.items():
        print(f"\n{'─'*40}")
        print(f"Ticker: {name}")

        if ENABLE_FMP:
            raw_q = fetch_fmp_fundamentals(screener_ticker)
        else:
            raw_q = scrape_screener(screener_ticker)
            time.sleep(3)   # polite crawl delay

        if raw_q.empty:
            print(f"  No data for {name}. Will use placeholder NaNs.")
            # Create placeholder so pipeline doesn't break
            daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
            placeholder = pd.DataFrame(index=daily_idx)
            for col in ["PE", "DebtEquity", "ROE", "EPS"]:
                placeholder[f"{name}_{col}"] = np.nan
            all_daily.append(placeholder)
            continue

        daily_df = align_to_daily(raw_q, name, reporting_lag_days=45)
        all_daily.append(daily_df)
        print(f"  Aligned to daily: {daily_df.shape}")

    if all_daily:
        fundamentals_panel = pd.concat(all_daily, axis=1)
        fundamentals_panel.to_csv(PROCESSED_DIR / "fundamentals_daily.csv")
        print(f"\nFundamentals panel saved: {fundamentals_panel.shape}")
        print(f"→ {PROCESSED_DIR}/fundamentals_daily.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
