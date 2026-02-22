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
    python fetch_fundamentals.py --refetch

Output schema (data/processed/fundamentals_daily.csv):
    Date, Ticker, PE, Debt_Equity, ROE, EPS, Revenue, EBITDA_Margin, Promoter_Holding

Notes:
  - If scraping fails (rate limiting, CAPTCHA), manually download the CSV from
    Screener.in: Company Page → "Export to Excel" → parse the sheets.
  - Alternative free source: financialmodelingprep.com (free tier: 250 calls/day)
    Set ENABLE_FMP=True below and add your free API key.
"""

import argparse
import re
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

FUNDAMENTAL_FEATURES = [
    "PE",
    "Debt_Equity",
    "ROE",
    "EPS",
    "Revenue",
    "EBITDA_Margin",
    "Promoter_Holding",
]

FUNDAMENTAL_ALIASES = {
    "PE": ["p e", "pe ratio", "price to earning", "price earnings"],
    "Debt_Equity": ["debt equity", "debt to equity", "debt / equity"],
    "ROE": ["roe", "return on equity"],
    "EPS": ["eps in rs", "eps", "earning per share"],
    "Revenue": ["revenue", "sales"],
    "EBITDA_Margin": ["ebitda margin", "opm", "operating margin"],
    "Promoter_Holding": ["promoter holding", "promoters holding", "promoter"],
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

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert mixed-format numeric columns (commas, %, blanks) to floats."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        cleaned = (
            out[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
            .replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})
        )
        out[col] = pd.to_numeric(cleaned, errors="coerce")
    return out


def normalize_label(label: str) -> str:
    text = str(label).replace("\u00a0", " ").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_standard_fundamentals(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only assignment-mandated fundamental metrics with a fixed schema.
    Missing metrics remain NaN.
    """
    if quarterly_df.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_FEATURES, index=quarterly_df.index)

    col_map = {col: normalize_label(col) for col in quarterly_df.columns}
    selected = pd.DataFrame(index=quarterly_df.index)

    for metric in FUNDAMENTAL_FEATURES:
        aliases = [normalize_label(a) for a in FUNDAMENTAL_ALIASES[metric]]
        matched_col = None
        for raw_col, norm_col in col_map.items():
            if any(alias in norm_col for alias in aliases):
                matched_col = raw_col
                break

        if matched_col is not None:
            selected[metric] = quarterly_df[matched_col]
        else:
            selected[metric] = np.nan

    return selected


def load_cached_quarterly(cache_paths: list[Path]) -> pd.DataFrame:
    """
    Load previously saved quarterly fundamentals (if present), so the pipeline
    can proceed even when fresh scraping fails.
    """
    for path in cache_paths:
        if not path.exists():
            continue
        try:
            qdf = pd.read_csv(path, index_col=0)
            qdf.index = pd.to_datetime(qdf.index, errors="coerce")
            qdf = qdf[~qdf.index.isna()].sort_index()
            qdf = qdf[~qdf.index.duplicated(keep="last")]
            qdf = coerce_numeric_columns(qdf)
            if not qdf.empty and qdf.notna().sum().sum() > 0:
                print(f"  Loaded cached quarterly data: {path}")
                return qdf
        except Exception as e:
            print(f"  WARNING: Could not load cache {path}: {e}")
    return pd.DataFrame()


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
        qdf = qdf[~qdf.index.isna()].sort_index()
        qdf = qdf[~qdf.index.duplicated(keep="last")]
        qdf = coerce_numeric_columns(qdf)

        # Save raw
        qdf.to_csv(RAW_FUND_DIR / f"{ticker}_quarterly_raw.csv")
        return qdf

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("    Rate limited. Sleeping 30 s ...")
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
    reporting_lag_days: Days after quarter-end before data is treated as available
    """
    if quarterly_df.empty:
        return pd.DataFrame()

    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    quarterly_df = quarterly_df.copy().sort_index()

    # Shift each observation forward by the reporting lag
    quarterly_df.index = quarterly_df.index + pd.Timedelta(days=reporting_lag_days)

    # Reindex and forward-fill
    daily_df = quarterly_df.reindex(daily_idx, method="ffill")
    daily_df.index.name = "Date"

    return daily_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(refetch: bool = False):
    print("=" * 60)
    print("Step 3 – Fundamental Data Ingestion")
    print("=" * 60)

    all_daily = []

    for name, screener_ticker in SCREENER_TICKERS.items():
        print(f"\n{'─'*40}")
        print(f"Ticker: {name}")

        cache_paths = [
            RAW_FUND_DIR / f"{screener_ticker}_quarterly_raw.csv",
            RAW_FUND_DIR / f"{name}_quarterly_raw.csv",
        ]

        # Prefer existing raw cache; fall back to network fetch only if needed.
        raw_q = pd.DataFrame()
        if not refetch:
            raw_q = load_cached_quarterly(cache_paths)

        if raw_q.empty:
            if ENABLE_FMP:
                raw_q = fetch_fmp_fundamentals(screener_ticker)
            else:
                raw_q = scrape_screener(screener_ticker)
                time.sleep(3)   # polite crawl delay

        # If fresh fetch failed but old raw exists, use old raw instead of NaN placeholders.
        if raw_q.empty:
            raw_q = load_cached_quarterly(cache_paths)

        if raw_q.empty:
            print(f"  No data for {name}. Will use placeholder NaNs.")
            # Keep schema fixed even when one ticker is missing.
            daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
            placeholder = pd.DataFrame(index=daily_idx, columns=FUNDAMENTAL_FEATURES, dtype=float)
            placeholder.index.name = "Date"
            placeholder["Ticker"] = name
            all_daily.append(placeholder.reset_index())
            continue

        standardized_q = extract_standard_fundamentals(raw_q)
        daily_df = align_to_daily(standardized_q, reporting_lag_days=45)
        daily_df["Ticker"] = name
        daily_df = daily_df.reset_index()
        daily_df = daily_df[["Date", "Ticker"] + FUNDAMENTAL_FEATURES]
        all_daily.append(daily_df)
        non_null = int(daily_df[FUNDAMENTAL_FEATURES].notna().sum().sum())
        print(f"  Aligned to daily: {daily_df.shape} | non-null cells: {non_null}")

    if all_daily:
        fundamentals_panel = pd.concat(all_daily, axis=0, ignore_index=True)
        fundamentals_panel["Date"] = pd.to_datetime(fundamentals_panel["Date"])
        fundamentals_panel.sort_values(["Date", "Ticker"], inplace=True)
        fundamentals_panel.to_csv(PROCESSED_DIR / "fundamentals_daily.csv", index=False)
        print(f"\nFundamentals panel saved: {fundamentals_panel.shape}")
        print(f"→ {PROCESSED_DIR}/fundamentals_daily.csv")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Ignore cached raw fundamentals and fetch fresh data.",
    )
    args = parser.parse_args()
    main(refetch=args.refetch)
