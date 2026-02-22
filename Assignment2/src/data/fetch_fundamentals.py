"""
fetch_fundamentals.py
=====================
Scrapes quarterly fundamental metrics for 6 Indian stocks from Screener.in.

No login required — Screener.in serves full HTML publicly.

Metrics extracted per quarter:
  • EPS          — from Quarterly Results table (direct)
  • Revenue      — from Quarterly Results table (Sales row)
  • EBITDA_Margin— from Quarterly Results table (OPM % row)
  • Debt_Equity  — CALCULATED from Balance Sheet (Borrowings ÷ (Equity Capital + Reserves))
  • Promoter_Holding — from Shareholding Pattern table
  • ROE          — from annual P&L summary block (forward-filled quarterly)
  • PE           — CALCULATED externally: daily_price / trailing_12m_EPS
                   (Screener only shows a live P/E in the header, not historically per quarter)

All quarterly values are forward-filled to daily frequency and lagged by 45 days
(reporting_lag_days) to approximate the announcement date and avoid look-ahead bias.

Usage:
    python fetch_fundamentals.py
    python fetch_fundamentals.py --refetch

Output:
    data/processed/fundamentals_daily.csv
    Columns: Date, Ticker, EPS, Revenue, EBITDA_Margin, Debt_Equity,
             Promoter_Holding, ROE, Trailing_EPS_4Q, PE
"""

import argparse
import re
import time
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

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

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

# Days after quarter-end to treat data as available (avoids look-ahead bias)
REPORTING_LAG_DAYS = 45

RAW_FUND_DIR  = Path("data/raw/fundamentals")
PROCESSED_DIR = Path("data/processed")
RAW_FUND_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy column matching."""
    text = str(text).replace("\u00a0", " ").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Strip commas, %, currency symbols; coerce to float."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})
        .pipe(pd.to_numeric, errors="coerce")
    )


def parse_screener_date(label: str) -> pd.Timestamp:
    """
    Convert Screener quarter labels to pandas Timestamps.
    Formats seen: 'Sep 2024', 'Mar 2023', 'Jun 2022', 'Dec 2021'
    """
    try:
        # Use month-end as period end; month-start would leak ~1 month.
        dt = pd.to_datetime(label.strip(), format="%b %Y")
        return (dt + pd.offsets.MonthEnd(0)).normalize()
    except Exception:
        return pd.NaT


def find_row(df: pd.DataFrame, *keywords) -> pd.Series | None:
    """
    Search a DataFrame (first column = row labels) for a row whose label
    contains ALL given keywords. Returns that row as a Series, or None.
    """
    label_col = df.columns[0]
    for _, row in df.iterrows():
        label = normalize(str(row[label_col]))
        if all(kw in label for kw in keywords):
            return row
    return None


def table_to_series(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    """
    Given a raw HTML table and a matched row, return a Series
    indexed by parsed quarter dates with numeric values.
    """
    label_col = df.columns[0]
    date_cols = [c for c in df.columns if c != label_col]
    dates = [parse_screener_date(c) for c in date_cols]
    values = [coerce_numeric(pd.Series([row[c]])).iloc[0] for c in date_cols]
    s = pd.Series(values, index=dates, name=row[label_col])
    s = s[s.index.notna()].sort_index()
    return s


# ─── Screener Fetcher ─────────────────────────────────────────────────────────

def fetch_html(ticker: str) -> str | None:
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    print(f"  Fetching {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 429:
            print("    Rate limited — sleeping 60s ...")
            time.sleep(60)
        else:
            print(f"    HTTP {status}: {e}")
    except Exception as e:
        print(f"    Fetch error: {e}")
    return None


def identify_tables(html: str) -> dict[str, pd.DataFrame]:
    """
    Parse all HTML tables and identify them by their section heading.
    Screener page structure:
      h2: "Quarterly Results"   → quarterly P&L
      h2: "Profit & Loss"       → annual P&L (has ROE summary)
      h2: "Balance Sheet"       → annual balance sheet
      h2: "Shareholding Pattern"→ quarterly shareholding
    Returns dict keyed by normalized heading name.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    result = {}

    # Walk through every h2 and grab the first table after it
    for h2 in soup.find_all("h2"):
        heading = normalize(h2.get_text())
        table = h2.find_next("table")
        if table is None:
            continue
        try:
            df = pd.read_html(StringIO(str(table)))[0]
            result[heading] = df
        except Exception:
            continue

    return result


# ─── Per-table Extraction ─────────────────────────────────────────────────────

def extract_quarterly_pl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract EPS, Revenue (Sales), EBITDA_Margin from the Quarterly Results table.
    Returns DataFrame indexed by quarter-end date.
    """
    out = {}

    # Sales / Revenue — row label contains "Sales" (first column)
    row = find_row(df, "sales")
    if row is not None:
        out["Revenue"] = table_to_series(df, row)
    else:
        print("    WARNING: 'Sales' row not found in Quarterly Results")

    # OPM % → EBITDA Margin
    row = find_row(df, "opm")
    if row is not None:
        out["EBITDA_Margin"] = table_to_series(df, row)
    else:
        print("    WARNING: 'OPM %' row not found in Quarterly Results")

    # EPS
    row = find_row(df, "eps")
    if row is not None:
        out["EPS"] = table_to_series(df, row)
    else:
        print("    WARNING: 'EPS' row not found in Quarterly Results")

    if not out:
        return pd.DataFrame()

    result = pd.DataFrame(out)
    result.index.name = "Quarter"
    return result


def extract_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Debt/Equity from Balance Sheet table.
    D/E = Borrowings / (Equity Capital + Reserves)
    Returns annual DataFrame indexed by fiscal year-end date.
    """
    borr_row = find_row(df, "borrowings")
    eq_row   = find_row(df, "equity", "capital")
    res_row  = find_row(df, "reserves")

    if borr_row is None or eq_row is None or res_row is None:
        print("    WARNING: Balance Sheet rows missing for D/E calculation")
        return pd.DataFrame()

    borrowings = table_to_series(df, borr_row)
    equity_cap = table_to_series(df, eq_row)
    reserves   = table_to_series(df, res_row)

    net_worth  = equity_cap.add(reserves, fill_value=0)
    de_ratio   = borrowings.div(net_worth)
    de_ratio.name = "Debt_Equity"

    result = de_ratio.to_frame()
    result.index.name = "Quarter"
    return result


def extract_shareholding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Promoter Holding % from the Shareholding Pattern table.
    Screener shows it as the 'Promoters +' row.
    """
    row = find_row(df, "promoter")
    if row is None:
        print("    WARNING: 'Promoters' row not found in Shareholding Pattern")
        return pd.DataFrame()

    s = table_to_series(df, row)
    s.name = "Promoter_Holding"
    return s.to_frame()


def extract_roe_from_pl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Screener's P&L page has a small 'Return on Equity' summary block
    with 10Y / 5Y / 3Y / Last Year values, but NOT per quarter.

    Instead, we can calculate an approximate annual ROE from the P&L table:
      ROE = Net Profit / (Equity Capital + Reserves)  [from Balance Sheet]

    Here we extract Net Profit from the annual P&L table and return it
    so it can be combined with balance sheet data downstream.
    Falls back to NaN series if rows are missing.
    """
    row = find_row(df, "net profit")
    if row is None:
        print("    WARNING: 'Net Profit' row not found in Profit & Loss table")
        return pd.DataFrame()

    s = table_to_series(df, row)
    s.name = "Net_Profit_Annual"
    return s.to_frame()


# ─── Align to Daily ───────────────────────────────────────────────────────────

def align_to_daily(
    quarterly_df: pd.DataFrame,
    lag_days: int = REPORTING_LAG_DAYS,
) -> pd.DataFrame:
    """
    Forward-fill quarterly fundamentals onto a daily business-day index.

    Each quarterly observation is shifted forward by `lag_days` to represent
    the earliest date the market could have known the data. This prevents
    look-ahead bias when merging with daily price data.

    Parameters
    ----------
    quarterly_df : DataFrame indexed by quarter-end (or fiscal-year-end) dates
    lag_days     : Approximate days from period-end to public announcement
    """
    if quarterly_df.empty:
        return pd.DataFrame()

    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")

    df = quarterly_df.copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[df.index.notna()]

    # Apply the reporting lag: data "becomes available" lag_days after period end
    df.index = df.index + pd.Timedelta(days=lag_days)

    # Reindex to daily business day calendar, forward-filling
    daily = df.reindex(daily_idx, method="ffill")
    daily.index.name = "Date"
    return daily


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def scrape_ticker(name: str, screener_ticker: str, refetch: bool) -> pd.DataFrame:
    """
    Full pipeline for one ticker: fetch → parse → extract → align to daily.
    Returns a daily DataFrame with all fundamental columns.
    """
    cache_html = RAW_FUND_DIR / f"{screener_ticker}_page.html"

    # Load from cache if available and refetch not forced
    html = None
    if not refetch and cache_html.exists():
        print(f"  Loading cached HTML: {cache_html}")
        html = cache_html.read_text(encoding="utf-8")

    if html is None:
        html = fetch_html(screener_ticker)
        if html:
            cache_html.write_text(html, encoding="utf-8")
            print(f"  HTML cached to {cache_html}")

    if html is None:
        print(f"  FAILED to obtain HTML for {name}. Returning NaN placeholder.")
        return _placeholder(name)

    # Identify all tables by their section heading
    try:
        tables = identify_tables(html)
    except ImportError:
        print("  ERROR: 'beautifulsoup4' not installed. Run: pip install beautifulsoup4 lxml")
        return _placeholder(name)

    if not tables:
        print(f"  No tables found for {name}")
        return _placeholder(name)

    print(f"  Tables found: {list(tables.keys())}")

    # ── Extract each section ──────────────────────────────────────────────

    # 1. Quarterly Results → EPS, Revenue, EBITDA_Margin
    q_key = next((k for k in tables if "quarterly" in k), None)
    quarterly_pl = pd.DataFrame()
    if q_key:
        quarterly_pl = extract_quarterly_pl(tables[q_key])
        quarterly_pl.to_csv(RAW_FUND_DIR / f"{screener_ticker}_quarterly_pl.csv")
    else:
        print("  WARNING: Quarterly Results table not found")

    # 2. Balance Sheet → Debt_Equity
    bs_key = next((k for k in tables if "balance" in k), None)
    balance_sheet = pd.DataFrame()
    if bs_key:
        balance_sheet = extract_balance_sheet(tables[bs_key])
        balance_sheet.to_csv(RAW_FUND_DIR / f"{screener_ticker}_balance_sheet.csv")
    else:
        print("  WARNING: Balance Sheet table not found")

    # 3. Shareholding Pattern → Promoter_Holding
    sh_key = next((k for k in tables if "shareholding" in k), None)
    shareholding = pd.DataFrame()
    if sh_key:
        shareholding = extract_shareholding(tables[sh_key])
        shareholding.to_csv(RAW_FUND_DIR / f"{screener_ticker}_shareholding.csv")
    else:
        print("  WARNING: Shareholding table not found")

    # 4. Annual P&L → Net Profit (for ROE computation)
    pl_key = next((k for k in tables if "profit" in k and "loss" in k), None)
    annual_pl = pd.DataFrame()
    if pl_key:
        annual_pl = extract_roe_from_pl(tables[pl_key])

    # ── Compute ROE from annual P&L + Balance Sheet ───────────────────────
    roe_series = pd.DataFrame()
    if not annual_pl.empty and not balance_sheet.empty:
        # Align on common annual dates (fiscal year-end)
        # Balance sheet has Equity Capital + Reserves already parsed;
        # re-extract raw rows to compute net worth
        bs_df = tables.get(bs_key, pd.DataFrame())
        eq_row  = find_row(bs_df, "equity", "capital")
        res_row = find_row(bs_df, "reserves")
        if eq_row is not None and res_row is not None:
            net_worth = (
                table_to_series(bs_df, eq_row)
                .add(table_to_series(bs_df, res_row), fill_value=0)
            )
            net_profit = annual_pl["Net_Profit_Annual"]
            # Align on common index
            common_idx = net_profit.index.intersection(net_worth.index)
            if len(common_idx) > 0:
                roe = net_profit.loc[common_idx] / net_worth.loc[common_idx] * 100
                roe.name = "ROE"
                roe_series = roe.to_frame()

    # ── Align every metric to daily ───────────────────────────────────────
    dfs_to_merge = []

    for df, label in [
        (quarterly_pl,  "quarterly_pl"),
        (balance_sheet, "balance_sheet"),
        (shareholding,  "shareholding"),
        (roe_series,    "roe"),
    ]:
        if not df.empty:
            daily = align_to_daily(df, lag_days=REPORTING_LAG_DAYS)
            dfs_to_merge.append(daily)

    if not dfs_to_merge:
        return _placeholder(name)

    combined = dfs_to_merge[0]
    for other in dfs_to_merge[1:]:
        combined = combined.join(other, how="left", rsuffix="_dup")
        dup_cols = [c for c in combined.columns if c.endswith("_dup")]
        combined.drop(columns=dup_cols, inplace=True)

    # ── Compute trailing 12-month EPS and P/E placeholder ─────────────────
    # Trailing EPS = sum of last 4 quarterly EPS values (TTM)
    # P/E cannot be computed here without daily price; we store Trailing_EPS_4Q
    # for the caller to compute: PE = daily_close / Trailing_EPS_4Q
    if "EPS" in combined.columns:
        # Re-derive from quarterly (not forward-filled) to sum 4 quarters properly
        if not quarterly_pl.empty and "EPS" in quarterly_pl.columns:
            eps_q = quarterly_pl["EPS"].dropna().sort_index()
            # Rolling 4-quarter sum on the quarterly series
            ttm_eps = eps_q.rolling(4, min_periods=1).sum()
            ttm_daily = align_to_daily(ttm_eps.to_frame(name="Trailing_EPS_4Q"))
            combined = combined.join(ttm_daily, how="left")
        combined["PE"] = np.nan  # Placeholder — fill with: price / Trailing_EPS_4Q

    combined["Ticker"] = name

    all_cols = ["Ticker", "EPS", "Revenue", "EBITDA_Margin",
                "Debt_Equity", "Promoter_Holding", "ROE",
                "Trailing_EPS_4Q", "PE"]
    for col in all_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    result = combined[all_cols].reset_index()
    non_null = int(result.drop(columns=["Ticker", "PE"]).notna().sum().sum())
    print(f"  Done: shape={result.shape} | non-null cells (excl PE): {non_null}")
    return result


def _placeholder(name: str) -> pd.DataFrame:
    """Return a correctly-shaped all-NaN DataFrame for a failed ticker."""
    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    cols = ["EPS", "Revenue", "EBITDA_Margin", "Debt_Equity",
            "Promoter_Holding", "ROE", "Trailing_EPS_4Q", "PE"]
    df = pd.DataFrame(np.nan, index=daily_idx, columns=cols)
    df.index.name = "Date"
    df["Ticker"] = name
    return df.reset_index()


# ─── PE Computation ───────────────────────────────────────────────────────────

def compute_pe(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the PE column by joining the fundamentals panel with daily price data.

    Looks for daily price data at:
        data/processed/panel_daily.csv   ← preferred (your OHLCV pipeline output)
        data/raw/prices_daily.csv         ← fallback

    Expected price file schema (any of these column names work):
        Date, Ticker, Close
        Date, Ticker, close
        Date, Symbol, Close

    If no price file is found, PE remains NaN and a clear warning is printed.

    PE formula:
        PE = daily_close_price / Trailing_EPS_4Q

    where Trailing_EPS_4Q is the rolling sum of the last 4 quarterly EPS values,
    already forward-filled with a 45-day reporting lag (no look-ahead bias).

    Returns the panel DataFrame with PE filled where possible.
    """
    price_candidates = [
        PROCESSED_DIR / "panel_daily.csv",
        # Path("data/raw/prices_daily.csv"),
        Path("panel_daily.csv"),
    ]

    price_df = None
    for path in price_candidates:
        if path.exists():
            print(f"\n  Loading price data from: {path}")
            price_df = pd.read_csv(path)
            break

    if price_df is None:
        print("\n  WARNING: No price file found. PE column will remain NaN.")
        print("  Searched locations:")
        for p in price_candidates:
            print(f"    {p.resolve()}")
        print("  Place your daily OHLCV file at: data/processed/panel_daily.csv")
        print("  Required columns: Date, Ticker, Close")
        return panel

    # ── Normalise price DataFrame ─────────────────────────────────────────
    price_df.columns = [c.strip().lower() for c in price_df.columns]

    # Accept common column name variants
    col_map = {}
    for c in price_df.columns:
        if c in ("date",):                          col_map[c] = "Date"
        elif c in ("ticker", "symbol", "stock"):    col_map[c] = "Ticker"
        elif c in ("close", "adj close", "adj_close", "price"): col_map[c] = "Close"
    price_df.rename(columns=col_map, inplace=True)

    missing = [c for c in ("Date", "Ticker", "Close") if c not in price_df.columns]
    if missing:
        print(f"\n  WARNING: Price file missing columns: {missing}. PE will remain NaN.")
        return panel

    price_df["Date"]   = pd.to_datetime(price_df["Date"])
    price_df["Close"]  = pd.to_numeric(price_df["Close"], errors="coerce")
    price_df           = price_df[["Date", "Ticker", "Close"]].dropna()

    # Align ticker names: price file may use NSE symbols (e.g. "M&M") while
    # our panel uses the internal name key (e.g. "MM"). Build a reverse map.
    reverse_ticker = {v: k for k, v in SCREENER_TICKERS.items()}
    price_df["Ticker"] = price_df["Ticker"].replace(reverse_ticker)

    # ── Merge and compute PE ──────────────────────────────────────────────
    panel = panel.copy()
    panel["Date"] = pd.to_datetime(panel["Date"])

    merged = panel.merge(
        price_df.rename(columns={"Close": "_close"}),
        on=["Date", "Ticker"],
        how="left",
    )

    # Guard: avoid division by zero or negative TTM EPS (meaningless P/E)
    valid_eps = merged["Trailing_EPS_4Q"].replace(0, np.nan)
    valid_eps = valid_eps.where(valid_eps > 0, np.nan)

    merged["PE"] = merged["_close"] / valid_eps
    merged.drop(columns=["_close"], inplace=True)

    filled = merged["PE"].notna().sum()
    total  = len(merged)
    print(f"  PE computed: {filled}/{total} rows filled "
          f"({100*filled/total:.1f}%)")

    return merged


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main(refetch: bool = False):
    print("=" * 60)
    print("Fundamental Data Ingestion — Screener.in")
    print("=" * 60)

    # ── Step 1: Scrape and align fundamentals for every ticker ────────────
    all_daily = []
    for name, screener_ticker in SCREENER_TICKERS.items():
        print(f"\n{'─'*50}")
        print(f"Ticker: {name}  (Screener ID: {screener_ticker})")
        df = scrape_ticker(name, screener_ticker, refetch=refetch)
        all_daily.append(df)
        time.sleep(3)   # polite crawl delay between requests

    panel = pd.concat(all_daily, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel.sort_values(["Date", "Ticker"], inplace=True)

    # ── Step 2: Compute PE by joining with daily price data ───────────────
    # This happens here, in this script, not "somewhere else".
    # PE = daily_close_price / Trailing_EPS_4Q (rolling 4-quarter EPS sum)
    # Trailing_EPS_4Q is already lagged by 45 days to avoid look-ahead bias.
    print(f"\n{'─'*50}")
    print("Computing PE from price data ...")
    panel = compute_pe(panel)

    # ── Step 3: Save final panel ──────────────────────────────────────────
    out_path = PROCESSED_DIR / "fundamentals_daily.csv"
    panel.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Fundamentals panel saved: {panel.shape}")
    print(f"→ {out_path}")
    print()
    print("Columns in output:")
    for col in panel.columns:
        n_valid = int(panel[col].notna().sum()) if panel[col].dtype != object else "—"
        print(f"  {col:<22} non-null: {n_valid}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch fundamentals from Screener.in")
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="Re-download HTML even if a cached copy exists.",
    )
    args = parser.parse_args()
    main(refetch=args.refetch)
