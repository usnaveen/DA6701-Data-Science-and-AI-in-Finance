"""
data.py: Data & Statistics
======================================
What this module does:
  - Download daily adjusted closing prices for 5 tickers via yfinance (Jan 2014 - Dec 2023)
  - Compute daily log returns
  - Compute annualised expected returns (x252), annualised std devs, and 5x5 covariance matrix
  - Build the 20-year annual savings schedule (₹20,000/month growing at 4% p.a.)

Returns:
  mu       : np.ndarray, shape (5,)   — annualised expected returns per ticker
  cov      : np.ndarray, shape (5, 5) — annualised covariance matrix
  savings  : np.ndarray, shape (20,)  — annual contributions in ₹ for years 1-20
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# CONSTANTS
TICKERS = ["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ITC.NS"]
START_DATE = "2014-01-01"
END_DATE = "2023-12-31"
TRADING_DAYS = 252  # annualisation factor

MONTHLY_SAVINGS_INITIAL = 20_000  # ₹ per month
SAVINGS_GROWTH_RATE = 0.04  # 4% annual growth
HORIZON_YEARS = 20


# 1. DATA DOWNLOAD
def download_prices(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """
    Download daily adjusted closing prices from yfinance.

    Parameters
    ----------
    tickers : list[str]  — Yahoo Finance ticker symbols
    start   : str        — start date (YYYY-MM-DD)
    end     : str        — end date   (YYYY-MM-DD)

    Returns
    -------
    prices : pd.DataFrame
        Date-indexed DataFrame of adjusted closing prices,
        columns = tickers, no missing values (forward-filled then dropped).
    """
    print(f"Downloading prices for: {tickers}")
    print(f"Period : {start}  ->  {end}")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex when >1 ticker; extract 'Close'
    if isinstance(raw.columns, pd.MultiIndex): # type: ignore
        prices = raw["Close"][tickers] # type: ignore
    else:
        prices = raw[["Close"]] # type: ignore
        prices.columns = tickers

    # Basic cleaning — forward-fill up to 5 consecutive NaNs, then drop remaining
    prices = prices.ffill(limit=5).dropna()

    print(
        f"Downloaded {len(prices)} trading days  |  "
        f"{prices.shape[1]} tickers  |  "
        f"{prices.index[0].date()} to {prices.index[-1].date()}"
    )
    return prices


# 2. RETURN CALCULATIONS
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price series.

    log_return_t = ln(P_t / P_{t-1})

    Returns
    -------
    log_returns : pd.DataFrame  — same shape as prices minus first row
    """
    log_returns = (prices / prices.shift(1)).apply(lambda x: np.log(x)).dropna()
    return log_returns


def compute_annualised_stats(
    log_returns: pd.DataFrame, trading_days: int = TRADING_DAYS
):
    """
    Compute annualised expected returns, standard deviations, and covariance matrix
    from daily log returns.

    Annualisation:
      - Expected return : mean_daily x trading_days
      - Std deviation   : std_daily  x sqrt(trading_days)
      - Covariance      : cov_daily  x trading_days

    Parameters
    ----------
    log_returns  : pd.DataFrame — daily log returns
    trading_days : int          — number of trading days per year (default 252)

    Returns
    -------
    mu  : np.ndarray, shape (n,)   — annualised expected returns
    std : np.ndarray, shape (n,)   — annualised standard deviations
    cov : np.ndarray, shape (n, n) — annualised covariance matrix
    """
    mean_daily = log_returns.mean()
    cov_daily = log_returns.cov()

    mu = (mean_daily * trading_days).values  # shape (5,)
    std = (log_returns.std() * np.sqrt(trading_days)).values  # shape (5,)
    cov = (cov_daily * trading_days).values  # shape (5, 5)

    return mu, std, cov


# 3. SAVINGS SCHEDULE
def build_savings_schedule(
    monthly_initial: float = MONTHLY_SAVINGS_INITIAL,
    growth_rate: float = SAVINGS_GROWTH_RATE,
    horizon: int = HORIZON_YEARS,
) -> np.ndarray:
    """
    Build the annual savings contribution array over the investment horizon.

    The client saves ₹20,000/month in Year 1, growing at 4% p.a.
    Annual contribution in year t  =  monthly_initial x 12 x (1 + growth_rate)^(t-1)

    Parameters
    ----------
    monthly_initial : float — initial monthly savings in ₹
    growth_rate     : float — annual growth rate of savings (e.g. 0.04)
    horizon         : int   — number of years (default 20)

    Returns
    -------
    savings : np.ndarray, shape (horizon,)
        savings[0] = contribution at end of Year 1
        savings[1] = contribution at end of Year 2
        ...
        savings[19] = contribution at end of Year 20
    """
    annual_base = monthly_initial * 12  # ₹2,40,000 in Year 1
    savings = np.array([annual_base * (1 + growth_rate) ** t for t in range(horizon)])
    return savings


# 4. SUMMARY / DISPLAY HELPERS
def print_stats_table(tickers, mu, std, cov):
    """Pretty-print the annualised return and risk stats."""
    print("\n" + "=" * 55)
    print(f"{'Ticker':<15} {'Ann. Return':>12} {'Ann. Std Dev':>13}")
    print("-" * 55)
    for i, t in enumerate(tickers):
        print(f"{t:<15} {mu[i]:>11.2%} {std[i]:>12.2%}")
    print("=" * 55)

    print("\nAnnualised Covariance Matrix:")
    cov_df = pd.DataFrame(cov, index=tickers, columns=tickers)
    print(cov_df.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nCorrelation Matrix (derived from cov):")
    corr = np.corrcoef(cov)  # correlation from cov matrix
    corr_df = pd.DataFrame(corr, index=tickers, columns=tickers)
    print(corr_df.to_string(float_format=lambda x: f"{x:.4f}"))


def print_savings_schedule(savings):
    """Display the first 5 and last 5 years of the savings schedule."""
    print("\nSavings Schedule (₹ per year):")
    print(f"  Year  1 : ₹{savings[0]:>12,.0f}")
    print(f"  Year  2 : ₹{savings[1]:>12,.0f}")
    print(f"  Year  3 : ₹{savings[2]:>12,.0f}")
    print("  ...      (growing at 4% p.a.)")
    print(f"  Year 18 : ₹{savings[17]:>12,.0f}")
    print(f"  Year 19 : ₹{savings[18]:>12,.0f}")
    print(f"  Year 20 : ₹{savings[19]:>12,.0f}")
    print(f"\n  Total contributions over 20 years: ₹{savings.sum():>,.0f}")


# 5. TOP-LEVEL CONVENIENCE FUNCTIONS
def get_returns(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    """Download prices and return the daily log-returns DataFrame."""
    prices = download_prices(tickers, start, end)
    return compute_log_returns(prices)


def get_stats(log_returns: pd.DataFrame | None = None, tickers=TICKERS):
    """
    Compute and return (mu, std, cov) from log returns.
    If log_returns is None, downloads data first.
    """
    if log_returns is None:
        log_returns = get_returns(tickers)
    return compute_annualised_stats(log_returns)


def get_cov_matrix(log_returns: pd.DataFrame | None = None, tickers=TICKERS) -> np.ndarray:
    """Return just the annualised covariance matrix."""
    _, _, cov = get_stats(log_returns, tickers)
    return cov


def get_savings() -> np.ndarray:
    """Return the 20-year annual savings schedule array."""
    return build_savings_schedule()


if __name__ == "__main__":
    print("=== Data & Statistics ===")

    # Step 1 — Download prices
    prices = download_prices()
    log_returns = compute_log_returns(prices)
    print(f"\nLog-returns shape : {log_returns.shape}")
    print(f"Date range        : {log_returns.index[0].date()} -> {log_returns.index[-1].date()}")
    print(f"\nFirst 3 rows of log-returns:\n{log_returns.head(3).to_string()}")

    # Step 2 — Annualised statistics
    mu, std, cov = compute_annualised_stats(log_returns)
    print_stats_table(TICKERS, mu, std, cov)

    # Step 3 — Savings schedule
    savings = build_savings_schedule()
    print_savings_schedule(savings)

    print('=== Final Confirmation ===\n')
    print(f"mu      shape : {mu.shape}   dtype: {mu.dtype}")
    print(f"cov     shape : {cov.shape}  dtype: {cov.dtype}")
    print(f"savings shape : {savings.shape}  dtype: {savings.dtype}")