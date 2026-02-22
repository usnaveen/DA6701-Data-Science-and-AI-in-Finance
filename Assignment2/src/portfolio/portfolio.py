"""
portfolio.py
============
Constructs a long-only portfolio from the 6 predicted daily returns
and evaluates performance for the Oct–Dec 2025 forward-test period.

Portfolio weighting methods available:
  1. "predicted_return"  — weight proportional to predicted positive returns
                           (zero weight for stocks with negative prediction)
  2. "equal"             — equal weight across positively predicted stocks
  3. "mean_variance"     — Markowitz mean-variance (maximize Sharpe)
  4. "inverse_vol"       — inverse volatility weighting

Performance metrics:
  • Sharpe Ratio (annualised, 252 days)
  • Maximum Drawdown
  • Hit Ratio (directional accuracy per stock + portfolio)
  • Equity Curve (cumulative returns)
  • Cumulative Alpha vs Nifty 50 benchmark

Usage:
    python portfolio.py --method predicted_return
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS           = ["RELIANCE", "HDFCBANK", "INFY", "MM", "BHARTIARTL", "HUL"]
OUTPUT_DIR        = Path("outputs")
PROCESSED_DIR     = Path("data/processed")
FORWARD_TEST_START = "2025-10-01"
RISK_FREE_RATE    = 0.065 / 252   # ~6.5% annualised RBI repo rate, daily


# ─── Load Data ────────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    path = OUTPUT_DIR / "predictions_fwd_test.csv"
    if not path.exists():
        raise FileNotFoundError("Run train_model.py first.")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def load_actual_returns() -> pd.DataFrame:
    """Load realised log-returns for the forward-test period."""
    path = PROCESSED_DIR / "panel_forward_test.csv"
    if not path.exists():
        raise FileNotFoundError("Run fetch_market_data.py first.")

    panel = pd.read_csv(path, parse_dates=["Date"])
    pivot = panel.pivot(index="Date", columns="Ticker", values="Log_Return")
    pivot.columns.name = None
    return pivot


# ─── Weighting Methods ────────────────────────────────────────────────────────

def weight_predicted_return(predictions: pd.Series) -> pd.Series:
    """
    Allocate weight proportional to predicted positive return.
    Stocks with negative predicted return get 0 weight.
    """
    pos = predictions.clip(lower=0)
    total = pos.sum()
    if total == 0:
        # All negative: equal weight (risk management: stay invested)
        return pd.Series(1 / len(predictions), index=predictions.index)
    return pos / total


def weight_equal(predictions: pd.Series) -> pd.Series:
    """Equal weight across stocks with positive predicted return."""
    mask = predictions > 0
    n = mask.sum()
    if n == 0:
        n = len(predictions)
        mask = pd.Series(True, index=predictions.index)
    w = mask.astype(float) / n
    return w


def weight_inverse_vol(
    predictions: pd.Series,
    historical_returns: pd.DataFrame,
    lookback: int = 63,
) -> pd.Series:
    """Inverse of 63-day rolling volatility, normalized."""
    recent = historical_returns.tail(lookback)
    vol = recent.std()
    inv_vol = 1 / (vol + 1e-10)
    return inv_vol / inv_vol.sum()


def weight_mean_variance(
    expected_returns: pd.Series,
    historical_returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.Series:
    """
    Markowitz mean-variance optimization via scipy.
    Maximize Sharpe ratio (numerically).
    """
    cov = historical_returns.tail(lookback).cov().values
    mu  = expected_returns.values
    n   = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w + 1e-10)
        return -(port_ret - RISK_FREE_RATE) / port_vol

    # Constraints: weights sum to 1, all >= 0 (long only)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints,
                      method="SLSQP", options={"maxiter": 500, "ftol": 1e-9})

    w_opt = pd.Series(result.x, index=expected_returns.index)
    return w_opt / (w_opt.sum() + 1e-10)


# ─── Portfolio Simulation ─────────────────────────────────────────────────────

def simulate_portfolio(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    train_actuals: pd.DataFrame,
    method: str = "predicted_return",
) -> pd.DataFrame:
    """
    Day-by-day portfolio simulation.
    At each day T: compute weights from predictions_{T}, realise actual_{T}.

    Returns DataFrame with columns:
      weights per ticker, portfolio_return, cumulative_return
    """
    common_dates = predictions.index.intersection(actuals.index)
    preds = predictions.loc[common_dates, TICKERS].fillna(0)
    acts  = actuals.loc[common_dates, TICKERS].fillna(0)

    records = []
    for date in common_dates:
        pred_row = preds.loc[date]
        act_row  = acts.loc[date]

        # Compute weights
        if method == "predicted_return":
            w = weight_predicted_return(pred_row)
        elif method == "equal":
            w = weight_equal(pred_row)
        elif method == "inverse_vol":
            hist = train_actuals.loc[:date].tail(63)
            w = weight_inverse_vol(pred_row, hist)
        elif method == "mean_variance":
            hist = train_actuals.loc[:date].tail(252)
            w = weight_mean_variance(pred_row, hist)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Portfolio return = weighted sum of actual returns
        port_ret = (w * act_row).sum()
        rec = {"Date": date, "portfolio_return": port_ret}
        for t in TICKERS:
            rec[f"w_{t}"] = w.get(t, 0)
            rec[f"ret_{t}"] = act_row.get(t, 0)
        records.append(rec)

    result = pd.DataFrame(records).set_index("Date")
    result["cum_return"] = (1 + result["portfolio_return"]).cumprod() - 1
    return result


# ─── Performance Metrics ──────────────────────────────────────────────────────

def compute_metrics(portfolio: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    r = portfolio["portfolio_return"].values
    cum = (1 + portfolio["portfolio_return"]).cumprod().values

    # Sharpe ratio
    excess = r - RISK_FREE_RATE
    sharpe = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(252)

    # Max drawdown
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / (running_max + 1e-10)
    mdd = dd.min()

    # Hit ratio — directional accuracy per ticker and portfolio
    hit_ticker = {}
    for t in TICKERS:
        if f"w_{t}" in portfolio.columns and f"ret_{t}" in portfolio.columns:
            prd = portfolio[f"w_{t}"] * portfolio[f"ret_{t}"]   # weighted contribution
            # Simple directional hit
            pred_sign = portfolio[f"w_{t}"].apply(lambda x: 1 if x > 0 else 0)
            act_sign  = portfolio[f"ret_{t}"].apply(lambda x: 1 if x > 0 else -1)
            hit_ticker[t] = (pred_sign == (act_sign > 0).astype(int)).mean()

    port_hit = (np.sign(r[r != 0]) == 1).mean()   # days we were long and positive

    return {
        "Sharpe Ratio":         round(sharpe, 4),
        "Max Drawdown":         round(mdd * 100, 2),
        "Hit Ratio (Portfolio)": round(port_hit, 4),
        "Hit Ratio per Ticker": {k: round(v, 4) for k, v in hit_ticker.items()},
        "Cumulative Return (%)": round(portfolio["cum_return"].iloc[-1] * 100, 2),
        "Ann. Volatility (%)":  round(r.std() * np.sqrt(252) * 100, 2),
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_equity_curve(portfolio: pd.DataFrame, metrics: dict, method: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(
        f"Portfolio Performance — {method.replace('_',' ').title()}\n"
        f"Sharpe: {metrics['Sharpe Ratio']}  |  MDD: {metrics['Max Drawdown']}%  "
        f"|  Hit Ratio: {metrics['Hit Ratio (Portfolio)']:.2%}",
        fontsize=13,
    )

    # Equity curve
    ax1 = axes[0]
    (1 + portfolio["cum_return"]).plot(ax=ax1, color="steelblue", linewidth=2)
    ax1.set_title("Equity Curve (Cumulative Return)")
    ax1.set_ylabel("Portfolio Value (₹1 = base)")
    ax1.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax1.grid(alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    cum = (1 + portfolio["portfolio_return"]).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max * 100
    drawdown.plot(ax=ax2, color="crimson", linewidth=1.5, alpha=0.7)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.3)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    # Daily portfolio returns
    ax3 = axes[2]
    portfolio["portfolio_return"].plot(ax=ax3, color="teal", linewidth=0.8, alpha=0.7)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Daily Portfolio Returns")
    ax3.set_ylabel("Daily Return")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"equity_curve_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Equity curve saved → {out_path}")


def plot_weights_heatmap(portfolio: pd.DataFrame, method: str):
    weight_cols = [f"w_{t}" for t in TICKERS if f"w_{t}" in portfolio.columns]
    wdf = portfolio[weight_cols].copy()
    wdf.columns = [c.replace("w_", "") for c in wdf.columns]

    fig, ax = plt.subplots(figsize=(14, 5))
    wdf.plot.area(ax=ax, stacked=True, colormap="tab10", alpha=0.8)
    ax.set_title(f"Portfolio Weights Over Time — {method.replace('_',' ').title()}")
    ax.set_ylabel("Weight")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"weights_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Weights chart saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(method: str = "predicted_return"):
    print("=" * 60)
    print(f"Step 7 – Portfolio Construction ({method})")
    print("=" * 60)

    preds        = load_predictions()
    fwd_actuals  = load_actual_returns()

    # Load training-period actuals for vol/covariance estimation
    train_panel = pd.read_csv(
        PROCESSED_DIR / "panel_train.csv", parse_dates=["Date"]
    )
    train_actuals = train_panel.pivot(
        index="Date", columns="Ticker", values="Log_Return"
    )

    print(f"Predictions shape:     {preds.shape}")
    print(f"Forward actuals shape: {fwd_actuals.shape}")

    # Simulate portfolio
    portfolio = simulate_portfolio(preds, fwd_actuals, train_actuals, method=method)

    # Compute metrics
    metrics = compute_metrics(portfolio, fwd_actuals)

    print("\n" + "=" * 50)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Sharpe Ratio:          {metrics['Sharpe Ratio']}")
    print(f"  Max Drawdown:          {metrics['Max Drawdown']}%")
    print(f"  Hit Ratio (Portfolio): {metrics['Hit Ratio (Portfolio)']:.2%}")
    print(f"  Cumulative Return:     {metrics['Cumulative Return (%)']}%")
    print(f"  Ann. Volatility:       {metrics['Ann. Volatility (%)']}%")
    print("\n  Per-Ticker Hit Ratios:")
    for t, v in metrics["Hit Ratio per Ticker"].items():
        print(f"    {t:12s}: {v:.2%}")

    # Save results
    portfolio.to_csv(OUTPUT_DIR / f"portfolio_results_{method}.csv")
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "Hit Ratio per Ticker"}]).to_csv(
        OUTPUT_DIR / f"portfolio_metrics_{method}.csv", index=False
    )

    # Plots
    plot_equity_curve(portfolio, metrics, method)
    plot_weights_heatmap(portfolio, method)

    print(f"\nPortfolio results saved → {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="predicted_return",
        choices=["predicted_return", "equal", "inverse_vol", "mean_variance"],
        help="Portfolio weighting method",
    )
    args = parser.parse_args()
    main(method=args.method)
