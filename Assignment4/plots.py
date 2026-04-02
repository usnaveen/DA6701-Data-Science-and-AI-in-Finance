"""plots.py — Visualisation helpers for Assignment IV.

All functions return a matplotlib Figure so callers can either display
inline (Jupyter) or save to disk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from eval import portfolio_returns


# ── colour palette ────────────────────────────────────────────────────────────
_COLOURS = {
    "Lasso":       "#2196F3",
    "Autoencoder": "#FF9800",
    "GA":          "#4CAF50",
    "Benchmark":   "#9E9E9E",
}


def _method_colour(label: str) -> str:
    for key, colour in _COLOURS.items():
        if key.lower() in label.lower():
            return colour
    return "#607D8B"


# ── 1. TE vs k ────────────────────────────────────────────────────────────────
def plot_te_vs_k(
    results_frames: dict,
    title: str = "Sparsity vs Tracking Error",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Line chart of TE (val + holdout) vs k for multiple methods.

    Parameters
    ----------
    results_frames : dict
        {method_label: pd.DataFrame}  — each df must have columns
        [k, TE_val, TE_hold].
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, df in results_frames.items():
        if df.empty:
            continue
        colour = _method_colour(label)
        df_sorted = df.sort_values("k")
        ax.plot(df_sorted["k"], df_sorted["TE_val"],
                color=colour, linestyle="-", linewidth=2,
                label=f"{label} (val)")
        ax.plot(df_sorted["k"], df_sorted["TE_hold"],
                color=colour, linestyle="--", linewidth=2,
                label=f"{label} (holdout)")

    ax.set_xlabel("Number of stocks  k", fontsize=12)
    ax.set_ylabel("Annualised Tracking Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    fig.tight_layout()
    return fig


# ── 2. Cumulative returns ─────────────────────────────────────────────────────
def plot_cumulative_returns(
    weights_per_method: dict,
    R: pd.DataFrame,
    bench: pd.Series,
    title: str = "Cumulative Returns",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Overlay cumulative return of each portfolio vs benchmark.

    Parameters
    ----------
    weights_per_method : dict
        {method_label: {ticker: weight}}
    R : pd.DataFrame
        Return matrix for the evaluation period.
    bench : pd.Series
        Benchmark return series.
    """
    fig, ax = plt.subplots(figsize=figsize)

    bench_cumret = (1 + bench).cumprod()
    ax.plot(bench_cumret.index, bench_cumret.values,
            color=_COLOURS["Benchmark"], linewidth=2.5,
            linestyle=":", label="S&P 500 (benchmark)")

    for label, weights in weights_per_method.items():
        p = portfolio_returns(weights, R)
        cumret = (1 + p).cumprod()
        colour = _method_colour(label)
        ax.plot(cumret.index, cumret.values,
                color=colour, linewidth=2, label=label)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (1 = start)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    fig.tight_layout()
    return fig


# ── 3. Sector drift ───────────────────────────────────────────────────────────
def plot_sector_drift(
    portfolio_weights: dict,
    sector_map_path: str = "data/sector_map.csv",
    portfolio_label: str = "Portfolio (k=50)",
    figsize: tuple = (13, 5),
) -> plt.Figure:
    """Bar chart comparing portfolio sector allocation vs S&P 500 baseline.

    Parameters
    ----------
    portfolio_weights : dict
        {ticker: weight} for the portfolio you want to inspect.
    sector_map_path : str
        Path to sector_map.csv with columns [ticker, sector].
    """
    sector_map = pd.read_csv(sector_map_path, index_col="ticker")

    sp500_weights = sector_map["sector"].value_counts(normalize=True)

    selected = list(portfolio_weights.keys())
    available = [t for t in selected if t in sector_map.index]
    if not available:
        raise ValueError("None of the portfolio tickers found in sector_map.")

    port_sectors = sector_map.loc[available, "sector"].value_counts(normalize=True)

    drift_df = pd.DataFrame(
        {"S&P 500": sp500_weights, portfolio_label: port_sectors}
    ).fillna(0).sort_values("S&P 500", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(drift_df))
    width = 0.38
    ax.bar(x - width / 2, drift_df["S&P 500"], width,
           color="#9E9E9E", label="S&P 500")
    ax.bar(x + width / 2, drift_df[portfolio_label], width,
           color="#2196F3", label=portfolio_label)
    ax.set_xticks(x)
    ax.set_xticklabels(drift_df.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Sector Weight", fontsize=12)
    ax.set_title(
        f"Sector Drift: {portfolio_label} vs S&P 500", fontsize=14, fontweight="bold"
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    return fig


# ── 4. Results summary table (pretty-print) ───────────────────────────────────
def make_results_table(rows: list) -> pd.DataFrame:
    """Build the assignment summary table from a list of result dicts.

    Each dict should have: Method, k, TE_val, TE_hold, IR_val, IR_hold.
    """
    df = pd.DataFrame(rows)
    df = df[["Method", "k", "TE_val", "IR_val", "TE_hold", "IR_hold"]]
    df = df.rename(columns={
        "TE_val": "TE (Val)",
        "IR_val": "IR (Val)",
        "TE_hold": "TE (Holdout)",
        "IR_hold": "IR (Holdout)",
    })
    for col in ["TE (Val)", "TE (Holdout)"]:
        df[col] = df[col].map("{:.4f}".format)
    for col in ["IR (Val)", "IR (Holdout)"]:
        df[col] = df[col].map("{:.4f}".format)
    return df
