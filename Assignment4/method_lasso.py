"""method_lasso.py — Lasso-based portfolio replication (Method 1).

Strategy
--------
Fit a non-negative Lasso regression  b_t = Σ w_i · r_{i,t}  on the training
set.  The L1 penalty drives most weights to exactly zero, leaving a sparse
portfolio.  Sweeping alpha (regularisation strength) across a log-scale grid
produces portfolios of varying cardinality k; we evaluate each on val and
holdout sets and keep the best alpha per k.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from eval import evaluate, load_splits


def run_lasso_sweep(
    R_train: pd.DataFrame,
    b_train: pd.Series,
    R_val: pd.DataFrame,
    b_val: pd.Series,
    R_hold: pd.DataFrame,
    b_hold: pd.Series,
    alpha_grid: np.ndarray | None = None,
    min_k: int = 5,
) -> pd.DataFrame:
    """Sweep alpha and return a DataFrame of evaluation results.

    Parameters
    ----------
    alpha_grid : array-like, optional
        Log-spaced alpha values.  Defaults to np.logspace(-4, -1, 40).
    min_k : int
        Skip alphas that select fewer than this many stocks.

    Returns
    -------
    pd.DataFrame with columns: label, k, TE_val, IR_val, TE_hold, IR_hold, alpha
        One row per (alpha, k) combination, filtered to one row per k
        (the alpha that minimises TE_val for each k).
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, -1, 40)

    X_train = R_train.fillna(0.0).values
    y_train = b_train.values

    records = []
    for alpha in alpha_grid:
        lasso = Lasso(alpha=alpha, positive=True, max_iter=10_000, tol=1e-4)
        lasso.fit(X_train, y_train)

        selected = {
            ticker: w
            for ticker, w in zip(R_train.columns, lasso.coef_)
            if w > 1e-6
        }
        k = len(selected)
        if k < min_k:
            continue

        total = sum(selected.values())
        weights = {t: w / total for t, w in selected.items()}

        res = evaluate(weights, R_val, b_val, R_hold, b_hold, label="Lasso")
        res["alpha"] = float(alpha)
        res["weights"] = weights
        records.append(res)

    if not records:
        return pd.DataFrame()

    results_df = pd.DataFrame(records)
    best_per_k = (
        results_df.sort_values("TE_val")
        .drop_duplicates(subset="k", keep="first")
        .sort_values("k")
        .reset_index(drop=True)
    )
    return best_per_k


def get_lasso_weights_for_k(
    results_df: pd.DataFrame, k: int = 50
) -> dict:
    """Extract the weight dict for the closest k in results_df."""
    if results_df.empty:
        raise ValueError("results_df is empty — run run_lasso_sweep first.")
    row = results_df.iloc[(results_df["k"] - k).abs().argsort()[:1]]
    return row["weights"].iloc[0]


if __name__ == "__main__":
    R_train, b_train, R_val, b_val, R_hold, b_hold = load_splits()
    print(f"Running Lasso sweep on {R_train.shape[1]} stocks ...")
    results = run_lasso_sweep(R_train, b_train, R_val, b_val, R_hold, b_hold)
    print(results[["k", "TE_val", "IR_val", "TE_hold", "IR_hold", "alpha"]].to_string())
    results.drop(columns="weights").to_csv("data/lasso_results.csv", index=False)
    print("Saved to data/lasso_results.csv")
