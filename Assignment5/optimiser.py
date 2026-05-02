"""
optimiser.py: Brute-Force Portfolio Optimiser
==============================================
What this module does:
  - Generate all valid discrete weight combinations from {0, 0.25, 0.5, 0.75, 1.0}
    where weights sum to exactly 1.0 across 5 securities (56 valid combinations)
  - For each combination, call simulate_portfolio() and record the success probability
  - Identify the single best portfolio for Sequence A and Sequence B separately

Returns:
  best_weights      : np.ndarray, shape (5,)  -- optimal portfolio weights
  best_success_prob : float                   -- highest success probability found
"""

from simulator import simulate_portfolio, compute_success_probability

import numpy as np
import pandas as pd
from itertools import product
import warnings

warnings.filterwarnings("ignore")


# CONSTANTS
DISCRETE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SECURITIES = 5
TERMINAL_GOAL = 15_000_000  # Rs 1.5 Crore
N_PATHS = 5000
WEIGHT_SUM_TOLERANCE = 1e-9  # floating point tolerance for sum == 1.0


# 1. COMBINATION GENERATION
def generate_valid_combinations() -> list:
    """
    Generate all discrete weight combinations that sum to exactly 1.0.

    Weights are drawn from {0, 0.25, 0.5, 0.75, 1.0} for each of the 5 securities.
    Total candidates: 5^5 = 3125. Valid (sum == 1.0): 56 combinations.

    Returns
    -------
    combos : list of np.ndarray, each shape (5,)
        All valid weight vectors summing to 1.0.
    """
    combos = []
    for weights in product(DISCRETE_WEIGHTS, repeat=N_SECURITIES):
        if abs(sum(weights) - 1.0) < WEIGHT_SUM_TOLERANCE:
            combos.append(np.array(weights))
    return combos


# 2. OPTIMISATION
def run_optimisation(
    mu: np.ndarray,
    cov: np.ndarray,
    savings: np.ndarray,
    goals: dict,
    n_paths: int = N_PATHS,
    seed: int | None = 42,
    verbose: bool = True,
) -> tuple:
    """
    Brute-force search over all valid discrete weight combinations.

    For each combination, runs a Monte Carlo simulation and records the
    probability of the final portfolio value at Year 20 meeting the terminal goal.

    Parameters
    ----------
    mu      : np.ndarray, shape (5,)   -- annualised expected returns from data.py
    cov     : np.ndarray, shape (5,5)  -- annualised covariance matrix from data.py
    savings : np.ndarray, shape (20,)  -- annual savings schedule from data.py
    goals   : dict {int: float}        -- full goal schedule including terminal year 20
    n_paths : int                      -- Monte Carlo paths per combination (default 5000)
    seed    : int | None               -- random seed for reproducibility
    verbose : bool                     -- print progress every 10 combinations

    Returns
    -------
    best_weights      : np.ndarray, shape (5,) -- weights of the optimal portfolio
    best_success_prob : float                  -- success probability of the optimal portfolio
    results_df        : pd.DataFrame           -- full results table, sorted by success prob
    """
    combos = generate_valid_combinations()
    total = len(combos)

    if verbose:
        print(f"Total valid combinations : {total}")
        print(f"Paths per combination    : {n_paths:,}")
        print(f"Terminal goal            : Rs {TERMINAL_GOAL:,.0f}")
        print("Running optimisation...\n")

    records = []

    for i, weights in enumerate(combos):
        final_values = simulate_portfolio(
            weights, mu, cov, savings, goals, n_paths=n_paths, seed=seed
        )
        prob = compute_success_probability(final_values, TERMINAL_GOAL)
        records.append(
            {
                "TCS.NS": weights[0],
                "HDFCBANK.NS": weights[1],
                "RELIANCE.NS": weights[2],
                "SUNPHARMA.NS": weights[3],
                "ITC.NS": weights[4],
                "success_prob": prob,
            }
        )

        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1:>3}/{total} combinations...")

    results_df = (
        pd.DataFrame(records)
        .sort_values("success_prob", ascending=False)
        .reset_index(drop=True)
    )

    best_row = results_df.iloc[0]
    best_weights = best_row[
        ["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ITC.NS"]
    ].values.astype(float)
    best_success_prob = float(best_row["success_prob"])

    if verbose:
        print("\nOptimisation complete.")
        print(f"Best success probability : {best_success_prob:.2%}")
        print(f"Best weights             : {best_weights}")

    return best_weights, best_success_prob, results_df


# 3. DISPLAY HELPERS
def print_optimal_portfolio(
    weights: np.ndarray, prob: float, tickers: list, label: str
):
    """Print a formatted summary of the optimal portfolio found."""
    print("\n" + "=" * 55)
    print(f"Optimal Portfolio -- {label}")
    print("-" * 55)
    for t, w in zip(tickers, weights):
        bar = "#" * int(w * 20)
        print(f"  {t:<15} : {w:.2f}  {bar}")
    print(f"\n  Success Probability : {prob:.2%}")
    print("=" * 55)


def print_top_n(results_df: pd.DataFrame, n: int = 10, label: str = ""):
    """Print the top N portfolios by success probability."""
    tickers = ["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ITC.NS"]
    print(f"\nTop {n} Portfolios -- {label}")
    print("-" * 75)
    header = (
        f"{'Rank':<6}" + "".join(f"{t:<14}" for t in tickers) + f"{'Success Prob':>13}"
    )
    print(header)
    print("-" * 75)
    for rank, row in results_df.head(n).iterrows():
        weights_str = "".join(f"{row[t]:<14.2f}" for t in tickers)
        print(f"  {rank + 1:<4}{weights_str}{row['success_prob']:>12.2%}")
    print("-" * 75)


# main entrypoint for testing (uses mock data -- no dependency on data.py)
if __name__ == "__main__":
    print("=== Brute-Force Portfolio Optimiser ===")

    TICKERS_MOCK = ["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ITC.NS"]

    mu_mock = np.array([0.17, 0.14, 0.13, 0.15, 0.12])
    std_mock = np.array([0.24, 0.26, 0.28, 0.25, 0.22])
    cov_mock = np.diag(std_mock**2)
    savings_mock = np.array([240_000 * (1.04**t) for t in range(20)])

    goals_A = {3: 1_500_000, 7: 2_500_000, 12: 3_000_000, 20: 15_000_000}
    goals_B = {8: 1_000_000, 12: 2_000_000, 16: 4_000_000, 20: 15_000_000}

    combos = generate_valid_combinations()
    print(f"\nValid combinations generated : {len(combos)}")
    print(f"First 3 combos : {[c.tolist() for c in combos[:3]]}")

    print("\n-- Sequence A (Aggressive Early Goals) --")
    best_w_A, best_p_A, results_A = run_optimisation(
        mu_mock, cov_mock, savings_mock, goals_A, n_paths=500, seed=42
    )
    print_optimal_portfolio(best_w_A, best_p_A, TICKERS_MOCK, label="Sequence A")
    print_top_n(results_A, n=5, label="Sequence A")

    print("\n-- Sequence B (Backloaded Goals) --")
    best_w_B, best_p_B, results_B = run_optimisation(
        mu_mock, cov_mock, savings_mock, goals_B, n_paths=500, seed=42
    )
    print_optimal_portfolio(best_w_B, best_p_B, TICKERS_MOCK, label="Sequence B")
    print_top_n(results_B, n=5, label="Sequence B")

    print(f"best_w_A shape : {best_w_A.shape}  dtype: {best_w_A.dtype}")
    print(f"results_A shape: {results_A.shape}")
    print(f"Columns        : {list(results_A.columns)}")
