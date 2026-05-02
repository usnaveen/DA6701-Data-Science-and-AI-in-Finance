"""
simulator.py: Monte Carlo Simulation Engine
============================================
What this module does:
  - Generate 5,000 annual portfolio return paths using multivariate normal sampling
  - Apply portfolio return each year to grow the portfolio value
  - Add the annual savings contribution at the end of each year
  - At each goal year, deduct the goal amount from the portfolio
  - If the portfolio is short of a goal, borrow the shortfall at 12% p.a. and track the loan
  - Return the array of final portfolio values at Year 20

Returns:
  final_values : np.ndarray, shape (n_paths,) — portfolio value at end of Year 20 per path
"""

import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CONSTANTS
BORROWING_RATE = 0.12  # 12% annual interest on shortfall loans
HORIZON_YEARS = 20
N_PATHS = 5000


# 1. CORE SIMULATION
def simulate_portfolio(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    savings: np.ndarray,
    goals: dict,
    n_paths: int = N_PATHS,
    seed: int | None = None,
) -> np.ndarray:
    """
    Run a Monte Carlo simulation for a static portfolio over 20 years.

    Each path evolves as:
      1. Sample annual portfolio return from N(mu_p, sigma_p^2)
      2. Grow portfolio: V = V * (1 + r_p)
      3. Add savings contribution for that year
      4. If it is a goal year:
           - Deduct the goal amount from the portfolio
           - If V < goal: borrow the shortfall, goal is fulfilled, portfolio goes to 0
             for that path; shortfall added to outstanding loan balance
      5. Grow outstanding loan by 12% (compounds annually)
      6. If portfolio has surplus above the loan, repay the loan in full

    Parameters
    ----------
    weights  : np.ndarray, shape (5,)  — portfolio weights, must sum to 1.0
    mu       : np.ndarray, shape (5,)  — annualised expected returns per ticker
    cov      : np.ndarray, shape (5,5) — annualised covariance matrix
    savings  : np.ndarray, shape (20,) — annual contributions in ₹, savings[t] for year t+1
    goals    : dict {int: float}       — {year: target_amount}, e.g. {3: 1_500_000, 7: 2_500_000}
    n_paths  : int                     — number of Monte Carlo paths (default 5,000)
    seed     : int | None              — random seed for reproducibility (default None)

    Returns
    -------
    final_values : np.ndarray, shape (n_paths,)
        Portfolio value at the end of Year 20 for each simulated path.
        Values can be negative if borrowing obligations exceed portfolio value.
    """
    if seed is not None:
        np.random.seed(seed)

    # Portfolio-level statistics
    mu_p = float(weights @ mu)  # scalar: weighted expected return
    sigma_p = float(np.sqrt(weights @ cov @ weights))  # scalar: portfolio std dev

    # Initialise all paths with zero portfolio value and zero outstanding loan
    portfolio = np.zeros(n_paths)  # shape (n_paths,)
    loan = np.zeros(n_paths)  # outstanding loan balance per path

    for year in range(1, HORIZON_YEARS + 1):
        # Step 1 — Sample annual returns for all paths at once
        annual_returns = np.random.normal(mu_p, sigma_p, n_paths)

        # Step 2 — Grow portfolio by the sampled return
        portfolio = portfolio * (1 + annual_returns)

        # Step 3 — Add this year's savings contribution
        portfolio += savings[year - 1]

        # Step 4 — Handle goal deduction if this is a goal year.
        # Done BEFORE loan compounding so portfolio has its full current value available.
        if year in goals:
            target = goals[year]

            # Paths that meet the goal: simply deduct the goal amount
            meets_goal = portfolio >= target
            portfolio = np.where(meets_goal, portfolio - target, portfolio)

            # Paths that fall short: borrow the exact shortfall to fulfill the goal.
            # The client receives the full goal amount regardless.
            # Portfolio on these paths is zeroed (it contributed what it had),
            # and the shortfall is added to the outstanding loan.
            shortfall = np.where(~meets_goal, target - portfolio, 0.0)
            loan += shortfall
            portfolio = np.where(~meets_goal, 0.0, portfolio)

        # Step 5 — Compound the outstanding loan for this year
        loan = loan * (1 + BORROWING_RATE)

        # Step 6 — Repay loan from portfolio surplus where possible.
        # Only repay if portfolio fully covers the loan — avoids starving
        # the portfolio of capital needed to grow toward future goals.
        # can_repay = portfolio >= loan
        # portfolio = np.where(can_repay, portfolio - loan, portfolio)
        # loan = np.where(can_repay, 0.0, loan)

        # Step 6 — Partial repayment: pay as much of the loan as the portfolio allows.
        # Using partial (not all-or-nothing) repayment because the assignment states
        # the loan is "paid off using future portfolio returns and savings" — meaning
        # the portfolio keeps contributing each year rather than holding everything
        # back until it can repay in one lump sum (which caused unbounded debt growth).
        repayment = np.minimum(portfolio, loan)  # pay as much as you can, every year
        portfolio = portfolio - repayment
        loan = loan - repayment

    # Any unpaid loan remaining after Year 20 is deducted from final portfolio value
    final_values = portfolio - loan

    return final_values


# 2. SUCCESS PROBABILITY
def compute_success_probability(
    final_values: np.ndarray,
    terminal_goal: float,
) -> float:
    """
    Compute the fraction of paths that meet or exceed the terminal goal.

    Parameters
    ----------
    final_values  : np.ndarray — output of simulate_portfolio()
    terminal_goal : float      — retirement target in ₹ (e.g. 1,50,00,000)

    Returns
    -------
    prob : float — probability of success in [0, 1]
    """
    return float(np.mean(final_values >= terminal_goal))


# 3. SUMMARY HELPERS
def print_simulation_summary(
    weights: np.ndarray,
    tickers: list,
    final_values: np.ndarray,
    terminal_goal: float,
    goals: dict,
):
    """Print a summary of the simulation results for a given portfolio."""
    prob = compute_success_probability(final_values, terminal_goal)

    print("\n" + "=" * 55)
    print("Simulation Summary")
    print("-" * 55)
    print("Portfolio Weights:")
    for t, w in zip(tickers, weights):
        print(f"  {t:<15} : {w:.2f}")
    print(f"\nGoal Schedule : {goals}")
    print(f"Terminal Goal : ₹{terminal_goal:>,.0f}")
    print("-" * 55)
    print(f"  Paths simulated     : {len(final_values):,}")
    print(f"  Mean final value    : ₹{final_values.mean():>15,.0f}")
    print(f"  Median final value  : ₹{np.median(final_values):>15,.0f}")
    print(f"  5th percentile      : ₹{np.percentile(final_values, 5):>15,.0f}")
    print(f"  95th percentile     : ₹{np.percentile(final_values, 95):>15,.0f}")
    print(f"  Success probability : {prob:.2%}")
    print("=" * 55)


# main entrypoint for testing (uses mock data — no dependency on data.py)
if __name__ == "__main__":
    print("=== Monte Carlo Simulation Engine ===")

    # Mock data
    TICKERS_MOCK = ["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ITC.NS"]
    np.random.seed(0)
    mu_mock = np.array([0.17, 0.14, 0.13, 0.15, 0.12])
    std_mock = np.array([0.24, 0.26, 0.28, 0.25, 0.22])
    # Build a simple diagonal covariance (no cross-correlations) for mock
    cov_mock = np.diag(std_mock**2)
    savings_mock = np.array([240_000 * (1.04**t) for t in range(20)])

    # Equal-weight portfolio
    weights_mock = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Goal sequences
    goals_A = {3: 1_500_000, 7: 2_500_000, 12: 3_000_000}
    goals_B = {8: 1_000_000, 12: 2_000_000, 16: 4_000_000}

    print("\n-- Sequence A (Aggressive Early Goals) --")
    final_A = simulate_portfolio(
        weights_mock,
        mu_mock,
        cov_mock,
        savings_mock,
        goals=goals_A,
        n_paths=5000,
        seed=42,
    )
    print_simulation_summary(
        weights_mock, TICKERS_MOCK, final_A, terminal_goal=15_000_000, goals=goals_A
    )

    print("\n-- Sequence B (Backloaded Goals) --")
    final_B = simulate_portfolio(
        weights_mock,
        mu_mock,
        cov_mock,
        savings_mock,
        goals=goals_B,
        n_paths=5000,
        seed=42,
    )
    print_simulation_summary(
        weights_mock, TICKERS_MOCK, final_B, terminal_goal=15_000_000, goals=goals_B
    )

    print("=== Final Confirmation ===\n")
    print(f"final_values shape : {final_A.shape}  dtype: {final_A.dtype}")
    print(
        f"Success prob (A)   : {compute_success_probability(final_A, 15_000_000):.2%}"
    )
    print(
        f"Success prob (B)   : {compute_success_probability(final_B, 15_000_000):.2%}"
    )
