"""
bonus.py: Continuous Portfolio Optimiser (SciPy)
=================================================
What this module does:
  - Replaces the discrete brute-force grid with a continuous weight optimiser
  - Weights are any float with no hard bounds, short-selling allowed (e.g. -0.5 in one security)
  - Only hard constraint: weights must sum to exactly 1.0
  - Uses scipy.optimize.minimize with the SLSQP method (handles equality constraints)
  - Objective: maximise success probability = minimise negative success probability
  - Runs multiple random restarts to escape local minima (success probability
    is non-convex and has flat regions, so a single start is unreliable)
  - Reports the best solution found across all restarts for Sequence A and B

Returns:
  best_weights      : np.ndarray, shape (5,)  -- optimal continuous weights
  best_success_prob : float                   -- highest success probability found
"""

from simulator import simulate_portfolio, compute_success_probability
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


# CONSTANTS
N_SECURITIES = 5
TERMINAL_GOAL = 15_000_000  # Rs 1.5 Crore
N_PATHS = 5000
N_RESTARTS = 30  # number of random starting points
SEED_BASE = 100  # base seed for optimiser restarts


# 1. OBJECTIVE FUNCTION
def negative_success_prob(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    savings: np.ndarray,
    goals: dict,
    n_paths: int = N_PATHS,
    seed: int | None = None,
) -> float:
    """
    Objective function for scipy.optimize.minimize.

    Returns the negative success probability because scipy minimises,
    so minimising -P(success) is equivalent to maximising P(success).

    Parameters
    ----------
    weights : np.ndarray, shape (5,)  -- current weight vector from the optimiser
    mu      : np.ndarray, shape (5,)  -- annualised expected returns
    cov     : np.ndarray, shape (5,5) -- annualised covariance matrix
    savings : np.ndarray, shape (20,) -- annual savings schedule
    goals   : dict {int: float}       -- full goal schedule including year 20
    n_paths : int                     -- Monte Carlo paths per evaluation
    seed    : int | None              -- random seed for this evaluation

    Returns
    -------
    float -- negative success probability (to be minimised)
    """
    final_values = simulate_portfolio(weights, mu, cov, savings, goals, n_paths, seed)
    prob = compute_success_probability(final_values, TERMINAL_GOAL)
    return -prob


# 2. SINGLE OPTIMISATION RUN
def run_single_optimisation(
    start_weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    savings: np.ndarray,
    goals: dict,
    n_paths: int,
    seed: int | None,
) -> tuple:
    """
    Run one SLSQP optimisation from a given starting point.

    Constraint  : sum(weights) == 1.0
    Bounds      : none -- weights are unconstrained (short-selling freely allowed)

    Parameters
    ----------
    start_weights : np.ndarray, shape (5,) -- initial weight vector
    (remaining params same as negative_success_prob)

    Returns
    -------
    weights : np.ndarray, shape (5,) -- optimised weights (or start if failed)
    prob    : float                  -- success probability at the solution
    success : bool                   -- whether the optimiser converged
    """
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        fun=negative_success_prob,
        x0=start_weights,
        args=(mu, cov, savings, goals, n_paths, seed),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    weights = result.x
    prob = -result.fun
    return weights, prob, result.success


# 3. MULTI-RESTART OPTIMISATION
def run_continuous_optimisation(
    mu: np.ndarray,
    cov: np.ndarray,
    savings: np.ndarray,
    goals: dict,
    n_paths: int = N_PATHS,
    n_restarts: int = N_RESTARTS,
    seed: int | None = 42,
    verbose: bool = True,
) -> tuple:
    """
    Multi-restart continuous optimisation using SciPy SLSQP.

    Because the objective (Monte Carlo success probability) is non-convex
    and stochastic, a single optimisation run may settle in a local optimum.
    We run N_RESTARTS times from different random starting points and keep
    the best result found.

    Starting points are generated as Dirichlet samples (guaranteed to sum
    to 1.0) and then perturbed to allow short positions.

    Parameters
    ----------
    mu         : np.ndarray, shape (5,)   -- annualised expected returns
    cov        : np.ndarray, shape (5,5)  -- annualised covariance matrix
    savings    : np.ndarray, shape (20,)  -- annual savings schedule
    goals      : dict {int: float}        -- full goal schedule
    n_paths    : int                      -- Monte Carlo paths per evaluation
    n_restarts : int                      -- number of random starting points
    seed       : int | None              -- master seed for reproducibility
    verbose    : bool                     -- print progress per restart

    Returns
    -------
    best_weights      : np.ndarray, shape (5,) -- best weights found
    best_success_prob : float                  -- success probability at best weights
    all_results       : pd.DataFrame           -- all restart results sorted by prob
    """
    rng = np.random.default_rng(seed)

    records = []
    best_prob = -np.inf
    best_weights = None

    if verbose:
        print(f"Running {n_restarts} restarts | {n_paths:,} paths per evaluation")
        print("Bounds per security: none (unconstrained, short-selling allowed)")
        print("Constraint: sum(weights) == 1.0\n")

    for i in range(n_restarts):
        # Generate a diverse starting point:
        # Mix of Dirichlet (long-only) and random (allows shorts)
        if i < n_restarts // 2:
            # First half: Dirichlet samples -- well-spread long-only starts
            start = rng.dirichlet(np.ones(N_SECURITIES))
        else:
            # Second half: unconstrained random (range -1 to 2), then rescale to sum=1
            # Wide range encourages the optimiser to explore short positions
            raw = rng.uniform(-1.0, 2.0, N_SECURITIES)
            start = raw / raw.sum()

        opt_seed = SEED_BASE + i  # different MC seed per restart for diversity

        weights, prob, converged = run_single_optimisation(
            start, mu, cov, savings, goals, n_paths, seed=opt_seed
        )

        records.append(
            {
                "restart": i + 1,
                "TCS.NS": weights[0],
                "HDFCBANK.NS": weights[1],
                "RELIANCE.NS": weights[2],
                "SUNPHARMA.NS": weights[3],
                "ITC.NS": weights[4],
                "success_prob": prob,
                "converged": converged,
            }
        )

        if prob > best_prob:
            best_prob = prob
            best_weights = weights.copy()

        if verbose:
            status = "OK" if converged else "no-conv"
            print(
                f"  Restart {i + 1:>2}/{n_restarts}  prob={prob:.4%}  "
                f"weights=[{', '.join(f'{w:.3f}' for w in weights)}]  [{status}]"
            )

    all_results = (
        pd.DataFrame(records)
        .sort_values("success_prob", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\nBest success probability : {best_prob:.2%}")
        print(f"Best weights             : {[round(w, 4) for w in best_weights]}")

    return best_weights, best_prob, all_results


# 4. DISPLAY HELPERS
def print_bonus_result(weights: np.ndarray, prob: float, tickers: list, label: str):
    """Print the continuous optimisation result with short-selling info."""
    print("\n" + "=" * 60)
    print(f"Continuous Optimiser Result -- {label}")
    print("-" * 60)
    for t, w in zip(tickers, weights):
        direction = "SHORT" if w < 0 else "LONG "
        bar_len = int(abs(w) * 20)
        bar = ("#" if w >= 0 else "-") * bar_len
        print(f"  {t:<15} : {w:>+7.4f}  [{direction}]  {bar}")
    print(f"\n  Sum of weights      : {weights.sum():.8f}  (should be 1.0)")
    print(f"  Success Probability : {prob:.2%}")
    print("=" * 60)


def compare_brute_vs_continuous(
    brute_weights: np.ndarray,
    brute_prob: float,
    cont_weights: np.ndarray,
    cont_prob: float,
    tickers: list,
    label: str,
):
    """Side-by-side comparison of brute-force and continuous optimiser results."""
    print(f"\nComparison -- {label}")
    print("-" * 65)
    print(f"{'Ticker':<15} {'Brute-Force':>15} {'Continuous':>15}")
    print("-" * 65)
    for t, bw, cw in zip(tickers, brute_weights, cont_weights):
        print(f"  {t:<15} {bw:>14.4f}  {cw:>14.4f}")
    print("-" * 65)
    print(f"  {'Success Prob':<13} {brute_prob:>15.2%} {cont_prob:>15.2%}")
    improvement = cont_prob - brute_prob
    print(f"  {'Improvement':<13} {'':>15} {improvement:>+14.2%}")
    print("-" * 65)


# main entrypoint for testing (uses mock data -- no dependency on data.py)
if __name__ == "__main__":
    print("=== Bonus: Continuous Portfolio Optimiser ===")

    TICKERS_MOCK = ["TCS.NS", "HDFCBANK.NS", "RELIANCE.NS", "SUNPHARMA.NS", "ITC.NS"]

    mu_mock = np.array([0.17, 0.14, 0.13, 0.15, 0.12])
    std_mock = np.array([0.24, 0.26, 0.28, 0.25, 0.22])
    cov_mock = np.diag(std_mock**2)
    savings_mock = np.array([240_000 * (1.04**t) for t in range(20)])

    goals_A = {3: 1_500_000, 7: 2_500_000, 12: 3_000_000, 20: 15_000_000}
    goals_B = {8: 1_000_000, 12: 2_000_000, 16: 4_000_000, 20: 15_000_000}

    print("\n-- Sequence A (Aggressive Early Goals) --")
    best_w_A, best_p_A, results_A = run_continuous_optimisation(
        mu_mock,
        cov_mock,
        savings_mock,
        goals_A,
        n_paths=500,
        n_restarts=5,
        seed=42,
        verbose=True,
    )
    print_bonus_result(best_w_A, best_p_A, TICKERS_MOCK, label="Sequence A")

    print("\n-- Sequence B (Backloaded Goals) --")
    best_w_B, best_p_B, results_B = run_continuous_optimisation(
        mu_mock,
        cov_mock,
        savings_mock,
        goals_B,
        n_paths=500,
        n_restarts=5,
        seed=42,
        verbose=True,
    )
    print_bonus_result(best_w_B, best_p_B, TICKERS_MOCK, label="Sequence B")

    print(f"best_w_A shape   : {best_w_A.shape}  sum: {best_w_A.sum():.8f}")
    print(f"results_A shape  : {results_A.shape}")
    print(f"Columns          : {list(results_A.columns)}")
