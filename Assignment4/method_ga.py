"""method_ga.py — Genetic Algorithm portfolio replication (Method 3, optional).

Strategy
--------
Each individual in the GA population is a set of k ticker indices.
Fitness = −TE(portfolio, benchmark) on the training set (maximise → minimise TE).
Crossover: two-point cut; mutation: random stock swap with probability p_mut.
After convergence the best individual's weights are solved via the same QP used
by the Autoencoder method.
"""

import random
import numpy as np
import pandas as pd

from eval import evaluate, load_splits
from method_autoencoder import fit_weights_qp
from eval import tracking_error, portfolio_returns


def _fitness(
    individual: list,
    tickers: list,
    R_train: pd.DataFrame,
    b_train: pd.Series,
) -> float:
    """Negative annualised TE on the training set (higher = better)."""
    selected = [tickers[i] for i in individual]
    try:
        w = fit_weights_qp(selected, R_train, b_train)
        p = portfolio_returns(w, R_train)
        return -tracking_error(p, b_train)
    except Exception:
        return -999.0


def ga_select(
    R_train: pd.DataFrame,
    b_train: pd.Series,
    k: int = 50,
    pop_size: int = 50,
    n_gen: int = 100,
    p_mut: float = 0.1,
    seed: int = 42,
    verbose: bool = False,
) -> list:
    """Run GA and return the list of k selected tickers.

    Parameters
    ----------
    pop_size : int
        Number of individuals per generation.
    n_gen : int
        Maximum number of generations.
    p_mut : float
        Per-individual mutation probability (swap one stock).
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    tickers = list(R_train.columns)
    N = len(tickers)

    population = [random.sample(range(N), k) for _ in range(pop_size)]

    def score(ind):
        return _fitness(ind, tickers, R_train, b_train)

    best_fitness_history = []

    fitness_cache: dict = {}

    def cached_score(ind: list) -> float:
        key = frozenset(ind)
        if key not in fitness_cache:
            fitness_cache[key] = score(ind)
        return fitness_cache[key]

    for gen in range(n_gen):
        scored = sorted(population, key=cached_score, reverse=True)
        elite = scored[: pop_size // 2]
        best_fitness_history.append(-cached_score(elite[0]))

        if verbose and gen % 10 == 0:
            print(f"  Gen {gen:4d} | best TE (train) = {best_fitness_history[-1]:.4f}")

        children = []
        while len(children) < pop_size // 2:
            p1, p2 = random.sample(elite, 2)
            cut = k // 2
            child = list(set(p1[:cut] + p2[cut:]))
            while len(child) < k:
                candidate = random.randint(0, N - 1)
                if candidate not in child:
                    child.append(candidate)
            child = child[:k]
            children.append(child)

        for ind in children:
            if random.random() < p_mut:
                swap_out = random.randint(0, k - 1)
                swap_in = random.choice([i for i in range(N) if i not in ind])
                ind[swap_out] = swap_in

        population = elite + children

    best_ind = sorted(population, key=score, reverse=True)[0]
    return [tickers[i] for i in best_ind]


def run_ga_sweep(
    R_train: pd.DataFrame,
    b_train: pd.Series,
    R_val: pd.DataFrame,
    b_val: pd.Series,
    R_hold: pd.DataFrame,
    b_hold: pd.Series,
    k_range: range | None = None,
    pop_size: int = 20,
    n_gen: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep k values using GA selection and return a results DataFrame.

    Reduced defaults (pop_size=20, n_gen=50) keep runtime manageable.
    """
    if k_range is None:
        k_range = range(10, 101, 10)

    records = []
    for k in k_range:
        if verbose:
            print(f"GA: selecting k={k} …")
        selected = ga_select(
            R_train, b_train, k=k, pop_size=pop_size, n_gen=n_gen, verbose=False
        )
        weights = fit_weights_qp(selected, R_train, b_train)
        res = evaluate(weights, R_val, b_val, R_hold, b_hold, label="GA")
        res["weights"] = weights
        records.append(res)
        if verbose:
            print(f"  k={k:3d} | TE_val={res['TE_val']:.4f} | TE_hold={res['TE_hold']:.4f}")

    return pd.DataFrame(records)


def get_ga_weights_for_k(results_df: pd.DataFrame, k: int = 50) -> dict:
    """Extract weight dict for the closest k in results_df."""
    if results_df.empty:
        raise ValueError("results_df is empty — run run_ga_sweep first.")
    row = results_df.iloc[(results_df["k"] - k).abs().argsort()[:1]]
    return row["weights"].iloc[0]


if __name__ == "__main__":
    R_train, b_train, R_val, b_val, R_hold, b_hold = load_splits()
    print(f"Running GA sweep on {R_train.shape[1]} stocks …")
    results = run_ga_sweep(
        R_train, b_train, R_val, b_val, R_hold, b_hold,
        pop_size=20, n_gen=50
    )
    print(results[["k", "TE_val", "IR_val", "TE_hold", "IR_hold"]].to_string())
    results.drop(columns="weights").to_csv("data/ga_results.csv", index=False)
    print("Saved → data/ga_results.csv")
