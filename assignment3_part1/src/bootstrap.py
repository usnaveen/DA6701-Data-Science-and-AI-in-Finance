import numpy as np
import matplotlib.pyplot as plt
from config import BOOTSTRAP_ITERATIONS, FIGURES_DIR

def bootstrap_covariance_stability(train_returns):
    condition_numbers = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = train_returns.sample(frac=1, replace=True)
        S_boot = sample.cov()
        kappa_boot = np.linalg.cond(S_boot)
        condition_numbers.append(kappa_boot)

    return condition_numbers


def plot_bootstrap_distribution(condition_numbers):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(condition_numbers, bins=20)
    plt.title("Bootstrap Condition Number Distribution")
    plt.xlabel("Condition Number")
    plt.ylabel("Frequency")
    plt.savefig(FIGURES_DIR / "bootstrap_condition_numbers.png")
    plt.close()