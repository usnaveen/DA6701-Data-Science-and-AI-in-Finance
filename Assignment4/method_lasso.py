# method_lasso.py
# P2 — Lasso-based Portfolio Replication
# What we're doing: We treat the benchmark (S&P 500) return as the "target"
# and each stock's daily return as a "feature". Lasso regression finds which
# stocks (features) best explain the benchmark, while forcing most weights to
# exactly zero. The non-zero weights = your selected portfolio stocks.

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
import sys
import os

# So we can import eval.py from the same folder
sys.path.append(os.path.dirname(__file__))
from eval import evaluate, portfolio_returns, tracking_error

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")

R_train = pd.read_parquet("p1d1_train_returns.parquet")
R_val   = pd.read_parquet("p1d1_val_returns.parquet")
R_hold  = pd.read_parquet("p1d1_test_returns.parquet")

bench   = pd.read_parquet("p1d1_benchmark_returns.parquet")

# The benchmark parquet might have a column name like 'bench' or '^GSPC'
# Check what columns exist:
print("Benchmark columns:", bench.columns.tolist())

# Adjust this depending on what the column is called:
BENCH_COL = bench.columns[0]  # grabs first column automatically

# Split benchmark into same periods as returns
b_train = bench.loc[R_train.index, BENCH_COL]
b_val   = bench.loc[R_val.index,   BENCH_COL]
b_hold  = bench.loc[R_hold.index,  BENCH_COL]

print(f"Train shape: {R_train.shape}")   # should be ~756 days × ~490 stocks
print(f"Val shape:   {R_val.shape}")
print(f"Hold shape:  {R_hold.shape}")

# ─────────────────────────────────────────
# 2. PREPARE MATRICES
# ─────────────────────────────────────────
# X = stock returns matrix (days × stocks)
# y = benchmark returns (days,)
# fillna(0) handles any remaining missing values safely

X_train = R_train.fillna(0).values
y_train = b_train.values

# ─────────────────────────────────────────
# 3. FIND BEST ALPHA WITH CROSS-VALIDATION
# ─────────────────────────────────────────
# LassoCV tries many alpha values automatically using 5-fold CV
# positive=True → long-only portfolio (no short selling)
# This takes a few minutes — that's normal

print("\nRunning LassoCV to find best alpha...")
lasso_cv = LassoCV(cv=5, max_iter=10000, positive=True, verbose=1, n_jobs=-1)
lasso_cv.fit(X_train, y_train)

best_alpha = lasso_cv.alpha_
print(f"Best alpha found: {best_alpha:.6f}")

# ─────────────────────────────────────────
# 4. SWEEP ALPHA → GET DIFFERENT k VALUES
# ─────────────────────────────────────────
# "k" = number of stocks selected (non-zero weights)
# Lower alpha  → less regularization → more stocks selected (higher k)
# Higher alpha → more regularization → fewer stocks selected (lower k)
# We sweep 40 values of alpha to get a range of k values

print("\nSweeping alpha to get different portfolio sizes (k)...")

results = []
seen_k = {}  # track best alpha per k value

alphas = np.logspace(-4, -1, 40)  # 40 values from 0.0001 to 0.1

for alpha in alphas:
    lasso = Lasso(alpha=alpha, positive=True, max_iter=10000)
    lasso.fit(X_train, y_train)

    # Extract non-zero weights → selected stocks
    selected = {
        ticker: w
        for ticker, w in zip(R_train.columns, lasso.coef_)
        if w > 1e-6  # threshold to ignore near-zero weights
    }

    k = len(selected)

    if k < 5:
        # Skip portfolios that are too small to be meaningful
        continue

    # Normalize weights so they sum to 1
    total = sum(selected.values())
    weights = {t: w / total for t, w in selected.items()}

    # Evaluate on validation AND holdout
    res = evaluate(weights, R_val, b_val, R_hold, b_hold, label="Lasso")
    res["alpha"] = alpha
    res["k"] = k

    # Keep only the best alpha for each k (lowest val tracking error)
    if k not in seen_k or res["TE_val"] < seen_k[k]["TE_val"]:
        seen_k[k] = res

    print(f"  alpha={alpha:.5f} → k={k:3d} | TE_val={res['TE_val']:.4f}")

# Convert to DataFrame, one row per unique k
lasso_results = pd.DataFrame(list(seen_k.values())).sort_values("k")

print(f"\nTotal unique k values: {len(lasso_results)}")
print(lasso_results[["k", "alpha", "TE_val", "TE_hold", "IR_val", "IR_hold"]].to_string())

# ─────────────────────────────────────────
# 5. SAVE RESULTS
# ─────────────────────────────────────────
lasso_results.to_csv("p2_lasso_results.csv", index=False)
print("\nSaved: p2_lasso_results.csv")

# ─────────────────────────────────────────
# 6. EXTRACT k=50 WEIGHTS (for P4's charts)
# ─────────────────────────────────────────
# Find the row closest to k=50
best_k50_row = lasso_results.iloc[(lasso_results["k"] - 50).abs().argsort()[:1]]
best_k50_alpha = best_k50_row["alpha"].values[0]
actual_k = best_k50_row["k"].values[0]

print(f"\nBest portfolio near k=50: actual k={actual_k}, alpha={best_k50_alpha:.6f}")

# Refit with that alpha to extract the weights dict
lasso_k50 = Lasso(alpha=best_k50_alpha, positive=True, max_iter=10000)
lasso_k50.fit(X_train, y_train)

selected_k50 = {
    ticker: w
    for ticker, w in zip(R_train.columns, lasso_k50.coef_)
    if w > 1e-6
}
total = sum(selected_k50.values())
lasso_k50_weights = {t: w / total for t, w in selected_k50.items()}

# Save weights for P4
weights_df = pd.DataFrame.from_dict(
    lasso_k50_weights, orient="index", columns=["weight"]
)
weights_df.index.name = "ticker"
weights_df.to_csv("p2_lasso_k50_weights.csv")
print(f"Saved: p2_lasso_k50_weights.csv ({len(lasso_k50_weights)} stocks)")

# ─────────────────────────────────────────
# 7. QUICK SANITY CHECK
# ─────────────────────────────────────────
print("\n--- Top 10 stocks in k=50 Lasso portfolio ---")
top10 = sorted(lasso_k50_weights.items(), key=lambda x: -x[1])[:10]
for ticker, w in top10:
    print(f"  {ticker:6s}: {w:.4f} ({w*100:.2f}%)")
