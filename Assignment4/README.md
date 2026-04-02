# Assignment IV — S&P 500 Portfolio Replication

**Course:** DA6701 — Data Science and AI in Finance  
**Topic:** Sparse Index Tracking via Heuristic & Regularised Methods

---

## Objective

Synthetically replicate the daily return profile of the **S&P 500 Index (^GSPC)** using a strictly limited subset of its constituent stocks (**k ≤ 100**). The core challenge is a Mixed-Integer Quadratic Programme (MIQP) — selecting exactly k out of N ≈ 553 stocks to minimise Tracking Error is NP-hard, so heuristic or regularised approaches are required.

**Primary metric — Tracking Error (TE):**

$$TE = \sqrt{Var(R_p - R_b)}$$

where $R_p$ is the portfolio return vector and $R_b$ is the S&P 500 benchmark.

---

## Data

| Item | Details |
|------|---------|
| Source | `Mega/` — daily OHLCV CSVs for ~580 S&P 500 constituents |
| Benchmark | `Mega/^GSPC.csv` |
| Date range | Jan 2020 – Dec 2025 |
| Universe after cleaning | **553 stocks** (6 empty files + 24 coverage exclusions + 1 gap exclusion removed) |

### Train / Validation / Holdout Splits

| Split | Period | Rows |
|-------|--------|------|
| **Train** | 2020-01-03 → 2024-12-31 | 1 257 |
| **Validation** | 2025-01-02 → 2025-06-30 | 122 |
| **Holdout** | 2025-07-01 → 2025-12-31 | 128 |

> The holdout set was **never used during model selection** — only for final evaluation.

---

## Methods

### Method 1 — Lasso Regression
Fits the non-negative linear model `b_t = Σ wᵢ · rᵢ,ₜ` with an L1 penalty. The penalty forces most weights to exactly zero, producing a naturally sparse portfolio. Sweeping the regularisation strength `alpha` across a log-scale grid yields portfolios of varying cardinality k.

- **Key property:** interpretable, convex, extremely fast.
- **Implementation:** `method_lasso.py` — sweeps `alpha` over `np.logspace(-7, -1, 100)`.

### Method 2 — Autoencoder + Quadratic Programme
1. Trains a symmetric autoencoder (`N → 128 → 32 → 128 → N`) on the T × N daily return matrix.  
2. Ranks stocks by **communality** — the R² of reconstructing each stock's return series from the latent bottleneck.  
3. Takes the top-k stocks and solves a **Quadratic Programme** to minimise variance of active returns subject to long-only, sum-to-1 constraints.

- **Key property:** captures latent co-movement structure; communality provides a principled ranking.
- **Implementation:** `method_autoencoder.py` — sweeps `k ∈ {10, 15, …, 100}`.

### Method 3 — Genetic Algorithm *(optional)*
Evolves a population of k-stock subsets. Fitness = −TE on the training set. Uses the same QP weight-solver as Method 2. A fitness cache avoids re-solving the same subsets across generations.

- **Key property:** direct cardinality-constrained search; stochastic but avoids local optima.
- **Implementation:** `method_ga.py` — set `RUN_GA = True` in `notebook.ipynb` to enable.

---

## Results (k ≈ 50)

| Method | k | TE (Val) | IR (Val) | TE (Holdout) | IR (Holdout) |
|--------|---|----------|----------|--------------|--------------|
| Lasso | 48 | 0.1134 | -0.5846 | 0.0623 | -0.1245 |
| Autoencoder | 50 | 0.0904 | 0.5276 | 0.0907 | -1.2298 |

### Generated Figures

| File | Description |
|------|-------------|
| `figures/te_vs_k.png` | **Sparsity vs Tracking Error** — both methods, val + holdout curves |
| `figures/cumret_val.png` | Cumulative returns — validation period |
| `figures/cumret_hold.png` | Cumulative returns — holdout period |
| `figures/sector_drift_lasso.png` | **Sector Drift** — Lasso k=50 vs S&P 500 |
| `figures/sector_drift_ae.png` | Sector Drift — Autoencoder k=50 vs S&P 500 |
| `figures/information_ratio.png` | Information Ratio bar chart |
| `figures/communality_distribution.png` | Autoencoder communality scores across all 553 stocks |

---

## Project Structure

```
Assignment4/
│
├── Mega/                          # Raw OHLCV CSVs (read-only, extracted from OHLCV_Data.zip)
│   ├── ^GSPC.csv                  # S&P 500 index
│   ├── AAPL.csv
│   └── ...                        # ~584 constituent CSVs
│
├── data/                          # All processed data artifacts
│   ├── train_returns.parquet      # 1257 × 554  (553 stocks + ^GSPC)
│   ├── val_returns.parquet        # 122  × 554
│   ├── test_returns.parquet       # 128  × 554
│   ├── returns_all.parquet        # Full 1507 × 554 return matrix
│   ├── bench_returns.parquet      # Benchmark-only series
│   ├── stock_returns.parquet      # Stock-only return matrix
│   ├── final_prices.parquet       # Price matrix pre-return computation
│   ├── kept_universe.parquet      # Metadata: tickers kept after coverage filter
│   ├── excluded_universe.parquet  # Metadata: tickers excluded + reason
│   ├── sector_map.csv             # 553 tickers → GICS sector (fetched via yfinance)
│   ├── summary.csv                # P1 pipeline summary statistics
│   ├── file_issues.csv            # CSVs excluded at load time
│   ├── train_gap_exclusions.csv   # Stocks with training-period gaps
│   ├── lasso_results.csv          # Per-k Lasso evaluation results
│   ├── ae_results.csv             # Per-k Autoencoder evaluation results
│   └── results_summary.csv        # Final k=50 summary table
│
├── figures/                       # All generated plots (PNG)
│   ├── te_vs_k.png
│   ├── cumret_val.png
│   ├── cumret_hold.png
│   ├── sector_drift_lasso.png
│   ├── sector_drift_ae.png
│   ├── information_ratio.png
│   └── communality_distribution.png
│
├── eval.py                        # Shared evaluation harness (TE, IR, portfolio_returns)
├── method_lasso.py                # Method 1: Lasso alpha sweep
├── method_autoencoder.py          # Method 2: Autoencoder + QP sweep
├── method_ga.py                   # Method 3: Genetic Algorithm (optional)
├── plots.py                       # Visualisation helpers
├── notebook.ipynb                 # Final executed notebook (all outputs embedded)
│
├── p1d1.ipynb                     # P1 data pipeline notebook (data cleaning & splits)
├── generate_sector_map.py         # One-time script to fetch sector labels via yfinance
│
├── OHLCV_Data.zip                 # Original zip (Git LFS)
├── Assignment IV.pdf              # Problem statement
├── A4_Plan.pdf                    # Detailed execution plan
├── changes.md                     # Development log
└── readme-parquet.md              # Parquet artifact guide for the data/ folder
```

---

## Setup

### Prerequisites

```bash
# Python 3.10+
pip install numpy pandas scipy scikit-learn torch pyarrow matplotlib yfinance jupyter
```

All dependencies are available in a standard ML environment. GPU is optional — the autoencoder trains on CPU in ~1–2 minutes.

### One-time Data Setup

The raw data lives in Git LFS. If cloning fresh:

```bash
# Install git-lfs if needed
sudo apt-get install git-lfs      # Ubuntu/Debian
brew install git-lfs               # macOS

# Pull LFS objects
git lfs install
git lfs pull

# Extract raw CSVs
unzip OHLCV_Data.zip               # creates Mega/
```

> If `data/` parquet files are already present (LFS pulled), you can skip directly to running the notebook.

---

## Running

### Option A — Run the final notebook

```bash
jupyter notebook notebook.ipynb
# or execute headlessly:
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

### Option B — Run individual method scripts

```bash
# From Assignment4/
python method_lasso.py          # writes data/lasso_results.csv
python method_autoencoder.py    # writes data/ae_results.csv
python method_ga.py             # writes data/ga_results.csv  (slow — ~10 min)
```

### Option C — Regenerate sector map

Only needed if `data/sector_map.csv` is missing:

```bash
python generate_sector_map.py   # fetches via yfinance, ~1 min
```

### Option D — Re-run the P1 data pipeline

Only needed if the `data/` parquet files are missing entirely:

```bash
jupyter nbconvert --to notebook --execute --inplace p1d1.ipynb
# Then copy outputs into data/ with canonical names (see readme-parquet.md)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Train end at Dec 2024 (not Dec 2022) | Maximises training data while preserving a 6-month val and 6-month holdout |
| One-month grace period for listing date | Allows stocks like CARR (listed Mar 2020) without shrinking the window |
| Log returns | Additive across time; numerically stable |
| Lasso `positive=True` | Enforces long-only weights; no shorting the index |
| QP for AE weights | Directly minimises active-return variance; respects long-only and budget constraints |
| GA fitness cache | Avoids re-solving QP for identical subsets across generations |
| GA disabled by default | Pop=20 × Gen=50 × 10 k-values ≈ thousands of QP solves; enable with `RUN_GA = True` |

---

## Module Reference

### `eval.py`
```python
from eval import load_splits, evaluate, tracking_error, information_ratio, portfolio_returns

R_train, b_train, R_val, b_val, R_hold, b_hold = load_splits(data_dir='data')
res = evaluate(weights_dict, R_val, b_val, R_hold, b_hold, label='MyMethod')
# res keys: label, k, TE_val, IR_val, TE_hold, IR_hold
```

### `method_lasso.py`
```python
from method_lasso import run_lasso_sweep, get_lasso_weights_for_k

results_df = run_lasso_sweep(R_train, b_train, R_val, b_val, R_hold, b_hold)
weights_k50 = get_lasso_weights_for_k(results_df, k=50)
```

### `method_autoencoder.py`
```python
from method_autoencoder import run_autoencoder_sweep, get_ae_weights_for_k

ae_results, communality, model = run_autoencoder_sweep(
    R_train, b_train, R_val, b_val, R_hold, b_hold, k_range=range(10, 101, 5)
)
weights_k50 = get_ae_weights_for_k(ae_results, k=50)
```

### `method_ga.py`
```python
from method_ga import ga_select, run_ga_sweep
from method_autoencoder import fit_weights_qp

selected_tickers = ga_select(R_train, b_train, k=50, pop_size=20, n_gen=50)
weights = fit_weights_qp(selected_tickers, R_train, b_train)
```

### `plots.py`
```python
from plots import plot_te_vs_k, plot_cumulative_returns, plot_sector_drift, make_results_table

fig = plot_te_vs_k({'Lasso': lasso_df, 'Autoencoder': ae_df})
fig = plot_cumulative_returns({'Lasso (k=50)': weights}, R_val, b_val)
fig = plot_sector_drift(weights_k50, sector_map_path='data/sector_map.csv')
table = make_results_table(rows)   # returns formatted pd.DataFrame
```
