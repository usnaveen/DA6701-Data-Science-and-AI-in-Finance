# Assignment IV — Portfolio Replication: Insights & Analysis Notes

## 1. Problem Setup

- **Goal:** Replicate the S&P 500 daily return profile using at most k ≤ 50 constituent stocks
- **Benchmark:** ^GSPC daily log returns
- **Splits:**
  - Train: Jan 2020 – Dec 2024
  - Validation: Jan 2025 – Jun 2025
  - Holdout: Jul 2025 – Dec 2025
- **Why it's hard:** Selecting exactly k stocks out of N=500 to minimise tracking variance is MIQP (NP-hard). Standard Markowitz won't work directly.

---

## 2. Methodology Overview

All three methods follow the same **two-stage paradigm**:

```
Stage 1 — Stock Selection (method-specific)
        ↓
Stage 2 — Weight Optimisation (QP, shared across all methods)
```

This is standard industry practice for cardinality-constrained portfolio optimisation. The QP in Stage 2 solves:

```
minimise   Var(Xw - y)
subject to Σwᵢ = 1
           wᵢ ≥ 0   (long-only)
```

where X = training stock returns matrix, y = benchmark return vector.

### Method 1 — Lasso
- Fits benchmark returns as a penalised linear combination of stock returns
- L1 penalty drives most coefficients to exactly zero → sparse selection
- Sweep alpha (regularisation strength) across a log-grid → different k values
- Selected tickers then passed to QP for weight optimisation

### Method 2 — Autoencoder
- Trains a symmetric autoencoder: N → 128 → latent_dim → 128 → N
- Computes **communality** per stock = R² of reconstructing that stock's returns from the bottleneck
- Ranks stocks by communality, selects top-k (high communality = stock behaviour well-explained by shared market factors)
- Selected tickers passed to QP

### Method 3 — Genetic Algorithm (GA)
- Population of candidate stock sets, each of size k
- Fitness = -TE(portfolio, benchmark) on training set
- Selection → Crossover → Mutation over n_gen generations
- QP used internally to score each candidate, and finally to get weights from the best individual

---

## 3. Key Metrics

### Tracking Error (TE)
```
TE = std(Rp - Rb) × √252     [annualised]
```
- Measures how closely portfolio tracks the index day-to-day
- **Lower is better** for a replication strategy
- TE = 0% → perfect replication
- TE ~5-7% → reasonable for a 50-stock sparse portfolio
- TE > 10% → poor tracking, behaving like an active fund

### Information Ratio (IR)
```
IR = mean(Rp - Rb) / std(Rp - Rb) × √252     [annualised]
```
- Measures systematic outperformance/underperformance relative to its own noise
- **For a tracker, ideal IR ≈ 0** — no systematic active tilt
- Positive IR → portfolio systematically outperforms benchmark (accidental active bet)
- Negative IR → portfolio systematically underperforms benchmark
- |IR| < 0.3 → essentially no drift, good tracker
- |IR| > 1.0 → large active tilt, behaving like an active fund

> **Important:** Large IR in either direction is a red flag for a replication strategy.
> A sign flip in IR between val and holdout means the active tilt is unstable — driven by
> period-specific sector bets, not genuine tracking.

---

## 4. Results at k=50

| Method | TE Val | IR Val | TE Holdout | IR Holdout |
|--------|--------|--------|------------|------------|
| Lasso | 5.86% | -2.29 | 6.78% | +1.60 |
| Autoencoder | 8.96% | +1.19 | 8.37% | -0.74 |
| GA | 4.68% | -1.44 | 4.63% | -0.10 |

### Interpretation
- **GA** is the best overall — lowest TE on both splits, and holdout IR of -0.10 is the closest to ideal (no systematic drift)
- **Lasso** has moderate TE but large IR sign flip (val: -2.29, holdout: +1.60) — April 2025 tariff shock crushed its tech/healthcare tilt on val; it recovered on holdout
- **Autoencoder** has the highest TE and a sign flip on IR — sector concentration (Financials/Utilities) drove this

---

## 5. Sparsity vs TE Trade-off

The TE vs k curve has the expected shape for all methods:
- Steep drop from k=10 to k~50 (each additional stock meaningfully reduces tracking error)
- Flattens out beyond k~60-70 (diminishing returns)

Notable observations:
- Lasso at k=10 hits ~31% TE on val — extremely high, caused by heavily concentrated selection at low k
- GA is already at ~8% TE at k=10, demonstrating better selection quality at low cardinality
- Val/Holdout TE gap for Lasso is inverted at low k (val worse than holdout) — unusual, likely because val contained the April 2025 shock

---

## 6. Sector Drift Analysis

The S&P 500 side uses **market-cap weighted sector shares** (fetched via yfinance `marketCap`).
The portfolio side uses **QP-optimised portfolio weight shares**.

### GA — Passes ✅
- All sectors within a few percentage points of the benchmark
- Technology slightly overweight (~33% vs 30%), everything else close
- Directly supports the claim that GA did not build a sector-biased portfolio

### Lasso — Partial ⚠️
- Technology close (35% vs 30%)
- Healthcare notably overweight (~21% vs 9%)
- Communication Services underweight
- Acceptable for a 50-stock sparse portfolio but visible drift exists

### Autoencoder — Fails ❌
- Financial Services: ~43% vs 12% in index — severe overconcentration
- Utilities: ~15% vs 2.5%
- Communication Services, Healthcare, Industrials, Consumer Cyclical all near zero
- Root cause: communality-based selection gravitates toward stocks that load heavily on shared factors, which in this dataset cluster in Financials and Utilities

---

## 7. Cumulative Returns Summary

### Validation Period (Jan–Jun 2025)
- April 2025 tariff shock is clearly visible — all portfolios draw down, but at different magnitudes
- Lasso craters to ~83% while benchmark only dips to ~90% — tech/healthcare tilt amplified the drawdown
- Autoencoder overshoots on recovery (+10% by June vs benchmark's +4%) — explains positive val IR
- GA stays closest to benchmark throughout

### Holdout Period (Jul–Dec 2025)
- All three portfolios track much more closely
- All finish within ~2% of the benchmark by year end
- Autoencoder diverges downward in Nov-Dec — consistent with its negative holdout IR

---

## 8. Bugs Fixed During Development

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | plots.py | Cumulative returns used `(1+r).cumprod()` — wrong for log returns | `np.exp(r.cumsum())` |
| 2 | plots.py | Sector drift used ticker counts not portfolio weights | Aggregate actual portfolio weights by sector |
| 3 | plots.py | S&P 500 sector weights used ticker counts not market-cap weights | Fetch `marketCap` per ticker, compute weighted sector shares |
| 4 | plots.py | Sector drift bar colour hardcoded to Lasso-blue | Use `_method_colour(portfolio_label)` |
| 5 | method_autoencoder.py | Device selection always fell back to CPU | Proper if/elif/else for cuda/mps/cpu |
| 6 | method_ga.py | Crossover padded children with random stocks after set deduplication | Use union of parents as pool, sample k from it |
| 7 | method_lasso.py | Weights came directly from Lasso coefficients, not QP — inconsistent with other methods | Pass selected tickers to `fit_weights_qp` after selection |
| 8 | generate_sector_map.py | Only stored ticker and sector, no market cap | Also fetch and store `marketCap` per ticker |

---

## 9. Design Decisions & Their Justification

### Why weighted sum of log returns is used as portfolio return
```
rp,t = Σ wᵢ · rᵢ,t
```
Theoretically, the exact portfolio log return is `log(Σ wᵢ · exp(rᵢ,t))`. The weighted sum is a first-order approximation. For daily returns (±1-2%), the error is negligible and this is standard practice.

### Why QP minimises variance not std
Same argmin — minimising `Var(active)` produces identical weights as minimising `std(active)`. TE is then reported as std for convention.

### Why the two-stage approach
Full simultaneous optimisation of stock selection + weights is MIQP (NP-hard). Two-stage decomposition (select via heuristic, then solve convex QP) is the standard industry solution. Stage 1 is where methods differentiate; Stage 2 is standardised.

---

## 10. Key Takeaways for the Report

1. **GA is the best method** — lowest and most stable TE, most honest IR, cleanest sector balance
2. **Autoencoder's sector concentration is the main weakness** — communality-based selection is unsupervised and has no explicit sector constraint; it will always tend to pick stocks that co-move, which clusters in specific sectors
3. **Lasso is sensitive to extreme market events** — its tilt amplified the April 2025 drawdown significantly; adding QP weights helped but did not eliminate the sector drift
4. **IR sign flips across val/holdout** for Lasso and AE indicate unstable active bets, not genuine replication quality
5. **GA's holdout IR of -0.10** is the closest to the ideal of 0 — this is the strongest evidence that GA actually replicates the index rather than accidentally running an active strategy
