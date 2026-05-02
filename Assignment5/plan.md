## 2-Day Team Plan — Assignment V: Goal-Based Portfolio Optimisation

---

### Member Roles & Ownership

| Member | Role | Core Responsibility |
|---|---|---|
| **Member 1** | Data & Statistics | Data pipeline + statistical foundation |
| **Member 2** | Simulation Engine | Monte Carlo core logic |
| **Member 3** | Optimiser & Results | Brute-force search + output |
| **Member 4** | Report & Bonus | Written report + SciPy bonus |

---

### Day 1 — Build Independently

**Member 1 — Data & Statistics**
- Download daily adjusted closing prices for all 5 tickers via `yfinance` (Jan 2014 – Dec 2023)
- Compute daily log returns
- Compute annualized expected returns (×252), annualized standard deviations, and the full 5×5 covariance matrix
- Export a clean module: `data.py` with functions `get_returns()`, `get_cov_matrix()`, `get_stats()`
- Also define the **savings schedule** — a 20-year array of annual contributions starting at ₹2,40,000 growing at 4% per year

**Member 2 — Simulation Engine**
- Build `simulator.py` — a single function `simulate_portfolio(weights, mu, cov, savings, goals, n_paths=5000)` that:
  - Generates 5,000 annual return paths using multivariate normal sampling
  - Applies portfolio return each year to grow the portfolio value
  - Adds the annual savings contribution
  - At each goal year, deducts the goal amount — if short, applies the 12% borrowing penalty and tracks the outstanding loan
  - Returns the array of final portfolio values at Year 20
- Work with **mock/hardcoded** `mu`, `cov`, `savings`, and `goals` values on Day 1 so you don't depend on Member 1

**Member 3 — Optimiser**
- Build `optimiser.py` that:
  - Generates all valid discrete weight combinations (weights from `{0, 0.25, 0.5, 0.75, 1.0}` summing to 1.0) — there are 56 valid combos
  - Loops over every combination, calls `simulate_portfolio()`, and records the **success probability** (fraction of 5,000 paths ending ≥ ₹1.5 Crores)
  - Identifies and stores the best portfolio for Sequence A and Sequence B
- On Day 1, use a **stub version** of `simulate_portfolio()` that returns random values so you can test your loop and output structure independently

**Member 4 — Report & Bonus**
- Draft the **report skeleton**: introduction, methodology summary, results table (placeholders), and discussion section
- Write the **bonus code** in `bonus.py` using `scipy.optimize.minimize` with a continuous weight optimizer (including short-selling), calling the same `simulate_portfolio()` stub
- Prepare the **final notebook structure** (`main.ipynb`) that imports from all three modules and runs end-to-end — leave result cells blank for Day 2

---

### Day 2 — Integrate & Finalise

**Morning — Integration (all members, ~2 hours)**

A brief sync where:
- Member 1 shares the finalised `data.py` outputs with everyone
- Member 2 replaces the mock values in their simulator with real `mu` and `cov`
- Member 3 replaces the stub `simulate_portfolio()` with Member 2's real function
- Member 4 plugs real outputs into the bonus optimizer and the report

**Afternoon — Polish & Wrap-up**

| Member | Task |
|---|---|
| **Member 1** | Validate data outputs, add docstrings, create the methodology flowchart |
| **Member 2** | Run edge-case checks on the simulator (e.g., zero portfolio, max borrowing), add inline comments |
| **Member 3** | Run the full brute-force for both Sequence A and B, generate comparison tables and success probability charts |
| **Member 4** | Fill in real results in the report, write the discussion comparing Seq A vs Seq B portfolios, run and verify the bonus optimizer |

---

### Interface Contract (agree on Day 1 morning — 15 min)

This is the shared "API" so everyone can work independently:

```python
# data.py — Member 1 exports these
mu        # np.array shape (5,)  — annualized returns
cov       # np.array shape (5,5) — annualized covariance matrix
savings   # np.array shape (20,) — annual contributions in ₹

# simulator.py — Member 2 exports this
simulate_portfolio(weights, mu, cov, savings, goals, n_paths=5000)
# weights: np.array (5,) summing to 1.0
# goals:   dict {year: target_amount}  e.g. {3: 1500000, 7: 2500000, ...}
# returns: np.array (5000,) of final portfolio values at Year 20

# optimiser.py — Member 3 exports this
run_optimisation(mu, cov, savings, goals)
# returns: (best_weights, best_success_prob)
```

---

### Deliverable Checklist by End of Day 2

- `data.py` — data download + stats *(Member 1)*
- `simulator.py` — Monte Carlo engine *(Member 2)*
- `optimiser.py` — brute-force search *(Member 3)*
- `bonus.py` — continuous SciPy optimizer *(Member 4)*
- `main.ipynb` — unified notebook with all results *(Member 4, assembled)*
- `report.pdf` — 1–2 page writeup with optimal weights + discussion *(Member 4)*
- Flowchart of methodology *(Member 1)*
