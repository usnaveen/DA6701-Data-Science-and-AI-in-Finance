# Changes

## 2026-04-01 - Planning Pass

### Approach

- Read the `P1` notes in `Assignment4/assignment4_plan.md`.
- Installed `pymupdf` into `.venv` so the assignment PDF could be read locally without guessing the rubric.
- Extracted the assignment text from `Assignment4/Assignment IV.pdf` and cross-checked the train/test and holdout windows.
- Scanned the `Assignment4/Mega` folder to identify invalid, empty, and late/incomplete CSVs before drafting the implementation plan.

### Changes Made

- Added [plan-of-action.md](/Users/tanmaygawande/Desktop/Code/Sem2/AI%20in%20Finance/DA6701-Data-Science-and-AI-in-Finance/plan-of-action.md) with the proposed notebook implementation flow and explicit assumptions for review.
- Added this `changes.md` file to log the planning work.
- Confirmed these files should be excluded as data issues:
  - `sp500_tickers_cache.csv` because it is not a price-history CSV
  - `ANDV.csv`
  - `ESRX.csv`
  - `EVHC.csv`
  - `NFX.csv`
  - `SCG.csv`
- Confirmed that some readable stock files still need to be excluded from the final aligned matrix because they do not cover the full assignment horizon, including examples such as `ABNB.csv`, `SNDK.csv`, `FB.csv`, and `Q.csv`.

### Notes

- No notebook implementation has been executed yet.
- Work is paused pending review of `plan-of-action.md`.

## 2026-04-01 - P1 Day 1 Implementation

### Approach

- Created a separate notebook `p1d1.ipynb` in `Assignment4/`, as requested.
- Updated the split logic to:
  - train: Jan 2020 to Dec 2024
  - validation: Jan 2025 to Jun 2025
  - final test: Jul 2025 to Dec 2025
- Used a one-month training-start grace rule ending on `2020-02-03`.
- Excluded late entrants and early exits from the final matrix instead of shrinking the window into late 2025.
- Enforced a clean training block first, then forward-filled slight validation/test holes at the price level before computing returns.
- Installed `pyarrow` in `.venv` so the final datasets could be saved as parquet files.

### Changes Made

- Added `Assignment4/p1d1.ipynb` with the full P1 data pipeline:
  - CSV validation and issue logging
  - benchmark loading
  - stock-universe screening
  - alignment to the benchmark trading calendar
  - common training start enforcement
  - validation/test forward-fill handling
  - log-return construction
  - split dataset export
- Executed the notebook successfully and generated these artifacts:
  - `p1d1_stock_returns.parquet`
  - `p1d1_benchmark_returns.parquet`
  - `p1d1_model_ready_returns.parquet`
  - `p1d1_train_returns.parquet`
  - `p1d1_val_returns.parquet`
  - `p1d1_test_returns.parquet`
  - `p1d1_final_prices.parquet`
  - `p1d1_kept_universe.parquet`
  - `p1d1_excluded_universe.parquet`
  - `p1d1_file_issues.csv`
  - `p1d1_train_gap_exclusions.csv`
  - `p1d1_summary.csv`

### Output Summary

- Readable stock price files: `578`
- File issues excluded immediately: `6`
  - `ANDV.csv`
  - `ESRX.csv`
  - `EVHC.csv`
  - `NFX.csv`
  - `SCG.csv`
  - `sp500_tickers_cache.csv`
- Coverage-based exclusions: `24`
  - `21` late entrants
  - `3` early exits
- Additional training-gap exclusion: `CVG` with `25` missing training prices
- Final stock universe: `553` stocks
- Final model-ready return matrix: `1507 x 554`
  - `553` stock columns
  - `1` benchmark column: `^GSPC`
- Final return date range: `2020-01-03` to `2025-12-31`
- Split sizes:
  - train: `1257 x 554`
  - validation: `122 x 554`
  - test: `128 x 554`
- Verified the saved return matrices contain no missing values.

### Notes

- The common training price start remained at `2020-01-02`, so the one-month grace rule did not force a later shared start.
- Validation and test gaps were handled via forward-fill at the price level before return construction.

## 2026-04-01 - Artifact Documentation Update

### Approach

- Added a short handoff note so teammates can immediately tell which parquet file to use for modeling, benchmarking, evaluation, and audit checks.

### Changes Made

- Added `Assignment4/readme-parquet.md`.
- Documented the purpose of each generated parquet and CSV artifact.
- Included recommended usage for `P1`, `P2`, `P3`, and `P4`.
- Added a minimal loading example showing how to separate stock returns from the `^GSPC` benchmark column.

### Notes

- No data artifacts were regenerated in this step.
- This was a documentation-only update.

## 2026-04-03 - Final Report Draft

### Approach

- Read the Assignment IV deliverables page, the report notes, the data-preparation and portfolio-replication notebooks, the plotting helpers, and all three method modules.
- Used the Assignment 3 LaTeX template as the styling reference, but rewrote the layout for a compact two-page Assignment IV report.
- Centered the report around the required deliverables from the assignment PDF:
  - sparsity vs tracking error
  - information ratio
  - sector drift

### Changes Made

- Created `Assignment4/final-compilation/assignment4_report.tex`.
- Wired the report to the generated figures:
  - `figures/te_vs_k.png`
  - `figures/information_ratio.png`
  - `figures/sector_drift_ga.png`
- Added a compact methodology summary, data-pipeline summary, and a `k=50` results table for Lasso, Autoencoder, and GA.
- Highlighted GA as the best overall method based on holdout tracking quality and sector alignment.
- Compiled the LaTeX successfully to confirm that the file is valid and stays within 2 pages.

### Notes

- The compiled output is `Assignment4/final-compilation/assignment4_report.pdf`.
- A few harmless LaTeX font-size substitution warnings appeared during compilation, but the document built successfully.

## 2026-04-03 - Member Cover Page and Template Update

### Approach

- Used the member-cover-page structure from the provided `Assignment3_Final_Report.tex`.
- Applied the same cover-page layout to the Assignment 4 report.
- Updated the root LaTeX template so future reports can reuse the same member-page format directly.

### Changes Made

- Updated `Assignment4/final-compilation/assignment4_report.tex` to include a front page with:
  - course header
  - assignment title
  - report topic
  - group member names and roll numbers
- Recompiled the Assignment 4 report successfully after adding the front page.
- Updated the project-root `Portfolio_Report_LaTeX_Template.tex` to include:
  - reusable cover-page variables
  - a generic member table
  - the same front-page pattern before the main report body

### Notes

- The compiled Assignment 4 PDF now has `3` pages total: `1` member cover page plus `2` report pages.
- I did not compile the root template because it still contains placeholder content and placeholder figure paths by design.

## 2026-04-03 - Final Report Layout Revision

### Approach

- Revised the Assignment 4 report structure to match the requested narrative flow and figure set more closely.
- Removed the summary KPI boxes so the content starts directly with the report sections.
- Expanded the visualization coverage to include all three sector-drift plots and both cumulative-return plots.

### Changes Made

- Removed the top metric boxes for readable CSVs, final stocks, issue counts, and best-method summary.
- Kept section labels in the `n | Title` format throughout the report.
- Renamed Sections 3 and 4 to remove the phrase `Required Deliverables`.
- Updated the sector-drift section to include:
  - `sector_drift_lasso.png`
  - `sector_drift_ae.png`
  - `sector_drift_ga.png`
- Added a new Section 5 for cumulative returns with:
  - `cumret_val.png`
  - `cumret_hold.png`
- Shifted the conclusion to Section 6 so the report ends with the conclusion as requested.
- Recompiled the report successfully after the layout revision.

### Notes

- The updated compiled output remains `Assignment4/final-compilation/assignment4_report.pdf`.
- Total compiled length remains `3` pages including the member cover page.
