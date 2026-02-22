# Assignment2 Pipeline Reference

This document explains:
- what each script/function does,
- what each generated data file contains,
- and the schema contracts between pipeline stages.

## 1. End-to-end flow

Pipeline runner:
- `run_pipeline.sh`

Execution order:
1. `src/data/fetch_market_data.py`
2. `src/data/fetch_macro_data.py`
3. `src/data/fetch_fundamentals.py`
4. `src/data/fetch_sentiment.py`
5. `src/features/build_features.py`
6. `src/models/train_model.py`
7. `src/portfolio/portfolio.py`

## 2. Fundamental-data contract (updated)

`src/data/fetch_fundamentals.py` now writes a long-format file with a fixed schema:
- `Date`
- `Ticker`
- `PE`
- `Debt_Equity`
- `ROE`
- `EPS`
- `Revenue`
- `EBITDA_Margin`
- `Promoter_Holding`

Design intent:
- Constant columns across all tickers.
- Row growth with `Date x Ticker` (business-day panel).
- 45-day reporting lag is applied before forward-fill to avoid look-ahead bias.

Important note:
- Current cached raw quarterly files mainly provide `EPS`, `Revenue/Sales`, and `OPM`-style margin.
- If `PE`, `Debt_Equity`, `ROE`, `Promoter_Holding` are not available in the source table, they remain `NaN` (column still exists to keep schema stable).

## 3. Function reference

### 3.1 `src/data/fetch_market_data.py`

- `fetch_single(name, ticker)`
  - Downloads OHLCV from Yahoo Finance for one ticker.
  - Computes `Log_Return` and `Return`.
  - Saves raw ticker CSV to `data/raw/yahoo/{ticker}.csv`.

- `build_panel(ticker_dfs)`
  - Stacks all ticker data into long panel format.
  - Writes `data/processed/panel_daily.csv`.

- `split_forward_test(panel)`
  - Splits panel into train and forward-test by `FORWARD_TEST_START`.
  - Writes:
    - `data/processed/panel_train.csv`
    - `data/processed/panel_forward_test.csv`

- `main()`
  - Orchestrates market data ingestion and splitting.

### 3.2 `src/data/fetch_macro_data.py`

- `fetch_yf_series(name, yf_ticker)`
  - Fetches one macro series from Yahoo Finance (`Close`).

- `fetch_fred_csv(series_id, name)`
  - Fetches one macro series from FRED CSV endpoint.

- `resample_to_daily(s, method="ffill")`
  - Reindexes a series to daily calendar frequency.

- `lag_series(df, lag=1)`
  - Applies lag to remove same-day leakage.

- `main()`
  - Builds merged macro panel, derived macro returns/changes, lagged features.
  - Writes:
    - `data/processed/macro_raw_daily.csv`
    - `data/processed/macro_daily.csv`

### 3.3 `src/data/fetch_fundamentals.py`

- `coerce_numeric_columns(df)`
  - Converts string-form numeric columns to floats.
  - Handles commas, `%`, blanks, NBSP.

- `normalize_label(label)`
  - Normalizes column labels for robust alias matching.

- `extract_standard_fundamentals(quarterly_df)`
  - Extracts only assignment-required metrics into fixed columns:
    `PE, Debt_Equity, ROE, EPS, Revenue, EBITDA_Margin, Promoter_Holding`.

- `load_cached_quarterly(cache_paths)`
  - Loads previously saved quarterly raw files if available.
  - Lets pipeline proceed even when scraping fails.

- `scrape_screener(ticker)`
  - Scrapes quarterly table from Screener consolidated page.
  - Saves raw quarterly CSV.

- `fetch_fmp_fundamentals(ticker)`
  - Optional alternative source (FMP), currently disabled by default.

- `align_to_daily(quarterly_df, reporting_lag_days=45)`
  - Applies reporting lag, then forward-fills on business-day index.

- `main(refetch=False)`
  - Per ticker:
    - uses cache first (unless `--refetch`),
    - falls back to fetch,
    - standardizes metrics,
    - builds daily long panel with `Ticker`.
  - Writes:
    - `data/processed/fundamentals_daily.csv`

### 3.4 `src/data/fetch_sentiment.py`

- `_headline_cache_candidates(ticker)`
  - Resolves canonical and legacy cache file names.

- `fetch_gdelt(ticker, start, end, refetch=False)`
  - Fetches and caches GDELT headlines in quarterly chunks.

- `fetch_newsapi(ticker, api_key, days_back=30)`
  - Optionally fetches recent headlines from NewsAPI.

- `load_finbert()`
  - Loads FinBERT pipeline (or fallback sentiment model).

- `score_headlines(headlines, model, model_type)`
  - Scores text headlines into sentiment scores.

- `compute_daily_sentiment(df, model, model_type)`
  - Aggregates headline sentiment to daily series.

- `main(newsapi_key="", refetch=False)`
  - Builds full sentiment panel and writes:
    - `data/processed/news_sentiment.csv`

### 3.5 `src/features/build_features.py`

- `compute_rsi(series, period=14)`
  - RSI indicator.

- `compute_macd(series, fast=12, slow=26, signal=9)`
  - MACD line and signal line.

- `compute_bollinger_width(series, period=20)`
  - Bollinger width volatility feature.

- `add_technical_features(df)`
  - Adds all technical signals and `target` (next-day log return).

- `load_panel()`
  - Loads `panel_daily.csv`.

- `load_macro()`
  - Loads lagged macro panel.

- `load_fundamentals()`
  - Loads long-format fundamentals panel.
  - Enforces expected columns and fixed schema.

- `load_sentiment()`
  - Loads sentiment panel with legacy `M&M` compatibility handling.

- `build_ticker_features(ticker_name, panel, macro, fundamentals, sentiment)`
  - Builds feature set for one ticker.
  - Joins:
    - market/technical,
    - macro,
    - standardized fundamentals,
    - sentiment.

- `scale_features(train_df, test_df)`
  - Robust-scaling on train, then transform test.

- `main()`
  - Generates and saves:
    - `data/processed/features_train.parquet`
    - `data/processed/features_forward_test.parquet`

### 3.6 `src/models/train_model.py`

- `make_xgb()`, `make_lgbm()`
  - Model factories with fallback estimators if packages unavailable.

- `directional_accuracy(y_true, y_pred)`
  - Sign-hit metric.

- `sharpe_ratio(returns, freq=252, risk_free_rate_annual=...)`
  - Annualized Sharpe using excess return over risk-free daily rate.

- `max_drawdown(equity_curve)`
  - Drawdown utility.

- `select_features_rfe(X_train, y_train, feature_cols)`
  - RFECV feature selection on train only.
  - Falls back gracefully to all features if RFECV fails.

- `winsorize_and_impute(X_train, X_test)`
  - Per-ticker preprocessing:
    - drop all-NaN columns,
    - winsorize by train quantiles,
    - median impute by train medians.

- `walk_forward_cv(X, y, n_splits=5)`
  - Time-series CV evaluation.

- `train_ticker(ticker, train_df, test_df)`
  - Trains one model pair (XGB + LGBM) for one ticker only.
  - Saves per-ticker model artifacts and feature list.

- `main()`
  - Runs training for all tickers, aggregates outputs:
    - `outputs/cv_scores.csv`
    - `outputs/feature_importance.csv`
    - `outputs/predictions_fwd_test.csv`
    - `outputs/model_xgb_{TICKER}.pkl`
    - `outputs/model_lgbm_{TICKER}.pkl`
    - `outputs/features_{TICKER}.pkl`

### 3.7 `src/portfolio/portfolio.py`

- `load_predictions()`
  - Loads model predictions for forward-test dates.

- `load_actual_returns()`
  - Loads realized forward-test returns from `panel_forward_test.csv`.

- `weight_predicted_return(predictions)`
  - Long-only proportional-to-positive-forecast weighting.

- `weight_equal(predictions)`
  - Equal weights among positive-forecast stocks.

- `weight_inverse_vol(predictions, historical_returns, lookback=63)`
  - Inverse volatility allocation.

- `weight_mean_variance(expected_returns, historical_returns, lookback=252)`
  - Markowitz optimization (long-only).

- `simulate_portfolio(predictions, actuals, train_actuals, method)`
  - Daily backtest loop over chosen weighting method.

- `compute_metrics(portfolio, actuals)`
  - Computes Sharpe, drawdown, hit ratio, cumulative return, volatility.

- `plot_equity_curve(portfolio, metrics, method)`
  - Saves equity curve and drawdown chart.

- `plot_weights_heatmap(portfolio, method)`
  - Saves stacked-area weights plot.

- `main(method="predicted_return")`
  - Runs portfolio construction and writes:
    - `outputs/portfolio_results_{method}.csv`
    - `outputs/portfolio_metrics_{method}.csv`
    - `outputs/equity_curve_{method}.png`
    - `outputs/weights_{method}.png`

## 4. Data file reference

### 4.1 Raw data files (`data/raw/`)

- `data/raw/yahoo/{TICKER}.csv`
  - Per-ticker OHLCV + returns from Yahoo.

- `data/raw/macro/{SERIES}_raw.csv`
  - Raw macro time series before panel merge/lag.

- `data/raw/news/{TICKER}_headlines.csv`
  - Raw headline cache used for sentiment.

- `data/raw/fundamentals/{TICKER}_quarterly_raw.csv`
  - Raw quarterly fundamentals pulled from source tables.

### 4.2 Processed data files (`data/processed/`)

- `panel_daily.csv`
  - Long market panel for all tickers.
  - Core columns: `Date, Open, High, Low, Close, Volume, Ticker, Log_Return, Return`.

- `panel_train.csv`
  - Subset of `panel_daily.csv` before forward-test start.

- `panel_forward_test.csv`
  - Subset of `panel_daily.csv` in forward-test window.

- `macro_raw_daily.csv`
  - Daily-aligned macro panel before lag suffixing.

- `macro_daily.csv`
  - Lagged macro panel (`*_lag1` columns).

- `news_sentiment.csv`
  - Daily sentiment panel with ticker-specific columns.

- `fundamentals_daily.csv`
  - Long standardized fundamentals panel:
    `Date, Ticker, PE, Debt_Equity, ROE, EPS, Revenue, EBITDA_Margin, Promoter_Holding`.

- `features_train.parquet`
  - Final train feature matrix (all tickers, with `ticker` and `target`).

- `features_forward_test.parquet`
  - Forward-test feature matrix (same schema as train).

### 4.3 Output files (`outputs/`)

- `model_xgb_{TICKER}.pkl`, `model_lgbm_{TICKER}.pkl`
  - Saved model objects per stock.

- `features_{TICKER}.pkl`
  - Selected feature names per stock model.

- `cv_scores.csv`
  - Fold-level validation metrics per ticker.

- `feature_importance.csv`
  - Aggregated feature importances across tickers.

- `predictions_fwd_test.csv`
  - Forward-test predictions per ticker with `Date` column.

- `portfolio_results_predicted_return.csv`
  - Daily simulated portfolio returns and weights.

- `portfolio_metrics_predicted_return.csv`
  - Summary portfolio KPIs.

- `equity_curve_predicted_return.png`, `weights_predicted_return.png`
  - Visualization artifacts for portfolio evaluation.

## 5. Current modeling safeguards

- No K-fold leakage: time-series split only.
- Risk-free-adjusted Sharpe in model validation.
- Per-ticker training is strictly separate.
- Per-ticker winsorization + median imputation before fitting.
- Sentiment naming backward compatibility (`M&M` -> `MM`).
- Fundamentals schema stability via fixed columns and long format.
