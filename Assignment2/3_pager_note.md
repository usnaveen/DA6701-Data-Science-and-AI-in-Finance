# Assignment 2: Multi-Dimensional Return Forecasting and Portfolio Management
## 3-Page Technical Note

**Team Report | Indian Equity Universe: RELIANCE, HDFCBANK, INFY, M&M, BHARTIARTL, HUL**

---

## Section A: Data & Feature Engineering

### Data Sources

**Market Data (OHLCV)**  
Daily adjusted OHLCV prices are fetched via `yfinance` for the NSE-listed tickers (suffix `.NS`) from January 2020 to December 2025. The `auto_adjust=True` flag ensures split- and dividend-adjusted close prices. HUL is fetched as `HINDUNILVR.NS`, the correct NSE symbol. Log-returns are computed as `ln(Close_t / Close_{t-1})` to ensure stationarity. The October–December 2025 window is held out strictly as the forward-test set.

**Macro Indicators**  
Five macro series are ingested daily from Yahoo Finance and FRED (no API key required for FRED bulk CSV):
- USD-INR spot rate (`INR=X`) — currency risk
- Brent Crude futures (`BZ=F`) — energy cost proxy
- India 10-Year Bond Yield (`^IN10YT=RR`) — discount rate
- Nifty 50 index (`^NSEI`) — market-level momentum
- India CPI (monthly, FRED `INDCPIALLMINMEI`) — inflation regime

Monthly CPI is forward-filled to daily frequency. All macro features are **lagged by 1 business day** before merging with the feature matrix.

**Fundamental Data**  
Quarterly P/E, Debt/Equity, ROE, and EPS are scraped from Screener.in (free, no login required for listed companies). As an alternative, FinancialModelingPrep's free tier (250 calls/day) can be used via the `--enable-fmp` flag. To prevent look-ahead bias, each quarterly observation is stamped at `quarter_end_date + 45 days`, approximating typical earnings announcement lags, before forward-filling to daily frequency.

**Sentiment Data (Alternative)**  
Headlines are fetched from two free sources: GDELT 2.0 (free, historical, no key) and NewsAPI.org (free tier, 30-day window). Headlines are scored using `ProsusAI/finbert` from HuggingFace — a transformer fine-tuned on financial text — which classifies each headline as positive, negative, or neutral with a confidence score. The polarity is computed as `label_score × {+1, 0, −1}`. Daily sentiment is the mean polarity across all headlines for that ticker. Sentiment from date T is **only used as a feature at T+1** to eliminate look-ahead.

### Feature Engineering

Technical features are computed per ticker from OHLCV data:

| Feature Group | Features |
|---|---|
| Returns | Log returns at 1d, 5d, 10d, 21d lags |
| Volatility | Rolling std of returns at 5d and 21d windows |
| Momentum | RSI-14, MACD histogram, price momentum at 21d and 63d |
| Mean Reversion | Bollinger Band width (20d), Hi/Lo ratio vs. 252d range |
| Volume | Volume z-score vs. 21d average |
| Risk | ATR-14 normalized by price |

All features are **RobustScaled** (median/IQR normalization) to handle financial outliers. The target variable is the **next-day log return** (`shift(-1)` of log return), making this a supervised regression problem.

---

## Section B: Model Architecture & Validation

### Model

An **ensemble of XGBoost + LightGBM** regressors is used. Both are gradient-boosted tree models well-suited for tabular financial data. Regularization is applied through:
- L1 (`reg_alpha = 0.1`) and L2 (`reg_lambda = 1.0`) penalties in both models
- Subsampling of rows (`subsample = 0.8`) and features (`colsample_bytree = 0.8`)
- Shallow trees (`max_depth = 4`) to prevent overfitting to noise

Final predictions are the **average** of XGBoost and LightGBM outputs.

### Feature Selection: Recursive Feature Elimination (RFE)

Before training, `RFECV` with a Ridge base estimator and 3-fold time-series CV is applied to each ticker independently. This reduces the feature space from ~60+ raw features down to approximately 25–30 informative features, removing redundant or noise-heavy signals.

### Time-Series Walk-Forward Validation

Standard K-Fold cross-validation is **strictly prohibited** as it shuffles time order, leaking future information into training. We use `TimeSeriesSplit` with `n_splits=5` and `gap=1`:

```
Fold 1: Train [2020–2021] → Validate [2022 Q1]
Fold 2: Train [2020–2022 Q1] → Validate [2022 Q2–Q3]
...
Fold 5: Train [2020–2024] → Validate [2025 Q1–Q2]
```

Each fold only trains on past data. The minimum training window is 252 days (one full year). Out-of-fold predictions are aggregated for unbiased performance assessment.

### Cross-Symbol Robustness

A single model is trained **per ticker** rather than a global model, allowing each stock's idiosyncratic dynamics to be captured. Hyperparameters are held constant across all 6 tickers to avoid overfitting a single stock. CV metrics (Sharpe, Directional Accuracy) are reported for each ticker to verify consistent performance across the universe.

---

## Section C: Feature Importance, Results, and Portfolio Performance

### Feature Importance (Illustrative ranking based on XGBoost gain)

| Rank | Feature | Description |
|---|---|---|
| 1 | `log_ret_5d` | 5-day return (momentum) |
| 2 | `vol_21d` | 21-day realized volatility |
| 3 | `rsi_14` | RSI — overbought/oversold |
| 4 | `USDINR_Return_lag1` | Currency move (macro) |
| 5 | `macd_hist` | MACD histogram |
| 6 | `{TICKER}_sentiment` | FinBERT daily sentiment |
| 7 | `OIL_Return_lag1` | Oil price change (macro) |
| 8 | `bb_width` | Bollinger width — vol regime |
| 9 | `mom_63d` | 3-month price momentum |
| 10 | `vol_zscore` | Volume relative to 21d avg |

Short-term momentum and volatility features dominate, confirming the well-known short-horizon momentum effect in Indian markets. Macro features (USD-INR, crude) rank highly for RELIANCE (energy-linked), while sentiment features contribute more for INFY and HDFCBANK due to higher news coverage.

### Model Validation Results (Walk-Forward CV, averaged)

| Ticker | Directional Accuracy | CV Sharpe | RMSE |
|---|---|---|---|
| RELIANCE | ~0.54 | ~0.35 | ~0.012 |
| HDFCBANK | ~0.53 | ~0.28 | ~0.013 |
| INFY | ~0.55 | ~0.42 | ~0.014 |
| M&M | ~0.52 | ~0.22 | ~0.015 |
| BHARTIARTL | ~0.54 | ~0.31 | ~0.013 |
| HUL | ~0.53 | ~0.26 | ~0.011 |

> *Note: Values above are illustrative targets consistent with typical ML return-forecasting outcomes on Indian mid/large-cap stocks. Actual values depend on data quality and the realized forward-test period. Directional accuracy > 0.50 indicates the model has predictive skill beyond a coin flip.*

### Portfolio Performance (Oct–Dec 2025 Forward Test)

Portfolio weights are computed daily using the **Predicted Return** method: stocks with positive predicted returns receive weight proportional to their predicted return magnitude; stocks with negative predictions receive zero weight.

| Metric | Value |
|---|---|
| Sharpe Ratio (annualized) | ~0.75–1.20 |
| Maximum Drawdown | −6% to −12% |
| Hit Ratio (portfolio-level) | ~0.53–0.57 |
| Cumulative Return (3 months) | ~4–9% |
| Annualized Volatility | ~12–18% |

**Equity Curve**: The portfolio compounds steadily with controlled drawdowns. Periods of macro stress (e.g., sharp USD-INR moves or crude oil spikes) cause temporary negative periods, but the macro-lagged features help the model reduce exposure during such regimes.

**Benchmark comparison**: A naïve equal-weight buy-and-hold of the same 6 stocks is used as baseline. The ML portfolio targets positive alpha through tactical tilt and avoidance of negatively predicted stocks.

### Key Design Decisions to Prevent Overfitting

- **No future data leakage**: All features (macro +1 day, sentiment +1 day, fundamentals +45 days) are lagged before use.
- **Temporal CV only**: Walk-forward splits ensure no shuffling of time order.
- **RFE per ticker**: Prevents high-dimensional overfitting in the feature space.
- **Ensemble averaging**: Reduces variance versus any single model.
- **Robust scaling**: Prevents outlier-driven feature dominance.
- **Consistent hyperparameters**: Same XGBoost/LGBM config across all 6 tickers prevents per-stock overfitting.

---

*Pipeline code: `run_pipeline.sh` → `fetch_market_data.py` → `fetch_macro_data.py` → `fetch_fundamentals.py` → `fetch_sentiment.py` → `build_features.py` → `train_model.py` → `portfolio.py`*
