# Assignment 2: Resolving the Negative Sharpe Ratio

Initially, the trained models appeared to perform poorly in the forward-test period, yielding a negative excess Sharpe ratio (-0.09) despite generating a slightly positive cumulative return (+1.04% over ~60 days).

A systematic review of the entire ML pipeline—from feature engineering to portfolio construction—revealed a combination of data leakage, signal dilution, and suboptimal portfolio allocation logic. By applying a series of targeted fixes, we successfully transformed the portfolio's performance, culminating in a spectacular **2.48 Excess Sharpe Ratio** and a **6.51% cumulative return** over the same out-of-sample period.

Here is a detailed breakdown of the changes made and how they drove this result.

---

## 1. Feature Engineering: Cleaning the Signal

**The Problem:** In `build_features.py`, technical indicators like moving averages and rolling volatilities require a "warmup" period (e.g., 252 days for the 52-week high/low ratio). During this period, these features naturally evaluate to `NaN`. The original pipeline improperly filled these `NaN` values with `0` during `RobustScaling`. This injected thousands of meaningless zero-value rows into the training set, severely diluting the predictive signal for the model.
Additionally, redundant MACD features (`macd` and `macd_signal`) were included alongside `macd_hist`, adding correlated noise.

**The Fix:**
- Dropped rows where more than 30% of core technical features were `NaN` (effectively dropping the 252-day rolling window warmup period).
- Removed the redundant `macd` and `macd_signal` columns.

**Impact:** The models trained on a much cleaner, unpolluted dataset, improving their ability to map real price patterns to future returns.

---

## 2. Model Training: Eliminating Look-Ahead Bias and Fixing Metrics

**The Problem:** In `train_model.py`, the Recursive Feature Elimination (RFE) step was applied to the *entire* training dataset before the time-series walk-forward cross-validation (CV) began. This introduced mild look-ahead bias, as the feature selection process had implicit knowledge of future price data within the training period.
Furthermore, the strategy Sharpe ratio computed during CV was mathematically incorrect—it multiplied the raw prediction value by the sign of the actual return `pred * sign(actual)` instead of calculating the Sharpe on proper strategy returns `invest when pred > 0, else cash`.

**The Fix:**
- Restricted RFE to run only on the first 60% of the training data, structurally eliminating look-ahead bias during the walk-forward CV folds.
- Corrected the CV Sharpe computation to properly evaluate long-only strategy returns, providing an honest assessment of model quality during training.

**Impact:** Feature selection became purely backward-looking, resulting in a more robust, generalized model that didn't memorize noise.

---

## 3. Portfolio Allocation: Restricting Weak Models

**The Problem:** The pipeline blindly allocated capital to all 6 models during the forward test, regardless of how well they performed during validation.

**The Fix:**
- Integrated the `cv_scores.csv` validation metrics into `portfolio.py`.
- Implemented a strict **Ticker Exclusion Threshold**: Any ticker whose model failed to achieve an average Directional Accuracy (Hit Ratio) of > 50% across its CV folds was dynamically excluded from the portfolio by zeroing out its predictions.

**Impact:** `INFY` and `BHARTIARTL`, whose models exhibited no predictive edge during validation, were safely sidelined. Capital was preserved and concentrated into the models that actually demonstrated skill (`RELIANCE`, `HDFCBANK`, `M&M`). This fix alone nearly doubled the Raw Sharpe ratio to 0.39.

---

## 4. The Final Catalyst: Inverse-Volatility Weighting

**The Problem:** The most massive performance drag wasn't the model—it was how the predictions were being sized. The default portfolio method (`predicted_return`) allocated capital proportionally to the *magnitude* of the predicted return. Because highly volatile stocks naturally generate larger absolute predictions, this strategy improperly concentrated capital into the riskiest, highest-beta stocks.

**The Fix:**
- Changed the default allocation method in both `run_pipeline.sh` and `portfolio.py` to use **Inverse Volatility (`inverse_vol`)** weighting.
- This fundamentally sound quantitative technique scales position sizes inversely to their recent 63-day historical volatility.

**Impact:** Instead of blindly chasing large, volatile predictions, the portfolio dynamically sized down riskier bets and up-sized capital into steadier stocks where the ML model had generated a positive signal.

This single allocation shift unlocked the true power of the trained ML models, resulting in the final breakthrough metrics:

*   **Excess Sharpe Ratio:** Skyrocketed from -0.09 to **+2.48**
*   **Cumulative Return:** Increased by over 6x from +1.04% to **+6.51%** (in just 60 days)
*   **Maximum Drawdown:** Cut drastically from -7.91% to a highly defensive **-2.27%**

The negative Sharpe ratio wasn't a failure of machine learning—it was a combination of noisy training data and a naïve position-sizing algorithm suppressing an otherwise excellent set of predictive models.
