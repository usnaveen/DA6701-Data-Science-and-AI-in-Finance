"""
train_model.py
==============
Trains return-prediction models for all 6 Indian equity tickers using:

  • Time-Series Walk-Forward Cross-Validation  (no data leakage)
  • XGBoost + LightGBM (ensemble)
  • Recursive Feature Elimination (RFE) for feature selection
  • Per-ticker winsorization + median imputation for missing/outlier robustness
  • Regularization (L1/L2 via model hyperparameters)
  • Evaluation: Sharpe (risk-free adjusted), MAE, RMSE, directional accuracy

Model architecture
------------------
  XGBRegressor   } averaged
  LGBMRegressor  }

Validation strategy
-------------------
  TimeSeriesSplit (sklearn) with n_splits=5.
  Each fold: train on historical, validate on next ~4-month window.
  Standard K-Fold is STRICTLY prohibited (destroys temporal order).

Output
------
  outputs/model_xgb_{TICKER}.json
  outputs/model_lgbm_{TICKER}.txt
  outputs/cv_scores.csv
  outputs/feature_importance.csv
  outputs/predictions_train.csv
  outputs/predictions_fwd_test.csv

Usage:
    python train_model.py
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

# Optional imports — graceful fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Will use sklearn Ridge as fallback.")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS = ["RELIANCE", "HDFCBANK", "INFY", "MM", "BHARTIARTL", "HUL"]

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_CV_SPLITS      = 5
MIN_TRAIN_SIZE   = 252     # ~1 year of trading days minimum
EXCLUDE_COLS     = ["ticker", "target", "Open", "High", "Low", "Close", "Volume",
                     "Log_Return", "Return", "Ticker"]
RISK_FREE_RATE_ANNUAL = 0.065
APPLY_WINSORIZATION = True
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99


# ─── Model Factory ────────────────────────────────────────────────────────────

def make_xgb():
    if HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=50, #300,
            max_depth=3, #4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,     # L1
            reg_lambda=1.0,    # L2
            n_jobs=-1,
            random_state=42,
            verbosity=0,
            min_child_weight=5,
        )
    else:
        return GradientBoostingRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )


def make_lgbm():
    if HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
    else:
        return Ridge(alpha=1.0)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def directional_accuracy(y_true, y_pred):
    """Hit ratio: fraction of days where predicted direction matches actual."""
    correct = np.sign(y_pred) == np.sign(y_true)
    return correct.mean()


def sharpe_ratio(
    returns: np.ndarray,
    freq: int = 252,
    risk_free_rate_annual: float = RISK_FREE_RATE_ANNUAL,
) -> float:
    """Annualised Sharpe ratio using excess returns over risk-free rate."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    rf_daily = risk_free_rate_annual / freq
    excess = r - rf_daily
    vol = excess.std()
    if vol == 0:
        return 0.0
    return (excess.mean() / vol) * np.sqrt(freq)


def max_drawdown(equity_curve: np.ndarray) -> float:
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / (cummax + 1e-10)
    return drawdown.min()


# ─── Feature Selection ────────────────────────────────────────────────────────

def select_features_rfe(X_train, y_train, feature_cols) -> list:
    """
    Use RFECV with Ridge as base estimator (fast) on training data only.
    Returns the selected feature names.
    """
    if len(feature_cols) <= 10:
        print(f"    RFE skipped (only {len(feature_cols)} features).")
        return list(feature_cols)

    print("    Running RFE feature selection ...")
    estimator = Ridge(alpha=1.0)
    tscv = TimeSeriesSplit(n_splits=3)

    rfe = RFECV(
        estimator=estimator,
        step=max(1, min(5, len(feature_cols) // 4)),
        cv=tscv,
        scoring="neg_mean_squared_error",
        min_features_to_select=min(25, len(feature_cols)),
        # n_jobs=1 avoids process-spawn failures in restricted environments.
        n_jobs=1,
    )
    try:
        rfe.fit(X_train, y_train)
        selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
        print(f"    Selected {len(selected)} / {len(feature_cols)} features")
        return selected
    except Exception as e:
        print(f"    RFE failed ({e}). Using all features.")
        return list(feature_cols)


def fit_preprocessor(X_train: pd.DataFrame) -> dict | None:
    """
    Fit per-ticker preprocessing using TRAIN ONLY:
      1) keep non-empty columns,
      2) winsorization caps from train quantiles,
      3) median imputation from train,
      4) RobustScaler fit on train.
    """
    valid_cols = [c for c in X_train.columns if X_train[c].notna().any()]
    if not valid_cols:
        return None

    X_tr = X_train[valid_cols].copy()

    if APPLY_WINSORIZATION:
        lower = X_tr.quantile(WINSOR_LOWER_Q)
        upper = X_tr.quantile(WINSOR_UPPER_Q)
        X_tr = X_tr.clip(lower=lower, upper=upper, axis=1)
    else:
        lower = pd.Series(-np.inf, index=valid_cols)
        upper = pd.Series(np.inf, index=valid_cols)

    medians = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(medians)

    scaler = RobustScaler()
    scaler.fit(X_tr)

    return {
        "valid_cols": valid_cols,
        "lower": lower,
        "upper": upper,
        "medians": medians,
        "scaler": scaler,
    }


def transform_with_preprocessor(X: pd.DataFrame, prep: dict) -> pd.DataFrame:
    """Apply a fitted preprocessor to any split (train/val/test)."""
    cols = prep["valid_cols"]
    Xt = X.reindex(columns=cols).copy()
    Xt = Xt.clip(lower=prep["lower"], upper=prep["upper"], axis=1)
    Xt = Xt.fillna(prep["medians"])
    Xt.loc[:, cols] = prep["scaler"].transform(Xt[cols])
    return Xt


# ─── Walk-Forward Validation ─────────────────────────────────────────────────

def walk_forward_cv(X_raw: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Time-series cross-validation.
    Returns dict of averaged metrics and fold-level predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)  # gap=1 avoids T overlap

    fold_metrics = []
    oof_preds    = pd.Series(index=y.index, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
        if len(train_idx) < MIN_TRAIN_SIZE:
            continue

        X_tr_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        prep = fit_preprocessor(X_tr_raw)
        if prep is None:
            continue

        X_tr = transform_with_preprocessor(X_tr_raw, prep)
        X_val = transform_with_preprocessor(X_val_raw, prep)

        fold_features = list(X_tr.columns)
        selected = select_features_rfe(X_tr.values, y_tr.values, fold_features)
        X_tr = X_tr[selected]
        X_val = X_val[selected]

        # Ensemble: fit both models
        m1 = make_xgb()
        m1.fit(X_tr, y_tr)

        m2 = make_lgbm()
        m2.fit(X_tr, y_tr)

        pred = (m1.predict(X_val) + m2.predict(X_val)) / 2
        oof_preds.iloc[val_idx] = pred

        mae    = mean_absolute_error(y_val, pred)
        rmse   = mean_squared_error(y_val, pred) ** 0.5
        da     = directional_accuracy(y_val.values, pred)
        # Long/short 1 unit based on predicted direction (no use of true sign).
        strategy_returns = np.sign(pred) * y_val.values
        sr = sharpe_ratio(strategy_returns, risk_free_rate_annual=RISK_FREE_RATE_ANNUAL)

        fold_metrics.append({
            "fold": fold + 1,
            "MAE": mae,
            "RMSE": rmse,
            "DirectionalAccuracy": da,
            "PseudoSharpe": sr,
        })
        print(f"    Fold {fold+1}: MAE={mae:.5f}  RMSE={rmse:.5f}  DA={da:.3f}  Sharpe={sr:.2f}")

    metrics_df = pd.DataFrame(fold_metrics)
    return metrics_df, oof_preds


# ─── Per-Ticker Training ──────────────────────────────────────────────────────

def train_ticker(
    ticker: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    print(f"\n{'='*50}")
    print(f"Training: {ticker}")
    print(f"{'='*50}")

    tr = train_df[train_df["ticker"] == ticker].copy().sort_index()
    te = test_df[test_df["ticker"] == ticker].copy().sort_index()

    if tr.empty:
        print(f"  No training data for {ticker}. Skipping.")
        return {}

    # Drop ALL non-numeric and excluded columns — catches 'ticker', 'Ticker', etc.
    feature_cols = [
        c for c in tr.columns
        if c not in EXCLUDE_COLS
        and tr[c].dtype != object
        and pd.api.types.is_numeric_dtype(tr[c])
    ]

    X_tr_raw = tr[feature_cols].copy()
    y_tr = tr["target"]
    X_te_raw = te[feature_cols].copy() if not te.empty else pd.DataFrame(index=te.index, columns=feature_cols)
    y_te = te["target"] if not te.empty else pd.Series(dtype=float)

    print(f"  Train raw: {X_tr_raw.shape}   Forward-test raw: {X_te_raw.shape}")

    # Step 1: Walk-forward CV with fold-wise preprocessing (no leakage)
    print("  Walk-forward cross-validation ...")
    cv_metrics, oof_preds = walk_forward_cv(X_tr_raw, y_tr, n_splits=N_CV_SPLITS)

    # Step 2: Fit final preprocessing on all train data only
    prep = fit_preprocessor(X_tr_raw)
    if prep is None:
        print(f"  No usable features for {ticker} after preprocessing. Skipping.")
        return {}

    X_tr = transform_with_preprocessor(X_tr_raw, prep)
    X_te = transform_with_preprocessor(X_te_raw, prep) if not X_te_raw.empty else pd.DataFrame(index=X_te_raw.index, columns=prep["valid_cols"])
    print(f"  Train preprocessed: {X_tr.shape}   Forward-test preprocessed: {X_te.shape}")

    # Step 3: Feature selection for final model (train only)
    selected_features = select_features_rfe(X_tr.values, y_tr.values, list(X_tr.columns))
    X_tr_sel = X_tr[selected_features]
    X_te_sel = X_te[selected_features] if not X_te.empty else pd.DataFrame(index=X_te.index, columns=selected_features)

    # Step 4: Final model fit on all training data
    print("  Training final model on full training set ...")
    m1 = make_xgb()
    m1.fit(X_tr_sel, y_tr)

    m2 = make_lgbm()
    m2.fit(X_tr_sel, y_tr)

    # Step 5: Feature importance
    if HAS_XGB and isinstance(m1, xgb.XGBRegressor):
        fi = pd.Series(m1.feature_importances_, index=selected_features, name=ticker)
    else:
        fi = pd.Series(np.abs(m1.coef_) if hasattr(m1, "coef_") else 0,
                       index=selected_features, name=ticker)

    # Step 6: Forward-test predictions
    fwd_preds = pd.Series(dtype=float)
    if not X_te_sel.empty:
        fwd_preds = pd.Series(
            (m1.predict(X_te_sel) + m2.predict(X_te_sel)) / 2,
            index=X_te_sel.index,
            name=ticker,
        )
        fwd_mae = mean_absolute_error(y_te.dropna(), fwd_preds.loc[y_te.dropna().index])
        fwd_da  = directional_accuracy(y_te.dropna().values, fwd_preds.loc[y_te.dropna().index].values)
        print(f"  Forward-test → MAE={fwd_mae:.5f}  DA={fwd_da:.3f}")

    # Save models
    joblib.dump(m1, OUTPUT_DIR / f"model_xgb_{ticker}.pkl")
    joblib.dump(m2, OUTPUT_DIR / f"model_lgbm_{ticker}.pkl")
    joblib.dump(selected_features, OUTPUT_DIR / f"features_{ticker}.pkl")
    joblib.dump(prep, OUTPUT_DIR / f"preprocess_{ticker}.pkl")

    return {
        "cv_metrics":    cv_metrics,
        "oof_preds":     oof_preds,
        "fwd_preds":     fwd_preds,
        "feature_imp":   fi,
        "selected_feats": selected_features,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 6 – Model Training & Validation")
    print("=" * 60)

    # Load feature matrices
    train_path = PROCESSED_DIR / "features_train.parquet"
    test_path  = PROCESSED_DIR / "features_forward_test.parquet"

    if not train_path.exists():
        raise FileNotFoundError("Run build_features.py first.")

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()

    # Train each ticker
    all_cv_metrics   = []
    all_oof_preds    = []
    all_fwd_preds    = []
    all_feature_imps = []

    for ticker in TICKERS:
        results = train_ticker(ticker, train_df, test_df)
        if not results:
            continue

        cv = results["cv_metrics"].copy()
        cv.insert(0, "ticker", ticker)
        all_cv_metrics.append(cv)

        oof = results["oof_preds"].rename(ticker)
        all_oof_preds.append(oof)

        if not results["fwd_preds"].empty:
            all_fwd_preds.append(results["fwd_preds"])

        all_feature_imps.append(results["feature_imp"])

    # Save aggregated outputs
    if all_cv_metrics:
        cv_df = pd.concat(all_cv_metrics, ignore_index=True)
        cv_df.to_csv(OUTPUT_DIR / "cv_scores.csv", index=False)
        print(f"\nCV scores → {OUTPUT_DIR}/cv_scores.csv")
        print(cv_df.groupby("ticker")[["DirectionalAccuracy", "PseudoSharpe"]].mean().round(3))

    if all_feature_imps:
        fi_df = pd.concat(all_feature_imps, axis=1).fillna(0)
        fi_df["mean"] = fi_df.mean(axis=1)
        fi_df.sort_values("mean", ascending=False, inplace=True)
        fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv")
        print("\nTop 10 features (avg importance):")
        print(fi_df["mean"].head(10).round(4))

    if all_fwd_preds:
        fwd_df = pd.concat(all_fwd_preds, axis=1).sort_index()
        fwd_df.index.name = "Date"
        fwd_df.to_csv(OUTPUT_DIR / "predictions_fwd_test.csv", index_label="Date")
        print(f"\nForward-test predictions → {OUTPUT_DIR}/predictions_fwd_test.csv")

    print("\nAll models trained. Done.")


if __name__ == "__main__":
    main()
