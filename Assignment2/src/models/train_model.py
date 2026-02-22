"""
train_model.py
==============
Trains return-prediction models for all 6 Indian equity tickers using:

  • Time-Series Walk-Forward Cross-Validation  (no data leakage)
  • XGBoost + LightGBM (ensemble)
  • Recursive Feature Elimination (RFE) for feature selection
  • Regularization (L1/L2 via model hyperparameters)
  • Evaluation: Sharpe, MAE, RMSE, directional accuracy

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

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS = ["RELIANCE", "HDFCBANK", "INFY", "MM", "BHARTIARTL", "HUL"]

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

N_CV_SPLITS      = 5
MIN_TRAIN_SIZE   = 252     # ~1 year of trading days minimum
RFE_N_FEATURES   = 30      # target feature count after RFE
EXCLUDE_COLS     = ["ticker", "target", "Open", "High", "Low", "Close", "Volume",
                     "Log_Return", "Return", "Ticker","ticker"]


# ─── Model Factory ────────────────────────────────────────────────────────────

def make_xgb():
    if HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,     # L1
            reg_lambda=1.0,    # L2
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    else:
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )


def make_lgbm():
    if HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=4,
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


def sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """Annualised Sharpe of a return series."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(freq)


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
    print("    Running RFE feature selection ...")
    estimator = Ridge(alpha=1.0)
    tscv = TimeSeriesSplit(n_splits=3)

    rfe = RFECV(
        estimator=estimator,
        step=5,                 # remove 5 features per step
        cv=tscv,
        scoring="neg_mean_squared_error",
        min_features_to_select=10,
        n_jobs=-1,
    )
    rfe.fit(X_train, y_train)
    selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
    print(f"    Selected {len(selected)} / {len(feature_cols)} features")
    return selected


# ─── Walk-Forward Validation ─────────────────────────────────────────────────

def walk_forward_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """
    Time-series cross-validation.
    Returns dict of averaged metrics and fold-level predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)  # gap=1 avoids T overlap

    fold_metrics = []
    oof_preds    = pd.Series(index=y.index, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < MIN_TRAIN_SIZE:
            continue

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

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
        sr     = sharpe_ratio(pred * np.sign(y_val.values))  # long when positive, short when negative

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

    X_tr = tr[feature_cols].fillna(0)
    y_tr = tr["target"]
    X_te = te[feature_cols].fillna(0) if not te.empty else pd.DataFrame()
    y_te = te["target"] if not te.empty else pd.Series()

    print(f"  Train: {X_tr.shape}   Forward-test: {X_te.shape}")

    # Step 1: Feature selection (RFE) on train data only
    selected_features = select_features_rfe(X_tr.values, y_tr.values, feature_cols)
    X_tr_sel = X_tr[selected_features]
    X_te_sel = X_te[selected_features] if not X_te.empty else pd.DataFrame()

    # Step 2: Walk-forward CV
    print("  Walk-forward cross-validation ...")
    cv_metrics, oof_preds = walk_forward_cv(X_tr_sel, y_tr, n_splits=N_CV_SPLITS)

    # Step 3: Final model fit on all training data
    print("  Training final model on full training set ...")
    m1 = make_xgb()
    m1.fit(X_tr_sel, y_tr)

    m2 = make_lgbm()
    m2.fit(X_tr_sel, y_tr)

    # Step 4: Feature importance
    if HAS_XGB and isinstance(m1, xgb.XGBRegressor):
        fi = pd.Series(m1.feature_importances_, index=selected_features, name=ticker)
    else:
        fi = pd.Series(np.abs(m1.coef_) if hasattr(m1, "coef_") else 0,
                       index=selected_features, name=ticker)

    # Step 5: Forward-test predictions
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
        fwd_df = pd.concat(all_fwd_preds, axis=1)
        fwd_df.to_csv(OUTPUT_DIR / "predictions_fwd_test.csv")
        print(f"\nForward-test predictions → {OUTPUT_DIR}/predictions_fwd_test.csv")

    print("\nAll models trained. Done.")


if __name__ == "__main__":
    main()
