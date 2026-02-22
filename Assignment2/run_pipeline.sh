#!/bin/bash
# run_pipeline.sh
# ===============
# Master pipeline: runs all steps in order.
#
# Usage:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --newsapi-key YOUR_KEY
#
# Prerequisites:
# pip install yfinance pandas numpy scikit-learn xgboost lightgbm \
#               requests beautifulsoup4 transformers torch tqdm scipy \
#               matplotlib joblib pyarrow

set -e  # exit on error

NEWSAPI_KEY="77f16e3455e64337aab74474739d4898"
PORTFOLIO_METHOD="predicted_return"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --newsapi-key) NEWSAPI_KEY="$2"; shift 2;;
        --method)      PORTFOLIO_METHOD="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  Indian Equity Return Forecasting Pipeline"
echo "  6 stocks: RELIANCE, HDFCBANK, INFY, M&M, BHARTIARTL, HUL"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

if [[ -x "../.venv/bin/python3" ]]; then
    PYTHON_BIN="../.venv/bin/python3"
elif [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: python3 not found. Activate your environment or install python3."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"

echo ">>> Step 1: Fetch Market Data (OHLCV)"
"$PYTHON_BIN" src/data/fetch_market_data.py

echo ""
echo ">>> Step 2: Fetch Macro Indicators"
"$PYTHON_BIN" src/data/fetch_macro_data.py

echo ""
echo ">>> Step 3: Fetch Fundamental Data"
"$PYTHON_BIN" src/data/fetch_fundamentals.py

echo ""
echo ">>> Step 4: Fetch News Sentiment"
if [[ -n "$NEWSAPI_KEY" ]]; then
    "$PYTHON_BIN" src/data/fetch_sentiment.py --newsapi-key "$NEWSAPI_KEY"
else
    "$PYTHON_BIN" src/data/fetch_sentiment.py
fi

echo ""
echo ">>> Step 5: Feature Engineering"
"$PYTHON_BIN" src/features/build_features.py

echo ""
echo ">>> Step 6: Train Models & Validate"
"$PYTHON_BIN" src/models/train_model.py

echo ""
echo ">>> Step 7: Portfolio Construction & Evaluation"
"$PYTHON_BIN" src/portfolio/portfolio.py --method "$PORTFOLIO_METHOD"

echo ""
echo "============================================================"
echo "  Pipeline complete. Outputs in: ./outputs/"
echo "============================================================"
