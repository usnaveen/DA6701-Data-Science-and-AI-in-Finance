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

NEWSAPI_KEY=""
PORTFOLIO_METHOD="inverse_vol"

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

echo ">>> Step 1: Fetch Market Data (OHLCV)"
python src/data/fetch_market_data.py

echo ""
echo ">>> Step 2: Fetch Macro Indicators"
python src/data/fetch_macro_data.py

echo ""
echo ">>> Step 3: Fetch Fundamental Data"
python src/data/fetch_fundamentals.py

echo ""
echo ">>> Step 4: Fetch News Sentiment"
if [[ -n "$NEWSAPI_KEY" ]]; then
    python src/data/fetch_sentiment.py --newsapi-key "$NEWSAPI_KEY"
else
    python src/data/fetch_sentiment.py
fi

echo ""
echo ">>> Step 5: Feature Engineering"
python src/features/build_features.py

echo ""
echo ">>> Step 6: Train Models & Validate"
python src/models/train_model.py

echo ""
echo ">>> Step 7: Portfolio Construction & Evaluation"
python src/portfolio/portfolio.py --method "$PORTFOLIO_METHOD"

echo ""
echo "============================================================"
echo "  Pipeline complete. Outputs in: ./outputs/"
echo "============================================================"
