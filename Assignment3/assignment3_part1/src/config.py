from pathlib import Path

# Project root (assignment3_part1)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

TICKERS = [
    "TCS.NS", "INFY.NS", "HCLTECH.NS",
    "TATASTEEL.NS", "HINDALCO.NS", "JINDALSTEL.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS"
]

YEARS_TOTAL = 5
TRAIN_YEARS = 4
BOOTSTRAP_ITERATIONS = 100