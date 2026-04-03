"""generate_sector_map.py — Fetch GICS sector labels for all universe tickers.

Run once: python generate_sector_map.py
Writes data/sector_map.csv with columns: ticker, sector, market_cap.
"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "sector_map.csv"


def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "sector": info.get("sector") or info.get("sectorDisp") or "Unknown",
            "market_cap": info.get("marketCap") or 0.0,
        }
    except Exception:
        return {"sector": "Unknown", "market_cap": 0.0}


def main():
    train = pd.read_parquet(DATA_DIR / "train_returns.parquet")
    tickers = [c for c in train.columns if c != "^GSPC"]
    print(f"Fetching sector and market cap for {len(tickers)} tickers ...")

    records = []
    for i, ticker in enumerate(tickers):
        info = fetch_info(ticker)
        records.append({"ticker": ticker, **info})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(tickers)} done ...")
        time.sleep(0.05)

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(df["sector"].value_counts().head(15))


if __name__ == "__main__":
    main()
