"""generate_sector_map.py — Fetch GICS sector labels for all universe tickers.

Run once: python generate_sector_map.py
Writes data/sector_map.csv with columns: ticker, sector.
"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "sector_map.csv"


def fetch_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or info.get("sectorDisp") or "Unknown"
    except Exception:
        return "Unknown"


def main():
    train = pd.read_parquet(DATA_DIR / "train_returns.parquet")
    tickers = [c for c in train.columns if c != "^GSPC"]
    print(f"Fetching sector for {len(tickers)} tickers …")

    records = []
    for i, ticker in enumerate(tickers):
        sector = fetch_sector(ticker)
        records.append({"ticker": ticker, "sector": sector})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} done …")
        time.sleep(0.05)

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUT_PATH}")
    print(df["sector"].value_counts().head(15))


if __name__ == "__main__":
    main()
