"""
fetch_sentiment.py
==================
Fetches financial news headlines for the 6 Indian stocks and scores them
with FinBERT (ProsusAI/finbert) to produce daily sentiment polarity scores.

Fixes vs previous version:
  - GDELT chunked quarterly (not yearly) to stay under rate limits
  - 429 rate-limit detection → 60s sleep before retry
  - Per-ticker headline cache: skips re-fetching if CSV already exists
  - Per-ticker sentiment cache: skips re-scoring if already computed
  - Longer inter-request sleep (5s between quarters, 15s between tickers)

Usage:
    python fetch_sentiment.py
    python fetch_sentiment.py --newsapi-key YOUR_KEY
    python fetch_sentiment.py --refetch   # ignore cache, re-fetch everything
"""

import argparse
import time
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TICKERS = ["RELIANCE", "HDFCBANK", "INFY", "M&M", "BHARTIARTL", "HUL"]
COMPANY_NAMES = {
    "RELIANCE":   "Reliance Industries",
    "HDFCBANK":   "HDFC Bank",
    "INFY":       "Infosys",
    "M&M":        "Mahindra Mahindra",
    "BHARTIARTL": "Bharti Airtel",
    "HUL":        "Hindustan Unilever",
}

START_DATE = "2020-01-01"
END_DATE   = "2025-12-31"

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
GDELT_BASE   = "https://api.gdeltproject.org/api/v2/doc/doc"

NEWS_DIR      = Path("data/raw/news")
PROCESSED_DIR = Path("data/processed")
NEWS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── GDELT Fetcher (with caching + rate-limit handling) ───────────────────────

def fetch_gdelt(ticker: str, start: str, end: str, refetch: bool = False) -> pd.DataFrame:
    """
    Fetch headlines from GDELT 2.0.
    - Chunks by QUARTER to reduce per-request size and avoid 429s.
    - Caches results to data/raw/news/{ticker}_headlines.csv.
    - On 429, sleeps 60s then retries up to 3 times.
    - 5s polite delay between each quarterly request.
    """
    cache_path = NEWS_DIR / f"{ticker}_headlines.csv"

    # Return cached data if available and refetch not forced
    if cache_path.exists() and not refetch:
        print(f"  GDELT: {ticker} — loading from cache ({cache_path})")
        return pd.read_csv(cache_path)

    query = COMPANY_NAMES.get(ticker, ticker)
    print(f"  GDELT: {ticker} ({query}) — fetching ...")

    all_rows = []

    # Chunk quarterly: smaller windows = lower chance of 429 / timeout
    quarters = pd.date_range(start=start, end=end, freq="QS")

    for q_start in quarters:
        q_end = min(
            q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1),
            pd.Timestamp(end),
        )
        gd_start = q_start.strftime("%Y%m%d%H%M%S")
        gd_end   = q_end.strftime("%Y%m%d%H%M%S")

        for attempt in range(3):
            try:
                params = dict(
                    query=f'"{query}" sourcelang:english',
                    mode="artlist",
                    maxrecords=250,
                    startdatetime=gd_start,
                    enddatetime=gd_end,
                    format="json",
                )
                resp = requests.get(GDELT_BASE, params=params, timeout=60)

                # Rate limited
                if resp.status_code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"    429 rate-limited ({q_start.year} Q{q_start.quarter}), "
                          f"sleeping {wait}s ...")
                    time.sleep(wait)
                    continue

                if resp.status_code != 200 or not resp.text.strip():
                    raise ValueError(f"Bad response: status {resp.status_code}")

                data = resp.json()
                articles = data.get("articles", [])
                for a in articles:
                    all_rows.append({
                        "date":     a.get("seendate", "")[:8],
                        "headline": a.get("title", ""),
                        "url":      a.get("url", ""),
                    })

                print(f"    {q_start.year} Q{q_start.quarter}: "
                      f"{len(articles)} articles (total so far: {len(all_rows)})")
                time.sleep(5)   # polite inter-request delay
                break           # success

            except Exception as e:
                wait = 10 * (attempt + 1)
                print(f"    Error ({q_start.year} Q{q_start.quarter}, "
                      f"attempt {attempt+1}/3): {e} — sleeping {wait}s")
                time.sleep(wait)

    if not all_rows:
        print(f"  WARNING: No headlines fetched for {ticker}.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["headline"] = df["headline"].fillna("").str.strip()
    df = df[df["headline"].str.len() > 5].drop_duplicates(subset=["headline"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Save to cache
    df.to_csv(cache_path, index=False)
    print(f"  Cached {len(df)} headlines → {cache_path}")
    return df


# ─── NewsAPI Fetcher ──────────────────────────────────────────────────────────

def fetch_newsapi(ticker: str, api_key: str, days_back: int = 30) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()

    query     = COMPANY_NAMES[ticker]
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    print(f"  NewsAPI: {ticker} ...")
    try:
        resp = requests.get(
            NEWSAPI_BASE,
            params=dict(q=query, from_date=from_date, language="en",
                        sortBy="publishedAt", pageSize=100, apiKey=api_key),
            timeout=20,
        )
        articles = resp.json().get("articles", [])
        rows = [{"date": a["publishedAt"][:10], "headline": a.get("title", "") or ""}
                for a in articles if a.get("title")]
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"    NewsAPI error: {e}")
        return pd.DataFrame()


# ─── FinBERT Loader ───────────────────────────────────────────────────────────

def load_finbert():
    try:
        from transformers import pipeline #,AutoTokenizer, AutoModelForSequenceClassification

        # tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        # model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        pipe = pipe = pipeline("text-classification", model="ProsusAI/finbert")
        print("FinBERT loaded.")
        return pipe, "finbert"
    except Exception as e:
        print(f"FinBERT unavailable ({e}). Falling back to TextBlob.")
        try:
            from textblob import TextBlob  # noqa
            return None, "textblob"
        except Exception as e:
            return None, "zero"


# ─── Sentiment Scoring ────────────────────────────────────────────────────────

def score_headlines(headlines: list, model, model_type: str) -> list:
    if not headlines:
        return []

    if model_type == "finbert":
        LABEL_MAP = {"positive": 1, "negative": -1, "neutral": 0}
        results = []
        batch_size = 32
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            try:
                for p in model(batch):
                    results.append(LABEL_MAP.get(p["label"].lower(), 0) * p["score"])
            except Exception as e:
                print(f"    Scoring error: {e}")
                results.extend([0.0] * len(batch))
        return results

    elif model_type == "textblob":
        from textblob import TextBlob
        return [TextBlob(h).sentiment.polarity for h in headlines]

    return [0.0] * len(headlines)


def compute_daily_sentiment(df: pd.DataFrame, model, model_type: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["headline"] = df["headline"].fillna("").astype(str)
    df = df[df["headline"].str.len() > 5]
    print(f"    Scoring {len(df)} headlines ...")
    df["sentiment"] = score_headlines(df["headline"].tolist(), model, model_type)
    daily = df.groupby("date")["sentiment"].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(newsapi_key: str = "", refetch: bool = False):
    print("=" * 60)
    print("Step 4 – News Sentiment Ingestion")
    print("=" * 60)

    # Check if full sentiment panel already cached
    sentiment_cache = PROCESSED_DIR / "news_sentiment.csv"
    if sentiment_cache.exists() and not refetch:
        print(f"Sentiment panel already exists at {sentiment_cache}. "
              f"Use --refetch to recompute.")
        return

    model, model_type = load_finbert()
    print(f"Sentiment model: {model_type}\n")

    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    sentiment_panel = pd.DataFrame(index=daily_idx)
    sentiment_panel.index.name = "Date"

    for ticker in TICKERS:
        print(f"\n{'─'*40}")
        print(f"Ticker: {ticker}")

        headlines_df = fetch_gdelt(ticker, START_DATE, END_DATE, refetch=refetch)

        if newsapi_key:
            recent = fetch_newsapi(ticker, newsapi_key)
            if not recent.empty:
                headlines_df = pd.concat([headlines_df, recent], ignore_index=True)
                headlines_df.drop_duplicates(subset=["headline"], inplace=True)

        if not headlines_df.empty:
            print(f"  Total headlines: {len(headlines_df)}")
            daily_sent = compute_daily_sentiment(headlines_df, model, model_type)
            daily_sent = daily_sent.reindex(daily_idx, fill_value=np.nan)
            daily_sent = daily_sent.ffill().fillna(0.0)
            daily_sent = daily_sent.shift(1)   # look-ahead guard
            sentiment_panel[f"{ticker}_sentiment"] = daily_sent
        else:
            print(f"  No headlines. Using 0.0 for {ticker}.")
            sentiment_panel[f"{ticker}_sentiment"] = 0.0

        # Save panel after every ticker so progress is preserved on crash
        sentiment_panel.to_csv(sentiment_cache)

        print("  Sleeping 15s before next ticker ...")
        time.sleep(15)

    print(f"\nSentiment panel saved: {sentiment_panel.shape}")
    print(f"→ {sentiment_cache}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--newsapi-key", default="")
    parser.add_argument("--refetch", action="store_true",
                        help="Ignore cache and re-fetch all headlines")
    args = parser.parse_args()
    main(newsapi_key=args.newsapi_key, refetch=args.refetch)