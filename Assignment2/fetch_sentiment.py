"""
fetch_sentiment.py
==================
Fetches financial news headlines for the 6 Indian stocks and scores them
with FinBERT (ProsusAI/finbert) to produce daily sentiment polarity scores.

Two data sources are tried in order:
  1. NewsAPI.org  — free tier: 100 req/day, 30-day history
                    Register at https://newsapi.org/ for a free key.
  2. GDELT Project — 100% free, no key, full historical data
                    Uses the GDELT 2.0 API (doc.category, sourceurl, etc.)

FinBERT is downloaded from Hugging Face on first run (~500 MB).
Offline fallback: TextBlob polarity if transformers are unavailable.

Output
------
  data/raw/news/{TICKER}_headlines.csv   — raw headlines with dates
  data/processed/news_sentiment.csv      — daily sentiment scores per ticker

Usage:
    python fetch_sentiment.py --newsapi-key YOUR_KEY

Look-ahead guard
----------------
Articles on date T are only used as features at T+1 (next trading day).
"""

import argparse
import os
import re
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


# ─── NewsAPI Fetcher ──────────────────────────────────────────────────────────

def fetch_newsapi(ticker: str, api_key: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch recent headlines from NewsAPI (free tier = last 30 days only).
    For historical data, use GDELT below.
    """
    if not api_key:
        return pd.DataFrame()

    query = COMPANY_NAMES[ticker]
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    print(f"  NewsAPI: {ticker} ({query}) ...")
    try:
        resp = requests.get(
            NEWSAPI_BASE,
            params=dict(
                q=query,
                from_date=from_date,
                language="en",
                sortBy="publishedAt",
                pageSize=100,
                apiKey=api_key,
            ),
            timeout=20,
        )
        data = resp.json()
        articles = data.get("articles", [])
        rows = [
            {
                "date":     a["publishedAt"][:10],
                "headline": a.get("title", "") or "",
                "source":   a.get("source", {}).get("name", ""),
            }
            for a in articles
            if a.get("title")
        ]
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


def fetch_gdelt(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch headlines from GDELT 2.0 Doc API (free, no key).
    Queries by company name, returns article metadata with dates.

    GDELT date format: YYYYMMDDHHMMSS
    """
    query = COMPANY_NAMES.get(ticker, ticker)
    print(f"  GDELT: {ticker} ({query}) ...")

    # GDELT free API limits to ~3 months per call; we chunk by year
    all_rows = []
    years = pd.date_range(start=start, end=end, freq="YS")  # year-start

    for yr_start in years:
        yr_end = min(yr_start + pd.DateOffset(years=1) - pd.Timedelta(days=1),
                     pd.Timestamp(end))
        gd_start = yr_start.strftime("%Y%m%d%H%M%S")
        gd_end   = yr_end.strftime("%Y%m%d%H%M%S")

        try:
            params = dict(
                query=f'"{query}" sourcelang:english',
                mode="artlist",
                maxrecords=250,
                startdatetime=gd_start,
                enddatetime=gd_end,
                format="json",
            )
            resp = requests.get(GDELT_BASE, params=params, timeout=30)
            data = resp.json()
            articles = data.get("articles", [])
            for a in articles:
                all_rows.append({
                    "date":     a.get("seendate", "")[:8],  # YYYYMMDD
                    "headline": a.get("title", ""),
                    "url":      a.get("url", ""),
                })
            time.sleep(1)  # polite
        except Exception as e:
            print(f"    GDELT error for {yr_start.year}: {e}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ─── Sentiment Scoring ────────────────────────────────────────────────────────

def load_finbert():
    """Load FinBERT from HuggingFace. Falls back to TextBlob if unavailable."""
    try:
        from transformers import pipeline
        print("Loading FinBERT (ProsusAI/finbert) from HuggingFace ...")
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
        print("FinBERT loaded.")
        return sentiment_pipe, "finbert"
    except Exception as e:
        print(f"FinBERT not available ({e}). Falling back to TextBlob.")
        try:
            from textblob import TextBlob
            return None, "textblob"
        except:
            print("TextBlob also not available. Using zero sentiment.")
            return None, "zero"


def score_headlines(headlines: list, model, model_type: str) -> list:
    """Score a list of headline strings, return list of float scores in [-1, 1]."""
    if not headlines:
        return []

    if model_type == "finbert":
        # FinBERT returns label: {positive, negative, neutral} + score (confidence)
        LABEL_MAP = {"positive": 1, "negative": -1, "neutral": 0}
        results = []
        batch_size = 32
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            try:
                preds = model(batch)
                for p in preds:
                    polarity = LABEL_MAP.get(p["label"].lower(), 0) * p["score"]
                    results.append(polarity)
            except Exception as e:
                print(f"    Scoring error: {e}")
                results.extend([0.0] * len(batch))
        return results

    elif model_type == "textblob":
        from textblob import TextBlob
        return [TextBlob(h).sentiment.polarity for h in headlines]

    else:  # zero
        return [0.0] * len(headlines)


def compute_daily_sentiment(df: pd.DataFrame, model, model_type: str) -> pd.Series:
    """
    Given a DataFrame with 'date' and 'headline' columns, score headlines
    and aggregate to daily mean sentiment.
    """
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

def main(newsapi_key: str = ""):
    print("=" * 60)
    print("Step 4 – News Sentiment Ingestion")
    print("=" * 60)

    model, model_type = load_finbert()
    print(f"Sentiment model: {model_type}\n")

    daily_idx = pd.date_range(start=START_DATE, end=END_DATE, freq="B")
    sentiment_panel = pd.DataFrame(index=daily_idx)
    sentiment_panel.index.name = "Date"

    for ticker in TICKERS:
        print(f"\n{'─'*40}")
        print(f"Ticker: {ticker}")

        # Try GDELT first (historical), supplement with NewsAPI
        headlines_df = fetch_gdelt(ticker, START_DATE, END_DATE)

        if newsapi_key:
            recent = fetch_newsapi(ticker, newsapi_key)
            if not recent.empty:
                headlines_df = pd.concat([headlines_df, recent], ignore_index=True)
                headlines_df.drop_duplicates(subset=["headline"], inplace=True)

        if not headlines_df.empty:
            headlines_df.to_csv(NEWS_DIR / f"{ticker}_headlines.csv", index=False)
            print(f"  Total headlines: {len(headlines_df)}")

            daily_sent = compute_daily_sentiment(headlines_df, model, model_type)

            # Reindex to business days, forward-fill gaps, then LAG by 1 day
            daily_sent = daily_sent.reindex(daily_idx, fill_value=np.nan)
            daily_sent = daily_sent.ffill().fillna(0.0)
            daily_sent = daily_sent.shift(1)  # <-- look-ahead guard: T+1 only

            sentiment_panel[f"{ticker}_sentiment"] = daily_sent
        else:
            print(f"  No headlines fetched. Using 0.0 for {ticker}.")
            sentiment_panel[f"{ticker}_sentiment"] = 0.0

    sentiment_panel.to_csv(PROCESSED_DIR / "news_sentiment.csv")
    print(f"\nSentiment panel saved: {sentiment_panel.shape}")
    print(f"→ {PROCESSED_DIR}/news_sentiment.csv")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--newsapi-key", default="", help="NewsAPI.org free API key")
    args = parser.parse_args()
    main(newsapi_key=args.newsapi_key)
