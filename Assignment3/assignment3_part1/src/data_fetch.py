import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from config import TICKERS, YEARS_TOTAL, DATA_RAW

def fetch_price_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * YEARS_TOTAL)

    data = yf.download(
        TICKERS,
        start=start_date,
        end=end_date,
        auto_adjust=True,   # <-- important
        progress=True
    )

    # If MultiIndex (multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Use Close (since auto_adjust=True already adjusted it)
        prices = data["Close"]
    else:
        prices = data["Close"]

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    prices.to_csv(DATA_RAW / "adjusted_close.csv")

    return prices