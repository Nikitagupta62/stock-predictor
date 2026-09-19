"""
data_fetch.py
Pulls historical daily OHLCV data for a given ticker using yfinance.
"""

import yfinance as yf
import pandas as pd


def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical price data for a ticker.

    Args:
        ticker: e.g. "AAPL", "TSLA", "RELIANCE.NS" (use .NS for NSE, .BO for BSE)
        period: how far back to pull data, e.g. "5y", "2y", "max"
        interval: bar size, "1d" for daily

    Returns:
        DataFrame indexed by date with columns: Open, High, Low, Close, Volume
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol is correct.")

    # yfinance sometimes returns MultiIndex columns for single tickers depending on version
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    # quick manual test
    ticker = "AAPL"
    data = fetch_price_history(ticker, period="2y")
    print(data.tail())
    print(f"\nFetched {len(data)} rows for {ticker}")
