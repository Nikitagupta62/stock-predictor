"""
universe.py

50 liquid large-cap NSE stocks (Nifty-50 style). Index membership and ticker
symbols change over time (renames, demergers, delistings), so treat this as
a starting list. Any symbol Yahoo can't serve is skipped with a warning by
intraday_data.download_universe(), so a stale entry never crashes the pipeline.
"""

from __future__ import annotations

SYMBOLS = [
    # Financials
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN", "SBILIFE", "HDFCLIFE",
    # IT
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    # Energy / utilities / materials
    "RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BPCL",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "GRASIM", "ULTRACEMCO",
    # Consumer
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM",
    "TITAN", "ASIANPAINT", "TRENT",
    # Auto
    "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    # Healthcare
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP",
    # Infra / telecom / conglomerates
    "LT", "BHARTIARTL", "ADANIPORTS", "ADANIENT",
]


def get_universe(n: int | None = None) -> list[str]:
    """Yahoo Finance tickers (NSE suffix .NS), optionally the first n."""
    tickers = [f"{s}.NS" for s in SYMBOLS]
    return tickers if n is None else tickers[:n]


if __name__ == "__main__":
    print(f"{len(SYMBOLS)} symbols")
