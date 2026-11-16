"""
intraday_data.py

Downloads and cleans hourly NSE bars from Yahoo Finance (via yfinance).

Facts this module is built around:
  * Yahoo serves at most ~730 days of 1-hour bars.
  * NSE trades 09:15-15:30 IST, so hourly bars start at
    09:15, 10:15, 11:15, 12:15, 13:15, 14:15 and a short 15:15 bar
    that only lasts until 15:30.
  * The newest bar returned during market hours is still FORMING. Using it
    would be look-ahead in live mode, so drop_incomplete_bars() removes it.
  * Yahoo is an unofficial source: data can be delayed, patchy or missing.
    Always look at the timestamp of the latest bar.
"""

from __future__ import annotations

import time
from datetime import time as dtime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
BAR = pd.Timedelta(hours=1)
SESSION_OPEN = dtime(9, 15)
LAST_BAR_START = dtime(15, 15)   # the short closing bar


def clean_bars(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Standardise one ticker's bars: IST index, NSE session hours only."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)
    df = df[[c for c in OHLCV if c in df.columns]].copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return pd.DataFrame(columns=OHLCV)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    in_session = np.array([SESSION_OPEN <= t <= LAST_BAR_START for t in df.index.time])
    df = df[in_session & (df.index.dayofweek < 5)]
    df["Volume"] = df["Volume"].fillna(0.0)
    return df


def bar_end_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When each bar closes: start + 1h, but never later than 15:30."""
    end = index + BAR
    close = index.normalize() + pd.Timedelta(hours=15, minutes=30)
    return end.where(end <= close, close)


def drop_incomplete_bars(df: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Remove bars that haven't closed yet (the one still forming)."""
    now = pd.Timestamp.now(tz=IST) if now is None else now
    return df[bar_end_times(df.index) <= now]


def download_universe(
    tickers: Iterable[str],
    days: Optional[int] = None,
    period: str = "1mo",
    interval: str = "1h",
    batch_size: int = 10,
    pause: float = 1.0,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Download hourly bars for many tickers in small batches (gentler on Yahoo).

    days   : history length for training (max ~730 for 1h). Uses start/end.
    period : used when days is None, e.g. "1mo" for live prediction.
    Tickers that fail or return too little data are skipped with a warning.
    """
    import yfinance as yf

    tickers = list(tickers)
    kwargs = dict(interval=interval, group_by="ticker", auto_adjust=True,
                  progress=False, threads=True)
    if days is not None:
        days = min(days, 725)
        kwargs["start"] = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        kwargs["period"] = period

    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, **kwargs)
        except Exception as exc:  # network / throttling
            print(f"  ! batch {batch[0]}..{batch[-1]} failed: {exc}")
            continue
        for t in batch:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                sub = None
            df = clean_bars(sub)
            if len(df) < 20:
                print(f"  ! {t}: no usable data, skipped")
            else:
                out[t] = df
        if verbose:
            print(f"  downloaded {min(i + batch_size, len(tickers))}/{len(tickers)}")
        time.sleep(pause)
    return out


def load_or_download(
    tickers: Iterable[str],
    cache_path: str = "data/intraday_cache.pkl",
    max_age_hours: float = 12,
    refresh: bool = False,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Use a cached download if it is fresh enough, otherwise re-download."""
    path = Path(cache_path)
    if path.exists() and not refresh:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            print(f"Using cache {path} ({age_h:.1f}h old). Use --refresh to re-download.")
            return pd.read_pickle(path)
    data = download_universe(tickers, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, path)
    return data
