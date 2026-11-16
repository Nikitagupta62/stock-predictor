"""
intraday_features.py

Features + labels for "will this stock close higher one hour from now?"

Timing convention
-----------------
A row is stamped with its bar's START time. The row's features use only that
bar and earlier bars, so they are known the moment the bar closes (start + 1h).
At that moment we predict the CLOSE of the next full hourly bar.

  bar start  9:15 10:15 11:15 12:15 13:15 14:15 15:15(short)
  bar_of_day   0     1     2     3     4     5     6

Predictions are made after bars 0-4 close (10:15 ... 14:15) and target the
close of bars 1-5 (11:15 ... 15:15). We never predict across the overnight
gap or into the short closing bar, so every label is a clean 1-hour move.

All features are scale-free (ratios, percentages, oscillators), which also
protects against stock splits distorting old price levels.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

BAR = pd.Timedelta(hours=1)
LAST_PREDICTABLE_BAR = 4        # bar_of_day 0..4 have a full next hour
FIRST_BAR_MINUTES = 9 * 60 + 15

STOCK_FEATURES: List[str] = [
    "bar_ret", "ret_1h", "ret_2h", "ret_4h",
    "ret_since_open", "ret_vs_prev_close", "gap",
    "sma10_ratio", "sma35_ratio", "macd_norm", "macd_diff_norm", "rsi_14",
    "bb_pct", "bb_width", "atr_pct", "volatility_20", "range_pct",
    "rel_volume", "vwap_dist",
    "bar_of_day", "day_of_week",
]
MARKET_FEATURES: List[str] = ["mkt_ret_1h", "mkt_ret_4h", "mkt_breadth", "rel_ret_1h"]
ALL_FEATURES: List[str] = STOCK_FEATURES + MARKET_FEATURES


def ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features and label for ONE ticker's hourly bars (index = bar start, IST)."""
    d = df.copy()
    o, h, l, c, v = d["Open"], d["High"], d["Low"], d["Close"], d["Volume"]
    day = d.index.normalize()
    minutes = d.index.hour * 60 + d.index.minute
    bar_of_day = pd.Series(np.round((minutes - FIRST_BAR_MINUTES) / 60.0), index=d.index)

    # --- returns & intraday position ---
    d["bar_ret"] = c / o - 1                      # candle body, no overnight gap
    d["ret_1h"] = c.pct_change(1)
    d["ret_2h"] = c.pct_change(2)
    d["ret_4h"] = c.pct_change(4)
    first_open = o.groupby(day).transform("first")
    d["ret_since_open"] = c / first_open - 1
    # previous trading day's close: shifted by a whole day, so never the current day
    prev_close = c.groupby(day).last().shift(1).reindex(day).to_numpy()
    d["ret_vs_prev_close"] = c / prev_close - 1
    d["gap"] = first_open / prev_close - 1

    # --- VWAP of the day so far (cumulative, uses only past bars) ---
    typical = (h + l + c) / 3
    cum_pv = (typical * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum().replace(0, np.nan)
    d["vwap_dist"] = c / (cum_pv / cum_v) - 1

    # --- trend / momentum / volatility ---
    d["sma10_ratio"] = c / SMAIndicator(c, window=10).sma_indicator() - 1
    d["sma35_ratio"] = c / SMAIndicator(c, window=35).sma_indicator() - 1   # ~5 days
    macd = MACD(c)
    d["macd_norm"] = macd.macd() / c
    d["macd_diff_norm"] = macd.macd_diff() / c
    d["rsi_14"] = RSIIndicator(c, window=14).rsi()
    bb = BollingerBands(c)
    d["bb_pct"] = bb.bollinger_pband()
    d["bb_width"] = bb.bollinger_wband()
    d["atr_pct"] = AverageTrueRange(h, l, c).average_true_range() / c
    d["volatility_20"] = d["ret_1h"].rolling(20).std()
    d["range_pct"] = (h - l) / c

    # --- volume relative to the same time-of-day over the last 20 sessions ---
    typical_vol = v.groupby(bar_of_day.to_numpy()).transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
    )
    d["rel_volume"] = v / typical_vol.replace(0, np.nan)

    # --- calendar ---
    d["bar_of_day"] = bar_of_day.astype(float)
    d["day_of_week"] = d.index.dayofweek.astype(float)

    # --- label: does the NEXT full hourly bar close higher? ---
    starts = pd.Series(d.index, index=d.index)
    next_is_next_hour = (starts.shift(-1) - starts) == BAR
    valid = next_is_next_hour & (d["bar_of_day"] <= LAST_PREDICTABLE_BAR)
    next_close = c.shift(-1)
    d["next_ret"] = (next_close / c - 1).where(valid)
    d["target"] = (next_close > c).astype(float).where(valid)
    d["can_predict"] = d["bar_of_day"] <= LAST_PREDICTABLE_BAR
    return d


def build_panel(data: Dict[str, pd.DataFrame], training: bool = True) -> pd.DataFrame:
    """
    Stack all tickers into one table (index = bar start time, column 'ticker')
    and add market-wide features computed across the whole universe.

    training=True  -> only rows with a valid label (for fit / validation).
    training=False -> keep rows without a label (the latest bar, for live use).
    """
    frames = []
    for ticker, df in data.items():
        if len(df) < 60:
            continue
        f = ticker_features(df)
        f["ticker"] = ticker
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames).sort_index(kind="stable")

    # market context = equal-weight average across the universe at the same bar
    g = panel.groupby(level=0)
    panel["mkt_ret_1h"] = g["ret_1h"].transform("mean")
    panel["mkt_ret_4h"] = g["ret_4h"].transform("mean")
    up = (panel["ret_1h"] > 0).astype(float).where(panel["ret_1h"].notna())
    panel["mkt_breadth"] = up.groupby(panel.index).transform("mean")
    panel["rel_ret_1h"] = panel["ret_1h"] - panel["mkt_ret_1h"]

    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel[ALL_FEATURES] = panel[ALL_FEATURES].astype("float64")
    if training:
        panel = panel.dropna(subset=ALL_FEATURES + ["target"])
        panel["target"] = panel["target"].astype("int64")
    else:
        panel = panel.dropna(subset=ALL_FEATURES)
    return panel
