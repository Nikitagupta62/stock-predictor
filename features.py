"""
features.py  (revised)

Turns raw OHLCV data into model-ready features + a next-day direction label.

Changes vs. the original version
--------------------------------
1. TARGET FIX: the last row now gets target = NaN (no next day exists yet).
   Before, `(close.shift(-1) > close).astype(int)` turned that NaN comparison
   into 0, so today's row was silently kept in training as a fake "down" day.
2. SCALE-FREE FEATURES: raw SMA / EMA / MACD / ATR / OBV values move with the
   stock's price, so the model could memorise price levels instead of
   behaviour. Everything is now a ratio, a percentage or an oscillator.
3. NEW FEATURES: overnight gap, relative volume, 20-day volatility,
   day-of-week, and (optionally) market context such as Nifty / S&P 500.
4. `training=False` keeps the latest row so predict.py can use it.

Every feature at row t uses only data available at the close of day t.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

STOCK_FEATURES: List[str] = [
    # returns / momentum
    "return_1d", "return_5d", "return_10d", "return_20d",
    "gap",
    # trend (all relative to price)
    "sma10_ratio", "sma50_ratio", "ema10_ratio", "sma10_vs_sma50",
    "macd_norm", "macd_signal_norm", "macd_diff_norm",
    # oscillators (already bounded / scale-free)
    "rsi_14", "stoch_k", "stoch_d",
    # volatility
    "bb_width", "bb_pct", "atr_pct", "volatility_20d", "high_low_range",
    # volume
    "rel_volume", "obv_flow_5d",
    # calendar
    "day_of_week",
]

MARKET_FEATURES: List[str] = ["mkt_return_1d", "mkt_return_5d", "rel_return_5d"]

# Kept for backwards compatibility with train_model.py / predict.py
FEATURE_COLUMNS: List[str] = STOCK_FEATURES


def get_feature_columns(with_market: bool = False) -> List[str]:
    """Feature list to use, depending on whether market data was supplied."""
    return STOCK_FEATURES + (MARKET_FEATURES if with_market else [])


def default_market_ticker(ticker: str) -> str:
    """Pick a sensible benchmark index for a Yahoo Finance ticker."""
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "^NSEI"   # Nifty 50
    return "^GSPC"       # S&P 500


def build_features(
    df: pd.DataFrame,
    market: Optional[pd.DataFrame] = None,
    training: bool = True,
) -> pd.DataFrame:
    """
    Args:
        df:       OHLCV DataFrame (Open, High, Low, Close, Volume), date-indexed.
        market:   optional OHLCV DataFrame of a benchmark index (needs 'Close').
        training: True  -> drop rows without a label (use for fit / validation).
                  False -> keep the latest row (target is NaN) for prediction.

    Returns:
        DataFrame with original columns, engineered features and 'target'.
    """
    data = df.copy()
    close, high, low = data["Close"], data["High"], data["Low"]
    open_, volume = data["Open"], data["Volume"]

    # --- Returns / momentum ---
    data["return_1d"] = close.pct_change(1)
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)
    data["return_20d"] = close.pct_change(20)
    data["gap"] = open_ / close.shift(1) - 1  # overnight gap

    # --- Trend, expressed relative to price ---
    sma10 = SMAIndicator(close, window=10).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    ema10 = EMAIndicator(close, window=10).ema_indicator()
    data["sma10_ratio"] = close / sma10 - 1
    data["sma50_ratio"] = close / sma50 - 1
    data["ema10_ratio"] = close / ema10 - 1
    data["sma10_vs_sma50"] = sma10 / sma50 - 1

    macd = MACD(close)
    data["macd_norm"] = macd.macd() / close
    data["macd_signal_norm"] = macd.macd_signal() / close
    data["macd_diff_norm"] = macd.macd_diff() / close

    # --- Oscillators ---
    data["rsi_14"] = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high, low, close)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    # --- Volatility ---
    bb = BollingerBands(close)
    data["bb_width"] = bb.bollinger_wband()
    data["bb_pct"] = bb.bollinger_pband()
    atr = AverageTrueRange(high, low, close).average_true_range()
    data["atr_pct"] = atr / close
    data["volatility_20d"] = data["return_1d"].rolling(20).std()
    data["high_low_range"] = (high - low) / close

    # --- Volume ---
    avg_vol_20 = volume.rolling(20).mean()
    data["rel_volume"] = volume / avg_vol_20
    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    # 5-day net volume flow, scaled by typical volume (OBV itself is unbounded)
    data["obv_flow_5d"] = (obv - obv.shift(5)) / avg_vol_20

    # --- Calendar ---
    data["day_of_week"] = pd.to_datetime(data.index).dayofweek.astype(float)

    # --- Market context (optional) ---
    has_market = market is not None and "Close" in market.columns
    if has_market:
        mkt_close = market["Close"].reindex(data.index).ffill()
        data["mkt_return_1d"] = mkt_close.pct_change(1)
        data["mkt_return_5d"] = mkt_close.pct_change(5)
        data["rel_return_5d"] = data["return_5d"] - data["mkt_return_5d"]

    # --- Target: did the NEXT day close higher? ---
    # NaN on the last row (no next day yet). Do not cast before masking.
    next_close = close.shift(-1)
    data["target"] = (next_close > close).astype(float).where(next_close.notna())

    # --- Cleaning ---
    cols = get_feature_columns(has_market)
    data = data.replace([np.inf, -np.inf], np.nan)
    data[cols] = data[cols].astype("float64")

    if training:
        data = data.dropna(subset=cols + ["target"])
        data["target"] = data["target"].astype("int64")
    else:
        # keep the newest row (its target is NaN); drop only warm-up rows
        data = data.dropna(subset=cols)

    return data


if __name__ == "__main__":
    from data_fetch import fetch_price_history

    raw = fetch_price_history("AAPL", period="2y")
    train = build_features(raw, training=True)
    live = build_features(raw, training=False)
    print(train[FEATURE_COLUMNS + ["target"]].tail())
    print(f"\n{len(train)} training rows, {len(live)} rows incl. latest day")
    print(f"Last training date: {train.index[-1]}  |  latest day: {live.index[-1]}")
    print(f"Target balance:\n{train['target'].value_counts(normalize=True)}")
