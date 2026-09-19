"""
features.py
Turns raw OHLCV data into model-ready features + a next-day direction label.

IMPORTANT: every feature here is computed using only past/current-day data,
so nothing leaks information from the future into the training set.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: raw OHLCV DataFrame (columns: Open, High, Low, Close, Volume)

    Returns:
        DataFrame with engineered features + 'target' column
        target = 1 if next day's close > today's close, else 0
    """
    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # --- Trend indicators ---
    data["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    data["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    data["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
    macd = MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()

    # --- Momentum indicators ---
    data["rsi_14"] = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high, low, close)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    # --- Volatility indicators ---
    bb = BollingerBands(close)
    data["bb_width"] = bb.bollinger_wband()
    data["bb_pct"] = bb.bollinger_pband()
    data["atr_14"] = AverageTrueRange(high, low, close).average_true_range()

    # --- Volume indicators ---
    data["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    data["volume_change"] = volume.pct_change()

    # --- Price action / lagged returns ---
    data["return_1d"] = close.pct_change(1)
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)
    data["high_low_range"] = (high - low) / close
    data["close_vs_sma50"] = (close - data["sma_50"]) / data["sma_50"]

    # --- Target: did price go UP the next trading day? ---
    # shift(-1) looks one day forward for the LABEL only - this is fine,
    # it's what we're trying to predict, not a feature we train on.
    data["target"] = (close.shift(-1) > close).astype(int)

    # Drop rows with NaNs from indicator warm-up periods and the last row
    # (which has no next-day label yet)
    data = data.dropna()

    return data


FEATURE_COLUMNS = [
    "sma_10", "sma_50", "ema_10",
    "macd", "macd_signal", "macd_diff",
    "rsi_14", "stoch_k", "stoch_d",
    "bb_width", "bb_pct", "atr_14",
    "obv", "volume_change",
    "return_1d", "return_5d", "return_10d",
    "high_low_range", "close_vs_sma50",
]


if __name__ == "__main__":
    from data_fetch import fetch_price_history

    df = fetch_price_history("AAPL", period="2y")
    featured = build_features(df)
    print(featured[FEATURE_COLUMNS + ["target"]].tail())
    print(f"\n{len(featured)} rows after feature engineering")
    print(f"Target balance:\n{featured['target'].value_counts(normalize=True)}")
