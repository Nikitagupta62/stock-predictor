"""
predict.py
Loads a trained model and generates a prediction for the next trading day
using the most recent available data.

Run: python predict.py --ticker AAPL
"""

import argparse
import joblib
import numpy as np
import pandas as pd

from data_fetch import fetch_price_history
from features import build_features


def predict_next_day(ticker: str, model_path: str = None):
    if model_path is None:
        model_path = f"model_{ticker.replace('.', '_')}.joblib"

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    raw = fetch_price_history(ticker, period="1y")

    # Note: build_features() drops the most recent row because its target
    # (next day's direction) is unknown. For a LIVE prediction we want
    # that latest row's features, so we rebuild it without the drop.
    full_with_latest = build_features_for_prediction(raw, feature_cols)

    proba = model.predict_proba(full_with_latest)[0]
    pred = model.predict(full_with_latest)[0]

    last_close = raw["Close"].iloc[-1]
    last_date = raw.index[-1].date()

    print(f"\nTicker: {ticker}")
    print(f"Last close ({last_date}): {last_close:.2f}")
    print(f"Prediction for next trading day: {'UP' if pred == 1 else 'DOWN'}")
    print(f"Confidence: up={proba[1]:.1%}  down={proba[0]:.1%}")

    return {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "prediction": "UP" if pred == 1 else "DOWN",
        "prob_up": float(proba[1]),
        "prob_down": float(proba[0]),
    }


def build_features_for_prediction(raw: pd.DataFrame, feature_cols):
    """
    Rebuilds features but keeps the final row (today), since normal
    build_features() drops it for lacking a next-day label.
    """
    data = raw.copy()
    featured = _compute_indicators_only(data)
    latest = featured.iloc[[-1]][feature_cols]
    return latest


def _compute_indicators_only(data: pd.DataFrame) -> pd.DataFrame:
    """Same indicator logic as features.build_features, minus the target/dropna."""
    from ta.trend import SMAIndicator, EMAIndicator, MACD
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
    from features import FEATURE_COLUMNS

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    data["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    data["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    data["ema_10"] = EMAIndicator(close, window=10).ema_indicator()
    macd = MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()
    data["rsi_14"] = RSIIndicator(close, window=14).rsi()
    stoch = StochasticOscillator(high, low, close)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()
    bb = BollingerBands(close)
    data["bb_width"] = bb.bollinger_wband()
    data["bb_pct"] = bb.bollinger_pband()
    data["atr_14"] = AverageTrueRange(high, low, close).average_true_range()
    data["obv"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    data["volume_change"] = volume.pct_change()
    data["return_1d"] = close.pct_change(1)
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)
    data["high_low_range"] = (high - low) / close
    data["close_vs_sma50"] = (close - data["sma_50"]) / data["sma_50"]

    # Same cleanup as build_features(): kill inf values (e.g. from
    # zero-volume holiday rows) before dropping NaNs, and enforce float
    # dtype so XGBoost never chokes on mixed/object columns.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    for col in FEATURE_COLUMNS:
        data[col] = data[col].astype("float64")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    predict_next_day(args.ticker)