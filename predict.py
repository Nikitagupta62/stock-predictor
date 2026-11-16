"""
predict.py  (updated)

Loads a trained daily model and predicts the next trading day.
Run: python predict.py --ticker AAPL

Change vs. the old version: it no longer carries its own copy of the indicator
code. Features now come from features.build_features(..., training=False), so
training and prediction can never drift apart.
"""

import argparse

import joblib
import pandas as pd

from data_fetch import fetch_price_history
from features import build_features


def build_features_for_prediction(raw: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Features for the LATEST day (whose next-day outcome is still unknown)."""
    featured = build_features(raw, training=False)
    if featured.empty or featured.index[-1] != raw.index[-1]:
        raise ValueError(
            "The most recent day has incomplete indicator values (for example a "
            "zero-volume or flat day). Try again after the next trading day."
        )
    missing = [c for c in feature_cols if c not in featured.columns]
    if missing:
        raise ValueError(
            f"This model was trained with different features ({missing[:3]}...). "
            "Retrain it: python train_model.py --ticker <TICKER>"
        )
    return featured.iloc[[-1]][list(feature_cols)]


def predict_next_day(ticker: str, model_path: str = None):
    if model_path is None:
        model_path = f"model_{ticker.replace('.', '_')}.joblib"

    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["features"]

    raw = fetch_price_history(ticker, period="1y")
    latest = build_features_for_prediction(raw, feature_cols)

    proba = model.predict_proba(latest)[0]
    pred = model.predict(latest)[0]
    last_close = raw["Close"].iloc[-1]
    last_date = raw.index[-1].date()

    print(f"\nTicker: {ticker}")
    print(f"Last close ({last_date}): {last_close:.2f}")
    print(f"Prediction for next trading day: {'UP' if pred == 1 else 'DOWN'}")
    print(f"Confidence: up={proba[1]:.1%} down={proba[0]:.1%}")

    return {
        "ticker": ticker,
        "last_date": str(last_date),
        "last_close": float(last_close),
        "prediction": "UP" if pred == 1 else "DOWN",
        "prob_up": float(proba[1]),
        "prob_down": float(proba[0]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    predict_next_day(args.ticker)
