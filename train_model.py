"""
train_model.py
Trains a baseline classifier to predict next-day price direction (up/down),
using walk-forward validation appropriate for time-series data.

Run: python train_model.py --ticker AAPL --period 5y
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from data_fetch import fetch_price_history
from features import build_features, FEATURE_COLUMNS


def walk_forward_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Time-series cross-validation: train on past, test on a later slice,
    never shuffle, never let future data leak into training.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "fold": fold,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            # naive baseline: always predict "up" (most markets drift up over time)
            "baseline_accuracy": (y_test == 1).mean(),
        }
        fold_metrics.append(metrics)
        print(
            f"Fold {fold}: acc={metrics['accuracy']:.3f} "
            f"(baseline={metrics['baseline_accuracy']:.3f})  "
            f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}"
        )

    return pd.DataFrame(fold_metrics)


def train_final_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train on ALL available data for the model that will make live predictions."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--period", type=str, default="5y")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    print(f"Fetching data for {args.ticker}...")
    raw = fetch_price_history(args.ticker, period=args.period)

    print("Building features...")
    data = build_features(raw)
    X = data[FEATURE_COLUMNS]
    y = data["target"]

    print(f"\n{len(X)} samples, {len(FEATURE_COLUMNS)} features")
    print(f"Class balance -> up days: {y.mean():.1%}\n")

    print("Running walk-forward validation...\n")
    results = walk_forward_validate(X, y, n_splits=args.splits)

    print("\n--- Summary across folds ---")
    print(results[["accuracy", "baseline_accuracy", "precision", "recall", "f1"]].mean())

    print("\nTraining final model on full dataset...")
    final_model = train_final_model(X, y)

    model_path = f"model_{args.ticker.replace('.', '_')}.joblib"
    joblib.dump({"model": final_model, "features": FEATURE_COLUMNS, "ticker": args.ticker}, model_path)
    print(f"Saved model to {model_path}")

    # feature importance, quick sanity check
    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop 5 most important features:")
    print(importances.sort_values(ascending=False).head(5))


if __name__ == "__main__":
    main()
