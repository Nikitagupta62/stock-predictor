"""
backtest.py

Honest evaluation of the next-day direction model.

1. Builds OUT-OF-SAMPLE probabilities with walk-forward validation
   (train on the past, predict a later block, never shuffle, with a gap).
2. Simulates a simple long-or-cash strategy on those probabilities,
   including transaction costs.
3. Reports ML metrics (AUC, log loss, accuracy vs. baseline with a standard
   error) AND trading metrics (return, Sharpe, drawdown vs. buy-and-hold).

Run:
    python backtest.py --ticker AAPL --period 10y
    python backtest.py --ticker RELIANCE.NS --period 10y --cost-bps 15 --threshold 0.55
    python backtest.py --ticker TCS.NS --no-market
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from features import build_features, default_market_ticker, get_feature_columns

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Model factory
# --------------------------------------------------------------------------
def default_model():
    """XGBoost if installed (matches train_model.py), else a sklearn fallback."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=42
        )


# --------------------------------------------------------------------------
# Step 1: out-of-sample predictions
# --------------------------------------------------------------------------
def walk_forward_probabilities(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 10,
    model_factory: Callable = default_model,
) -> pd.Series:
    """
    P(up) for every day that appears in a test fold, using a model that never
    saw that day or the `gap` days before it (features look back up to 20 days,
    so a gap stops train/test windows from overlapping).

    The first chunk of data is only ever used for training, so predictions
    start part-way through the history.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    parts = []
    for train_idx, test_idx in tscv.split(X):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        parts.append(pd.Series(proba, index=X.index[test_idx]))
    return pd.concat(parts)


# --------------------------------------------------------------------------
# Step 2: strategy simulation
#
# Long-or-cash: at the close of day t, if P(up) > threshold we hold the stock
# through day t+1 (earning close[t+1]/close[t] - 1), otherwise we hold cash.
# Every change in position pays `cost_bps` basis points per side
# (brokerage + taxes + slippage; use higher values for Indian equities).
# --------------------------------------------------------------------------
def _next_day_returns(full_close: pd.Series) -> pd.Series:
    return full_close.shift(-1) / full_close - 1


def run_backtest(
    full_close: pd.Series,
    proba: pd.Series,
    threshold: float,
    cost_bps: float,
) -> pd.DataFrame:
    next_ret = _next_day_returns(full_close).reindex(proba.index)
    position = (proba > threshold).astype(float)

    # position[t] is decided at close t, so trades happen when it changes.
    # The first day counts as a trade if we start invested.
    trades = position.diff().abs()
    trades.iloc[0] = position.iloc[0]
    cost = trades * (cost_bps / 10_000)

    out = pd.DataFrame(
        {
            "proba": proba,
            "position": position,
            "next_ret": next_ret,
            "strategy_ret": position * next_ret - cost,
            "buyhold_ret": next_ret,
        }
    ).dropna(subset=["next_ret"])  # last day has no next-day return
    return out


# --------------------------------------------------------------------------
# Step 3: metrics
# --------------------------------------------------------------------------
def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def _perf(returns: pd.Series) -> Dict[str, float]:
    n = len(returns)
    total = float((1 + returns).prod() - 1)
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (returns.mean() * TRADING_DAYS) / vol if vol > 0 else float("nan")
    return {
        "total_return": total,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(returns),
    }


def ml_metrics(y_true: pd.Series, proba: pd.Series) -> Dict[str, float]:
    y = y_true.reindex(proba.index)
    preds = (proba > 0.5).astype(int)
    n = len(y)
    acc = accuracy_score(y, preds)
    baseline = max(y.mean(), 1 - y.mean())  # best constant guess
    return {
        "n_samples": n,
        "accuracy": acc,
        "baseline_accuracy": float(baseline),
        # standard error of an accuracy estimate: differences smaller than
        # ~2x this are indistinguishable from luck
        "accuracy_std_error": float(np.sqrt(acc * (1 - acc) / n)),
        "roc_auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba),
        "log_loss_baseline": log_loss(y, np.full(n, y.mean())),
    }


def report(ml: Dict[str, float], bt: pd.DataFrame, threshold: float, cost_bps: float):
    strat, bh = _perf(bt["strategy_ret"]), _perf(bt["buyhold_ret"])
    trades = int(bt["position"].diff().abs().fillna(bt["position"].iloc[0]).sum())
    edge = ml["accuracy"] - ml["baseline_accuracy"]
    se = ml["accuracy_std_error"]

    print("\n=== Out-of-sample ML quality ===")
    print(f"Samples tested        : {ml['n_samples']}")
    print(f"Accuracy              : {ml['accuracy']:.3f}  (+/- {se:.3f} std error)")
    print(f"Baseline (best const.): {ml['baseline_accuracy']:.3f}")
    print(f"ROC-AUC               : {ml['roc_auc']:.3f}   (0.500 = coin flip)")
    print(f"Log loss              : {ml['log_loss']:.4f}  (constant guess: {ml['log_loss_baseline']:.4f})")
    verdict = "within noise" if abs(edge) < 2 * se else ("above baseline" if edge > 0 else "BELOW baseline")
    print(f"Edge over baseline    : {edge:+.3f}  -> {verdict}")

    print(f"\n=== Strategy (long if P(up) > {threshold}, cost {cost_bps:g} bps/side) ===")
    print(f"{'':<14}{'Strategy':>12}{'Buy & hold':>14}")
    for label, key, pct in [
        ("Total return", "total_return", True),
        ("CAGR", "cagr", True),
        ("Sharpe", "sharpe", False),
        ("Max drawdown", "max_drawdown", True),
    ]:
        f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.2f}")
        print(f"{label:<14}{f(strat[key]):>12}{f(bh[key]):>14}")
    print(f"Time in market: {bt['position'].mean():.0%}   |   Trades: {trades}")
    print(
        "\nNote: a good-looking result here is a hypothesis, not proof. Check it "
        "on other tickers and other time periods before believing it."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    from data_fetch import fetch_price_history

    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--period", default="10y")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--gap", type=int, default=10, help="days skipped between train and test")
    p.add_argument("--threshold", type=float, default=0.55, help="min P(up) to go long")
    p.add_argument("--cost-bps", type=float, default=10.0, help="cost per side in basis points")
    p.add_argument("--no-market", action="store_true", help="skip index features")
    args = p.parse_args()

    print(f"Fetching {args.ticker}...")
    raw = fetch_price_history(args.ticker, period=args.period)

    market: Optional[pd.DataFrame] = None
    if not args.no_market:
        mkt = default_market_ticker(args.ticker)
        print(f"Fetching market context ({mkt})...")
        market = fetch_price_history(mkt, period=args.period)

    data = build_features(raw, market=market, training=True)
    cols = get_feature_columns(market is not None)
    X, y = data[cols], data["target"]
    print(f"{len(X)} samples, {len(cols)} features")

    proba = walk_forward_probabilities(X, y, n_splits=args.splits, gap=args.gap)
    ml = ml_metrics(y, proba)
    bt = run_backtest(data["Close"], proba, args.threshold, args.cost_bps)
    report(ml, bt, args.threshold, args.cost_bps)


if __name__ == "__main__":
    main()
