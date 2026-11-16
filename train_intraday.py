"""
train_intraday.py

Trains ONE pooled model on many NSE stocks to predict whether a stock's next
full hourly candle closes higher, then checks honestly whether that is real.

    python train_intraday.py                 # 50 stocks, ~2 years of hourly bars
    python train_intraday.py --n 20 --cost-bps 6
    python train_intraday.py --refresh       # force a fresh download

Why pooled? One stock gives ~5,000 hourly samples; 50 stocks give ~250,000,
which is much harder to overfit. Features are scale-free, so one model can
learn patterns shared across stocks.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from intraday_data import load_or_download
from intraday_features import ALL_FEATURES, build_panel
from strategy import rank_ic
from universe import get_universe

MODEL_PATH = "models/intraday_model.joblib"


def make_model():
    """XGBoost if installed, else scikit-learn's gradient boosting."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=250,
            min_samples_leaf=200, random_state=42,
        )


def time_folds(panel: pd.DataFrame, n_splits: int, gap_days: int = 1
               ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Split by TRADING DAY, not by row, so no day is ever shared between train
    and test and every stock is cut at the same moment. gap_days skips days
    between the two so the 1-hour label can't straddle the boundary.
    """
    row_day = panel.index.normalize()
    days = row_day.unique().sort_values()
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap_days).split(days):
        yield row_day.isin(days[tr]), row_day.isin(days[te])


def walk_forward(panel: pd.DataFrame, n_splits: int, gap_days: int):
    X, y = panel[ALL_FEATURES], panel["target"]
    proba = np.full(len(panel), np.nan)
    fold_auc: List[float] = []
    for k, (tr, te) in enumerate(time_folds(panel, n_splits, gap_days), 1):
        tr_i, te_i = np.flatnonzero(tr), np.flatnonzero(te)
        model = make_model()
        model.fit(X.iloc[tr_i], y.iloc[tr_i])
        proba[te_i] = model.predict_proba(X.iloc[te_i])[:, 1]
        auc = roc_auc_score(y.iloc[te_i], proba[te_i])
        fold_auc.append(auc)
        print(f"  fold {k}: train {len(tr_i):>7,} | test {len(te_i):>7,} | AUC {auc:.4f}")
    return proba, fold_auc


def make_regressor(n_estimators: int = 200):
    """Regressor for expected return / expected move size (XGBoost or sklearn fallback)."""
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=n_estimators, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=100,
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=n_estimators,
            min_samples_leaf=200, random_state=42,
        )


def fit_models(X: pd.DataFrame, y: pd.Series, next_ret: np.ndarray):
    """
    Three models on the same features:
      clf : P(next hour closes higher)
      reg : expected next-hour return
      vol : expected size of the move, |return| (how much it will move, either way)
    Returns are winsorised (0.5% / 99.5%) so a few freak bars can't dominate.
    """
    lo, hi = np.quantile(next_ret, [0.005, 0.995])
    ret = np.clip(next_ret, lo, hi)
    clf = make_model().fit(X, y)
    reg = make_regressor().fit(X, ret)
    vol = make_regressor().fit(X, np.abs(ret))
    return clf, reg, vol


def walk_forward_full(panel: pd.DataFrame, n_splits: int, gap_days: int):
    """Walk-forward for all three models. Returns out-of-sample arrays (NaN where untested)."""
    X, y = panel[ALL_FEATURES], panel["target"]
    r = panel["next_ret"].to_numpy()
    n = len(panel)
    proba, exp_ret, exp_move = (np.full(n, np.nan) for _ in range(3))
    fold_auc: List[float] = []
    fold_ic: List[float] = []
    for k, (tr, te) in enumerate(time_folds(panel, n_splits, gap_days), 1):
        tr_i, te_i = np.flatnonzero(tr), np.flatnonzero(te)
        clf, reg, vol = fit_models(X.iloc[tr_i], y.iloc[tr_i], r[tr_i])
        Xt = X.iloc[te_i]
        proba[te_i] = clf.predict_proba(Xt)[:, 1]
        exp_ret[te_i] = reg.predict(Xt) * 1e4      # basis points
        exp_move[te_i] = vol.predict(Xt) * 1e4
        auc = roc_auc_score(y.iloc[te_i], proba[te_i])
        ic = rank_ic(exp_ret[te_i], r[te_i])
        fold_auc.append(auc)
        fold_ic.append(ic)
        print(f"  fold {k}: train {len(tr_i):>7,} | test {len(te_i):>7,} | "
              f"AUC {auc:.4f} | return-IC {ic:+.4f}")
    return proba, exp_ret, exp_move, fold_auc, fold_ic


def report_magnitude(exp_ret: np.ndarray, realized: np.ndarray, fold_ic: List[float]) -> None:
    """Does the expected-return model rank stocks usefully?"""
    ic = rank_ic(exp_ret, realized)
    print("\n=== Expected-return model (out-of-sample) ===")
    print(f"Rank IC (correlation of predicted vs actual return): {ic:+.4f}   "
          f"(0 = no skill; ~0.02-0.05 is a good result)")
    print(f"IC by fold: {', '.join(f'{x:+.3f}' for x in fold_ic)}")
    df = pd.DataFrame({"e": exp_ret, "r": realized * 1e4})
    df["bucket"] = pd.qcut(df["e"], 5, labels=False, duplicates="drop")
    by = df.groupby("bucket")["r"].mean()
    print("Actual average next-hour return by predicted-return quintile (bps):")
    print("  " + "  ".join(f"Q{int(i) + 1}: {v:+.2f}" for i, v in by.items()))
    print("  (a useful model has Q5 clearly above Q1; the cost of a round trip is the bar to clear)")


def report_ml(y: np.ndarray, p: np.ndarray, fold_auc: List[float]) -> None:
    n = len(y)
    acc = accuracy_score(y, p > 0.5)
    base = max(y.mean(), 1 - y.mean())
    se = np.sqrt(acc * (1 - acc) / n)
    print("\n=== Out-of-sample prediction quality ===")
    print(f"Samples tested   : {n:,}")
    print(f"Accuracy         : {acc:.4f}  (+/- {se:.4f} std error)")
    print(f"Baseline         : {base:.4f}  (best constant guess)")
    print(f"ROC-AUC          : {roc_auc_score(y, p):.4f}   (0.5000 = coin flip)")
    print(f"AUC by fold      : {', '.join(f'{a:.3f}' for a in fold_auc)}")
    print(f"Log loss         : {log_loss(y, p):.4f}  (constant guess: "
          f"{log_loss(y, np.full(n, y.mean())):.4f})")
    edge = acc - base
    verdict = "within noise" if abs(edge) < 2 * se else ("above baseline" if edge > 0 else "BELOW baseline")
    print(f"Edge vs baseline : {edge:+.4f} -> {verdict}")


def report_trades(next_ret: np.ndarray, p: np.ndarray, cost_bps: float,
                  thresholds=(0.50, 0.52, 0.55, 0.58, 0.60)) -> None:
    """
    Trade-by-trade view: buy when P(up) > threshold, sell one hour later.
    Cost is charged on entry AND exit. 'net' is what you'd actually keep.
    """
    print(f"\n=== If you bought every 'UP' signal and sold 1h later "
          f"({cost_bps:g} bps per side) ===")
    print(f"Average 1h move of ALL stocks (no model): {np.mean(next_ret) * 1e4:+.2f} bps")
    print(f"{'P(up) >':>8} {'trades':>9} {'hit rate':>9} {'gross bps':>10} "
          f"{'net bps':>9} {'t-stat':>7}")
    for thr in thresholds:
        sel = p > thr
        n = int(sel.sum())
        if n < 30:
            print(f"{thr:>8.2f} {n:>9,}   (too few trades)")
            continue
        gross = next_ret[sel]
        net = gross - 2 * cost_bps / 1e4
        t = net.mean() / (net.std(ddof=1) / np.sqrt(n))
        print(f"{thr:>8.2f} {n:>9,} {np.mean(gross > 0):>9.1%} "
              f"{gross.mean() * 1e4:>10.2f} {net.mean() * 1e4:>9.2f} {t:>7.2f}")
    print("Reading this: 'net bps' must be clearly positive at a threshold you chose "
          "BEFORE looking.\nThe t-stat is optimistic because stocks move together.")


def train_pipeline(n: int = 50, days: int = 720, splits: int = 5, gap_days: int = 1,
                   cost_bps: float = 5.0, refresh: bool = False,
                   model_path: str = MODEL_PATH) -> dict:
    """
    Download -> features -> walk-forward validation -> final models -> save.
    Also saves the out-of-sample predictions next to the model (oos_predictions.pkl);
    the dashboard's Results tab replays them as a paper-trading strategy.
    Prints a full report and returns the key numbers (used by the dashboard).
    """
    tickers = get_universe(n)
    print(f"Loading hourly data for {len(tickers)} stocks...")
    data = load_or_download(tickers, cache_path=f"data/intraday_{len(tickers)}_{days}.pkl",
                            refresh=refresh, days=days)
    if not data:
        raise RuntimeError("No data downloaded. Check your internet connection.")
    print(f"{len(data)} stocks loaded.")

    panel = build_panel(data, training=True)
    print(f"{len(panel):,} labelled samples | {panel.index.min().date()} -> "
          f"{panel.index.max().date()} | {len(ALL_FEATURES)} features | "
          f"up-rate {panel['target'].mean():.1%}")

    print("\nWalk-forward validation (train on past days, test on later days):")
    proba, exp_ret, exp_move, fold_auc, fold_ic = walk_forward_full(panel, splits, gap_days)
    mask = ~np.isnan(proba)
    idx = np.flatnonzero(mask)
    y = panel["target"].to_numpy()[mask]
    realized = panel["next_ret"].to_numpy()[mask]
    report_ml(y, proba[mask], fold_auc)
    report_magnitude(exp_ret[mask], realized, fold_ic)
    report_trades(realized, proba[mask], cost_bps)

    oos = pd.DataFrame({
        "t": panel.index[idx], "ticker": panel["ticker"].to_numpy()[idx],
        "p_up": proba[idx], "exp_ret_bps": exp_ret[idx], "exp_move_bps": exp_move[idx],
        "target": y, "next_ret": realized,
    })
    oos_path = str(Path(model_path).with_name("oos_predictions.pkl"))
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    oos.to_pickle(oos_path)

    print("\nTraining final models on all data...")
    clf, reg, vol = fit_models(panel[ALL_FEATURES], panel["target"], panel["next_ret"].to_numpy())
    oos_auc = float(roc_auc_score(y, proba[mask]))
    oos_ic = rank_ic(exp_ret[mask], realized)
    trained_at = datetime.now().isoformat(timespec="seconds")
    joblib.dump({"model": clf, "reg_model": reg, "vol_model": vol, "features": ALL_FEATURES,
                 "tickers": list(data), "trained_at": trained_at,
                 "oos_auc": oos_auc, "oos_ic": oos_ic}, model_path)
    print(f"Saved {model_path} and {oos_path}")

    if hasattr(clf, "feature_importances_"):   # XGBoost has it; the sklearn fallback doesn't
        imp = pd.Series(clf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
        print("\nTop 8 features:\n" + imp.head(8).to_string())

    acc = float(accuracy_score(y, proba[mask] > 0.5))
    return {
        "n_stocks": len(data), "n_samples": int(mask.sum()), "oos_auc": oos_auc,
        "oos_ic": oos_ic, "fold_auc": fold_auc, "accuracy": acc,
        "baseline": float(max(y.mean(), 1 - y.mean())),
        "std_error": float(np.sqrt(acc * (1 - acc) / len(y))),
        "trained_at": trained_at, "oos_path": oos_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of stocks")
    ap.add_argument("--days", type=int, default=720, help="history in days (max ~730)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--gap-days", type=int, default=1)
    ap.add_argument("--cost-bps", type=float, default=5.0,
                    help="cost per side in basis points (brokerage+taxes+slippage)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--model-path", default=MODEL_PATH)
    a = ap.parse_args()
    try:
        train_pipeline(a.n, a.days, a.splits, a.gap_days, a.cost_bps, a.refresh, a.model_path)
    except RuntimeError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
