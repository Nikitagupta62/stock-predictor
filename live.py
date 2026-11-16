"""
live.py

Live tracking: fetch the latest hourly bars, predict each stock's move over
the NEXT hour, log every prediction, and later score it against what really
happened.

    python live.py                  # one pass
    python live.py --loop 300       # refresh every 5 minutes (Ctrl+C to stop)

Notes
-----
* Only COMPLETED hourly bars are used. A prediction is made right after a bar
  closes (10:15, 11:15, 12:15, 13:15, 14:15 IST) for the close one hour later.
  There is no prediction after 14:15: the next hour would run past the close.
* Yahoo data can lag. Every row shows the time of the bar it is based on.
* Two log files, merged when read (so they never conflict in git):
    predictions_log.csv   written by whatever machine runs the app (git-ignored)
    data/live_log.csv     written by the scheduled GitHub job (log_job.py)
* Integrity guard: a prediction is only logged if it is made within
  MAX_LOG_AGE_MIN minutes of its bar closing. Otherwise part of the target hour
  has already happened and the "prediction" would be contaminated.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from intraday_data import IST, bar_end_times, download_universe, drop_incomplete_bars
from intraday_features import BAR, build_panel
from universe import get_universe

MODEL_PATH = "models/intraday_model.joblib"
LOG_PATH = "predictions_log.csv"           # this machine
SHARED_LOG_PATH = "data/live_log.csv"      # the scheduled GitHub job
MAX_LOG_AGE_MIN = 30
LOG_COLS = ["made_at", "ticker", "entry_price", "p_up", "exp_ret_bps", "exp_move_bps",
            "target_start", "target_end", "logged_at", "outcome_price", "outcome_up"]
PRED_COLS = ["ticker", "made_at", "price", "p_up", "exp_ret_bps", "exp_move_bps", "signal",
             "target_start", "target_end", "predictable"]


# ------------------------------------------------------------------ status --
def market_status(now: Optional[pd.Timestamp] = None) -> Tuple[bool, str]:
    """Is NSE in its regular session? (Exchange holidays are not modelled.)"""
    now = pd.Timestamp.now(tz=IST) if now is None else now.tz_convert(IST)
    mins = now.hour * 60 + now.minute
    if now.dayofweek >= 5:
        return False, "Market closed (weekend)"
    if mins < 9 * 60 + 15:
        return False, "Market not open yet (opens 09:15 IST)"
    if mins >= 15 * 60 + 30:
        return False, "Market closed for the day (closed 15:30 IST)"
    return True, "Market open"


# -------------------------------------------------------------------- data --
def load_bundle(path: str = MODEL_PATH) -> dict:
    if not Path(path).exists():
        raise SystemExit(f"No model at {path}. Run: python train_intraday.py")
    return joblib.load(path)


def fetch_recent(tickers, period: str = "1mo") -> Dict[str, pd.DataFrame]:
    """Last month of hourly bars (enough warm-up for every indicator), closed bars only."""
    data = download_universe(tickers, period=period, verbose=False)
    return {t: drop_incomplete_bars(df) for t, df in data.items()}


# -------------------------------------------------------------- prediction --
def predict_latest(bundle: dict, data: Dict[str, pd.DataFrame], threshold: float = 0.55,
                   hurdle_bps: float = 0.0) -> pd.DataFrame:
    """
    One row per stock, based on its most recent completed bar.

    p_up         probability the next hour closes higher
    exp_ret_bps  expected return over the next hour, in basis points (if available)
    exp_move_bps expected SIZE of the move, either direction (if available)
    signal       UP            p_up >= threshold and expected return > hurdle_bps
                 UP (small move) direction looks up but the expected move is below the hurdle
                 DOWN / NEUTRAL / "-" (no prediction possible for this bar)
    """
    panel = build_panel(data, training=False)
    if panel.empty:
        return pd.DataFrame(columns=PRED_COLS)

    last = panel.groupby("ticker").tail(1)
    # keep a stock only if its newest bar produced valid features (not stale)
    fresh = [t for t, ts in zip(last["ticker"], last.index) if ts == data[t].index[-1]]
    last = last[last["ticker"].isin(fresh)]
    if last.empty:
        return pd.DataFrame(columns=PRED_COLS)

    X = last[bundle["features"]]
    p = bundle["model"].predict_proba(X)[:, 1]
    nan = np.full(len(last), np.nan)
    exp_ret = bundle["reg_model"].predict(X) * 1e4 if bundle.get("reg_model") is not None else nan
    exp_move = bundle["vol_model"].predict(X) * 1e4 if bundle.get("vol_model") is not None else nan

    starts = last.index
    out = pd.DataFrame({
        "ticker": last["ticker"].to_numpy(),
        "made_at": bar_end_times(starts),
        "price": last["Close"].to_numpy(),
        "p_up": p, "exp_ret_bps": exp_ret, "exp_move_bps": exp_move,
        "target_start": starts + BAR,
        "target_end": starts + 2 * BAR,
        "predictable": last["can_predict"].to_numpy(),
    })
    up = out["p_up"] >= threshold
    big_enough = out["exp_ret_bps"].isna() | (out["exp_ret_bps"] > hurdle_bps)
    out["signal"] = np.where(up & big_enough, "UP",
                     np.where(up, "UP (small move)",
                     np.where(out["p_up"] <= 1 - threshold, "DOWN", "NEUTRAL")))
    out.loc[~out["predictable"], "signal"] = "-"
    return out[PRED_COLS].sort_values("p_up", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- the log --
def _read_one(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=LOG_COLS)
    log = pd.read_csv(path)
    for c in LOG_COLS:                      # older logs lack the newer columns
        if c not in log:
            log[c] = np.nan
    for c in ["made_at", "target_start", "target_end"]:
        log[c] = pd.to_datetime(log[c], utc=True)
    log["logged_at"] = pd.to_datetime(log["logged_at"], utc=True, errors="coerce")
    return log[LOG_COLS]


def read_log(path: Optional[str] = None) -> pd.DataFrame:
    """
    path=None -> every known log merged (job log + this machine's log), one row
    per (made_at, ticker), preferring the copy that already has its outcome.
    path=...  -> just that file.
    """
    if path is not None:
        return _read_one(path)
    frames = [_read_one(p) for p in (SHARED_LOG_PATH, LOG_PATH) if Path(p).exists()]
    if not frames:
        return pd.DataFrame(columns=LOG_COLS)
    log = pd.concat(frames, ignore_index=True).sort_values("outcome_price", na_position="last")
    log = log.drop_duplicates(["made_at", "ticker"], keep="first")
    return log.sort_values(["made_at", "ticker"]).reset_index(drop=True)


def log_predictions(preds: pd.DataFrame, path: str = LOG_PATH, now: Optional[pd.Timestamp] = None,
                    max_age_min: Optional[float] = None) -> int:
    """Append new, timely, predictable rows (never duplicates). Returns rows added."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now.tz_convert("UTC")
    limit = MAX_LOG_AGE_MIN if max_age_min is None else max_age_min
    new = preds[preds["predictable"]]
    if new.empty:
        return 0
    age_min = (now - new["made_at"].dt.tz_convert("UTC")).dt.total_seconds() / 60
    new = new[age_min <= limit]                # too late -> the hour is already under way
    if new.empty:
        return 0
    rows = pd.DataFrame({
        "made_at": new["made_at"].dt.tz_convert("UTC"),
        "ticker": new["ticker"], "entry_price": new["price"], "p_up": new["p_up"],
        "exp_ret_bps": new["exp_ret_bps"], "exp_move_bps": new["exp_move_bps"],
        "target_start": new["target_start"].dt.tz_convert("UTC"),
        "target_end": new["target_end"].dt.tz_convert("UTC"),
        "logged_at": now, "outcome_price": np.nan, "outcome_up": np.nan,
    })
    log = _read_one(path)
    if len(log):
        seen = set(zip(log["made_at"], log["ticker"]))
        rows = rows[[(m, t) not in seen for m, t in zip(rows["made_at"], rows["ticker"])]]
    if rows.empty:
        return 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat([log, rows], ignore_index=True)[LOG_COLS].to_csv(path, index=False)
    return len(rows)


def update_outcomes(data: Dict[str, pd.DataFrame], path: str = LOG_PATH) -> int:
    """Fill in what actually happened for predictions whose hour has now closed."""
    log = _read_one(path)
    if log.empty:
        return 0
    closes = {}
    for t, df in data.items():
        s = df["Close"].copy()
        s.index = s.index.tz_convert("UTC")
        closes[t] = s
    filled = 0
    for i in log.index[log["outcome_price"].isna()]:
        s = closes.get(log.at[i, "ticker"])
        ts = log.at[i, "target_start"]
        if s is not None and ts in s.index:       # closed bars only (see fetch_recent)
            log.at[i, "outcome_price"] = float(s.loc[ts])
            log.at[i, "outcome_up"] = float(s.loc[ts] > log.at[i, "entry_price"])
            filled += 1
    if filled:
        log.to_csv(path, index=False)
    return filled


def scorecard(log: pd.DataFrame, confident: float = 0.55) -> dict:
    """How have the live (forward) predictions actually done?"""
    r = log.dropna(subset=["outcome_up"])
    n = len(r)
    if n == 0:
        return {"resolved": 0, "pending": len(log)}
    hit = (r["p_up"] > 0.5).astype(float) == r["outcome_up"]
    up_rate = r["outcome_up"].mean()
    acc = hit.mean()
    conf = r[(r["p_up"] > confident) | (r["p_up"] < 1 - confident)]
    out = {
        "resolved": n, "pending": len(log) - n, "accuracy": acc,
        "baseline": max(up_rate, 1 - up_rate),
        "std_error": float(np.sqrt(acc * (1 - acc) / n)),
        "confident_n": len(conf),
        "confident_accuracy": float(((conf["p_up"] > 0.5).astype(float)
                                     == conf["outcome_up"]).mean()) if len(conf) else np.nan,
        "auc": np.nan,
    }
    if r["outcome_up"].nunique() == 2:
        out["auc"] = roc_auc_score(r["outcome_up"], r["p_up"])
    return out


# --------------------------------------------------------------------- CLI --
def run_once(bundle: dict, tickers, threshold: float, hurdle_bps: float = 0.0):
    data = fetch_recent(tickers)
    resolved = update_outcomes(data)
    preds = predict_latest(bundle, data, threshold, hurdle_bps)
    added = log_predictions(preds)
    return data, preds, resolved, added


def print_snapshot(preds: pd.DataFrame, resolved: int, added: int, threshold: float) -> None:
    _, msg = market_status()
    print(f"\n{'=' * 70}\n{pd.Timestamp.now(tz=IST):%a %d %b %Y %H:%M IST} | {msg}")
    if preds.empty:
        print("No predictions available (no fresh data).")
        return
    latest = preds["made_at"].max()
    print(f"Latest completed bar closed at {latest:%H:%M IST} on {latest:%d %b} "
          f"| {len(preds)} stocks | logged {added} new, scored {resolved}")
    ok = preds[preds["predictable"]]
    if ok.empty:
        print("No prediction for this bar: the next hour would run past 15:30. "
              "Next predictions: 10:15 IST on the next trading day.")
        return
    tgt = ok["target_end"].iloc[0]
    print(f"Predicting the close at {tgt:%H:%M IST} (one hour ahead)\n")
    show = ["ticker", "price", "p_up", "exp_ret_bps", "signal"]
    print(f"Most likely UP (P >= {threshold}):")
    print(ok[ok.signal.str.startswith("UP")].head(8)[show].round(3).to_string(index=False)
          or "  none")
    print(f"\nMost likely DOWN (P <= {1 - threshold:.2f}):")
    print(ok[ok.signal == "DOWN"].tail(8)[show].round(3).to_string(index=False) or "  none")
    s = scorecard(read_log())
    if s.get("resolved"):
        print(f"\nLive scorecard: {s['resolved']} scored | accuracy {s['accuracy']:.3f} "
              f"(+/-{s['std_error']:.3f}) vs baseline {s['baseline']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--hurdle-bps", type=float, default=0.0)
    ap.add_argument("--loop", type=int, default=0, help="seconds between refreshes (0 = once)")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    bundle = load_bundle(args.model_path)
    tickers = get_universe(args.n)
    while True:
        _, preds, resolved, added = run_once(bundle, tickers, args.threshold, args.hurdle_bps)
        print_snapshot(preds, resolved, added, args.threshold)
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
