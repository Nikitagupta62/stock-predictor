"""
test_intraday.py  --  python test_intraday.py   (or: pytest -q)

Offline checks on synthetic NSE-style hourly data. They verify the logic that
most often goes wrong: label alignment, look-ahead leaks, forming-bar removal,
live prediction timing and the prediction log. They do NOT test Yahoo access.
"""

import os
import tempfile

import numpy as np
import pandas as pd

from intraday_data import IST, bar_end_times, drop_incomplete_bars
from intraday_features import ALL_FEATURES, LAST_PREDICTABLE_BAR, build_panel
from live import predict_latest, log_predictions, update_outcomes, read_log, scorecard
from train_intraday import make_model, walk_forward

STARTS = ["09:15", "10:15", "11:15", "12:15", "13:15", "14:15", "15:15"]


def make_ticker(seed: int, days: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2024-01-01", periods=days):
        idx += [pd.Timestamp(f"{d:%Y-%m-%d} {s}", tz=IST) for s in STARTS]
    idx = pd.DatetimeIndex(idx)
    ret = rng.normal(0.00002, 0.004, len(idx))
    close = 500 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(1e5, 5e5, len(idx)).astype(float)
    vol[idx.hour == 15] *= 0.25
    return pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.001,
                         "Low": np.minimum(open_, close) * 0.999,
                         "Close": close, "Volume": vol}, index=idx)


def make_universe(n: int = 8):
    return {f"T{i}.NS": make_ticker(i) for i in range(n)}


def test_bar_times_and_forming_bar():
    df = make_ticker(0, days=3)
    ends = bar_end_times(df.index)
    assert ends[6].strftime("%H:%M") == "15:30"      # short closing bar
    assert ends[0].strftime("%H:%M") == "10:15"
    now = pd.Timestamp("2024-01-02 11:00", tz=IST)   # 10:15 bar is still forming
    kept = drop_incomplete_bars(df, now=now)
    assert kept.index.max() == pd.Timestamp("2024-01-02 09:15", tz=IST)


def test_target_alignment():
    data = make_universe(3)
    panel = build_panel(data, training=True)
    assert set(panel["bar_of_day"].unique()) == {0.0, 1.0, 2.0, 3.0, 4.0}
    for t, grp in panel.groupby("ticker"):
        full = data[t]
        nxt = full["Close"].shift(-1).reindex(grp.index)
        expected = (nxt > grp["Close"]).astype(int)
        assert (grp["target"] == expected).all()
        # never crosses the overnight gap
        assert ((full.index.to_series().shift(-1) - full.index.to_series())
                .reindex(grp.index) == pd.Timedelta(hours=1)).all()


def test_no_lookahead():
    data = make_universe(5)
    cut = pd.Timestamp("2024-04-10 12:15", tz=IST)
    part = {t: df[df.index <= cut] for t, df in data.items()}
    a = build_panel(part, training=False)
    b = build_panel(data, training=False)
    b = b.reset_index().merge(a.reset_index()[["index", "ticker"]], on=["index", "ticker"])
    a = a.reset_index().sort_values(["index", "ticker"])
    b = b.sort_values(["index", "ticker"])
    assert np.allclose(a[ALL_FEATURES].to_numpy(), b[ALL_FEATURES].to_numpy(), equal_nan=True), \
        "features changed when future bars were removed -> look-ahead leak"


def test_walk_forward_runs_and_random_data_has_no_edge():
    panel = build_panel(make_universe(8), training=True)
    proba, aucs = walk_forward(panel, n_splits=4, gap_days=1)
    assert not np.isnan(proba).all()
    assert abs(np.mean(aucs) - 0.5) < 0.04


def _bundle(panel):
    model = make_model()
    model.fit(panel[ALL_FEATURES], panel["target"])
    return {"model": model, "features": ALL_FEATURES}


def test_live_prediction_timing_and_log():
    data = make_universe(6)
    panel = build_panel(data, training=True)
    bundle = _bundle(panel)

    # Live moment: the 11:15 bar has just closed (it is 12:15 IST). We predict
    # the bar that starts at 12:15 and closes at 13:15.
    cut = pd.Timestamp("2024-05-01 11:15", tz=IST)
    live = {t: df[df.index <= cut] for t, df in data.items()}
    preds = predict_latest(bundle, live)
    assert len(preds) == 6 and preds["predictable"].all()
    assert preds["p_up"].between(0, 1).all()
    assert (preds["target_start"] == cut + pd.Timedelta(hours=1)).all()

    # after the 14:15 bar closes no prediction is allowed (next hour past close)
    late = {t: df[df.index <= pd.Timestamp("2024-05-01 14:15", tz=IST)] for t, df in data.items()}
    assert not predict_latest(bundle, late)["predictable"].any()

    # log + outcome scoring
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        assert log_predictions(preds, path) == 6
        assert log_predictions(preds, path) == 0            # no duplicates
        assert update_outcomes(live, path) == 0             # hour not closed yet
        later = {t: df[df.index <= cut + pd.Timedelta(hours=1)] for t, df in data.items()}
        assert update_outcomes(later, path) == 6
        log = read_log(path)
        truth = {t: float(data[t].loc[cut + pd.Timedelta(hours=1), "Close"] > data[t].loc[cut, "Close"])
                 for t in data}
        assert all(truth[t] == o for t, o in zip(log["ticker"], log["outcome_up"]))
        assert scorecard(log)["resolved"] == 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASSED  {name}")
