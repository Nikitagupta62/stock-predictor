"""
results_view.py

Two screens for the hourly view:
  render_results() - "Would these signals have made money?" (historical, out-of-sample)
  render_paper()   - virtual portfolio built from the LIVE prediction log

Both use strategy.py, so the rules are identical in the backtest and live.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from live import MODEL_PATH, read_log
from strategy import (backtest, brier, calibration_table, equity_curve, paper_portfolio,
                      perf_metrics, random_baseline, rank_ic, verdict)

OOS_PATH = str(Path(MODEL_PATH).with_name("oos_predictions.pkl"))
CAUTION = ("Entries assume you are filled at the closing price of the signal bar. Real fills are "
           "slightly worse, so keep a realistic cost setting. Educational only, not investment advice.")


def _show(level: str, text: str) -> None:
    {"success": st.success, "warning": st.warning, "info": st.info}.get(level, st.info)(text)


@st.cache_resource
def _load_oos(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_pickle(path)


@st.cache_data(show_spinner="Replaying the strategy on historical data...")
def _simulate(mtime: float, threshold: float, k: int, cost: float, hurdle: float) -> dict:
    oos = _load_oos(OOS_PATH, mtime)
    out = {}
    for name, use_mag in [("Direction only", False), ("Direction + expected move", True)]:
        bt = backtest(oos, threshold, k, cost, hurdle if use_mag else 0.0, use_mag)
        rb = random_baseline(oos, bt["selected"], n_perm=200)
        out[name] = {"bt": {key: bt[key] for key in
                            ("strategy", "n_trades", "gross_bps", "net_bps", "hit_rate", "exposure")},
                     "perf": perf_metrics(bt["strategy"]), "rb": rb}
    bench = backtest(oos, threshold, k, cost)["benchmark"]
    out["_benchmark"] = {"series": bench, "perf": perf_metrics(bench)}
    return out


def render_results() -> None:
    st.subheader("Historical test: would these signals have made money?")
    st.caption("Replays the model's out-of-sample predictions (made by models that had never seen "
               "those days) as a paper-trading strategy, after trading costs.")
    if not Path(OOS_PATH).exists():
        st.info("The test data does not exist yet. Retrain the hourly model (Model tab) to create it.")
        return
    mtime = os.path.getmtime(OOS_PATH)
    oos = _load_oos(OOS_PATH, mtime)

    with st.expander("Strategy settings", expanded=True):
        threshold = st.slider("Buy only if P(up) is above", 0.50, 0.70, 0.55, step=0.01, key="r_thr")
        k = st.slider("Max stocks held at once", 1, 10, 5, key="r_k")
        cost = st.slider("Cost per side (bps: brokerage + taxes + slippage)", 0.0, 20.0, 5.0,
                         step=0.5, key="r_cost")
        hurdle = st.slider("Minimum expected move to trade, 'Direction + expected move' only (bps)",
                           0.0, 20.0, 0.0, step=0.5, key="r_hurdle")

    sim = _simulate(mtime, threshold, k, cost, hurdle)
    names = ["Direction only", "Direction + expected move"]

    # --- equity curves ---
    curves = {"Hold all stocks (same hours, no costs)": equity_curve(sim["_benchmark"]["series"])}
    for nm in names:
        curves[nm] = equity_curve(sim[nm]["bt"]["strategy"])
    chart = pd.DataFrame(curves)
    chart.index = pd.DatetimeIndex(chart.index).tz_localize(None)
    st.line_chart(chart)

    # --- metrics table ---
    rows = {"Hold all stocks": {"Total return": sim["_benchmark"]["perf"]["total_return"],
                                "Sharpe": sim["_benchmark"]["perf"]["sharpe"],
                                "Max drawdown": sim["_benchmark"]["perf"]["max_drawdown"]}}
    for nm in names:
        p, b = sim[nm]["perf"], sim[nm]["bt"]
        rows[nm] = {"Total return": p["total_return"], "Sharpe": p["sharpe"],
                    "Max drawdown": p["max_drawdown"], "Trades": b["n_trades"],
                    "Net bps/trade": b["net_bps"], "Hit rate": b["hit_rate"],
                    "Time in market": b["exposure"],
                    "p-value vs random": sim[nm]["rb"]["p_value"]}
    table = pd.DataFrame(rows).T
    for c in ["Total return", "Max drawdown", "Hit rate", "Time in market"]:
        if c in table:
            table[c] = table[c].map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    for c, f in [("Sharpe", "{:.2f}"), ("Net bps/trade", "{:+.2f}"), ("p-value vs random", "{:.3f}")]:
        if c in table:
            table[c] = table[c].map(lambda v, f=f: "-" if pd.isna(v) else f.format(v))
    if "Trades" in table:
        table["Trades"] = table["Trades"].map(lambda v: "-" if pd.isna(v) else f"{int(v):,}")
    st.dataframe(table)

    for nm in names:
        b, rb = sim[nm]["bt"], sim[nm]["rb"]
        st.markdown(f"**{nm}**")
        _show(*verdict(sim[nm]["perf"]["total_return"], rb["p_value"], b["n_trades"]))
    st.caption("p-value vs random: how often picking random stocks, the same number per hour, did "
               "at least as well before costs. Under 0.05 suggests the picks add something beyond "
               "luck. 'Hold all stocks' only counts these 5 hours per day, so it ignores overnight moves.")

    # --- calibration ---
    st.markdown("#### Are the probabilities honest?")
    cal = calibration_table(oos["p_up"], oos["target"])
    cal_chart = pd.DataFrame({"Actual up-rate": cal["actual_up_rate"].to_numpy(),
                              "Perfectly calibrated": cal["mean_pred"].to_numpy()},
                             index=cal["mean_pred"].round(3).to_numpy())
    st.line_chart(cal_chart)
    base = float(oos["target"].mean())
    st.caption(f"Each point is a bucket of predictions. If the model said about 55%, did the stock "
               f"rise about 55% of the time? Brier score {brier(oos['p_up'], oos['target']):.4f} vs "
               f"{brier([base] * len(oos), oos['target']):.4f} for always guessing the average "
               "(lower is better). Expected-return rank IC: "
               f"{rank_ic(oos['exp_ret_bps'], oos['next_ret']):+.4f} (0 = no skill).")
    st.caption(CAUTION)


def render_paper() -> None:
    st.subheader("Live paper portfolio (virtual money, real forward predictions)")
    st.caption("Built from the prediction log: every qualifying signal becomes a virtual trade, "
               "bought at the signal bar's close and sold one hour later.")
    log = read_log()
    if log.empty:
        st.info("No live predictions logged yet. Keep the app open during market hours (9:15-15:30 IST), "
                "or enable the scheduled GitHub job so it logs automatically.")
        return

    with st.expander("Portfolio settings", expanded=False):
        capital = st.select_slider("Starting capital (INR)", [50_000, 100_000, 250_000, 500_000, 1_000_000],
                                   value=100_000, key="p_cap")
        threshold = st.slider("Buy only if P(up) is above", 0.50, 0.70, 0.55, step=0.01, key="p_thr")
        k = st.slider("Max stocks held at once", 1, 10, 5, key="p_k")
        cost = st.slider("Cost per side (bps)", 0.0, 20.0, 5.0, step=0.5, key="p_cost")
        use_mag = st.checkbox("Also require an expected move above the minimum", value=False, key="p_mag")
        hurdle = st.slider("Minimum expected move (bps)", 0.0, 20.0, 0.0, step=0.5, key="p_hurdle")

    trades, s, equity = paper_portfolio(log, threshold, k, cost, float(capital), hurdle, use_mag)
    if trades.empty:
        st.info("No logged prediction meets these settings yet. Try a lower threshold.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Total P&L", f"INR {s['total_pnl']:,.0f}", f"{s['return_pct']:+.2%}")
    b.metric("Closed / open trades", f"{s['closed']} / {s['open']}")
    c.metric("Win rate", "-" if pd.isna(s["win_rate"]) else f"{s['win_rate']:.1%}")
    d.metric("Max drawdown", "-" if pd.isna(s["max_drawdown"]) else f"{s['max_drawdown']:.1%}")

    if len(equity):
        e = equity.copy()
        e.index = pd.DatetimeIndex(e.index).tz_localize(None)
        st.line_chart(e.rename("Portfolio value (INR)"))
    if s["closed"] < 200:
        st.info(f"Too early to judge: only {s['closed']} closed trades. Luck dominates small samples; "
                "aim for several hundred before drawing conclusions.")
    show = trades[["t", "ticker", "entry_price", "outcome_price", "p_up", "exp_ret_bps",
                   "status", "net_ret", "pnl_inr"]].rename(columns={"t": "signal_time"})
    show["signal_time"] = pd.DatetimeIndex(show["signal_time"]).tz_convert("Asia/Kolkata").tz_localize(None)
    st.dataframe(show.sort_values("signal_time", ascending=False).head(200), hide_index=True)
    st.caption(CAUTION)
