"""
hourly_view.py

The "Hourly live" screen of the single app. app.py calls render_hourly().
(This replaces the separate live_app.py: there is only ONE app now.)
"""

import contextlib
import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from live import (MODEL_PATH, fetch_recent, load_bundle, log_predictions, market_status,
                  predict_latest, read_log, scorecard, update_outcomes)
from results_view import render_paper, render_results
from train_intraday import train_pipeline
from universe import get_universe

_cached_bundle = st.cache_resource(load_bundle)


@st.cache_data(ttl=120, show_spinner="Fetching latest hourly bars...")
def _get_data(tickers: tuple):
    return fetch_recent(list(tickers))


def _training_panel() -> None:
    """Shown when no hourly model exists yet (or when the user wants to retrain)."""
    st.info("The hourly model has not been trained yet. Training downloads about 2 years of "
            "hourly data for the stocks and takes a few minutes the first time.")
    n = st.slider("Stocks to train on", 10, 50, 50, step=5, key="train_n")
    if st.button("Train hourly model", type="primary"):
        buf = io.StringIO()
        with st.spinner("Downloading data and training... (a few minutes)"):
            try:
                with contextlib.redirect_stdout(buf):
                    res = train_pipeline(n=n)
            except Exception as exc:  # network errors, empty data, etc.
                st.error(f"Training failed: {exc}")
                st.code(buf.getvalue()[-3000:])
                return
        st.cache_resource.clear()
        st.success(f"Model trained on {res['n_stocks']} stocks. Out-of-sample AUC "
                   f"{res['oos_auc']:.3f} (0.500 = coin flip).")
        with st.expander("Full training report"):
            st.code(buf.getvalue())
        st.rerun()


def render_hourly() -> None:
    st.title("⏱️ Next-Hour Direction (live, NSE stocks)")
    st.caption("Educational project, not investment advice. One-hour direction is very close "
               "to a coin flip; judge it by the live scorecard, not by a single signal.")

    with st.sidebar:
        st.header("Hourly settings")
        n = st.slider("Number of stocks", 10, 50, 50, step=5)
        threshold = st.slider("Signal threshold P(up)", 0.50, 0.70, 0.55, step=0.01)
        hurdle = st.slider("Minimum expected move (bps)", 0.0, 20.0, 0.0, step=0.5,
                           help="Signals whose expected 1-hour move is smaller than this "
                                "are shown as 'UP (small move)': too small to pay costs.")
        auto = st.checkbox("Auto-refresh", value=False)
        every = st.select_slider("Refresh every (sec)", [60, 120, 300, 600], value=300)
        if st.button("Refresh now"):
            st.cache_data.clear()

    if not Path(MODEL_PATH).exists():
        _training_panel()
        return

    bundle = _cached_bundle()
    data = _get_data(tuple(get_universe(n)))
    if not data:
        st.warning("No data available right now (Yahoo may be rate limiting). Try again shortly.")
        return
    update_outcomes(data)
    preds = predict_latest(bundle, data, threshold, hurdle)
    log_predictions(preds)

    is_open, status = market_status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market", "OPEN" if is_open else "CLOSED", status)
    c2.metric("Stocks with fresh data", len(preds))
    ok = preds[preds["predictable"]] if len(preds) else preds
    if len(preds):
        latest = preds["made_at"].max()
        c3.metric("Based on bar closed at", f"{latest:%H:%M IST}", f"{latest:%d %b}")
        c4.metric("Predicting close at",
                  f"{ok['target_end'].iloc[0]:%H:%M IST}" if len(ok) else "n/a")

    tab_pred, tab_chart, tab_res, tab_paper, tab_score, tab_model = st.tabs(
        ["Predictions", "Chart", "Results", "Paper portfolio", "Live scorecard", "Model"])

    with tab_pred:
        if preds.empty:
            st.warning("No fresh predictions right now.")
        elif ok.empty:
            st.info("No prediction for this bar: the next hour would run past the 15:30 close. "
                    "Next predictions arrive at 10:15 IST on the next trading day.")
        else:
            st.dataframe(
                ok[["ticker", "price", "p_up", "exp_ret_bps", "exp_move_bps", "signal"]],
                hide_index=True,
                column_config={
                    "p_up": st.column_config.ProgressColumn(
                        "P(up in 1h)", min_value=0.0, max_value=1.0, format="%.3f"),
                    "price": st.column_config.NumberColumn("Last close", format="%.2f"),
                    "exp_ret_bps": st.column_config.NumberColumn(
                        "Expected move (bps)", format="%+.1f",
                        help="Model's expected return over the next hour. 1 bp = 0.01%."),
                    "exp_move_bps": st.column_config.NumberColumn(
                        "Typical size (bps)", format="%.1f",
                        help="Expected size of the move in either direction."),
                },
            )

    with tab_chart:
        pick = st.selectbox("Stock", sorted(data))
        st.line_chart(data[pick]["Close"].tail(35).tz_localize(None))
        row = preds[preds["ticker"] == pick] if len(preds) else preds
        if len(row) and bool(row["predictable"].iloc[0]):
            st.write(f"**{pick}**: P(up in next hour) = {row['p_up'].iloc[0]:.3f} "
                     f"-> {row['signal'].iloc[0]}")

    with tab_res:
        render_results()

    with tab_paper:
        render_paper()

    with tab_score:
        s = scorecard(read_log())
        if not s.get("resolved"):
            st.info(f"No scored predictions yet ({s.get('pending', 0)} waiting for their hour to "
                    "close). Keep the app open during market hours (9:15-15:30 IST).")
        else:
            a, b, c, d = st.columns(4)
            a.metric("Scored predictions", s["resolved"])
            b.metric("Accuracy", f"{s['accuracy']:.3f}", f"+/- {s['std_error']:.3f} std err")
            c.metric("Baseline (best constant)", f"{s['baseline']:.3f}")
            d.metric("AUC", "n/a" if pd.isna(s["auc"]) else f"{s['auc']:.3f}")
            st.write(f"Confident calls (P > 0.55 or < 0.45): {s['confident_n']} "
                     f"| accuracy {s['confident_accuracy']:.3f}")
            st.caption("Until you have thousands of scored predictions, differences of 1-2 "
                       "points from the baseline are indistinguishable from luck.")
            st.dataframe(read_log().sort_values("made_at", ascending=False).head(200),
                         hide_index=True)

    with tab_model:
        st.write(f"Trained at: {bundle.get('trained_at', 'unknown')} | stocks: "
                 f"{len(bundle.get('tickers', []))} | out-of-sample AUC at training: "
                 f"{bundle.get('oos_auc', float('nan')):.3f} (0.500 = no skill) | return rank-IC "
                 f"{bundle.get('oos_ic', float('nan')):+.3f} (0 = no skill)")
        with st.expander("Retrain the hourly model"):
            _training_panel()

    if auto:
        time.sleep(every)
        st.cache_data.clear()
        st.rerun()
