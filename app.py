"""
app.py
Streamlit dashboard: pick a ticker, train/load a model, see the next-day
prediction plus recent price history and indicators.

Run: streamlit run app.py
"""

import os
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_fetch import fetch_price_history
from features import build_features, FEATURE_COLUMNS
from train_model import walk_forward_validate, train_final_model
from predict import build_features_for_prediction

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")

st.title("📈 Daily Stock Direction Predictor")
st.caption(
    "For personal, educational use only. This is a probabilistic model based on "
    "historical technical indicators — it is NOT financial advice, and daily "
    "market movements are inherently very hard to predict. Treat outputs as "
    "one weak signal among many, not a trading instruction."
)

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    period = st.selectbox("Historical data window", ["1y", "2y", "5y", "10y", "max"], index=2)
    n_splits = st.slider("Validation folds", min_value=3, max_value=8, value=5)
    run_button = st.button("Fetch data & run model", type="primary")

if run_button and ticker:
    model_path = f"model_{ticker.replace('.', '_')}.joblib"

    with st.spinner(f"Fetching {ticker} data..."):
        try:
            raw = fetch_price_history(ticker, period=period)
        except Exception as e:
            st.error(f"Couldn't fetch data: {e}")
            st.stop()

    # --- Price chart ---
    st.subheader(f"{ticker} — Price History")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=raw.index, open=raw["Open"], high=raw["High"],
        low=raw["Low"], close=raw["Close"], name=ticker
    ))
    fig.update_layout(height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Build features & validate ---
    with st.spinner("Building features and running walk-forward validation..."):
        data = build_features(raw)
        X = data[FEATURE_COLUMNS]
        y = data["target"]
        results = walk_forward_validate(X, y, n_splits=n_splits)

    st.subheader("Model Validation (walk-forward, out-of-sample)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Accuracy", f"{results['accuracy'].mean():.1%}")
    col2.metric("Naive 'always up' baseline", f"{results['baseline_accuracy'].mean():.1%}")
    col3.metric("Avg. Precision", f"{results['precision'].mean():.1%}")
    col4.metric("Avg. Recall", f"{results['recall'].mean():.1%}")

    if results["accuracy"].mean() <= results["baseline_accuracy"].mean() + 0.02:
        st.warning(
            "The model is barely beating (or is below) the naive baseline of "
            "always predicting 'up'. This is common and expected for daily "
            "stock prediction — treat any signal here with heavy skepticism."
        )

    with st.expander("See per-fold validation results"):
        st.dataframe(results)

    # --- Train final model & predict ---
    with st.spinner("Training final model and generating prediction..."):
        final_model = train_final_model(X, y)
        joblib.dump({"model": final_model, "features": FEATURE_COLUMNS, "ticker": ticker}, model_path)

        latest_features = build_features_for_prediction(raw, FEATURE_COLUMNS)
        proba = final_model.predict_proba(latest_features)[0]
        pred = final_model.predict(latest_features)[0]

    st.subheader("Next Trading Day Prediction")
    last_close = raw["Close"].iloc[-1]
    last_date = raw.index[-1].date()

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Last Close", f"{last_close:.2f}", help=f"as of {last_date}")
    pcol2.metric("Prediction", "⬆️ UP" if pred == 1 else "⬇️ DOWN")
    pcol3.metric("Confidence", f"{max(proba):.1%}")

    st.progress(float(proba[1]), text=f"Probability UP: {proba[1]:.1%}  |  Probability DOWN: {proba[0]:.1%}")

    # --- Feature importance ---
    st.subheader("What's driving this prediction?")
    importances = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False).head(10)
    fig2 = go.Figure(go.Bar(x=importances.values, y=importances.index, orientation="h"))
    fig2.update_layout(height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Enter a ticker symbol in the sidebar and click 'Fetch data & run model' to get started.")
    st.markdown(
        """
        **Examples of valid tickers:**
        - US stocks: `AAPL`, `TSLA`, `MSFT`, `NVDA`
        - Indian stocks (NSE): `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
        - Indian stocks (BSE): `RELIANCE.BO`
        """
    )
