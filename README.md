# Daily Stock Direction Predictor (Personal Project)

A starter project that predicts whether a stock will close **up or down**
the next trading day, using technical indicators and a gradient-boosted
tree model (XGBoost). Includes a Streamlit dashboard.

⚠️ **This is a learning/personal project, not a trading system.** Daily
stock movements are extremely hard to predict — professional quant funds
with far more data and compute struggle to reliably beat ~55-56% accuracy
on this exact problem. Do not use this to make real trading decisions
without a lot more validation, risk management, and skepticism.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Option A — Dashboard (recommended, easiest)
```bash
streamlit run app.py
```
Opens a browser window where you can enter a ticker, see the price chart,
validation metrics, and next-day prediction.

### Option B — Command line

Train a model and see validation metrics:
```bash
python train_model.py --ticker AAPL --period 5y
```

Get a prediction from an already-trained model:
```bash
python predict.py --ticker AAPL
```

## How it works

1. **`data_fetch.py`** — pulls historical daily OHLCV data via `yfinance`
2. **`features.py`** — computes technical indicators (RSI, MACD, Bollinger
   Bands, moving averages, ATR, OBV, etc.) as model features, and creates
   the "did price go up tomorrow?" label
3. **`train_model.py`** — trains an XGBoost classifier with **walk-forward
   validation** (never shuffles time — always trains on the past and
   tests on a later, unseen period, to avoid lookahead bias)
4. **`predict.py`** — loads a saved model and predicts tomorrow's direction
   using today's data
5. **`app.py`** — Streamlit UI wrapping all of the above

## Ticker symbol formats

- US stocks: `AAPL`, `MSFT`, `TSLA`
- Indian stocks (NSE): `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- Indian stocks (BSE): `RELIANCE.BO`
- Full list of supported formats: https://finance.yahoo.com

## Where to take this further

- **Add more data sources**: news sentiment (NewsAPI), macro indicators
  (FRED), earnings calendar, options flow
- **Try predicting magnitude, not just direction** (regression instead of
  classification) — e.g. "how much will it move," which is often more
  useful for position sizing
- **Ensemble multiple models** (XGBoost + LSTM + logistic regression) and
  compare/blend their predictions
- **Backtest a simple trading strategy** based on the signal (with
  transaction costs and slippage included!) before ever trusting it
- **Track live prediction accuracy over time** in a small database, so
  you can see if the model actually holds up out-of-sample going forward,
  not just in historical backtests

## A note on realistic expectations

If your validation accuracy is only a couple points above the "always
predict up" baseline (which the dashboard shows you), that's actually
the normal, expected result for this problem — not a bug in your code.
Markets are close to (though not perfectly) efficient, and any
easily-found edge tends to get arbitraged away. Treat this as a genuinely
hard ML problem worth learning from, not a shortcut to consistent profit.
