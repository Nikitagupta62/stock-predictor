"""
strategy.py

The trading logic shared by the historical backtest (Results tab) and the live
paper portfolio, so both use EXACTLY the same rules.

Rules (one decision per hourly bar, per stock):
  * eligible  : P(up) > threshold  (and, optionally, expected return > hurdle)
  * ranking   : by P(up), or by expected return when magnitude is used
  * portfolio : at most `max_positions` stocks per bar, equal weight, each held
                for exactly one hour; unused slots stay in cash
  * costs     : charged on entry AND exit of every trade (round trip = 2 x cost)

Everything here is plain pandas/numpy (no Streamlit), so it is easy to test.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 5 * 248      # 5 predicted hours per day x ~248 trading days


# ----------------------------------------------------------------- selection --
def select_trades(df: pd.DataFrame, threshold: float = 0.55, max_positions: int = 5,
                  hurdle_bps: float = 0.0, use_magnitude: bool = False) -> np.ndarray:
    """
    Boolean mask of the trades the strategy takes.

    df needs columns: t (decision time), p_up, and exp_ret_bps when use_magnitude.
    """
    d = df.reset_index(drop=True)
    eligible = d["p_up"] > threshold
    if use_magnitude:
        if "exp_ret_bps" not in d:
            return np.zeros(len(d), dtype=bool)
        eligible = eligible & (d["exp_ret_bps"] > hurdle_bps)   # NaN -> False
        score = d["exp_ret_bps"]
    else:
        score = d["p_up"]
    rank = score.where(eligible).groupby(d["t"]).rank(method="first", ascending=False)
    return (rank <= max_positions).to_numpy()


# ------------------------------------------------------------------ backtest --
def backtest(oos: pd.DataFrame, threshold: float, max_positions: int, cost_bps: float,
             hurdle_bps: float = 0.0, use_magnitude: bool = False) -> Dict:
    """
    Replay out-of-sample predictions as a trading strategy.

    oos columns: t, ticker, p_up, next_ret (and exp_ret_bps for magnitude mode).
    Returns per-bar strategy returns, a benchmark and trade statistics.
    The benchmark is an equal-weight basket of ALL stocks held during the same
    hours, with no costs -- the fairest "just hold" comparison for these hours.
    """
    d = oos.reset_index(drop=True)
    sel = select_trades(d, threshold, max_positions, hurdle_bps, use_magnitude)
    ret = d["next_ret"].to_numpy()
    net = ret - 2 * cost_bps / 1e4
    pnl = pd.Series(np.where(sel, net, 0.0) / max_positions, index=d.index)
    strategy = pnl.groupby(d["t"]).sum().sort_index()
    benchmark = d.groupby("t")["next_ret"].mean().sort_index()

    gross = ret[sel]
    n = int(sel.sum())
    in_market = pd.Series(sel).groupby(d["t"]).sum().gt(0).mean() if n else 0.0
    return {
        "strategy": strategy, "benchmark": benchmark, "selected": sel, "n_trades": n,
        "gross_bps": float(gross.mean() * 1e4) if n else float("nan"),
        "net_bps": float(gross.mean() * 1e4 - 2 * cost_bps) if n else float("nan"),
        "hit_rate": float((gross > 0).mean()) if n else float("nan"),
        "exposure": float(in_market),
    }


def perf_metrics(returns: pd.Series) -> Dict[str, float]:
    """Total return, CAGR, Sharpe and max drawdown for a series of per-bar returns."""
    r = returns.dropna()
    nan = float("nan")
    if r.empty:
        return {"total_return": nan, "cagr": nan, "sharpe": nan, "max_drawdown": nan}
    equity = (1 + r).cumprod()
    total = float(equity.iloc[-1] - 1)
    years = len(r) / PERIODS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else nan
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd and sd > 0 else nan
    mdd = float((equity / equity.cummax() - 1).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd}


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0.0)).cumprod()


# ------------------------------------------------------- is it better than luck? --
def random_baseline(oos: pd.DataFrame, selected: np.ndarray, n_perm: int = 200,
                    seed: int = 0) -> Dict[str, float]:
    """
    Compare the strategy with random stock-picking that trades the SAME number
    of stocks in the SAME hours (so market-wide moves are held constant).
    p_value = share of random runs that did at least as well (gross, per trade).
    """
    d = oos.reset_index(drop=True)
    sel = np.asarray(selected, dtype=bool)
    nan = float("nan")
    if sel.sum() == 0:
        return {"observed_bps": nan, "random_mean_bps": nan, "p_value": nan}
    codes = pd.factorize(d["t"], sort=True)[0]
    k = np.bincount(codes[sel], minlength=codes.max() + 1)
    sizes = np.bincount(codes)
    starts = np.cumsum(sizes) - sizes
    ret = d["next_ret"].to_numpy()
    observed = ret[sel].mean()
    rng = np.random.default_rng(seed)
    n = len(d)
    stats = np.empty(n_perm)
    for i in range(n_perm):
        order = np.lexsort((rng.random(n), codes))      # grouped by time, random inside
        pos = np.arange(n) - starts[codes[order]]
        take = order[pos < k[codes[order]]]
        stats[i] = ret[take].mean()
    p = (1 + int((stats >= observed).sum())) / (n_perm + 1)
    return {"observed_bps": float(observed * 1e4), "random_mean_bps": float(stats.mean() * 1e4),
            "p_value": float(p)}


def verdict(net_total_return: float, p_value: float, n_trades: int,
            min_trades: int = 200) -> Tuple[str, str]:
    """Plain-English reading of a backtest: (level, message)."""
    if n_trades < min_trades:
        return "info", (f"Only {n_trades} trades: too few to conclude anything. "
                        "Lower the threshold or the minimum expected move.")
    if not np.isfinite(net_total_return) or net_total_return <= 0:
        return "warning", ("This strategy LOSES money after costs in the historical test. "
                           "Any raw predictive skill is smaller than trading costs.")
    if not np.isfinite(p_value) or p_value > 0.05:
        return "info", (f"It makes money after costs, but random picking with the same number of "
                        f"trades does as well {p_value:.0%} of the time, so luck cannot be ruled out.")
    return "success", (f"It beats random picking (p = {p_value:.3f}) after costs in this historical "
                       "test. Treat that as a hypothesis: confirm it on the live paper portfolio "
                       "before trusting it.")


# --------------------------------------------------------------- calibration --
def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Do predicted probabilities match reality? One row per probability bucket."""
    df = pd.DataFrame({"p": np.asarray(p, dtype=float), "y": np.asarray(y, dtype=float)})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), actual_up_rate=("y", "mean"), n=("y", "size"))
    return g.reset_index(drop=True)


def brier(p, y) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def rank_ic(pred, realized) -> float:
    """Spearman correlation between predicted and realised returns (the 'IC')."""
    a, b = pd.Series(np.asarray(pred, dtype=float)), pd.Series(np.asarray(realized, dtype=float))
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() > 2 else float("nan")


# ------------------------------------------------------------ paper portfolio --
def paper_portfolio(log: pd.DataFrame, threshold: float, max_positions: int, cost_bps: float,
                    capital: float = 100_000.0, hurdle_bps: float = 0.0,
                    use_magnitude: bool = False
                    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    """
    Virtual portfolio built from the LIVE prediction log.

    Each qualifying prediction becomes a trade: buy at the signal bar's close,
    sell one hour later. Position size = capital / max_positions (no compounding,
    for simplicity). Entry at the signal bar's close is slightly optimistic;
    the cost setting is where to account for slippage.
    Returns (trades, summary, equity_curve_by_exit_time).
    """
    nan = float("nan")
    empty_summary = {"closed": 0, "open": 0, "total_pnl": 0.0, "return_pct": 0.0,
                     "win_rate": nan, "avg_net_bps": nan, "max_drawdown": nan}
    if log is None or len(log) == 0:
        return pd.DataFrame(), empty_summary, pd.Series(dtype=float)

    d = log.rename(columns={"made_at": "t"}).reset_index(drop=True)
    sel = select_trades(d, threshold, max_positions, hurdle_bps, use_magnitude)
    tr = d[sel].copy()
    if tr.empty:
        return pd.DataFrame(), empty_summary, pd.Series(dtype=float)

    slot = capital / max_positions
    tr["gross_ret"] = tr["outcome_price"] / tr["entry_price"] - 1
    tr["net_ret"] = tr["gross_ret"] - 2 * cost_bps / 1e4
    tr["status"] = np.where(tr["gross_ret"].isna(), "open", "closed")
    tr["position_inr"] = slot
    tr["pnl_inr"] = slot * tr["net_ret"]
    tr = tr.sort_values("t").reset_index(drop=True)

    closed = tr[tr["status"] == "closed"].sort_values("target_end")
    if closed.empty:
        summary = dict(empty_summary, open=int((tr["status"] == "open").sum()))
        return tr, summary, pd.Series(dtype=float)

    equity = capital + closed["pnl_inr"].cumsum()
    equity.index = pd.DatetimeIndex(closed["target_end"]).tz_convert("Asia/Kolkata")
    peak = equity.cummax().clip(lower=capital)
    summary = {
        "closed": len(closed), "open": int((tr["status"] == "open").sum()),
        "total_pnl": float(closed["pnl_inr"].sum()),
        "return_pct": float(closed["pnl_inr"].sum() / capital),
        "win_rate": float((closed["gross_ret"] > 0).mean()),
        "avg_net_bps": float(closed["net_ret"].mean() * 1e4),
        "max_drawdown": float((equity / peak - 1).min()),
    }
    return tr, summary, equity
