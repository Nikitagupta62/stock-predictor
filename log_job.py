"""
log_job.py

One headless pass of live tracking, meant to be run on a schedule (GitHub Actions)
but fine to run by hand:

    python log_job.py

It fetches the latest hourly bars, scores earlier predictions whose hour has now
closed, then logs fresh predictions. Everything goes to data/live_log.csv, which
the dashboard reads and merges with its own local log.

Exit code 1 means no data could be downloaded (the job shows as failed in
GitHub so you notice). Running outside market hours is harmless: nothing new is
logged because stale predictions are rejected.
"""

import sys

from live import (SHARED_LOG_PATH, fetch_recent, load_bundle, log_predictions, market_status,
                  predict_latest, update_outcomes)
from universe import get_universe


def main() -> int:
    bundle = load_bundle()
    tickers = bundle.get("tickers") or get_universe(50)
    data = fetch_recent(tickers)
    if not data:
        print("No data downloaded (Yahoo unavailable or rate-limiting).")
        return 1

    scored = update_outcomes(data, SHARED_LOG_PATH)
    preds = predict_latest(bundle, data)
    added = log_predictions(preds, SHARED_LOG_PATH)

    latest = preds["made_at"].max() if len(preds) else None
    _, status = market_status()
    print(f"{status} | stocks with data: {len(data)} | latest bar closed: {latest} "
          f"| new predictions logged: {added} | outcomes scored: {scored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
