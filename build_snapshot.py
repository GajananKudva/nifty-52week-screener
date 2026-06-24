#!/usr/bin/env python3
"""
build_snapshot.py — Headless 52-week High/Low screener for scheduled runs.
==========================================================================

Runs DataEngine.screen_universe() with the dashboard's DEFAULT parameters and
writes the results to `data/latest_screen.json`.

This is the "24/7" engine: a GitHub Actions cron job runs this script on a
schedule and commits the snapshot. The Streamlit dashboard (app.py) then reads
the snapshot so users see pre-computed results INSTANTLY — no waiting for a live
scan on page load.

Pure screening only — no AI narrative, no email. That means it needs NO API
keys, just yfinance (free).

Run locally:
    python build_snapshot.py
    python build_snapshot.py --limit 10     # quick test on first 10 tickers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine import DataEngine, ScreenerConfig
from mailer import NIFTY_500_TICKERS, fetch_nifty500_live, fetch_all_nse_live

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
SNAPSHOT = DATA_DIR / "latest_screen.json"

IST = timezone(timedelta(hours=5, minutes=30))


def build_universe(limit: int | None = None) -> list[str]:
    """Mirror app.py's default Nifty 500 universe selection."""
    env = os.getenv("TICKER_UNIVERSE", "").strip()
    if env:
        tickers = [t.strip().upper() for t in env.split(",") if t.strip()]
    else:
        # Full NSE main board (~2000); degrade gracefully if NSE blocks a run.
        tickers = (fetch_all_nse_live()
                   or fetch_nifty500_live()
                   or NIFTY_500_TICKERS)
    # Drop placeholder tickers (e.g. DUMMYVEDL*.NS) that have no price data.
    tickers = [t for t in tickers if not t.upper().startswith("DUMMY")]
    if limit:
        tickers = tickers[:limit]
    return tickers


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 52-week screen snapshot.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only screen the first N tickers (for quick local tests).")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    tickers = build_universe(limit=args.limit)
    print(f"Universe: {len(tickers)} tickers")

    # Defaults mirror the dashboard sidebar: strict 52-week touch, 252-day window,
    # volume filter OFF (volume_surge_threshold = 0 => every stock passes vol check).
    cfg = ScreenerConfig(
        tickers                = tickers,
        high_low_window        = int(os.getenv("HIGH_LOW_WINDOW", "252")),
        breakout_threshold     = float(os.getenv("BREAKOUT_THRESHOLD", "0.0")),
        volume_window          = 20,
        volume_surge_threshold = 0.0,
        rate_limit_sleep       = float(os.getenv("RATE_LIMIT_SLEEP", "0.25")),
        rate_limit_batch_size  = int(os.getenv("RATE_LIMIT_BATCH", "50")),
    )

    eng = DataEngine(cfg)
    raw = eng.screen_universe()

    highs = eng.format_for_display(raw.get("highs"))
    lows = eng.format_for_display(raw.get("lows"))
    errors = raw.get("errors", [])

    finished = datetime.now(timezone.utc)
    snapshot = {
        "schema_version": 1,
        "generated_at_utc": finished.isoformat(),
        "generated_at_ist": finished.astimezone(IST).strftime("%d %b %Y · %H:%M IST"),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "universe_size": len(tickers),
        "params": {
            "high_low_window": cfg.high_low_window,
            "breakout_threshold": cfg.breakout_threshold,
            "volume_surge_threshold": cfg.volume_surge_threshold,
        },
        "counts": {"highs": len(highs), "lows": len(lows), "errors": len(errors)},
        "highs": highs,
        "lows": lows,
        "errors": errors,
    }

    DATA_DIR.mkdir(exist_ok=True)
    # Atomic write so a half-written file is never committed/read.
    tmp = SNAPSHOT.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(SNAPSHOT)

    print(f"Wrote {SNAPSHOT}")
    print(f"  highs={len(highs)}  lows={len(lows)}  errors={len(errors)}  "
          f"in {snapshot['duration_seconds']}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
