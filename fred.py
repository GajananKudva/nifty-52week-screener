"""
fred.py — FRED (Federal Reserve Economic Data) macro context
=============================================================
Fetches global macro indicators that directly influence Indian markets
and returns a formatted string ready to inject into the Gemini prompt.

Free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
Add to .env:  FRED_API_KEY=your_key_here

Series fetched:
  DCOILBRENTEU      — Brent Crude Oil (USD/barrel)
  DTWEXBGS          — US Dollar Index (DXY Broad)
  DFF               — Fed Funds Effective Rate (%)
  VIXCLS            — CBOE VIX (market fear gauge)
  GS10              — US 10-Year Treasury Yield (%)
  GOLDAMGBD228NLBM  — Gold Price (USD/troy oz)
  T10Y2Y            — US Yield Curve Spread (10Y–2Y, recession signal)
  CPIAUCSL          — US CPI (MoM, proxy for inflation)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_KEY  = os.getenv("FRED_API_KEY", "").strip()
_TIMEOUT   = 8


# ──────────────────────────────────────────────────────────────────────────────
# Series definitions
# ──────────────────────────────────────────────────────────────────────────────

_SERIES: dict[str, dict] = {
    "DCOILBRENTEU":     {"label": "Brent Crude Oil",        "unit": "USD/bbl",  "freq": "daily"},
    "DTWEXBGS":         {"label": "US Dollar Index (DXY)",  "unit": "",         "freq": "daily"},
    "DFF":              {"label": "Fed Funds Rate",          "unit": "%",        "freq": "daily"},
    "VIXCLS":           {"label": "CBOE VIX",               "unit": "",         "freq": "daily"},
    "GS10":             {"label": "US 10Y Treasury Yield",  "unit": "%",        "freq": "monthly"},
    "GOLDAMGBD228NLBM": {"label": "Gold Price",             "unit": "USD/oz",   "freq": "daily"},
    "T10Y2Y":           {"label": "Yield Curve (10Y-2Y)",   "unit": "%",        "freq": "daily"},
}

# How each indicator affects Indian markets
_INDIA_IMPACT: dict[str, str] = {
    "DCOILBRENTEU":     "Higher crude → India CAD widens, rupee weakens, oil companies pressured; lower crude → opposite",
    "DTWEXBGS":         "Stronger DXY → FII outflows from India, rupee weakens; weaker DXY → FII inflows rise",
    "DFF":              "Higher Fed rate → capital flows to USD, EM selling pressure; cuts → EM relief rally",
    "VIXCLS":           "High VIX → risk-off, FII sell EMs including India; low VIX → risk-on, inflows",
    "GS10":             "Rising US 10Y → India bonds less attractive, FII debt outflows; falling → inflows",
    "GOLDAMGBD228NLBM": "Rising gold → risk-off signal; affects jewellery demand (Titan, Kalyan), import costs",
    "T10Y2Y":           "Negative spread = inverted curve = US recession risk → global risk-off, India hit",
}


# ──────────────────────────────────────────────────────────────────────────────
# Fetcher
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_series(series_id: str, lookback_days: int = 30) -> Optional[dict]:
    """
    Fetch the most recent observation for a FRED series.
    Returns dict with {value, date, prev_value, prev_date, change_pct} or None.
    """
    if not _FRED_KEY:
        return None

    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            _FRED_BASE,
            params={
                "series_id":   series_id,
                "api_key":     _FRED_KEY,
                "file_type":   "json",
                "sort_order":  "desc",
                "limit":       5,
                "observation_start": start,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        obs = [o for o in r.json().get("observations", [])
               if o.get("value", ".") != "."]
        if not obs:
            return None

        def to_float(v):
            try:   return float(v)
            except: return None

        latest  = obs[0]
        prev    = obs[1] if len(obs) > 1 else None
        val     = to_float(latest["value"])
        prev_v  = to_float(prev["value"]) if prev else None

        chg_pct = None
        if val is not None and prev_v is not None and prev_v != 0:
            chg_pct = (val - prev_v) / abs(prev_v) * 100

        return {
            "value":      val,
            "date":       latest["date"],
            "prev_value": prev_v,
            "prev_date":  prev["date"] if prev else None,
            "change_pct": chg_pct,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def build_fred_context() -> str:
    """
    Fetch all FRED series and return a formatted macro context block
    for injection into the Gemini prompt.
    """
    if not _FRED_KEY:
        return ""

    lines: list[str] = [
        "=" * 60,
        "GLOBAL MACRO INDICATORS  (Source: FRED — Federal Reserve)",
        f"As of {datetime.today().strftime('%d %B %Y')}",
        "=" * 60,
    ]

    any_data = False
    for series_id, meta in _SERIES.items():
        data = _fetch_series(series_id)
        if not data or data["value"] is None:
            continue

        any_data  = True
        val       = data["value"]
        date      = data["date"]
        chg       = data["change_pct"]
        unit      = meta["unit"]
        label     = meta["label"]
        impact    = _INDIA_IMPACT.get(series_id, "")

        val_str   = f"{val:,.2f}{(' ' + unit) if unit else ''}"
        chg_str   = f"  ({chg:+.2f}% vs prior)" if chg is not None else ""
        lines.append(f"\n{label} [{date}]")
        lines.append(f"  Current: {val_str}{chg_str}")
        lines.append(f"  India impact: {impact}")

    if not any_data:
        return ""

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def get_series_snapshot() -> dict[str, dict]:
    """
    Returns a dict of all series with their latest values.
    Used by the dashboard to display macro tiles.
    """
    result = {}
    for series_id, meta in _SERIES.items():
        data = _fetch_series(series_id)
        if data:
            result[series_id] = {**meta, **data}
    return result
