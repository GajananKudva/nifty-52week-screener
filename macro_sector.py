"""
macro_sector.py — Macro + sector backdrop for the AI analysis prompts.
=====================================================================
Builds ONE compact context block combining:
  • Global macro  — World Bank (India/US/World GDP & inflation) + FRED
                    (Brent, DXY, US 10Y, VIX, gold, yield curve via fred.py).
  • Sector        — FMP sector-performance tape.
  • Global news   — Finnhub general market headlines (world risk-on/off tone).

Design rules:
  • Every network call is wrapped in try/except with a short timeout and
    returns "" on failure — a dead API can NEVER break the analysis.
  • No keys are hard-coded. All read from the environment at call time:
        FRED_API_KEY, FMP_API_KEY, FINNHUB_API_KEY
    (locally via .env; on Streamlit Cloud via Secrets, which are exposed as
    env vars). Nothing here is ever committed.

The caller (app.py) wraps macro_sector_context() in st.cache_data so this only
hits the network once every few hours, not per stock.
"""
from __future__ import annotations

import os
from datetime import date

import requests

_TIMEOUT = 10


def _env(key: str) -> str:
    return os.getenv(key, "").strip()


# ── World Bank (free, no key) ─────────────────────────────────────────────────
_WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "Real GDP growth %",
    "FP.CPI.TOTL.ZG":    "Inflation (CPI) %",
}
_WB_COUNTRIES = {"IND": "India", "USA": "US", "WLD": "World"}


def world_bank_block() -> str:
    lines = []
    try:
        for code, ind_label in _WB_INDICATORS.items():
            row = []
            for iso, cty in _WB_COUNTRIES.items():
                try:
                    r = requests.get(
                        f"https://api.worldbank.org/v2/country/{iso}/indicator/{code}",
                        params={"format": "json", "per_page": 5, "mrnev": 1},
                        timeout=_TIMEOUT,
                    )
                    data = r.json()
                    if isinstance(data, list) and len(data) > 1 and data[1]:
                        obs = data[1][0]
                        val, yr = obs.get("value"), obs.get("date")
                        if val is not None:
                            row.append(f"{cty} {float(val):.1f}% ({yr})")
                except Exception:
                    continue
            if row:
                lines.append(f"  {ind_label}: " + " · ".join(row))
    except Exception:
        return ""
    return "World Bank (latest annual):\n" + "\n".join(lines) if lines else ""


# ── FRED global macro (reuse the existing builder) ────────────────────────────
def fred_block() -> str:
    try:
        import fred
        return (fred.build_fred_context() or "").strip()
    except Exception:
        return ""


# ── FMP sector-performance tape ───────────────────────────────────────────────
def sector_block(sector: str) -> str:
    key = _env("FMP_API_KEY")
    if not key:
        return ""
    try:
        r = requests.get(
            "https://financialmodelingprep.com/api/v3/sectors-performance",
            params={"apikey": key}, timeout=_TIMEOUT,
        )
        data = r.json()
        if not isinstance(data, list) or not data:
            return ""
        rows, mine = [], None
        for d in data:
            sec = str(d.get("sector", "")).strip()
            chg = str(d.get("changesPercentage", "")).strip()
            if not sec:
                continue
            rows.append((sec, chg))
            if sector and sec.lower() == sector.lower():
                mine = (sec, chg)
        if not rows:
            return ""
        out = ["Sector tape (FMP, 1-day):"]
        if mine:
            out.append(f"  > {mine[0]}: {mine[1]}  (this stock's sector)")
        out.append("  Across sectors: " + " · ".join(f"{s} {c}" for s, c in rows[:8]))
        return "\n".join(out)
    except Exception:
        return ""


# ── Finnhub global market news (free tier) ────────────────────────────────────
def finnhub_news_block(max_items: int = 4) -> str:
    key = _env("FINNHUB_API_KEY")
    if not key:
        return ""
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": key},
            timeout=_TIMEOUT,
        )
        data = r.json()
        if not isinstance(data, list) or not data:
            return ""
        heads = []
        for it in data[:max_items]:
            h = str(it.get("headline", "")).strip()
            src = str(it.get("source", "")).strip()
            if h:
                heads.append(f"  - {h}" + (f" [{src}]" if src else ""))
        return "Global market headlines (Finnhub):\n" + "\n".join(heads) if heads else ""
    except Exception:
        return ""


def global_macro_block() -> str:
    """Stock-agnostic global macro + world news."""
    parts = [world_bank_block(), fred_block(), finnhub_news_block()]
    return "\n".join(p for p in parts if p)


def macro_sector_context(sector: str = "") -> str:
    """Full backdrop = global macro + this stock's sector tape. '' if nothing."""
    parts = [global_macro_block(), sector_block(sector or "")]
    body = "\n".join(p for p in parts if p)
    if not body:
        return ""
    return f"[MACRO & SECTOR BACKDROP - as of {date.today():%d %b %Y}]\n{body}"


if __name__ == "__main__":
    print(macro_sector_context("Consumer Cyclical") or "(no backdrop — check API keys/network)")
