"""
india_macro.py — India-specific macro & institutional flow data
===============================================================
Fetches three categories of data without paid APIs:

1. Market rates       — USD/INR, India 10Y bond yield (via yfinance)
2. RBI policy rates   — repo rate, SDF, MSF from RBI DBIE website
3. FII / DII flows    — daily net buy/sell from NSE (session-based fetch)

Returns a formatted context string for the Gemini prompt, plus helper
functions for dashboard display.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import requests
import yfinance as yf

_TIMEOUT    = 10
_NSE_HDRS   = {
    "user-agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
    "accept":          "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "referer":         "https://www.nseindia.com/",
    "x-requested-with": "XMLHttpRequest",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Market rates via yfinance
# ──────────────────────────────────────────────────────────────────────────────

def get_market_rates() -> dict:
    """
    Fetch USD/INR spot rate and India 10Y bond yield via yfinance.
    Returns dict with keys: usdinr, usdinr_chg, india_10y, india_10y_chg.
    """
    result = {}

    # USD/INR
    try:
        tk  = yf.Ticker("USDINR=X")
        inf = tk.fast_info
        result["usdinr"]     = getattr(inf, "last_price", None)
        result["usdinr_prev"]= getattr(inf, "previous_close", None)
        if result["usdinr"] and result["usdinr_prev"]:
            result["usdinr_chg"] = (
                (result["usdinr"] - result["usdinr_prev"])
                / result["usdinr_prev"] * 100
            )
    except Exception:
        pass

    # India 10Y Government Bond Yield
    try:
        tk2 = yf.Ticker("^IN10YT=RR")
        inf2 = tk2.fast_info
        result["india_10y"]      = getattr(inf2, "last_price", None)
        result["india_10y_prev"] = getattr(inf2, "previous_close", None)
        if result["india_10y"] and result["india_10y_prev"]:
            result["india_10y_chg"] = (
                result["india_10y"] - result["india_10y_prev"]
            )   # in bps terms, keep as absolute change
    except Exception:
        pass

    # Nifty 50 as proxy for market level context
    try:
        nk = yf.Ticker("^NSEI")
        ni = nk.fast_info
        result["nifty"]     = getattr(ni, "last_price", None)
        result["nifty_prev"]= getattr(ni, "previous_close", None)
        if result["nifty"] and result["nifty_prev"]:
            result["nifty_chg"] = (
                (result["nifty"] - result["nifty_prev"])
                / result["nifty_prev"] * 100
            )
    except Exception:
        pass

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. RBI policy rates
# ──────────────────────────────────────────────────────────────────────────────

# Known current rates as a reliable fallback (update manually after each MPC)
_RBI_FALLBACK = {
    "repo_rate":     6.25,
    "sdf_rate":      6.00,
    "msf_rate":      6.50,
    "crr":           4.00,
    "slr":           18.00,
    "last_updated":  "April 2025",
    "stance":        "Neutral",
}


def get_rbi_rates() -> dict:
    """
    Attempt to fetch current RBI policy rates from the DBIE portal.
    Falls back to hardcoded known values on failure.
    """
    try:
        # RBI DBIE key statistics page — policy rates table
        url = "https://data.rbi.org.in/DBIE/dbie.rbi?site=home"
        r   = requests.get(url, timeout=_TIMEOUT,
                            headers={"user-agent": "Mozilla/5.0"})
        html = r.text

        # Extract repo rate — look for pattern like "6.25" near "Repo Rate"
        # RBI DBIE embeds rates in a JSON blob or table
        match = re.search(
            r"[Rr]epo\s+[Rr]ate[^0-9]{1,30}(\d+\.\d+)", html)
        if match:
            repo = float(match.group(1))
            if 3.0 <= repo <= 12.0:   # sanity check
                return {**_RBI_FALLBACK, "repo_rate": repo, "source": "live"}
    except Exception:
        pass

    return {**_RBI_FALLBACK, "source": "fallback"}


# ──────────────────────────────────────────────────────────────────────────────
# 3. FII / DII daily flows from NSE
# ──────────────────────────────────────────────────────────────────────────────

def get_fii_dii_flows() -> Optional[dict]:
    """
    Fetch today's FII and DII net activity from NSE using a cookie session.
    Returns dict with keys: date, fii_net, dii_net, fii_buy, fii_sell,
                            dii_buy, dii_sell  (all in Rs. crore).
    Returns None if the fetch fails.
    """
    try:
        session = requests.Session()
        # Warm up — get cookies from the homepage first
        session.get("https://www.nseindia.com",
                    headers=_NSE_HDRS, timeout=_TIMEOUT)

        r = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=_NSE_HDRS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()

        # Response is a list; find the "Cash Market" row
        # Structure: [{"category":"FII/FPI","buyValue":…,"sellValue":…,"netValue":…}, …]
        rows = data if isinstance(data, list) else data.get("data", [])

        fii_row = next((x for x in rows
                        if "FII" in str(x.get("category", "")).upper()
                        or "FPI" in str(x.get("category", "")).upper()), None)
        dii_row = next((x for x in rows
                        if "DII" in str(x.get("category", "")).upper()), None)

        def _crore(v):
            try:   return float(str(v).replace(",", ""))
            except: return None

        result = {"date": datetime.today().strftime("%d %b %Y"), "source": "live"}

        if fii_row:
            result["fii_buy"]  = _crore(fii_row.get("buyValue"))
            result["fii_sell"] = _crore(fii_row.get("sellValue"))
            result["fii_net"]  = _crore(fii_row.get("netValue"))

        if dii_row:
            result["dii_buy"]  = _crore(dii_row.get("buyValue"))
            result["dii_sell"] = _crore(dii_row.get("sellValue"))
            result["dii_net"]  = _crore(dii_row.get("netValue"))

        return result if (fii_row or dii_row) else None

    except Exception:
        return None


def get_fii_dii_monthly() -> Optional[list[dict]]:
    """
    Fetch last 30 days of FII/DII data from NSE archives.
    Falls back gracefully if unavailable.
    """
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com",
                    headers=_NSE_HDRS, timeout=_TIMEOUT)
        r = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact?type=historical",
            headers=_NSE_HDRS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()
        return raw if isinstance(raw, list) else raw.get("data", [])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v, prefix="", suffix="", dec=2) -> str:
    try:
        return f"{prefix}{float(v):,.{dec}f}{suffix}"
    except Exception:
        return "N/A"


def _signed(v, suffix="") -> str:
    try:
        f = float(v)
        sign = "+" if f >= 0 else ""
        return f"{sign}{f:,.2f}{suffix}"
    except Exception:
        return "N/A"


def build_india_macro_context() -> str:
    """
    Fetch all India macro data and return a formatted context block
    ready for injection into the Gemini prompt.
    """
    lines: list[str] = [
        "=" * 60,
        "INDIA MACRO DATA  (Sources: NSE, RBI, yfinance)",
        f"As of {datetime.today().strftime('%d %B %Y')}",
        "=" * 60,
    ]

    any_data = False

    # ── Market rates ──────────────────────────────────────────────────────────
    rates = get_market_rates()
    if rates:
        any_data = True
        lines.append("\n[MARKET RATES]")

        usdinr = rates.get("usdinr")
        if usdinr:
            chg = rates.get("usdinr_chg", 0) or 0
            dir_ = "weaker" if chg > 0 else "stronger"
            lines.append(
                f"  USD/INR           : {_fmt(usdinr, 'Rs.')}  "
                f"({_signed(chg, '%')} — rupee {dir_} vs USD)"
            )
            lines.append(
                f"  India impact      : {'Weaker rupee raises import costs, pressures CAD, may deter FII inflows' if chg > 0 else 'Stronger rupee reduces import costs, supports FII inflows'}"
            )

        india_10y = rates.get("india_10y")
        if india_10y:
            chg_bps = (rates.get("india_10y_chg") or 0) * 100
            lines.append(
                f"  India 10Y Yield   : {_fmt(india_10y, '', '%')}  "
                f"({_signed(chg_bps, ' bps') if chg_bps else ''})"
            )
            lines.append(
                f"  India impact      : {'Rising yields tighten financial conditions, pressure rate-sensitive sectors (banks, RE, NBFCs)' if chg_bps > 0 else 'Falling yields ease financial conditions, positive for rate-sensitives'}"
            )

        nifty = rates.get("nifty")
        if nifty:
            chg = rates.get("nifty_chg", 0) or 0
            lines.append(
                f"  Nifty 50          : {_fmt(nifty)}  ({_signed(chg, '%')} today)"
            )

    # ── RBI policy ────────────────────────────────────────────────────────────
    rbi = get_rbi_rates()
    lines.append("\n[RBI MONETARY POLICY]")
    any_data = True
    src_note = " (live)" if rbi.get("source") == "live" else " (last known)"
    lines.append(f"  Repo Rate         : {_fmt(rbi.get('repo_rate'), '', '%')}{src_note}")
    lines.append(f"  SDF (floor)       : {_fmt(rbi.get('sdf_rate'), '', '%')}")
    lines.append(f"  MSF / Bank Rate   : {_fmt(rbi.get('msf_rate'), '', '%')}")
    lines.append(f"  CRR               : {_fmt(rbi.get('crr'), '', '%')}")
    lines.append(f"  Stance            : {rbi.get('stance', 'N/A')}")
    lines.append(f"  Last updated      : {rbi.get('last_updated', 'N/A')}")
    lines.append(
        f"  India impact      : Repo at {_fmt(rbi.get('repo_rate'), '', '%')} — "
        f"{'accommodative relative to historic norms; supports credit growth' if (rbi.get('repo_rate') or 99) < 6.5 else 'restrictive; moderating credit growth and inflation'}"
    )

    # ── FII / DII flows ───────────────────────────────────────────────────────
    flows = get_fii_dii_flows()
    lines.append("\n[FII / DII DAILY FLOWS — Cash Market]")
    if flows:
        any_data = True
        src = " (live)" if flows.get("source") == "live" else " (delayed)"
        lines.append(f"  Date              : {flows.get('date', 'Today')}{src}")

        fii_net = flows.get("fii_net")
        if fii_net is not None:
            direction = "NET BUYER" if fii_net >= 0 else "NET SELLER"
            lines.append(
                f"  FII Net           : {_signed(fii_net, ' Cr')}  [{direction}]"
            )
            lines.append(
                f"    Buy / Sell      : {_fmt(flows.get('fii_buy'), 'Rs.', ' Cr')} / "
                f"{_fmt(flows.get('fii_sell'), 'Rs.', ' Cr')}"
            )

        dii_net = flows.get("dii_net")
        if dii_net is not None:
            direction = "NET BUYER" if dii_net >= 0 else "NET SELLER"
            lines.append(
                f"  DII Net           : {_signed(dii_net, ' Cr')}  [{direction}]"
            )
            lines.append(
                f"    Buy / Sell      : {_fmt(flows.get('dii_buy'), 'Rs.', ' Cr')} / "
                f"{_fmt(flows.get('dii_sell'), 'Rs.', ' Cr')}"
            )

        # Interpretation
        fii_pos = fii_net is not None and fii_net > 0
        dii_pos = dii_net is not None and dii_net > 0
        if fii_pos and dii_pos:
            interp = "Both FIIs and DIIs net buyers — strong institutional conviction, broad-based accumulation"
        elif fii_pos and not dii_pos:
            interp = "FIIs buying, DIIs selling — foreign conviction; domestic profit-taking"
        elif not fii_pos and dii_pos:
            interp = "FIIs selling, DIIs buying — domestic support absorbing foreign selling (SIP flows)"
        else:
            interp = "Both FIIs and DIIs net sellers — broad-based institutional selling, risk-off"
        lines.append(f"  Reading           : {interp}")
    else:
        lines.append("  Data unavailable — NSE session could not be established")
        lines.append("  (Check NSE website directly for FII/DII flows)")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines) if any_data else ""


def get_dashboard_snapshot() -> dict:
    """
    Returns a combined dict of market rates + RBI + FII/DII for dashboard tiles.
    """
    return {
        "rates": get_market_rates(),
        "rbi":   get_rbi_rates(),
        "flows": get_fii_dii_flows(),
    }
