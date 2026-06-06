"""
fmp.py — Financial Modeling Prep API client
============================================
Fetches structured fundamental data for a given NSE ticker and returns
a formatted context string ready to be injected into the Gemini prompt.

Endpoints used:
  - /earnings-surprises      → actual vs estimated EPS, beat/miss %
  - /income-statement        → quarterly revenue, net income, margins
  - /key-metrics             → ROE, ROIC, debt/equity, FCF yield
  - /analyst-estimates       → consensus EPS/revenue, revision direction
  - /earning_call_transcript → latest transcript text (management quotes)
  - /financial-growth        → revenue growth, EPS growth, FCF growth
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

_FMP_BASE    = "https://financialmodelingprep.com/api/v3"
_FMP_BASE_V4 = "https://financialmodelingprep.com/api/v4"
_FMP_KEY     = os.getenv("FMP_API_KEY", "").strip()
_TIMEOUT     = 10   # seconds per request


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None) -> Any:
    """GET wrapper (v3) — returns parsed JSON or empty list/dict on failure."""
    if not _FMP_KEY:
        return []
    p = {"apikey": _FMP_KEY}
    if params:
        p.update(params)
    try:
        r = requests.get(f"{_FMP_BASE}/{endpoint}", params=p, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _get_v4(endpoint: str, params: dict | None = None) -> Any:
    """GET wrapper (v4) — returns parsed JSON or empty list/dict on failure."""
    if not _FMP_KEY:
        return []
    p = {"apikey": _FMP_KEY}
    if params:
        p.update(params)
    try:
        r = requests.get(f"{_FMP_BASE_V4}/{endpoint}", params=p, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def _clean(symbol: str) -> str:
    """
    FMP accepts NSE tickers as-is (e.g. RELIANCE.NS) for most endpoints.
    For transcript / earnings endpoints the plain symbol sometimes works better.
    We try with .NS first; callers can strip it if needed.
    """
    return symbol.upper()


def _fmt(v, prefix="", suffix="", decimals=2, fallback="N/A") -> str:
    try:
        f = float(v)
        return f"{prefix}{f:,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return fallback


def _pct_chg(new, old) -> str:
    try:
        chg = (float(new) - float(old)) / abs(float(old)) * 100
        return f"{chg:+.1f}%"
    except Exception:
        return "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Individual fetchers
# ──────────────────────────────────────────────────────────────────────────────

def get_earnings_surprises(symbol: str, limit: int = 6) -> list[dict]:
    data = _get(f"earnings-surprises/{_clean(symbol)}")
    return data[:limit] if isinstance(data, list) else []


def get_income_statement(symbol: str, limit: int = 5) -> list[dict]:
    data = _get(f"income-statement/{_clean(symbol)}",
                params={"period": "quarter", "limit": limit})
    return data if isinstance(data, list) else []


def get_key_metrics(symbol: str, limit: int = 4) -> list[dict]:
    data = _get(f"key-metrics/{_clean(symbol)}",
                params={"period": "quarter", "limit": limit})
    return data if isinstance(data, list) else []


def get_analyst_estimates(symbol: str, limit: int = 4) -> list[dict]:
    data = _get(f"analyst-estimates/{_clean(symbol)}",
                params={"period": "quarter", "limit": limit})
    return data if isinstance(data, list) else []


def get_financial_growth(symbol: str, limit: int = 4) -> list[dict]:
    data = _get(f"financial-growth/{_clean(symbol)}",
                params={"period": "quarter", "limit": limit})
    return data if isinstance(data, list) else []


def get_institutional_ownership(symbol: str, limit: int = 4) -> list[dict]:
    sym  = _clean(symbol).replace(".NS", "").replace(".BO", "")
    data = _get_v4(f"institutional-ownership/symbol-ownership",
                   params={"symbol": sym, "limit": limit, "includeCurrentQuarter": "true"})
    return data if isinstance(data, list) else []


def get_insider_transactions(symbol: str, limit: int = 10) -> list[dict]:
    sym  = _clean(symbol).replace(".NS", "").replace(".BO", "")
    data = _get_v4(f"insider-trading", params={"symbol": sym, "limit": limit})
    return data if isinstance(data, list) else []


def get_price_targets(symbol: str, limit: int = 5) -> list[dict]:
    sym  = _clean(symbol).replace(".NS", "").replace(".BO", "")
    data = _get_v4(f"price-target", params={"symbol": sym})
    return data[:limit] if isinstance(data, list) else []


def get_latest_transcript(symbol: str) -> str:
    """
    Returns the most recent earnings call transcript (first 2,000 chars).
    FMP transcript list endpoint → pick latest quarter/year → fetch text.
    """
    # Step 1: get list of available transcripts
    sym   = _clean(symbol).replace(".NS", "").replace(".BO", "")
    lst   = _get(f"earning_call_transcript/{sym}")
    if not isinstance(lst, list) or not lst:
        # Try with full symbol
        lst = _get(f"earning_call_transcript/{_clean(symbol)}")
    if not isinstance(lst, list) or not lst:
        return ""

    # lst items: {"symbol":…, "quarter":…, "year":…, "date":…}
    latest = sorted(lst, key=lambda x: (x.get("year", 0),
                                         x.get("quarter", 0)), reverse=True)[0]
    from datetime import datetime as _dt
    q, y   = latest.get("quarter", 1), latest.get("year", _dt.now().year - 1)

    # Step 2: fetch actual transcript
    transcript = _get(f"earning_call_transcript/{sym}",
                      params={"quarter": q, "year": y})
    if not isinstance(transcript, list) or not transcript:
        transcript = _get(f"earning_call_transcript/{_clean(symbol)}",
                          params={"quarter": q, "year": y})

    if isinstance(transcript, list) and transcript:
        text = transcript[0].get("content", "")
        # Return first 2,000 chars — enough for key management quotes
        return text[:2000].strip()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Context builder — formats all data into a Gemini-ready string
# ──────────────────────────────────────────────────────────────────────────────

def build_fmp_context(ticker: str) -> str:
    """
    Fetch all FMP data for a ticker and return a structured context block
    ready to be injected into the Gemini analyst prompt.
    Returns empty string if FMP key is missing or all calls fail.
    """
    if not _FMP_KEY:
        return ""

    lines: list[str] = [
        "=" * 60,
        "VERIFIED FUNDAMENTAL DATA  (Source: Financial Modeling Prep)",
        "=" * 60,
    ]

    # ── 1. Earnings surprises ────────────────────────────────────────────────
    surprises = get_earnings_surprises(ticker)
    if surprises:
        lines.append("\n[EARNINGS SURPRISES — Last 6 Quarters]")
        for s in surprises:
            date     = s.get("date", "")
            actual   = s.get("actualEarningResult", None)
            estimate = s.get("estimatedEarning", None)
            try:
                beat = ((float(actual) - float(estimate)) /
                        abs(float(estimate)) * 100)
                flag = "BEAT" if beat > 0 else "MISS"
                lines.append(
                    f"  {date}: Actual EPS {_fmt(actual, 'Rs.')}  |  "
                    f"Estimate {_fmt(estimate, 'Rs.')}  |  "
                    f"{flag} {beat:+.1f}%"
                )
            except Exception:
                lines.append(f"  {date}: Actual {actual}  Estimate {estimate}")

    # ── 2. Quarterly income statement ────────────────────────────────────────
    income = get_income_statement(ticker)
    if income:
        lines.append("\n[QUARTERLY INCOME STATEMENT]")
        lines.append(
            f"  {'Quarter':<14} {'Revenue':>14} {'Rev YoY':>10} "
            f"{'Net Income':>14} {'Op Margin':>12} {'Net Margin':>12}"
        )
        for i, q in enumerate(income):
            date    = q.get("date", "")[:7]
            rev     = q.get("revenue", None)
            ni      = q.get("netIncome", None)
            rev_yoy = _pct_chg(rev, income[i+1].get("revenue")) \
                      if i + 1 < len(income) else "N/A"
            op_mar  = q.get("operatingIncomeRatio", None)
            net_mar = q.get("netIncomeRatio", None)
            lines.append(
                f"  {date:<14} {_fmt(rev, 'Rs.', '', 0):>14} {rev_yoy:>10} "
                f"{_fmt(ni, 'Rs.', '', 0):>14} "
                f"{_fmt(op_mar, '', '%', 1):>12} "
                f"{_fmt(net_mar, '', '%', 1):>12}"
            )

    # ── 3. Key metrics ────────────────────────────────────────────────────────
    metrics = get_key_metrics(ticker)
    if metrics and metrics[0]:
        m = metrics[0]   # most recent quarter
        lines.append("\n[KEY METRICS — Most Recent Quarter]")
        kv = [
            ("ROE",              _fmt(m.get("roe"),                "", "%", 1)),
            ("ROIC",             _fmt(m.get("roic"),               "", "%", 1)),
            ("Debt / Equity",    _fmt(m.get("debtToEquity"),        "", "x", 2)),
            ("Current Ratio",    _fmt(m.get("currentRatio"),        "", "x", 2)),
            ("FCF Yield",        _fmt(m.get("freeCashFlowYield"),   "", "%", 1)),
            ("EV / EBITDA",      _fmt(m.get("enterpriseValueOverEBITDA"), "", "x", 1)),
            ("P/B Ratio",        _fmt(m.get("pbRatio"),             "", "x", 2)),
            ("Dividend Yield",   _fmt(m.get("dividendYield"),       "", "%", 2)),
        ]
        for label, val in kv:
            lines.append(f"  {label:<22}: {val}")

    # ── 4. Analyst estimates ─────────────────────────────────────────────────
    estimates = get_analyst_estimates(ticker)
    if estimates:
        lines.append("\n[ANALYST CONSENSUS ESTIMATES]")
        for e in estimates[:3]:
            date       = e.get("date", "")[:7]
            rev_est    = e.get("estimatedRevenueAvg", None)
            eps_est    = e.get("estimatedEpsAvg", None)
            rev_high   = e.get("estimatedRevenueHigh", None)
            rev_low    = e.get("estimatedRevenueLow", None)
            num_analyst= e.get("numberAnalystEstimatedRevenue", "?")
            lines.append(
                f"  {date}: EPS est {_fmt(eps_est, 'Rs.')}  |  "
                f"Revenue est {_fmt(rev_est, 'Rs.', '', 0)}  "
                f"(range {_fmt(rev_low, 'Rs.', '', 0)}–{_fmt(rev_high, 'Rs.', '', 0)})  "
                f"| {num_analyst} analysts"
            )

    # ── 5. Growth rates ──────────────────────────────────────────────────────
    growth = get_financial_growth(ticker)
    if growth and growth[0]:
        g = growth[0]
        lines.append("\n[GROWTH RATES — Most Recent Quarter YoY]")
        gv = [
            ("Revenue Growth",       _fmt(g.get("revenueGrowth"),         "", "%", 1)),
            ("Net Income Growth",    _fmt(g.get("netIncomeGrowth"),        "", "%", 1)),
            ("EPS Growth",           _fmt(g.get("epsgrowth"),              "", "%", 1)),
            ("FCF Growth",           _fmt(g.get("freeCashFlowGrowth"),     "", "%", 1)),
            ("Operating Income Gr.", _fmt(g.get("operatingIncomeGrowth"),  "", "%", 1)),
            ("Gross Profit Growth",  _fmt(g.get("grossProfitGrowth"),      "", "%", 1)),
        ]
        for label, val in gv:
            lines.append(f"  {label:<26}: {val}")

    # ── 6. Earnings call transcript excerpt ──────────────────────────────────
    transcript = get_latest_transcript(ticker)
    if transcript:
        lines.append("\n[LATEST EARNINGS CALL TRANSCRIPT — Excerpt]")
        for para in transcript.split("\n"):
            wrapped = textwrap.fill(para.strip(), width=90)
            if wrapped:
                lines.append(f"  {wrapped}")

    # ── 7. Institutional ownership ───────────────────────────────────────────
    inst = get_institutional_ownership(ticker)
    if inst:
        lines.append("\n[INSTITUTIONAL OWNERSHIP — Latest Quarter]")
        for row in inst[:5]:
            investor  = row.get("investorName", "?")
            shares    = row.get("sharesNumber", "?")
            chg       = row.get("change", None)
            chg_str   = f"  (chg: {int(chg):+,})" if chg is not None else ""
            lines.append(f"  {investor:<35}: {shares:,} shares{chg_str}" if isinstance(shares, int)
                         else f"  {investor:<35}: {shares}{chg_str}")

    # ── 8. Insider transactions ──────────────────────────────────────────────
    insider = get_insider_transactions(ticker)
    if insider:
        lines.append("\n[INSIDER TRANSACTIONS — Recent]")
        for tx in insider[:6]:
            name       = tx.get("reportingName") or tx.get("name") or "?"
            trans_type = tx.get("transactionType") or tx.get("acquistionOrDisposition") or "?"
            shares     = tx.get("securitiesTransacted") or tx.get("shares") or "?"
            price      = tx.get("price") or "?"
            date       = tx.get("transactionDate") or tx.get("date") or ""
            lines.append(f"  {date:<12} | {name[:25]:<25} | {trans_type:<10} | "
                         f"Qty: {shares}  @ ${price}")

    # ── 9. Analyst price targets ─────────────────────────────────────────────
    targets = get_price_targets(ticker)
    if targets:
        lines.append("\n[ANALYST PRICE TARGETS]")
        for t in targets[:4]:
            analyst    = t.get("analystName") or t.get("analyst") or "?"
            company_nm = t.get("analystCompany") or ""
            target     = t.get("priceTarget") or "?"
            prev       = t.get("priceWhenPosted") or t.get("stockPriceWhenPosted") or "?"
            date       = t.get("publishedDate", "")[:10]
            lines.append(f"  {date:<12} | {analyst[:20]:<20} ({company_nm[:15]}) | "
                         f"Target: ${target}  (was ${prev})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
