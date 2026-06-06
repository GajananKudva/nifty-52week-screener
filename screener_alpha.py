"""
screener_alpha.py — Screener.in scraper + yfinance deep fundamentals
=====================================================================
Provides two context-building functions used by app.py's AI analyst:

  build_screener_context(symbol)           → str
      Scrapes screener.in for Indian-specific fundamentals:
      10-year financials, quarterly results, pros/cons, shareholding.

  build_yfinance_context(symbol)           → str
      Pulls 80+ fields from yfinance .info: analyst targets, consensus
      rating, EPS (trailing/forward), growth rates, margins, DCF metrics.
      Free, no API key, works natively with NSE .NS symbols.
"""

from __future__ import annotations

import re
from typing import Optional

import requests

# ── BeautifulSoup (optional — graceful fallback if not installed) ─────────────
try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _get(url: str, timeout: int = 10) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
    except Exception:
        pass
    return None


def _clean(text: str) -> str:
    """Strip whitespace and collapse multiple spaces/newlines."""
    return re.sub(r"\s{2,}", " ", text.strip())


def _trim(text: str, max_chars: int) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Screener.in  (Indian fundamentals: 10-yr financials, pros/cons, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def build_screener_context(symbol: str) -> str:
    """
    Scrape screener.in for the given NSE symbol and return a structured
    text block suitable for injection into an AI prompt.

    Falls back gracefully if the symbol is not found or the page structure
    changes.
    """
    if not _BS4_OK:
        return ""

    # Screener uses the NSE symbol without the .NS suffix
    clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "")
    url = f"https://www.screener.in/company/{clean_sym}/consolidated/"

    resp = _get(url)
    if resp is None:
        # Try standalone (non-consolidated) page
        url = f"https://www.screener.in/company/{clean_sym}/"
        resp = _get(url)

    if resp is None:
        return ""

    try:
        soup = BeautifulSoup(resp.text, "html.parser")  # built-in parser, no lxml needed
    except Exception:
        return ""

    lines: list[str] = [f"=== Screener.in: {clean_sym} ==="]

    # ── Company description ───────────────────────────────────────────────────
    about = soup.find("div", {"id": "company-info"})
    if about:
        about_text = _clean(about.get_text(separator=" "))
        lines.append(f"About: {_trim(about_text, 500)}")

    # ── Key ratios ────────────────────────────────────────────────────────────
    ratios_section = soup.find("section", {"id": "top-ratios"})
    if ratios_section:
        ratios: list[str] = []
        for li in ratios_section.find_all("li"):
            name_tag  = li.find("span", class_="name")
            value_tag = li.find("span", class_="number")
            if name_tag and value_tag:
                k = _clean(name_tag.get_text())
                v = _clean(value_tag.get_text())
                ratios.append(f"{k}: {v}")
        if ratios:
            lines.append("Key Ratios: " + " | ".join(ratios[:12]))

    # ── Pros & Cons ───────────────────────────────────────────────────────────
    pros_section = soup.find("div", {"id": "pros"})
    if pros_section:
        pros = [_clean(li.get_text()) for li in pros_section.find_all("li")]
        if pros:
            lines.append("Pros: " + "; ".join(pros[:6]))

    cons_section = soup.find("div", {"id": "cons"})
    if cons_section:
        cons = [_clean(li.get_text()) for li in cons_section.find_all("li")]
        if cons:
            lines.append("Cons: " + "; ".join(cons[:6]))

    # ── Quarterly results (last 4 quarters) ──────────────────────────────────
    quarterly_section = soup.find("section", {"id": "quarters"})
    if quarterly_section:
        table = quarterly_section.find("table")
        if table:
            rows = table.find_all("tr")
            # Header row
            header_cells = rows[0].find_all("th") if rows else []
            headers = [_clean(c.get_text()) for c in header_cells[-5:]]  # last 4 qtrs + label
            # Data rows (Revenue, Net Profit, EPS)
            key_rows = ["Sales", "Net Profit", "EPS"]
            q_lines: list[str] = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                row_label = _clean(cells[0].get_text())
                if any(k.lower() in row_label.lower() for k in key_rows):
                    values = [_clean(c.get_text()) for c in cells[1:]][-4:]
                    q_lines.append(f"  {row_label}: " + " | ".join(values))
            if q_lines and headers:
                lines.append(f"Quarterly Results (last 4Q — {' / '.join(headers[-4:])}):")
                lines.extend(q_lines)

    # ── Annual P&L (last 5 years) ─────────────────────────────────────────────
    pl_section = soup.find("section", {"id": "profit-loss"})
    if pl_section:
        table = pl_section.find("table")
        if table:
            rows = table.find_all("tr")
            header_cells = rows[0].find_all("th") if rows else []
            yr_headers = [_clean(c.get_text()) for c in header_cells[-6:]]
            key_rows_pl = ["Sales", "Net Profit", "OPM %", "EPS"]
            pl_lines: list[str] = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                row_label = _clean(cells[0].get_text())
                if any(k.lower() in row_label.lower() for k in key_rows_pl):
                    values = [_clean(c.get_text()) for c in cells[1:]][-5:]
                    pl_lines.append(f"  {row_label}: " + " | ".join(values))
            if pl_lines:
                lines.append(f"Annual P&L ({' / '.join(yr_headers[-5:])}):")
                lines.extend(pl_lines)

    # ── Balance sheet snapshot ────────────────────────────────────────────────
    bs_section = soup.find("section", {"id": "balance-sheet"})
    if bs_section:
        table = bs_section.find("table")
        if table:
            rows = table.find_all("tr")
            key_rows_bs = ["Borrowings", "Total Assets", "Reserves"]
            bs_lines: list[str] = []
            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                row_label = _clean(cells[0].get_text())
                if any(k.lower() in row_label.lower() for k in key_rows_bs):
                    values = [_clean(c.get_text()) for c in cells[1:]][-3:]
                    bs_lines.append(f"  {row_label}: " + " | ".join(values))
            if bs_lines:
                lines.append("Balance Sheet (last 3 years):")
                lines.extend(bs_lines)

    # ── Shareholding pattern ──────────────────────────────────────────────────
    sh_section = soup.find("section", {"id": "shareholding"})
    if sh_section:
        table = sh_section.find("table")
        if table:
            rows = table.find_all("tr")
            sh_lines: list[str] = []
            for row in rows[:5]:   # Promoter, FII, DII, Public
                cells = row.find_all(["td", "th"])
                if len(cells) >= 3:
                    label  = _clean(cells[0].get_text())
                    latest = _clean(cells[-1].get_text())
                    prev   = _clean(cells[-2].get_text()) if len(cells) >= 4 else ""
                    if label and latest:
                        delta = f" (prev: {prev})" if prev and prev != latest else ""
                        sh_lines.append(f"  {label}: {latest}{delta}")
            if sh_lines:
                lines.append("Shareholding Pattern (latest):")
                lines.extend(sh_lines)

    # ── Peer comparison ───────────────────────────────────────────────────────
    peer_section = soup.find("section", {"id": "peers"})
    if peer_section:
        table = peer_section.find("table")
        if table:
            rows = table.find_all("tr")
            peers: list[str] = []
            header_cells = rows[0].find_all("th") if rows else []
            peer_headers = [_clean(c.get_text()) for c in header_cells[:6]]
            for row in rows[1:5]:  # top 4 peers
                cells = row.find_all("td")
                if cells:
                    peer_vals = [_clean(c.get_text()) for c in cells[:6]]
                    peers.append(" | ".join(peer_vals))
            if peers:
                lines.append(f"Peers ({' | '.join(peer_headers)}):")
                for p in peers:
                    lines.append(f"  {p}")

    result = "\n".join(lines)
    return result if len(result) > 80 else ""


# ─────────────────────────────────────────────────────────────────────────────
# 2.  yfinance deep fundamentals
#     (analyst targets, EPS, growth, margins, ratings — free, no key needed)
# ─────────────────────────────────────────────────────────────────────────────

def build_yfinance_context(symbol: str) -> str:
    """
    Pull 80+ fundamental fields from yfinance for the given NSE/BSE symbol.
    Returns a rich text block for the AI analyst prompt.

    Covers:
    - Analyst consensus: target price (mean/high/low), recommendation, # of analysts
    - Valuation: P/E, forward P/E, PEG, P/B, EV/EBITDA, EV/Revenue
    - Earnings: trailing EPS, forward EPS, earnings growth, revenue growth
    - Profitability: gross/operating/net margins, ROE, ROA, ROCE
    - Balance sheet: debt/equity, current ratio, quick ratio
    - Cash flow: free cash flow, operating cash flow
    - Dividends: yield, payout ratio
    - Market: beta, market cap, enterprise value
    """
    try:
        import yfinance as yf
    except ImportError:
        return ""

    # Ensure .NS suffix for NSE stocks
    clean = symbol.upper().replace(".BO", "")
    if not clean.endswith(".NS"):
        clean = clean.replace(".NS", "") + ".NS"

    try:
        info = yf.Ticker(clean).info
    except Exception:
        return ""

    # Bail only if yfinance returned a clearly empty/invalid response.
    # Do NOT require price fields — newer yfinance versions often omit
    # regularMarketPrice/currentPrice for NSE tickers but still return
    # analyst targets, EPS, margins, etc.
    if not info or len(info) < 5:
        return ""

    lines = [f"=== yfinance Fundamentals: {clean} ==="]

    def _fmt(v, prefix="", suffix="", decimals=2, scale=1):
        """Format a numeric value, return empty string if None/0/NaN."""
        try:
            n = float(v) * scale
            if n == 0:
                return ""
            return f"{prefix}{n:,.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return ""

    def _fmt_cr(v):
        """Convert raw INR value to Rs Cr."""
        try:
            return f"Rs{float(v)/1e7:,.0f} Cr"
        except Exception:
            return ""

    # ── Analyst consensus ─────────────────────────────────────────────────────
    target_mean   = info.get("targetMeanPrice")
    target_high   = info.get("targetHighPrice")
    target_low    = info.get("targetLowPrice")
    n_analysts    = info.get("numberOfAnalystOpinions")
    rec_key       = info.get("recommendationKey", "")
    rec_mean      = info.get("recommendationMean")  # 1=Strong Buy … 5=Strong Sell

    rec_label = {
        "strongBuy": "Strong Buy", "buy": "Buy",
        "hold": "Hold", "sell": "Sell", "strongSell": "Strong Sell",
    }.get(rec_key, rec_key.title() if rec_key else "")

    analyst_parts = []
    if target_mean:  analyst_parts.append(f"Target(mean)=Rs{target_mean:,.0f}")
    if target_high:  analyst_parts.append(f"Target(high)=Rs{target_high:,.0f}")
    if target_low:   analyst_parts.append(f"Target(low)=Rs{target_low:,.0f}")
    if rec_label:    analyst_parts.append(f"Consensus={rec_label}")
    if n_analysts:   analyst_parts.append(f"Analysts={n_analysts}")
    if analyst_parts:
        lines.append("Analyst: " + " | ".join(analyst_parts))

    # ── Valuation multiples ───────────────────────────────────────────────────
    val_pairs = [
        ("P/E (TTM)",     _fmt(info.get("trailingPE"),   suffix="x")),
        ("P/E (Fwd)",     _fmt(info.get("forwardPE"),    suffix="x")),
        ("PEG",           _fmt(info.get("pegRatio"),     suffix="x")),
        ("P/B",           _fmt(info.get("priceToBook"),  suffix="x")),
        ("EV/EBITDA",     _fmt(info.get("enterpriseToEbitda"), suffix="x")),
        ("EV/Revenue",    _fmt(info.get("enterpriseToRevenue"), suffix="x")),
    ]
    val_str = " | ".join(f"{k}={v}" for k, v in val_pairs if v)
    if val_str:
        lines.append("Valuation: " + val_str)

    # ── Earnings & growth ─────────────────────────────────────────────────────
    eps_pairs = [
        ("EPS (TTM)",   _fmt(info.get("trailingEps"),  prefix="Rs")),
        ("EPS (Fwd)",   _fmt(info.get("forwardEps"),   prefix="Rs")),
        ("EPS Growth",  _fmt(info.get("earningsGrowth"), suffix="%", scale=100)),
        ("Rev Growth",  _fmt(info.get("revenueGrowth"),  suffix="%", scale=100)),
    ]
    eps_str = " | ".join(f"{k}={v}" for k, v in eps_pairs if v)
    if eps_str:
        lines.append("Earnings: " + eps_str)

    # ── Profitability ─────────────────────────────────────────────────────────
    prof_pairs = [
        ("Gross Margin",  _fmt(info.get("grossMargins"),     suffix="%", scale=100)),
        ("Op Margin",     _fmt(info.get("operatingMargins"), suffix="%", scale=100)),
        ("Net Margin",    _fmt(info.get("profitMargins"),    suffix="%", scale=100)),
        ("ROE",           _fmt(info.get("returnOnEquity"),   suffix="%", scale=100)),
        ("ROA",           _fmt(info.get("returnOnAssets"),   suffix="%", scale=100)),
    ]
    prof_str = " | ".join(f"{k}={v}" for k, v in prof_pairs if v)
    if prof_str:
        lines.append("Profitability: " + prof_str)

    # ── Balance sheet & cash flow ─────────────────────────────────────────────
    bs_pairs = [
        ("D/E",          _fmt(info.get("debtToEquity"),  suffix="x", scale=0.01)),
        ("Current Ratio", _fmt(info.get("currentRatio"), suffix="x")),
        ("Quick Ratio",   _fmt(info.get("quickRatio"),   suffix="x")),
        ("Free CF",       _fmt_cr(info.get("freeCashflow"))),
        ("Op CF",         _fmt_cr(info.get("operatingCashflow"))),
    ]
    bs_str = " | ".join(f"{k}={v}" for k, v in bs_pairs if v)
    if bs_str:
        lines.append("Balance Sheet: " + bs_str)

    # ── Market & size ─────────────────────────────────────────────────────────
    mkt_pairs = [
        ("Mkt Cap",    _fmt_cr(info.get("marketCap"))),
        ("Ent Value",  _fmt_cr(info.get("enterpriseValue"))),
        ("Beta",       _fmt(info.get("beta"), suffix="x")),
        ("Div Yield",  _fmt(info.get("dividendYield"), suffix="%", scale=100)),
    ]
    mkt_str = " | ".join(f"{k}={v}" for k, v in mkt_pairs if v)
    if mkt_str:
        lines.append("Market: " + mkt_str)

    # ── Company description (trimmed) ─────────────────────────────────────────
    desc = info.get("longBusinessSummary", "")
    if desc and len(desc) > 50:
        lines.append(f"Business: {_trim(desc, 350)}")

    result = "\n".join(lines)
    return result if len(result) > 80 else ""
