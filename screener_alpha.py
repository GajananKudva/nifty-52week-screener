"""
screener_alpha.py — Screener.in scraper + Alpha Vantage fundamentals
=====================================================================
Provides two context-building functions used by app.py's AI analyst:

  build_screener_context(symbol)           → str
      Scrapes screener.in for Indian-specific fundamentals:
      10-year financials, quarterly results, pros/cons, shareholding.

  build_alpha_vantage_context(symbol, api_key)  → str
      Calls Alpha Vantage for EPS surprises, analyst price targets,
      income statement highlights, and overview metrics.
"""

from __future__ import annotations

import re
import time
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
# 2.  Alpha Vantage  (EPS surprises, analyst targets, income highlights)
# ─────────────────────────────────────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"


def _av_get(function: str, symbol: str, api_key: str, **extra) -> Optional[dict]:
    """Fetch one Alpha Vantage endpoint. Returns parsed JSON or None."""
    params = {"function": function, "symbol": symbol, "apikey": api_key, **extra}
    try:
        r = requests.get(_AV_BASE, params=params, timeout=12)
        if r.status_code == 200:
            data = r.json()
            # AV returns {"Information": "..."} when rate-limited
            if "Information" in data or "Note" in data:
                return None
            return data
    except Exception:
        pass
    return None


def build_alpha_vantage_context(symbol: str, api_key: str) -> str:
    """
    Pull company overview, EPS surprises, and earnings calendar from
    Alpha Vantage and return a structured text block for the AI prompt.

    Alpha Vantage free tier: 25 req/day, 5 req/min — we make ≤3 calls.
    The NSE symbol (without .NS) maps to AV's BSE-listed symbol as-is for
    most large-cap Nifty 500 companies.
    """
    if not api_key:
        return ""

    clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "")
    # Alpha Vantage uses BSE/NSE symbol directly for Indian stocks
    av_sym = clean_sym

    lines: list[str] = [f"=== Alpha Vantage: {clean_sym} ==="]

    # ── 1. Company Overview ───────────────────────────────────────────────────
    overview = _av_get("OVERVIEW", av_sym, api_key)
    if overview and "Symbol" in overview:
        fields = [
            ("Market Cap",        overview.get("MarketCapitalization", "")),
            ("P/E (TTM)",         overview.get("PERatio", "")),
            ("EPS (TTM)",         overview.get("EPS", "")),
            ("Revenue (TTM)",     overview.get("RevenueTTM", "")),
            ("Gross Margin",      overview.get("GrossProfitTTM", "")),
            ("Operating Margin",  overview.get("OperatingMarginTTM", "")),
            ("52W High",          overview.get("52WeekHigh", "")),
            ("52W Low",           overview.get("52WeekLow", "")),
            ("Analyst Target",    overview.get("AnalystTargetPrice", "")),
            ("PEG Ratio",         overview.get("PEGRatio", "")),
            ("Dividend Yield",    overview.get("DividendYield", "")),
            ("Book Value",        overview.get("BookValue", "")),
            ("EBITDA",            overview.get("EBITDA", "")),
            ("Debt/Equity",       overview.get("DebtToEquityRatio", "")),
            ("ROE",               overview.get("ReturnOnEquityTTM", "")),
            ("ROA",               overview.get("ReturnOnAssetsTTM", "")),
            ("Beta",              overview.get("Beta", "")),
            ("Sector",            overview.get("Sector", "")),
            ("Industry",          overview.get("Industry", "")),
        ]
        valid_fields = [(k, v) for k, v in fields if v and v not in ("None", "-", "0")]
        if valid_fields:
            lines.append("Overview: " + " | ".join(f"{k}={v}" for k, v in valid_fields[:12]))

        desc = overview.get("Description", "")
        if desc and len(desc) > 30:
            lines.append(f"Description: {_trim(desc, 400)}")

        # Analyst ratings if available
        target = overview.get("AnalystTargetPrice", "")
        rating = overview.get("AnalystRatingStrongBuy", "")
        buy    = overview.get("AnalystRatingBuy", "")
        hold   = overview.get("AnalystRatingHold", "")
        sell   = overview.get("AnalystRatingSell", "")
        if any([rating, buy, hold, sell]):
            lines.append(
                f"Analyst Ratings: Strong Buy={rating} Buy={buy} Hold={hold} Sell={sell} | Target Price={target}"
            )

    time.sleep(0.3)  # stay within 5 req/min free tier

    # ── 2. Earnings (EPS surprises) ───────────────────────────────────────────
    earnings = _av_get("EARNINGS", av_sym, api_key)
    if earnings:
        quarterly = earnings.get("quarterlyEarnings", [])
        if quarterly:
            lines.append("Quarterly EPS Surprises (last 4Q):")
            for q in quarterly[:4]:
                date         = q.get("fiscalDateEnding", "")
                reported_eps = q.get("reportedEPS", "N/A")
                estimated    = q.get("estimatedEPS", "N/A")
                surprise     = q.get("surprisePercentage", "")
                surprise_str = f" (surprise: {float(surprise):+.1f}%)" if surprise and surprise not in ("None", "") else ""
                lines.append(
                    f"  {date}: Reported={reported_eps} | Estimated={estimated}{surprise_str}"
                )

    time.sleep(0.3)

    # ── 3. Income Statement (annual, last 3 years) ────────────────────────────
    income = _av_get("INCOME_STATEMENT", av_sym, api_key)
    if income:
        annual = income.get("annualReports", [])
        if annual:
            lines.append("Annual Income (last 3 years):")
            for yr in annual[:3]:
                fy          = yr.get("fiscalDateEnding", "")[:4]
                revenue     = yr.get("totalRevenue", "")
                gross       = yr.get("grossProfit", "")
                net_income  = yr.get("netIncome", "")
                ebitda      = yr.get("ebitda", "")

                def _fmt_cr(v: str) -> str:
                    """Convert raw USD/INR value to readable format (Rs Cr)."""
                    try:
                        n = float(v)
                        cr = n / 1e7
                        return f"Rs{cr:,.0f}Cr"
                    except Exception:
                        return v

                parts = [f"FY{fy}"]
                if revenue:  parts.append(f"Revenue={_fmt_cr(revenue)}")
                if gross:    parts.append(f"GrossProfit={_fmt_cr(gross)}")
                if net_income: parts.append(f"NetIncome={_fmt_cr(net_income)}")
                if ebitda:   parts.append(f"EBITDA={_fmt_cr(ebitda)}")
                lines.append("  " + " | ".join(parts))

    result = "\n".join(lines)
    return 