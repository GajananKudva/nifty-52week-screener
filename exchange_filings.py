"""
exchange_filings.py — NSE/BSE corporate announcements as catalyst items
=======================================================================
Many Indian mid/small-cap catalysts break first as exchange filings (board
meetings, order wins, fundraises, results) hours before they reach the news
domains the main pipeline whitelists. This module pulls recent corporate
announcements for a ticker from NSE (cookie session) with a BSE fallback, and
returns them in the SAME shape as _fetch_all_news_items() so they can be merged
straight into the catalyst pool.

Public API:
  fetch_filings(ticker, company, days=21) -> list[{date, source, title, snippet, url}]

Never raises; returns [] on any failure (NSE is often geo-blocked on cloud).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import requests

_TIMEOUT = 10
_NSE_HDRS = {
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "referer": "https://www.nseindia.com/",
    "x-requested-with": "XMLHttpRequest",
}


def _clean(sym: str) -> str:
    return str(sym or "").upper().replace(".NS", "").replace(".BO", "").strip()


def _parse_nse_dt(s: str) -> str:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip()[:len(fmt) + 4], fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return str(s)[:10]


def _fetch_nse(symbol: str, days: int) -> list[dict]:
    out: list[dict] = []
    try:
        sess = requests.Session()
        sess.get("https://www.nseindia.com", headers=_NSE_HDRS, timeout=_TIMEOUT)
        frm = (datetime.today() - timedelta(days=days)).strftime("%d-%m-%Y")
        to = datetime.today().strftime("%d-%m-%Y")
        url = ("https://www.nseindia.com/api/corporate-announcements"
               f"?index=equities&symbol={symbol}&from_date={frm}&to_date={to}")
        r = sess.get(url, headers=_NSE_HDRS, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        rows = data if isinstance(data, list) else data.get("data", [])
        for it in rows[:12]:
            subject = (it.get("subject") or it.get("desc") or it.get("attchmntText")
                       or "").strip()
            if not subject:
                continue
            dt = _parse_nse_dt(it.get("an_dt") or it.get("sort_date") or "")
            out.append({
                "date": dt, "source": "NSE-filing", "title": subject[:150],
                "snippet": (it.get("attchmntText") or "")[:200],
                "url": it.get("attchmntFile") or "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
            })
    except Exception:
        pass
    return out


def _fetch_bse(symbol: str, days: int) -> list[dict]:
    """BSE announcement API fallback (works by scrip code; best-effort by name)."""
    out: list[dict] = []
    try:
        # BSE's public announcement endpoint accepts a scrip code; without a
        # symbol->code map we query the generic latest feed and filter by name.
        url = ("https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
               f"?pageno=1&strCat=-1&strPrevDate=&strScrip=&strSearch=P&strType=C")
        hdrs = {"user-agent": _NSE_HDRS["user-agent"],
                "referer": "https://www.bseindia.com/"}
        r = requests.get(url, headers=hdrs, timeout=_TIMEOUT)
        if r.status_code != 200:
            return out
        rows = (r.json() or {}).get("Table", [])
        sym_low = symbol.lower()
        for it in rows:
            name = (it.get("SLONGNAME") or it.get("NEWSSUB") or "").lower()
            if sym_low[:5] not in name:
                continue
            head = (it.get("NEWSSUB") or it.get("HEADLINE") or "").strip()
            if not head:
                continue
            out.append({
                "date": str(it.get("NEWS_DT") or "")[:10],
                "source": "BSE-filing", "title": head[:150],
                "snippet": (it.get("MORE") or "")[:200], "url": "",
            })
            if len(out) >= 8:
                break
    except Exception:
        pass
    return out


def fetch_filings(ticker: str, company: str = "", days: int = 21) -> list[dict]:
    """Recent exchange filings for a ticker, newest-first, as news-item dicts."""
    sym = _clean(ticker)
    if not sym:
        return []
    items = _fetch_nse(sym, days)
    if not items:
        items = _fetch_bse(sym, days)
    # newest first
    try:
        items.sort(key=lambda x: x.get("date", ""), reverse=True)
    except Exception:
        pass
    return items


if __name__ == "__main__":
    rows = fetch_filings("RELIANCE.NS")
    print(f"{len(rows)} filings")
    for r in rows[:5]:
        print(" ", r["date"], r["title"])
