"""
policy_news.py — India trade / policy catalyst feed
===================================================
The macro layer tracks GDP, inflation, FX, RBI rates, FII/DII and a sector
tape — but NOT the events that most often move whole sectors: FTAs and
tariff/duty changes, Union Budget allocations, PLI schemes, GST changes,
export/import bans, and sector regulation. Today those are only caught if a
per-stock headline happens to mention them.

This module makes ONE stock-agnostic web search per run for India policy/trade
headlines, tags each with the sectors/themes it affects, and exposes:

  policy_context_block()            -> str   (for the macro prompt)
  policy_drivers_for(sector, theme) -> list  (for the deterministic diagnosis)

Uses Tavily if TAVILY_KEY is set, else NewsAPI, else returns empty — never
raises, never blocks analysis (same house style as macro_sector.py). Results
are cached in-process so all stocks in one run share ONE fetch.
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Optional

import requests

_TIMEOUT = 12


def _env(*keys: str) -> str:
    for k in keys:
        v = os.getenv(k, "").strip()
        if v:
            return v
    return ""


_QUERY = ("India tariff duty cut FTA free trade agreement PLI scheme budget "
          "GST export ban import duty policy incentive — sector impact stocks")

_DOMAINS = [
    "economictimes.indiatimes.com", "moneycontrol.com", "livemint.com",
    "business-standard.com", "financialexpress.com", "ndtvprofit.com",
    "reuters.com", "cnbctv18.com", "thehindubusinessline.com", "pib.gov.in",
]

# headline keyword -> sectors/themes it most affects (match themes.py labels).
_TAG_MAP: list[tuple[tuple[str, ...], list[str]]] = [
    (("textile", "apparel", "garment", "cotton", "yarn"),
     ["Textiles & Apparel", "Consumer Discretionary"]),
    (("pharma", "drug", "api ", "generic", "usfda"), ["Pharma & Healthcare"]),
    (("steel", "aluminium", "aluminum", "metal", "copper", "zinc", "iron ore"),
     ["Materials"]),
    (("chemical", "specialty chem", "agrochem"),
     ["Materials", "China+1 / Specialty Chem"]),
    (("auto", "vehicle", "ev ", "electric vehicle", "two-wheeler", "fame"),
     ["Consumer Discretionary", "EV & Mobility"]),
    (("solar", "renewable", "wind", "green hydrogen", "module", "almm"),
     ["Energy", "Solar & Renewables"]),
    (("defence", "defense", "military", "indigenisation"),
     ["Industrials", "Defence"]),
    (("railway", "rail ", "vande bharat"), ["Industrials", "Railways"]),
    (("sugar", "ethanol", "molasses"), ["Consumer Staples", "Sugar & Ethanol"]),
    (("software", "information technology", "it services", "it sector",
      "tech services", "h-1b", "h1-b", "saas", "bpo", "it exports"), ["IT"]),
    (("bank", "nbfc", "rbi", "lending", "microfinance"), ["Financials"]),
    (("fmcg", "food", "edible oil", "palm"), ["Consumer Staples"]),
    (("real estate", "realty", "housing", "rera"), ["Real Estate"]),
    (("power", "electricity", "discom", "coal", "thermal"), ["Energy"]),
    (("telecom", "spectrum", "5g", "agr "), ["Communication"]),
    (("infra", "capex", "road", "highway", "nhai", "construction"),
     ["Industrials", "Capex & Infra"]),
]

_POS_HINT = ("cut", "cuts", "scrap", "removed", "removes", "zero", "duty-free",
             "free trade", "fta", "pli", "incentive", "subsidy", "boost",
             "exempt", "relief", "approved", "cleared", "signed", "pact",
             "lower", "reduce", "reduces", "slash")
_NEG_HINT = ("hike", "raise", "raises", "ban", "curb", "curbs", "anti-dumping",
             "duty imposed", "tax imposed", "withdraw", "withdrawn",
             "trade war", "restrict", "restriction")


def _tag_sectors(text: str) -> list[str]:
    low = (text or "").lower()
    out: list[str] = []
    for keys, sectors in _TAG_MAP:
        if any(k in low for k in keys):
            for s in sectors:
                if s not in out:
                    out.append(s)
    return out


def _polarity(text: str) -> str:
    low = (text or "").lower()
    pos = sum(1 for w in _POS_HINT if w in low)
    neg = sum(1 for w in _NEG_HINT if w in low)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


_CACHE: dict = {"ts": 0.0, "items": None}
_TTL = 60 * 60  # 1 hour


def _fetch_policy_items(max_items: int = 12) -> list[dict]:
    now = time.time()
    if _CACHE["items"] is not None and (now - _CACHE["ts"]) < _TTL:
        return _CACHE["items"]

    items: list[dict] = []

    tav = _env("TAVILY_KEY", "TAVILY_API_KEY")
    if tav:
        try:
            r = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tav, "query": _QUERY, "search_depth": "advanced",
                      "max_results": max_items, "days": 21, "include_domains": _DOMAINS},
                timeout=_TIMEOUT,
            )
            if r.status_code == 200:
                for res in r.json().get("results", []):
                    title = (res.get("title") or "").strip()
                    if not title:
                        continue
                    items.append({
                        "title": title,
                        "date": (res.get("published_date") or "")[:10],
                        "url": res.get("url", ""),
                        "snippet": (res.get("content") or "")[:200].replace("\n", " "),
                    })
        except Exception:
            pass

    if not items:
        na = _env("NEWSAPI_KEY", "NEWS_API_KEY")
        if na:
            try:
                r = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"apiKey": na,
                            "q": "India (tariff OR duty OR FTA OR PLI OR GST OR \"export ban\")",
                            "sortBy": "publishedAt", "language": "en", "pageSize": max_items},
                    timeout=_TIMEOUT,
                )
                if r.status_code == 200:
                    for art in r.json().get("articles", []):
                        title = (art.get("title") or "").strip()
                        if not title or "[Removed]" in title:
                            continue
                        items.append({
                            "title": title,
                            "date": (art.get("publishedAt") or "")[:10],
                            "url": art.get("url", ""),
                            "snippet": (art.get("description") or "")[:200],
                        })
            except Exception:
                pass

    for it in items:
        blob = f"{it['title']} {it.get('snippet','')}"
        it["sectors"] = _tag_sectors(blob)
        it["polarity"] = _polarity(blob)

    _CACHE["items"] = items
    _CACHE["ts"] = now
    return items


def _is_listicle(title: str) -> bool:
    """Drop 'Top 5 stocks to benefit'-style recommendation pieces from policy."""
    try:
        import analysis_utils as _au
        if _au.is_junk_headline(title):
            return True
    except Exception:
        pass
    low = (title or "").lower()
    return any(p in low for p in
               ("top 5", "top 10", "5 stocks", "10 stocks", "stocks to",
                "shares to", "to benefit", "stocks &", "best stocks"))


# Generic / non-actionable documents that aren't a real catalyst.
_GENERIC_DOC = (
    "[pdf]", "after the crisis", "overview", "explained", "explainer", "history",
    "what is", "guide to", "report on", "analysis of", "introduction",
    "policy framework", "white paper", "primer", "background", "factsheet",
    "frequently asked questions", "faq", "q&a", "everything you need to know",
    "all you need to know", "how does", "meaning of", "definition of",
)
# A real policy catalyst names a concrete action.
_ACTION_KW = (
    "tariff", "duty", "cut", "hike", "slash", "scrap", "pli", "gst", "ban",
    "fta", "free trade", "incentive", "subsidy", "budget", "export", "import",
    "quota", "anti-dumping", "customs", "sop", "exempt", "levy", "waiver",
)


def _is_actionable_policy(title: str) -> bool:
    """True only for headlines that describe a concrete, dated policy ACTION
    (not a generic explainer / document / overview)."""
    low = (title or "").lower()
    if any(g in low for g in _GENERIC_DOC):
        return False
    return any(a in low for a in _ACTION_KW)


def policy_context_block(max_items: int = 8) -> str:
    """Stock-agnostic policy/trade headlines for the macro prompt block."""
    def _keep(i):
        blob = f"{i.get('title','')} {i.get('snippet','')}".lower()
        return (i.get("sectors") and not _is_listicle(i.get("title", ""))
                and _is_actionable_policy(i.get("title", ""))
                and not any(g in blob for g in _GENERIC_DOC))
    items = [i for i in _fetch_policy_items() if _keep(i)]
    if not items:
        return ""
    lines = [f"[INDIA POLICY & TRADE WATCH — as of {date.today():%d %b %Y}]"]
    for it in items[:max_items]:
        tag = f"[{it['date']}]" if it.get("date") else ""
        secs = ", ".join(it["sectors"][:3])
        lines.append(f"  - {tag} {it['title']}  ({it['polarity']} for: {secs})")
    return "\n".join(lines)


# Explicit industry keywords that must appear in a policy headline/snippet for
# it to count as relevant to THAT industry. Trailing spaces avoid substring
# false-positives (e.g. "it " won't match "deposit"). Generic pieces about
# "Indian markets and companies" carry none of these and are dropped.
_SECTOR_KEYWORDS = {
    "it": ("software", "information technology", "it services", "it sector",
           "it exports", "it firms", "it companies", "saas", "fintech", "bfsi",
           "tech services", "bpo", "h-1b", "h1-b"),
    "financials": ("bank", "banking", "nbfc", "lending", "insurance", "insurer",
                   "finance", "financial", "microfinance", "fintech", "credit"),
    "pharma & healthcare": ("pharma", "drug", "api ", "generic", "healthcare",
                            "usfda", "fda ", "formulation", "medicine", "hospital"),
    "materials": ("steel", "metal", "aluminium", "aluminum", "copper", "zinc",
                  "chemical", "cement", "iron ore", "mining", "specialty chem"),
    "energy": ("power", "electricity", "coal", "oil ", "gas ", "renewable",
               "solar", "wind ", "energy", "discom", "petroleum", "ethanol"),
    "consumer discretionary": ("auto", "vehicle", "textile", "apparel", "garment",
                               "retail", "consumer durable", "appliance", "footwear"),
    "consumer staples": ("fmcg", "food ", "edible oil", "sugar", "dairy", "staple",
                         "agri", "fertiliser", "fertilizer"),
    "industrials": ("infra", "capex", "defence", "defense", "railway", "rail ",
                    "construction", "engineering", "capital goods", "shipbuilding"),
    "real estate": ("real estate", "realty", "housing", "property", "rera"),
    "communication": ("telecom", "spectrum", "5g", "media", "broadband", "agr "),
}


def _industry_keywords(targets: set) -> set:
    """Industry words a policy headline/snippet must mention to be relevant."""
    kw: set = set()
    for t in targets:
        if t in _SECTOR_KEYWORDS:
            kw |= set(_SECTOR_KEYWORDS[t])
        else:
            # theme label (e.g. "textiles & apparel", "defence") → its own words
            kw |= {w for w in re.findall(r"[a-z]{4,}", t)}
    return kw


def policy_drivers_for(sector: str = "", theme: str = "", is_hi: bool = True,
                       max_items: int = 2) -> list[dict]:
    """
    Policy headlines relevant to THIS stock's sector/theme and directionally
    consistent with its move. Returns [{text, url, date}] for the diagnosis.

    Two-stage relevance: (1) EXACT canonical sector/theme tag overlap, then
    (2) the headline OR snippet must EXPLICITLY name the industry (so generic
    "impact on Indian markets and companies" pieces are dropped). The returned
    text names the sector and carries the specific 'why' from the snippet.
    """
    targets = {s.strip().lower() for s in (sector, theme) if s and s.strip()}
    if not targets:
        return []
    ind_kw = _industry_keywords(targets)
    want_pol = "positive" if is_hi else "negative"
    out: list[dict] = []
    for it in _fetch_policy_items():
        secs = {s.strip().lower() for s in it.get("sectors", [])}
        inter = targets & secs
        if not inter:                              # exact canonical-label overlap
            continue
        title = it.get("title", "")
        snip = (it.get("snippet", "") or "").strip()
        blob = f"{title} {snip}".lower()
        if any(g in blob for g in _GENERIC_DOC):   # drop explainers/FAQ/overviews
            continue
        if ind_kw and not any(k in blob for k in ind_kw):
            continue                               # must explicitly name the industry
        if _is_listicle(title) or not _is_actionable_policy(title):
            continue
        if it.get("polarity") not in (want_pol, "neutral"):
            continue
        _lab0 = sorted(inter)[0]
        label = "IT" if _lab0 == "it" else _lab0.title()   # nice display label
        text = f"{title} — bears on the {label} sector"
        if snip:
            text += f": {snip[:160]}"
        out.append({"text": text, "url": it.get("url", ""), "date": it.get("date", "")})
        if len(out) >= max_items:
            break
    return out


if __name__ == "__main__":
    print(policy_context_block() or "(no policy items — set TAVILY_KEY / NEWSAPI_KEY)")
