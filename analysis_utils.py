"""
analysis_utils.py — deterministic post-processing for the AI deep-dive output
=============================================================================
The LLM is creative but unreliable on three things the app cares about:
  1. Dates   — it tends to use a news article's publish date, not the event date.
  2. Entities — it leaks sibling/peer companies (e.g. ABFRL under ABSLAMC).
  3. Confidence — its self-reported confidence is not grounded in real signals.

These helpers run AFTER the JSON is parsed and apply cheap, deterministic
checks against ground truth (FMP verified data) so the rendered output is
trustworthy regardless of how the model behaved. No network calls here.
"""

from __future__ import annotations

import re
from datetime import datetime

# Acronyms that look like tickers but are safe to ignore in entity checks.
_SAFE_TOKENS = {
    "Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY", "YOY", "QOQ", "MOM",
    "NSE", "BSE", "SEBI", "RBI", "GST", "IPO", "FPO", "QIP", "OFS",
    "EPS", "PAT", "PBT", "EBITDA", "EBIT", "PROFIT", "REVENUE",
    "ROE", "ROCE", "ROA", "ROIC", "NIM", "AUM", "NPA", "GNPA", "CASA",
    "CEO", "CFO", "COO", "CMD", "MD", "AGM", "EGM", "BOARD",
    "USD", "INR", "EUR", "GBP", "AED", "USA", "UAE", "UK", "EU",
    "GDP", "CPI", "WPI", "MOU", "JV", "EV", "AI", "IT", "PE", "PB",
    "MW", "GW", "KWH", "TWH", "CR", "LAKH", "CRORE", "BPS",
    "YOY", "USD", "DII", "FII", "FPI", "NBFC", "AMC", "PSU",
}

# Keywords that mark a catalyst as earnings/results-related (date-sensitive).
_EARNINGS_KW = (
    "earnings", "results", "quarterly", "quarter", "q1 ", "q2 ", "q3 ", "q4 ",
    "q1fy", "q2fy", "q3fy", "q4fy", "revenue", "net profit", "net income",
    "pat", "profit after tax", "topline", "bottomline",
)


# ──────────────────────────────────────────────────────────────────────────────
# Date parsing / verification (#1)
# ──────────────────────────────────────────────────────────────────────────────

_DATE_FORMATS = (
    "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
    "%d-%m-%Y", "%d/%m/%Y", "%B %Y", "%b %Y", "%Y-%m",
)


def _parse_date(text: str):
    """Best-effort parse of the many date shapes the model emits. Returns date or None."""
    if not text:
        return None
    t = text.strip().strip(":").strip()
    # Pull a leading "DD Mon YYYY" / "Month DD, YYYY" chunk out of a headline.
    m = re.search(
        r"(\d{4}-\d{2}-\d{2})"
        r"|([A-Za-z]+ \d{1,2},? \d{4})"
        r"|(\d{1,2} [A-Za-z]+ \d{4})"
        r"|([A-Za-z]+ \d{4})",
        t,
    )
    cand = next((g for g in (m.groups() if m else []) if g), t)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cand.replace(",", "").strip(),
                                     fmt.replace(",", "")).date()
        except Exception:
            continue
    return None


def extract_verified_dates(fmp_context: str) -> list:
    """All YYYY-MM-DD dates that appear in the FMP verified-data block."""
    if not fmp_context:
        return []
    out = []
    for s in re.findall(r"\d{4}-\d{2}-\d{2}", fmp_context):
        d = _parse_date(s)
        if d:
            out.append(d)
    return sorted(set(out), reverse=True)


def _is_earnings_item(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in _EARNINGS_KW)


def verify_dates(result: dict, fmp_context: str, tolerance_days: int = 12) -> dict:
    """
    For earnings-type catalysts, flag any date that doesn't line up with a
    real reported date from FMP filings. We flag (annotate) rather than silently
    rewrite, since a catalyst could legitimately be a non-earnings June event.
    Adds counters under result['_verification'].
    """
    verified = extract_verified_dates(fmp_context)
    flagged = 0
    confirmed = 0
    if not verified:
        result.setdefault("_verification", {})
        result["_verification"].update({"dates_confirmed": 0, "dates_flagged": 0})
        return result

    def _check(item: dict):
        nonlocal flagged, confirmed
        if not isinstance(item, dict):
            return
        blob = f"{item.get('headline','')} {item.get('detail','')} {item.get('event','')}"
        if not _is_earnings_item(blob):
            return
        d = _parse_date(item.get("date", "") or item.get("headline", ""))
        if not d:
            return
        nearest = min(verified, key=lambda v: abs((v - d).days))
        if abs((nearest - d).days) <= tolerance_days:
            confirmed += 1
        else:
            flagged += 1
            note = (f" (⚠ date unverified — company filings show results on "
                    f"{nearest.strftime('%d %b %Y')})")
            if note not in (item.get("detail") or ""):
                item["detail"] = (item.get("detail", "") or "") + note

    for c in (result.get("catalysts") or []):
        _check(c)
    _check(result.get("primary_catalyst") or {})
    for e in (result.get("catalyst_timeline") or []):
        _check(e)

    result.setdefault("_verification", {})
    result["_verification"].update({"dates_confirmed": confirmed, "dates_flagged": flagged})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Entity-isolation guardrail (#2)
# ──────────────────────────────────────────────────────────────────────────────

def _significant_name_tokens(company: str) -> list:
    stop = {"limited", "ltd", "company", "co", "corporation", "corp", "india",
            "indian", "the", "and", "group", "of", "sun", "life", "&"}
    toks = re.findall(r"[A-Za-z]{3,}", company or "")
    return [t.lower() for t in toks if t.lower() not in stop]


def _foreign_tickers(text: str, clean_ticker: str) -> list:
    ct = (clean_ticker or "").upper()
    out = []
    for tok in set(re.findall(r"\b[A-Z][A-Z0-9&]{2,11}\b", text or "")):
        if tok == ct:
            continue
        # Skip the company's own abbreviation/prefix (e.g. ABSL ↔ ABSLAMC).
        if ct and len(tok) >= 3 and (ct.startswith(tok) or tok.startswith(ct)):
            continue
        if tok in _SAFE_TOKENS:
            continue
        if re.fullmatch(r"(FY|Q)\d+", tok):
            continue
        out.append(tok)
    return out


def filter_foreign_entities(result: dict, company: str, clean_ticker: str) -> dict:
    """
    Drop catalysts/timeline items that are clearly about a DIFFERENT listed
    entity (a foreign ticker in the headline and no mention of this company),
    and flag borderline ones. Records how many were removed.
    """
    name_tokens = _significant_name_tokens(company)
    ct = (clean_ticker or "").upper()
    removed = 0

    def _names_company(text: str) -> bool:
        low = (text or "").lower()
        if ct and ct.lower() in low:
            return True
        return any(tok in low for tok in name_tokens)

    def _keep(item: dict) -> bool:
        nonlocal removed
        if not isinstance(item, dict):
            return True
        headline = item.get("headline", "") or item.get("event", "")
        foreign = _foreign_tickers(headline, ct)
        if foreign and not _names_company(headline):
            removed += 1
            return False
        if foreign:
            tag = "⚠ mentions other entity (" + ", ".join(foreign[:2]) + ")"
            if tag not in (item.get("detail") or ""):
                item["detail"] = (item.get("detail", "") or "") + f" [{tag} — verify]"
        return True

    if isinstance(result.get("catalysts"), list):
        result["catalysts"] = [c for c in result["catalysts"] if _keep(c)]
    if isinstance(result.get("catalyst_timeline"), list):
        result["catalyst_timeline"] = [e for e in result["catalyst_timeline"] if _keep(e)]

    result.setdefault("_verification", {})
    result["_verification"]["entities_removed"] = removed
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Data-driven confidence (#3)
# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence(result: dict) -> str:
    """Confidence grounded in real signals, not the model's self-report."""
    cats = [c for c in (result.get("catalysts") or []) if isinstance(c, dict)]
    n = len(cats)
    n_url = sum(1 for c in cats if str(c.get("url", "")).startswith("http"))
    fresh = ""
    if result.get("news_freshness"):
        fresh = str(result["news_freshness"]).split()[0].upper()
    ver = result.get("_verification", {})
    flagged = ver.get("dates_flagged", 0)
    confirmed = ver.get("dates_confirmed", 0)

    score = 0
    score += 2 if n_url >= 2 else 1 if n_url == 1 else 0
    score += 1 if n >= 4 else 0
    score += 2 if fresh == "FRESH" else -1 if fresh in ("STALE", "THIN") else 0
    score += 1 if confirmed > 0 else 0
    score -= flagged                       # each unverified date erodes trust
    score -= ver.get("entities_removed", 0)

    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


# ──────────────────────────────────────────────────────────────────────────────
# Sector breadth (#5)
# ──────────────────────────────────────────────────────────────────────────────

def sector_breadth(sector: str, ticker: str, highs_df, lows_df, is_hi: bool) -> str:
    """
    Substantiate (or refute) a 'momentum / sector-driven' claim with data:
    how many same-sector names also hit a 52-week extreme in the same screen.
    """
    if not sector:
        return ""
    df = highs_df if is_hi else lows_df
    try:
        if df is None or len(df) == 0 or "sector" not in df.columns:
            return ""
        same = df[df["sector"] == sector]
        n = len(same)
        word = "high" if is_hi else "low"
        if n <= 1:
            return (f"Only {ticker.replace('.NS','').replace('.BO','')} from the "
                    f"{sector} sector hit a 52-week {word} in today's screen — "
                    f"this move looks company-specific, not sector-wide.")
        names = []
        for _, r in same.iterrows():
            nm = str(r.get("company_name") or r.get("ticker") or "").strip()
            if nm:
                names.append(nm.replace(" Limited", "").replace(" Ltd", ""))
        shown = ", ".join(names[:5]) + ("…" if len(names) > 5 else "")
        return (f"{n} {sector} names hit 52-week {word}s in today's screen "
                f"({shown}) — consistent with a sector-wide move rather than a "
                f"single company event.")
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# No-news move diagnosis (deterministic — options A/B/C/D/E)
# ──────────────────────────────────────────────────────────────────────────────
#
# When a stock hits a 52-week extreme but there is no fresh single-event news,
# "stale" is unhelpful. This builds a grounded explanation from data the app has
# already fetched — sector breadth, earnings trajectory, analyst actions, sector
# tailwind and price/volume momentum — with ZERO LLM tokens and no new network
# calls (callers pass in already-fetched context strings / values).

def _fmt_pct(v):
    try:
        if v is None:
            return None
        return f"{float(v):+.1f}%"
    except Exception:
        return None


def _scan_earnings(ctx: str):
    """(beats, misses, latest_surprise) parsed from the earnings-surprise block."""
    if not ctx:
        return 0, 0, ""
    beats  = len(re.findall(r"\bBEAT\b", ctx))
    misses = len(re.findall(r"\bMISS\b", ctx))
    m = re.search(r"(BEAT|MISS)\s*(?:by\s*)?([+-]?\d+(?:\.\d+)?)%", ctx)
    latest = f"{m.group(1).title()} {m.group(2)}%" if m else ""
    return beats, misses, latest


def _scan_upgrades(ctx: str):
    """(target_upside_pct, recent_action_line) parsed from the upgrades block."""
    if not ctx:
        return None, ""
    up = None
    m = re.search(r"Consensus Target:[^()]*\(([+-]?\d+(?:\.\d+)?)%", ctx)
    if m:
        try:
            up = float(m.group(1))
        except Exception:
            up = None
    action = ""
    am = re.search(r"Recent Rating Actions[^\n]*\n\s*(.+)", ctx)
    if am:
        action = am.group(1).strip().lstrip()
    return up, action


def diagnose_move(*, sector="", ticker="", is_hi=True, highs_df=None, lows_df=None,
                  vsurge=None, ret_1m=None, ret_3m=None, ret_6m=None, ret_1y=None,
                  earnings_surprise_context="", upgrades_context="",
                  index_move_pct=None, index_name="",
                  momentum_origin=None) -> dict:
    """
    Deterministic 'why did this move?' diagnosis for the no-fresh-news case.
    Returns {classification, headline, drivers:[{icon,label,text,url}], breadth_line}.
    """
    word    = "high" if is_hi else "low"
    drivers = []
    signals = []

    def _aligned(v):
        return isinstance(v, (int, float)) and ((v > 0) == is_hi) and abs(v) >= 20

    # ── A. Momentum origin (the event that likely started the run) ────────────
    if momentum_origin and momentum_origin.get("title"):
        _d = momentum_origin.get("date", "")
        drivers.append({
            "icon": "🏁", "label": "Momentum origin",
            "text": (f"The run likely began around {_d}: "
                     if _d else "Likely trigger of the run: ")
                    + momentum_origin["title"],
            "url": momentum_origin.get("url", ""),
        })
        signals.append("origin")

    # ── B. Sector breadth ─────────────────────────────────────────────────────
    breadth_line = sector_breadth(sector, ticker, highs_df, lows_df, is_hi)
    n_sector = 0
    try:
        df = highs_df if is_hi else lows_df
        if sector and df is not None and "sector" in getattr(df, "columns", []):
            n_sector = int((df["sector"] == sector).sum())
    except Exception:
        n_sector = 0
    if n_sector >= 3:
        signals.append("sector")
        drivers.append({"icon": "📊", "label": "Sector move",
                        "text": breadth_line or
                        f"Multiple {sector} names hit 52-week {word}s today.", "url": ""})
    elif breadth_line:
        drivers.append({"icon": "📊", "label": "Breadth", "text": breadth_line, "url": ""})

    # ── E. Sector / macro tailwind ────────────────────────────────────────────
    try:
        if index_move_pct is not None:
            mv = float(index_move_pct)
            if abs(mv) >= 2 and ((mv > 0) == is_hi):
                signals.append("macro")
                drivers.append({"icon": "🌐", "label": "Sector tailwind",
                                "text": (f"{index_name or 'The sector index'} is {mv:+.1f}% "
                                         f"recently — broad sector "
                                         f"{'strength' if is_hi else 'weakness'} is lifting "
                                         f"this name, not a single company event."), "url": ""})
    except Exception:
        pass

    # ── C. Earnings / valuation re-rating ─────────────────────────────────────
    beats, misses, latest = _scan_earnings(earnings_surprise_context)
    if is_hi and beats > misses and beats > 0:
        signals.append("fundamental")
        drivers.append({"icon": "📈", "label": "Earnings momentum",
                        "text": (f"{beats} of the last 4 quarters beat estimates"
                                 + (f" (latest {latest})" if latest else "")
                                 + " — a fundamental re-rating, not just sentiment."), "url": ""})
    elif (not is_hi) and misses >= beats and misses > 0:
        signals.append("fundamental")
        drivers.append({"icon": "📉", "label": "Earnings pressure",
                        "text": (f"{misses} of the last 4 quarters missed estimates"
                                 + (f" (latest {latest})" if latest else "")
                                 + " — weak fundamentals are driving the breakdown."), "url": ""})

    # ── D. Analyst signal ─────────────────────────────────────────────────────
    upside, action = _scan_upgrades(upgrades_context)
    if action:
        signals.append("analyst")
        drivers.append({"icon": "🏦", "label": "Analyst action",
                        "text": f"Recent rating action — {action}", "url": ""})
    elif upside is not None and abs(upside) >= 8 and ((upside > 0) == is_hi):
        drivers.append({"icon": "🏦", "label": "Analyst view",
                        "text": f"Consensus target implies {upside:+.1f}% vs the current price.",
                        "url": ""})

    # ── Momentum (price + volume) ─────────────────────────────────────────────
    rets  = {"1M": ret_1m, "3M": ret_3m, "6M": ret_6m, "1Y": ret_1y}
    shown = [f"{k} {_fmt_pct(v)}" for k, v in rets.items() if _fmt_pct(v) is not None]
    if shown:
        signals.append("momentum")
        big = any(_aligned(v) for v in rets.values())
        drivers.append({"icon": "🚀" if is_hi else "🔻", "label": "Momentum",
                        "text": ("Trailing returns: " + "   ".join(shown)
                                 + (" — a sustained run; " if big else " — ")
                                 + ("buyers in control." if is_hi else "sellers in control.")),
                        "url": ""})
    try:
        if vsurge and float(vsurge) >= 2:
            drivers.append({"icon": "🔊", "label": "Volume",
                            "text": f"Volume {float(vsurge):.1f}× the 20-day average — "
                                    f"real conviction behind the move.", "url": ""})
    except Exception:
        pass

    # ── Classification + one-line headline ────────────────────────────────────
    if "sector" in signals or "macro" in signals:
        classification = f"Sector-wide {word}"
    elif "fundamental" in signals:
        classification = "Fundamental re-rating"
    elif "analyst" in signals:
        classification = "Analyst-driven re-rating"
    elif "momentum" in signals or "origin" in signals:
        classification = "Momentum-driven"
    else:
        classification = "Technical / low-news"

    headline = {
        f"Sector-wide {word}":      f"Part of a broad {sector or 'sector'} move — not a single-company event.",
        "Fundamental re-rating":    f"Backed by the company's earnings trajectory rather than a one-off headline.",
        "Analyst-driven re-rating": f"Aligns with recent analyst actions and price targets.",
        "Momentum-driven":          f"A momentum run — price and volume are driving this {word}, with no single fresh catalyst.",
        "Technical / low-news":     f"No fresh company-specific catalyst — this {word} looks technical / momentum-driven.",
    }[classification]

    return {
        "classification": classification,
        "headline":       headline,
        "drivers":        drivers,
        "breadth_line":   breadth_line,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def post_process(result: dict, company: str, clean_ticker: str,
                 fmp_context: str, override_confidence: bool = True) -> dict:
    """Run all deterministic checks in order and return the cleaned result."""
    if not isinstance(result, dict):
        return result
    result = filter_foreign_entities(result, company, clean_ticker)
    result = verify_dates(result, fmp_context)
    if override_confidence:
        result["confidence"] = compute_confidence(result)
    return result



def needs_critique(result: dict) -> bool:
    """True if deterministic checks found problems worth a self-critique LLM pass."""
    ver = result.get("_verification", {}) if isinstance(result, dict) else {}
    return bool(ver.get("dates_flagged", 0) or ver.get("entities_removed", 0))
