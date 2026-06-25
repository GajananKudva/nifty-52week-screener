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
from datetime import datetime, timedelta

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
# Momentum-origin selection (shared by the card + the deep-dive Move Diagnosis)
# ──────────────────────────────────────────────────────────────────────────────
#
# Picks the news event most plausibly DRIVING a 52-week move, with four guards:
#   • hard recency floor (no months-old stragglers),
#   • entity isolation   (headline must name THIS company),
#   • direction guard    (no net-negative news for a high; no net-positive for a low),
#   • most-recent-first   (the latest qualifying catalyst, not the oldest).
# Returns None when nothing qualifies — an honest "no company-specific catalyst".

_NEG_WORDS = (
    "fear", "fears", "shock", "fall", "falls", "fell", "drop", "drops", "plunge",
    "plunges", "slump", "crash", "decline", "declines", "weak", "weakness", "loss",
    "losses", "probe", "raid", "fraud", "downgrade", "downgrades", "downgraded",
    "cut", "cuts", "warn", "warns", "warning", "concern", "concerns", "lawsuit",
    "ban", "penalty", "fine", "default", "scam", "resign", "resigns", "exit",
    "slowdown", "miss", "misses", "tariff", "tariffs", "headwind", "selloff",
    "sell-off", "bearish", "worry", "worries", "risk", "risks", "pressure", "hurt",
    "drag", "slide",
    # expanded — distress / regulatory / operational
    "halt", "halts", "halted", "suspend", "suspends", "suspended", "suspension",
    "recall", "recalls", "recalled", "writeoff", "write-off", "writedown",
    "impairment", "delist", "delisted", "delisting", "insolvency", "nclt",
    "bankrupt", "bankruptcy", "litigation", "strike", "shutdown", "glitch",
    "outage", "moratorium", "freeze", "frozen", "curb", "curbs", "embezzle",
    "siphon", "qualified", "downturn", "stake-sale", "offload", "dumping",
)
_POS_WORDS = (
    "surge", "surges", "soar", "soars", "rally", "rallies", "jump", "jumps", "gain",
    "gains", "record", "profit", "profits", "beat", "beats", "win", "wins", "won",
    "order", "orders", "bag", "bags", "bagged", "secure", "secures", "secured",
    "upgrade", "upgrades", "upgraded", "raise", "raises", "expansion", "expand",
    "expands", "acquire", "acquires", "acquisition", "launch", "launches",
    "growth", "strong", "bullish", "approval", "approves", "approved", "cleared",
    "deal", "partnership", "buyback", "bonus", "dividend", "rise", "rises", "boost",
    "outperform", "outperforms", "rerating", "re-rating", "rerated",
    # expanded — capital / balance-sheet / momentum catalysts
    "commissioned", "commissioning", "fundraise", "fundraising", "qip",
    "allotment", "demerger", "spinoff", "spin-off", "milestone", "highest",
    "robust", "doubles", "doubled", "triples", "tripled", "accumulate",
    "reappoint", "monetise", "monetize", "monetisation", "highs", "lifetime",
)
_ORIGIN_KW = (
    "result", "results", "earnings", "profit", "revenue", "order", "orders", "win",
    "wins", "contract", "acqui", "merger", "stake", "dividend", "bonus", "approval",
    "launch", "deal", "expansion", "capex", "guidance", "upgrade", "downgrade",
    "rating", "target", "buyback", "partnership", "qip", "fundrais",
    "tariff", "duty", "fta", "pli", "subsidy", "ban", "policy",   # policy/trade catalysts
    "demerger", "spin", "allotment", "preferential", "commission", "monet",
    "pledge", "block deal", "bulk deal", "insolvency", "nclt", "recall",  # event types
)

# ── Phrase-level polarity overrides ───────────────────────────────────────────
# Many single words flip meaning by context: a tariff / duty / tax / rate CUT is
# bullish even though "tariff" and "cut" are both in _NEG_WORDS; a probe that is
# DROPPED or a ban that is LIFTED is bullish even though "probe"/"ban" are
# negative. Multi-word phrases are scored first and weighted more heavily than
# the loose single-word lists, and their matched text is removed before
# single-word scoring so the constituent words aren't double-counted.
_POS_PHRASES = (
    # trade / tariff / duty becoming cheaper or freer
    "tariff cut", "tariff cuts", "tariff reduction", "tariff removed",
    "tariff removal", "cuts tariff", "scraps tariff", "removes tariff",
    "slashes tariff", "zero tariff", "tariff-free", "duty cut", "duty cuts",
    "cuts duty", "scraps duty", "removes duty", "duty removed", "duty scrapped",
    "zero duty", "duty free", "duty-free", "import duty cut", "free trade",
    "trade deal", "trade pact", "trade agreement", "fta", "export incentive",
    "export incentives", "pli scheme", "production linked", "production-linked",
    "subsidy", "incentive scheme",
    # taxes / rates becoming cheaper
    "tax cut", "tax cuts", "gst cut", "rate cut", "rate cuts", "repo cut",
    "duty hike on imports",   # protects domestic producers → bullish for them
    # negative events being resolved
    "ban lifted", "ban removed", "curbs eased", "curbs removed", "cleared of",
    "clean chit", "no fraud", "no wrongdoing", "probe dropped", "probe closed",
    "case dismissed", "case closed", "penalty waived", "fine waived",
    "charges dropped",
    # balance-sheet / ownership / ratings improving
    "debt reduction", "debt reduced", "debt free", "stake buy", "stake purchase",
    "promoter buying", "rating upgrade", "upgraded to", "credit rating upgrade",
    # order / contract wins
    "order win", "bags order", "wins order", "bags contract", "wins contract",
    "l1 bidder", "lowest bidder",
)
_NEG_PHRASES = (
    "tariff hike", "tariff hikes", "tariff war", "raises tariff", "duty hike",
    "import duty hike", "tax hike", "gst hike", "rate hike", "repo hike",
    "export ban", "import ban", "export curb", "export curbs", "export duty",
    "anti-dumping", "trade war", "stake sale", "stake sold", "sells stake",
    "block deal", "rating downgrade", "downgraded to", "credit rating downgrade",
    "guidance cut", "guidance cuts", "dividend cut", "profit warning",
    "ban imposed", "subsidy cut", "subsidy removed", "incentive withdrawn",
)


def _count_word_hits(words, text: str) -> int:
    """Count whole-word matches (word-boundary aware, so 'win' won't match
    'winter' and 'cut' won't match 'execute')."""
    n = 0
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"\b", text):
            n += 1
    return n


def _sentiment_ok(title: str, is_hi: bool) -> bool:
    """Direction guard with phrase-awareness.

    Single keywords are context-blind ("tariff CUT" is bullish even though both
    'tariff' and 'cut' are negative words), so multi-word phrases are scored
    first, weighted x2, and stripped from the text before the single-word lists
    are applied. Single words are matched on word boundaries to avoid substring
    false positives. Rejects net-negative headlines for highs (and vice versa).
    """
    low = f" {(title or '').lower()} "
    pos = neg = 0
    for p in _POS_PHRASES:
        if p in low:
            pos += 2
            low = low.replace(p, " ")
    for p in _NEG_PHRASES:
        if p in low:
            neg += 2
            low = low.replace(p, " ")
    pos += _count_word_hits(_POS_WORDS, low)
    neg += _count_word_hits(_NEG_WORDS, low)
    return (pos >= neg) if is_hi else (neg >= pos)


def _title_names_company(title: str, company: str, clean_ticker: str) -> bool:
    """True if the headline explicitly names this company or its ticker."""
    low = (title or "").lower()
    ct = (clean_ticker or "").lower()
    if ct and ct in low:
        return True
    return any(tok in low for tok in _significant_name_tokens(company))


# ──────────────────────────────────────────────────────────────────────────────
# Optional LLM sentiment + entity guard (#6b)
# ──────────────────────────────────────────────────────────────────────────────
# The keyword guard above is fast and offline but still finite. When a tiny,
# cheap LLM client is available, this resolves the two judgements the keyword
# lists do crudely — (a) does the headline NAME this company, and (b) is it
# bullish/bearish for it — in one call. Returns None on ANY failure so callers
# transparently fall back to the keyword guard. `client` is an OpenAI-compatible
# client (works with Groq via base_url), `model` a small/fast model id.

def llm_relevance_check(title: str, company: str, clean_ticker: str, is_hi: bool,
                        client, model: str) -> Optional[dict]:
    """Return {'names_company': bool, 'direction_ok': bool} or None on failure."""
    if not (client and model and title):
        return None
    try:
        want = "bullish (would push the stock toward a 52-week HIGH)" if is_hi \
            else "bearish (would push the stock toward a 52-week LOW)"
        sys = ("You are a precise equity-news classifier. Given a headline and a "
               "target company, answer ONLY with compact JSON: "
               '{"names_company": true/false, "direction": "bullish"/"bearish"/"neutral"}. '
               "names_company is true only if the headline is about THAT company "
               "(not a peer/parent/subsidiary).")
        usr = (f"Target company: {company} (ticker {clean_ticker}).\n"
               f"Headline: {title!r}\n"
               f"We are checking whether this could be the catalyst for a move that is {want}.")
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=40,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": usr}],
        )
        import json as _json
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = _json.loads(raw)
        names = bool(data.get("names_company"))
        direction = str(data.get("direction", "")).lower()
        direction_ok = (direction == "bullish") if is_hi else (direction == "bearish")
        # neutral is permissive (don't reject), mirroring the keyword guard's >=.
        if direction == "neutral":
            direction_ok = True
        return {"names_company": names, "direction_ok": direction_ok}
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Catalyst materiality scoring + junk filter (shared by the card AND deep-dive)
# ──────────────────────────────────────────────────────────────────────────────
# A small model on a thin news slice tends to grab whatever headline is on top —
# often a listicle ("5 stocks to watch"), a brokerage "buy" note, a pure
# technical/price piece, or the circular "stock hits 52-week high". These must
# never be presented as the MAIN catalyst. We (1) blocklist those shapes and
# (2) score the rest by how concrete/price-moving the event is, so selection is
# materiality-first instead of recency-first.

_JUNK_PATTERNS = (
    "stocks to buy", "stocks to watch", "stocks to add", "stocks to keep",
    "top picks", "stock picks", "buy or sell", "should you buy", "should you sell",
    "should investors", "what should", "here's why", "heres why", "multibagger",
    "stocks in focus", "stocks in news", "best stocks", "top gainers",
    "top losers", "f&o ban", "market wrap", "closing bell", "opening bell",
    "trade setup", "technical view", "technical outlook", "chart check",
    "stocks that hit", "hits 52-week high", "hits 52 week high", "hit 52-week",
    "at 52-week high", "at 52-week low", "52-week high today", "52-week low today",
    "near 52-week", "nears 52-week", "nearing 52-week", "price target by",
    "5 stocks", "10 stocks", "3 stocks", "7 stocks", "watchlist", "nifty today",
    "sensex today", "muhurat", "stocks to look", "shares to buy", "what to expect",
    "things to know", "to watch out", "stock radar", "stock of the day",
)

# concrete, price-moving company events -> weight
_MATERIAL_KW = (
    ("order", 3), ("contract", 3), ("bags", 3), ("awarded", 3), ("win", 3),
    ("won", 3), ("result", 3), ("results", 3), ("profit", 3), ("revenue", 3),
    ("earnings", 3), ("acqui", 3), ("merger", 3), ("buyback", 3), ("qip", 3),
    ("fundrais", 3), ("approval", 3), ("approved", 3), ("demerger", 3),
    ("tariff", 3), ("duty", 3), ("fta", 3), ("pli", 3), ("patent", 3),
    ("stake", 2), ("dividend", 2), ("bonus", 2), ("preferential", 2),
    ("launch", 2), ("capacity", 2), ("expansion", 2), ("capex", 2),
    ("guidance", 2), ("upgrade", 2), ("downgrade", 2), ("rating", 2),
    ("deal", 2), ("partnership", 2), ("commission", 2), ("ban", 2),
    ("acquisition", 3),
)

# Minimum score for a headline to count as a genuine "major catalyst".
MIN_MATERIALITY = 3


def is_junk_headline(title: str) -> bool:
    """True for listicles / recommendations / pure-price commentary that must
    never be shown as the main catalyst."""
    low = (title or "").lower().strip()
    if not low:
        return True
    return any(p in low for p in _JUNK_PATTERNS)


def _has_figure(title: str) -> bool:
    return bool(re.search(
        r"(\d+(?:\.\d+)?\s?%|rs\.?\s?\d|₹\s?\d|\$\s?\d|"
        r"\d+\s?(?:cr|crore|lakh|million|billion|bn|mn))",
        (title or "").lower()))


def score_catalyst_materiality(title: str) -> int:
    """0 = junk/no real event; higher = more concrete & price-moving."""
    if is_junk_headline(title):
        return 0
    low = (title or "").lower()
    score = 0
    seen = set()
    for kw, w in _MATERIAL_KW:
        if kw in seen:
            continue
        if re.search(r"\b" + re.escape(kw), low):
            score += w
            seen.add(kw)
    if _has_figure(title):
        score += 3
    return score


def rank_catalysts(items, company, clean_ticker, is_hi,
                   max_age_days: int = 75, today=None, min_score: int = 1):
    """
    Shared catalyst selector: material, company-named, direction-consistent news,
    sorted by (materiality desc, recency desc). Junk/listicles removed. Returns a
    list of the original item dicts (best first); empty when nothing qualifies.
    """
    if not items:
        return []
    today = today or datetime.now().date()
    floor = today - timedelta(days=max_age_days)
    scored = []
    for it in items:
        title = it.get("title", "")
        if not title or is_junk_headline(title):
            continue
        if "summary" in str(it.get("source", "")).lower():
            continue
        d = _parse_date(it.get("date", ""))
        if d is not None and (d < floor or d > today):
            continue
        if not _title_names_company(title, company, clean_ticker):
            continue
        if not _sentiment_ok(title, is_hi):
            continue
        s = score_catalyst_materiality(title)
        if s < min_score:
            continue
        scored.append((s, d or floor, it))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [it for _s, _d, it in scored]


# Tokens that don't help match a headline to its underlying news item.
_DATE_STOP = {
    "limited", "ltd", "company", "reports", "report", "strong", "results",
    "result", "revenue", "profit", "growth", "stock", "shares", "share", "week",
    "high", "low", "surges", "surge", "jumps", "rises", "gains", "with", "from",
    "year", "quarter", "crore", "million", "billion", "this", "that", "after",
    "amid", "over", "into", "says", "will", "news", "today", "update",
}


def resolve_event_date(headline, items, prefer_sources=("filing",)):
    """
    Return the ACTUAL EVENT date for a chosen catalyst headline, not the
    publish/re-report date. News about an event breaks on the event day, so the
    EARLIEST matching report is the event date; exchange filings (NSE/BSE) carry
    the true board-meeting/results date and are preferred when present. Matches
    the headline to the underlying news items by shared content words.
    Returns 'YYYY-MM-DD' or None when nothing matches.
    """
    if not headline or not items:
        return None
    h = {w for w in re.findall(r"[a-z0-9]{4,}", headline.lower())
         if w not in _DATE_STOP}
    if not h:
        return None
    matched = []
    for it in items:
        toks = {w for w in re.findall(r"[a-z0-9]{4,}", (it.get("title") or "").lower())
                if w not in _DATE_STOP}
        if len(h & toks) < 2:                 # needs ≥2 shared content words
            continue
        d = _parse_date(it.get("date", ""))
        if not d:
            continue
        is_filing = any(p in str(it.get("source", "")).lower() for p in prefer_sources)
        matched.append((0 if is_filing else 1, d))
    if not matched:
        return None
    # Exchange filings first; then the earliest date (= the event day).
    matched.sort(key=lambda x: (x[0], x[1]))
    return matched[0][1].strftime("%Y-%m-%d")


def pick_momentum_origin(items, company, clean_ticker, is_hi,
                         max_age_days: int = 45, today=None,
                         llm_client=None, llm_model: str = ""):
    """
    Choose the news event most plausibly behind a 52-week move. See module note
    above for the guards. Returns {title, date, url} or None.

    If `llm_client`/`llm_model` are supplied, the entity + direction judgement is
    made by the LLM (with the keyword guard as automatic fallback per-headline).
    """
    if not items:
        return None
    today = today or datetime.now().date()
    floor = today - timedelta(days=max_age_days)
    qualified = []
    for it in items:
        title = it.get("title", "")
        if not title:
            continue
        if "summary" in str(it.get("source", "")).lower():
            continue
        if is_junk_headline(title):       # never let listicles/price pieces win
            continue
        d = _parse_date(it.get("date", ""))
        if d is None or d < floor or d > today:
            continue

        verdict = llm_relevance_check(title, company, clean_ticker, is_hi,
                                      llm_client, llm_model) if llm_client else None
        if verdict is not None:
            if not verdict["names_company"] or not verdict["direction_ok"]:
                continue
        else:
            if not _title_names_company(title, company, clean_ticker):
                continue
            if not _sentiment_ok(title, is_hi):
                continue
        qualified.append((d, it))
    if not qualified:
        return None
    # Prefer genuine catalyst-type headlines; among those, the most MATERIAL,
    # then the most recent (materiality-first beats recency-first).
    cat = [(d, it) for (d, it) in qualified
           if any(k in it["title"].lower() for k in _ORIGIN_KW)]
    pool = cat or qualified
    d, o = max(pool, key=lambda x: (score_catalyst_materiality(x[1]["title"]), x[0]))
    return {"title": o["title"],
            "date": o.get("date") or d.strftime("%Y-%m-%d"),
            "url": o.get("url", "")}


def no_catalyst_summary(sector: str, is_hi: bool, breadth_line: str = "") -> str:
    """Honest one-liner when no recent company-specific catalyst exists."""
    word = "high" if is_hi else "low"
    if breadth_line and "sector-wide" in breadth_line.lower():
        return (f"Sector-wide {word} — part of a broad {sector or 'sector'} move, "
                f"no single-company catalyst.")
    return f"No recent company-specific catalyst — technical / momentum-led {word}."


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
                  momentum_origin=None, theme_breadth=None, policy_drivers=None) -> dict:
    """
    Deterministic 'why did this move?' diagnosis for the no-fresh-news case.
    Returns {classification, headline, drivers:[{icon,label,text,url}], breadth_line}.

    Extra (optional, already-fetched by the caller — zero work here):
      theme_breadth  : dict from themes.theme_breadth()        -> cross-sector thematic move
      policy_drivers : list from policy_news.policy_drivers_for -> trade/policy catalyst
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
            "icon": "🗓", "label": "Momentum origin",
            "text": (f"Move traces back to {_d}: "
                     if _d else "Likely origin of the move: ")
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

    # ── B2. Thematic breadth (cross-sector basket: defence, railways, EV…) ─────
    if isinstance(theme_breadth, dict) and theme_breadth.get("line"):
        signals.append("theme")
        drivers.append({"icon": "🧩", "label": "Thematic move",
                        "text": theme_breadth["line"], "url": ""})

    # ── B3. Policy / trade catalyst (FTA, tariff/duty, PLI, GST, ban) ──────────
    for _pdrv in (policy_drivers or [])[:2]:
        if isinstance(_pdrv, dict) and _pdrv.get("text"):
            signals.append("policy")
            _pdt = _pdrv.get("date", "")
            drivers.append({"icon": "🏛", "label": "Policy / trade",
                            "text": _pdrv["text"] + (f" ({_pdt})" if _pdt else ""),
                            "url": _pdrv.get("url", "")})

    # ── E. Sector / macro tailwind ────────────────────────────────────────────
    try:
        if index_move_pct is not None:
            mv = float(index_move_pct)
            if abs(mv) >= 2 and ((mv > 0) == is_hi):
                signals.append("macro")
                drivers.append({"icon": "🌐", "label": "Sector tailwind",
                                "text": (f"{index_name or 'The sector index'} is {mv:+.1f}% over the "
                                         f"past month — broad sector "
                                         f"{'strength' if is_hi else 'weakness'} is supporting the "
                                         f"move rather than a single company event."), "url": ""})
    except Exception:
        pass

    # ── C. Earnings / valuation re-rating ─────────────────────────────────────
    beats, misses, latest = _scan_earnings(earnings_surprise_context)
    if is_hi and beats > misses and beats > 0:
        signals.append("fundamental")
        drivers.append({"icon": "💹", "label": "Earnings trajectory",
                        "text": (f"{beats} of the last four quarters exceeded estimates"
                                 + (f" (most recent: {latest})" if latest else "")
                                 + " — the move is underpinned by a fundamental re-rating."), "url": ""})
    elif (not is_hi) and misses >= beats and misses > 0:
        signals.append("fundamental")
        drivers.append({"icon": "💹", "label": "Earnings trajectory",
                        "text": (f"{misses} of the last four quarters missed estimates"
                                 + (f" (most recent: {latest})" if latest else "")
                                 + " — deteriorating fundamentals are weighing on the shares."), "url": ""})

    # ── D. Analyst signal ─────────────────────────────────────────────────────
    upside, action = _scan_upgrades(upgrades_context)
    if action:
        signals.append("analyst")
        drivers.append({"icon": "🏦", "label": "Analyst action",
                        "text": f"Recent brokerage action — {action}.", "url": ""})
    elif upside is not None and abs(upside) >= 8 and ((upside > 0) == is_hi):
        drivers.append({"icon": "🏦", "label": "Analyst view",
                        "text": f"Consensus price target implies {upside:+.1f}% versus the current price.",
                        "url": ""})

    # ── Momentum (price + volume) ─────────────────────────────────────────────
    rets  = {"1M": ret_1m, "3M": ret_3m, "6M": ret_6m, "1Y": ret_1y}
    shown = [f"{k} {_fmt_pct(v)}" for k, v in rets.items() if _fmt_pct(v) is not None]
    if shown:
        signals.append("momentum")
        big = any(_aligned(v) for v in rets.values())
        drivers.append({"icon": "📈" if is_hi else "📉", "label": "Price momentum",
                        "text": ("Trailing returns: " + "   ".join(shown)
                                 + (" — an extended advance; the uptrend remains intact."
                                    if big and is_hi else
                                    " — an extended decline; the downtrend remains intact."
                                    if big else
                                    " — the prevailing trend remains in force.")),
                        "url": ""})
    try:
        if vsurge and float(vsurge) >= 2:
            drivers.append({"icon": "🔊", "label": "Volume",
                            "text": f"Volume at {float(vsurge):.1f}× the 20-day average — "
                                    f"elevated participation is confirming the move.", "url": ""})
    except Exception:
        pass

    # ── Classification + one-line headline ────────────────────────────────────
    # Priority: a genuine COMPANY-SPECIFIC catalyst (earnings trajectory or a
    # named momentum origin) always wins the headline over sector-level signals
    # (policy / theme / sector). Otherwise policy/theme/sector explain the move.
    if "fundamental" in signals:
        classification = "Fundamental re-rating"
    elif "origin" in signals:
        classification = "Company-catalyst-led"
    elif "theme" in signals:
        classification = f"Thematic {word}"
    elif "policy" in signals:
        classification = "Policy / trade-driven"
    elif "sector" in signals or "macro" in signals:
        classification = f"Sector-wide {word}"
    elif "analyst" in signals:
        classification = "Analyst-driven re-rating"
    elif "momentum" in signals:
        classification = "Momentum-driven"
    else:
        classification = "Technical / low-news"

    headline = {
        "Fundamental re-rating":    "Driven by the company's earnings trajectory rather than a one-off headline.",
        "Company-catalyst-led":     f"Driven by a company-specific catalyst (see the momentum origin) — not a sector or policy move.",
        f"Thematic {word}":         f"Part of a cross-sector thematic move rather than a single-company event.",
        "Policy / trade-driven":    f"Linked to a trade/policy change affecting the {sector or 'sector'} — a sector-wide catalyst, not a single-company event.",
        f"Sector-wide {word}":      f"Part of a broad {sector or 'sector'} move rather than a single-company event.",
        "Analyst-driven re-rating": "Consistent with recent brokerage actions and price-target revisions.",
        "Momentum-driven":          f"Momentum-driven: price and volume are extending this {word}, with no single fresh catalyst this week.",
        "Technical / low-news":     f"No fresh company-specific catalyst — this {word} is technical and momentum-led.",
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


# Thematic + stock-specific catalyst helpers: themes.theme_breadth() feeds
# diagnose_move; exchange_filings feeds the stock-specific news pool in app.py.
# Catalyst selection (rank_catalysts / score_catalyst_materiality / is_junk_headline)
# is shared by the pre-analysis card and the deep-dive so both pick the same,
# materiality-first, junk-filtered main catalyst.
