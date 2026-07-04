#!/usr/bin/env python3
"""
build_context.py — Background pre-fetch of NON-Perplexity data for flagged stocks.
==================================================================================

Runs a few times a day (10:30, 13:00, 15:45 IST). For every stock currently
flagged at a 52-week extreme (from data/latest_screen.json), it fetches the
"cheap"/free data — FMP fundamentals, Screener.in, yfinance, earnings, exchange
announcements, macro, and news — and caches it to data/context/{TICKER}.json.

Perplexity is NEVER called here (that stays on-click only, in the dashboard).

Dedup: a stock already cached with TODAY's date is skipped, so the 2nd and 3rd
runs only spend calls on stocks that are new since the previous run.

Run locally:
    python build_context.py            # all flagged, skip those done today
    python build_context.py --limit 5  # first 5 (testing)
    python build_context.py --force    # ignore the per-day cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CTX_DIR = DATA_DIR / "context"
SNAPSHOT = DATA_DIR / "latest_screen.json"
IST = timezone(timedelta(hours=5, minutes=30))


def today_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _safe(fn) -> str:
    try:
        return (fn() or "")
    except Exception:
        return ""


def flagged() -> list[tuple[str, str]]:
    """(ticker, company_name) for every flagged high/low in the snapshot."""
    try:
        d = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except Exception:
        return []
    seen, out = set(), []
    for key in ("highs", "lows"):
        for row in d.get(key, []) or []:
            t = (row.get("ticker") or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append((t, row.get("company_name") or t))
    return out


def _ctx_path(ticker: str) -> Path:
    return CTX_DIR / f"{ticker.replace('/', '_')}.json"


def done_today(ticker: str) -> bool:
    p = _ctx_path(ticker)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("date") == today_ist()
    except Exception:
        return False


def shared_macro() -> dict:
    """Macro blocks are the same for every stock — fetch once per run."""
    m = {}
    try:
        import fred
        m["fred"] = _safe(fred.build_fred_context)
    except Exception:
        m["fred"] = ""
    try:
        import india_macro
        m["india"] = _safe(india_macro.build_india_macro_context)
    except Exception:
        m["india"] = ""
    try:
        import macro_sector as ms
        m["world_bank"] = _safe(ms.world_bank_block)
        m["finnhub"] = _safe(ms.finnhub_news_block)
    except Exception:
        pass
    return m


def _yf_news(ticker: str, n: int = 6) -> str:
    try:
        import yfinance as yf
        out = []
        for it in (yf.Ticker(ticker).news or [])[:n]:
            c = it.get("content", {}) if isinstance(it, dict) else {}
            title = (c.get("title") or it.get("title") or "").strip()
            pub = c.get("pubDate") or it.get("providerPublishTime") or ""
            if title:
                out.append(f"[{str(pub)[:10]}] {title}")
        return "\n".join(out)
    except Exception:
        return ""


def _tavily_news(company: str, ticker: str) -> str:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return ""
    try:
        import requests
        clean = ticker.replace(".NS", "").replace(".BO", "")
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key,
                  "query": f"{company} {clean} India stock results order deal approval",
                  "search_depth": "basic", "max_results": 6, "days": 30},
            timeout=15,
        )
        if r.status_code == 200:
            lines = []
            for res in (r.json().get("results", []) or [])[:6]:
                t = (res.get("title") or "").strip()
                u = res.get("url", "")
                if t:
                    lines.append(f"- {t} ({u})")
            return "\n".join(lines)
    except Exception:
        return ""
    return ""


def _fmt_earnings(rows) -> str:
    if not rows:
        return ""
    out = ["Earnings surprises (most recent first):"]
    for r in rows[:6]:
        d = str(r.get("date", ""))[:10]
        act = r.get("actualEarningResult")
        est = r.get("estimatedEarning")
        if d:
            out.append(f"  {d}: EPS actual {act} vs estimate {est}")
    return "\n".join(out) if len(out) > 1 else ""


def _fmt_analyst(targets, estimates) -> str:
    out = []
    if targets:
        out.append("Analyst price targets (recent):")
        for t in targets[:5]:
            d = str(t.get("publishedDate", ""))[:10]
            pt = t.get("priceTarget") or t.get("adjPriceTarget")
            an = t.get("analystCompany") or t.get("analystName") or ""
            out.append(f"  {d} {an}: target {pt}")
    if estimates:
        e = estimates[0] if isinstance(estimates, list) and estimates else {}
        rev = e.get("estimatedRevenueAvg")
        eps = e.get("estimatedEpsAvg")
        if rev or eps:
            out.append(f"Consensus (next period): revenue ~{rev}, EPS ~{eps}")
    return "\n".join(out)


def _fmt_profile(p) -> str:
    if isinstance(p, list):
        p = p[0] if p else {}
    if not isinstance(p, dict):
        return ""
    name = p.get("companyName", "")
    sector = p.get("sector", "")
    desc = (p.get("description") or "")[:600]
    return f"{name} ({sector}): {desc}" if desc else ""


def build_one(ticker: str, company: str, macro: dict) -> dict:
    import fmp
    import screener_alpha
    import nse_bse
    return {
        "ticker": ticker,
        "company": company,
        "date": today_ist(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fmp": _safe(lambda: fmp.build_fmp_context(ticker)),
        "screener": _safe(lambda: screener_alpha.build_screener_context(ticker)),
        "yfinance": _safe(lambda: screener_alpha.build_yfinance_context(ticker)),
        "announcements": _safe(lambda: nse_bse.build_nse_bse_context(ticker)),
        "news_yf": _yf_news(ticker),
        "news_web": _tavily_news(company, ticker),
        "earnings": _safe(lambda: _fmt_earnings(fmp.get_earnings_surprises(ticker, limit=6))),
        "analyst": _safe(lambda: _fmt_analyst(fmp.get_price_targets(ticker, limit=5),
                                              fmp.get_analyst_estimates(ticker, limit=2))),
        "profile": _safe(lambda: _fmt_profile(fmp.get_company_profile(ticker))),
        "macro": macro,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-fetch non-Perplexity context for flagged stocks.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="ignore the per-day cache")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    CTX_DIR.mkdir(parents=True, exist_ok=True)
    names = flagged()
    if args.limit:
        names = names[: args.limit]

    todo = [(t, c) for (t, c) in names if args.force or not done_today(t)]
    print(f"Flagged: {len(names)} | already done today: {len(names) - len(todo)} | "
          f"to fetch: {len(todo)}")
    if not todo:
        print("Nothing new to fetch — all flagged stocks already have today's context.")
        return 0

    macro = shared_macro()
    written = 0

    def _work(pair):
        t, c = pair
        ctx = build_one(t, c, macro)
        tmp = _ctx_path(t).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ctx, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        tmp.replace(_ctx_path(t))
        return t

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, p): p[0] for p in todo}
        for f in as_completed(futs):
            try:
                f.result()
                written += 1
            except Exception as exc:
                print(f"  ! {futs[f]}: {exc}")

    print(f"Wrote {written} context file(s) to {CTX_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
