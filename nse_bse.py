"""
nse_bse.py — NSE & BSE stock-level institutional data
======================================================
Fetches per-stock signals using public NSE endpoints (no API key needed).
Session-based cookie approach identical to india_macro.py.

Data fetched per ticker:
  1. Delivery %         — genuine accumulation vs intraday speculation
  2. F&O OI buildup     — long buildup / short covering / unwinding
  3. Bulk & block deals — named institutional trades with price & quantity
  4. Promoter shareholding — quarterly % stake changes (last 4 quarters)
  5. Corporate announcements — dividends, buybacks, results, orders (last 90 days)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import requests

_TIMEOUT = 12
_NSE_HDRS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "accept":           "application/json, text/plain, */*",
    "accept-language":  "en-US,en;q=0.9",
    "accept-encoding":  "gzip, deflate, br",
    "referer":          "https://www.nseindia.com/",
    "x-requested-with": "XMLHttpRequest",
}

_ANNOUNCEMENT_KEYWORDS = [
    "dividend", "buyback", "results", "qip", "bonus", "split",
    "merger", "acquisition", "demerger", "order", "contract",
    "guidance", "capex", "expansion", "rights", "award", "win",
    "agreement", "delisting", "fpo",
]


def _session() -> requests.Session:
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=_NSE_HDRS, timeout=_TIMEOUT)
    except Exception:
        pass
    return s


def _clean(symbol: str) -> str:
    return symbol.upper().replace(".NS", "").replace(".BO", "")


def _fmt_num(v) -> str:
    try:
        f = float(str(v).replace(",", ""))
        if abs(f) >= 10_000_000:
            return f"{f/10_000_000:.2f}Cr"
        elif abs(f) >= 100_000:
            return f"{f/100_000:.2f}L"
        elif abs(f) >= 1_000:
            return f"{f/1_000:.1f}K"
        return f"{f:.0f}"
    except Exception:
        return str(v)


def _fmt_pct(v) -> str:
    try:
        return f"{float(str(v).replace('%','').strip()):.2f}%"
    except Exception:
        return "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Delivery %
# ──────────────────────────────────────────────────────────────────────────────

def get_delivery_data(symbol: str) -> Optional[dict]:
    """Delivery % from NSE trade info."""
    sym = _clean(symbol)
    try:
        s = _session()
        r = s.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={sym}&section=trade_info",
            headers=_NSE_HDRS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        ti = (r.json().get("marketDeptOrderBook") or {}).get("tradeInfo") or {}
        del_pct = ti.get("deliveryToTradedQuantity")
        if del_pct is None:
            return None
        return {
            "delivery_pct": float(str(del_pct).replace("%", "").strip()),
            "delivery_qty": ti.get("deliveryQuantity"),
            "total_qty":    ti.get("totalTradedVolume"),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 2. F&O buildup
# ──────────────────────────────────────────────────────────────────────────────

def get_fno_buildup(symbol: str) -> Optional[dict]:
    """Near-month futures OI buildup: long buildup / short buildup / covering / unwinding."""
    sym = _clean(symbol)
    try:
        s = _session()
        r = s.get(
            f"https://www.nseindia.com/api/quote-derivative?symbol={sym}",
            headers=_NSE_HDRS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        stocks = r.json().get("stocks") or []
        futures = [
            x for x in stocks
            if "FUTURES" in str(x.get("metadata", {}).get("instrumentType", "")).upper()
            and "STOCK" in str(x.get("metadata", {}).get("instrumentType", "")).upper()
        ]
        if not futures:
            return {"available": False, "reason": "Not in F&O segment"}

        near  = sorted(futures, key=lambda x: x.get("metadata", {}).get("expiryDate", "9999"))[0]
        meta  = near.get("metadata", {})
        other = (near.get("marketDeptOrderBook") or {}).get("otherInfo") or {}

        price_chg = float(str(meta.get("change", 0) or 0))
        oi_chg    = float(str(other.get("changeinOpenInterest", 0) or 0))

        if price_chg > 0 and oi_chg > 0:
            buildup = "LONG BUILDUP"
            interp  = "Fresh longs added — bullish conviction, move likely sustainable"
        elif price_chg < 0 and oi_chg > 0:
            buildup = "SHORT BUILDUP"
            interp  = "Fresh shorts added — institutional bears entering, bearish pressure"
        elif price_chg > 0 and oi_chg < 0:
            buildup = "SHORT COVERING"
            interp  = "Shorts squared off — rally present but lacks fresh conviction"
        else:
            buildup = "LONG UNWINDING"
            interp  = "Longs exiting — weak hands selling, bearish signal"

        return {
            "available":      True,
            "buildup":        buildup,
            "interpretation": interp,
            "price_change":   price_chg,
            "oi_change":      oi_chg,
            "open_interest":  other.get("openInterest", "N/A"),
            "expiry":         meta.get("expiryDate", ""),
        }
    except Exception:
        return {"available": False, "reason": "F&O data unavailable"}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Bulk & block deals
# ──────────────────────────────────────────────────────────────────────────────

def get_bulk_block_deals(symbol: str) -> dict:
    """Today's bulk and block deals for the symbol."""
    sym    = _clean(symbol)
    result = {"bulk": [], "block": []}
    try:
        s = _session()
        for deal_type, endpoint in [("bulk", "bulk-deals"), ("block", "block-deals")]:
            try:
                r = s.get(
                    f"https://www.nseindia.com/api/{endpoint}",
                    headers=_NSE_HDRS, timeout=_TIMEOUT,
                )
                if not r.ok:
                    continue
                raw   = r.json()
                items = raw if isinstance(raw, list) else raw.get("data", [])
                result[deal_type] = [
                    d for d in items
                    if str(d.get("symbol") or d.get("Symbol") or "").upper() == sym
                ]
            except Exception:
                pass
    except Exception:
        pass
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. Promoter shareholding pattern
# ──────────────────────────────────────────────────────────────────────────────

def get_promoter_shareholding(symbol: str) -> Optional[list]:
    """Last 4 quarters of promoter / FII / DII / retail holding %."""
    sym = _clean(symbol)
    try:
        s = _session()
        r = s.get(
            f"https://www.nseindia.com/api/corporate-shareholding-patterns?index=equities&symbol={sym}",
            headers=_NSE_HDRS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw     = r.json()
        records = raw if isinstance(raw, list) else raw.get("data", [])

        quarters = []
        for rec in records[:4]:
            entry = {
                "quarter":      rec.get("date") or rec.get("quarter") or rec.get("shareholdingDate") or "",
                "promoter_pct": None,
                "fii_pct":      None,
                "dii_pct":      None,
                "retail_pct":   None,
            }
            patterns = (
                rec.get("shareholdingPatterns", {}).get("items")
                or rec.get("shareHolding")
                or rec.get("data")
                or []
            )
            for item in (patterns if isinstance(patterns, list) else []):
                cat = str(
                    item.get("category") or item.get("shareHolderClass") or ""
                ).upper()
                raw_pct = (
                    item.get("percentage")
                    or item.get("sharesPercentage")
                    or item.get("percentageHolding")
                )
                try:
                    pct = float(str(raw_pct).replace("%", "").strip())
                except Exception:
                    pct = None

                if "PROMOTER" in cat:
                    entry["promoter_pct"] = pct
                elif "FII" in cat or "FPI" in cat or "FOREIGN PORTFOLIO" in cat:
                    entry["fii_pct"] = pct
                elif "DII" in cat or "MUTUAL FUND" in cat or "INSURANCE" in cat:
                    entry["dii_pct"] = pct
                elif "RETAIL" in cat or "PUBLIC" in cat or "INDIVIDUAL" in cat:
                    entry["retail_pct"] = pct

            if entry["quarter"]:
                quarters.append(entry)

        return quarters if quarters else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Corporate announcements
# ──────────────────────────────────────────────────────────────────────────────

def get_corporate_announcements(symbol: str) -> Optional[list]:
    """Recent material corporate announcements from NSE."""
    sym = _clean(symbol)
    try:
        s = _session()
        r = s.get(
            f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={sym}",
            headers=_NSE_HDRS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw   = r.json()
        items = raw if isinstance(raw, list) else raw.get("data", [])

        filtered = []
        for item in items[:40]:
            subject = str(
                item.get("desc") or item.get("subject") or item.get("attchmntText") or ""
            ).strip()
            date = str(
                item.get("excDt") or item.get("bcastDt") or item.get("an_dt") or ""
            ).strip()
            if any(kw in subject.lower() for kw in _ANNOUNCEMENT_KEYWORDS):
                filtered.append({"date": date, "subject": subject[:100]})
            if len(filtered) >= 8:
                break

        return filtered if filtered else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 6. Stock-specific FII/DII flows (Rs Cr, derived from shareholding x market cap)
# ──────────────────────────────────────────────────────────────────────────────

def get_fiidii_stock_flows(symbol: str) -> Optional[dict]:
    """
    Estimate stock-level FII and DII holding changes in Rs Cr.

    Method:
      1. Fetch market cap from NSE equity quote API.
      2. Use latest two quarters of shareholding pattern data.
      3. Compute DeltaHolding % x market cap -> approximate Rs Cr change.

    Returns dict with keys: market_cap_cr, latest_quarter, prev_quarter,
      fii_pct_now, fii_pct_prev, fii_chg_pct, fii_chg_cr,
      dii_pct_now, dii_pct_prev, dii_chg_pct, dii_chg_cr,
      interpretation.
    Returns None on failure or insufficient data.
    """
    sym = _clean(symbol)
    try:
        s = _session()
        r = s.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={sym}",
            headers=_NSE_HDRS, timeout=_TIMEOUT,
        )
        r.raise_for_status()
        info     = r.json()
        price_info = info.get("priceInfo", {})
        mkt_cap_raw = (
            (info.get("industryInfo") or {}).get("marketCap")
            or price_info.get("marketCap")
        )
        # Try metadata block
        if mkt_cap_raw is None:
            meta_block = info.get("metadata") or info.get("securityInfo") or {}
            mkt_cap_raw = meta_block.get("marketCap") or meta_block.get("mktCap")

        market_cap_cr: Optional[float] = None
        if mkt_cap_raw is not None:
            try:
                raw_val = float(str(mkt_cap_raw).replace(",", ""))
                # NSE often returns market cap in Rs (not Cr); convert
                market_cap_cr = raw_val / 1e7 if raw_val > 1e9 else raw_val
            except Exception:
                pass

        # Fallback: estimate from last price x shares outstanding
        if market_cap_cr is None:
            last_price = float(str(price_info.get("lastPrice") or 0).replace(",", "") or 0)
            shares_raw = (info.get("securityInfo") or {}).get("issuedSize") or 0
            try:
                shares = float(str(shares_raw).replace(",", ""))
                if last_price > 0 and shares > 0:
                    market_cap_cr = (last_price * shares) / 1e7
            except Exception:
                pass

        if market_cap_cr is None or market_cap_cr <= 0:
            return None

        # Get shareholding pattern (latest 2 quarters)
        holding = get_promoter_shareholding(symbol)
        if not holding or len(holding) < 2:
            return None

        latest = holding[0]
        prev   = holding[1]

        fii_now  = latest.get("fii_pct")
        fii_prev = prev.get("fii_pct")
        dii_now  = latest.get("dii_pct")
        dii_prev = prev.get("dii_pct")

        if fii_now is None or fii_prev is None:
            return None

        fii_chg_pct = round(float(fii_now) - float(fii_prev), 2)
        dii_chg_pct = round(float(dii_now or 0) - float(dii_prev or 0), 2)

        fii_chg_cr = round(fii_chg_pct / 100 * market_cap_cr, 0)
        dii_chg_cr = round(dii_chg_pct / 100 * market_cap_cr, 0)

        def _interp(chg_cr: float, who: str) -> str:
            if chg_cr > 500:
                return f"{who} added heavily (+Rs{chg_cr:,.0f}Cr) -- strong conviction buying"
            elif chg_cr > 100:
                return f"{who} increased stake (+Rs{chg_cr:,.0f}Cr) -- moderate accumulation"
            elif chg_cr > 0:
                return f"{who} marginally increased (+Rs{chg_cr:,.0f}Cr)"
            elif chg_cr < -500:
                return f"{who} sold heavily (-Rs{abs(chg_cr):,.0f}Cr) -- significant distribution"
            elif chg_cr < -100:
                return f"{who} reduced stake (-Rs{abs(chg_cr):,.0f}Cr) -- moderate selling"
            elif chg_cr < 0:
                return f"{who} marginally reduced (-Rs{abs(chg_cr):,.0f}Cr)"
            return f"{who} holding unchanged"

        return {
            "market_cap_cr":  round(market_cap_cr, 0),
            "latest_quarter": latest.get("quarter", ""),
            "prev_quarter":   prev.get("quarter", ""),
            "fii_pct_now":    fii_now,
            "fii_pct_prev":   fii_prev,
            "fii_chg_pct":    fii_chg_pct,
            "fii_chg_cr":     fii_chg_cr,
            "dii_pct_now":    dii_now,
            "dii_pct_prev":   dii_prev,
            "dii_chg_pct":    dii_chg_pct,
            "dii_chg_cr":     dii_chg_cr,
            "fii_interpretation": _interp(fii_chg_cr, "FII/FPI"),
            "dii_interpretation": _interp(dii_chg_cr, "DII/MF"),
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────────────────────────────────────

def build_nse_bse_context(symbol: str) -> str:
    sym   = _clean(symbol)
    lines = [
        "=" * 60,
        f"NSE STOCK-LEVEL DATA  ({sym})",
        f"As of {datetime.today().strftime('%d %B %Y')}",
        "=" * 60,
    ]
    any_data = False

    # -- Delivery %
    delivery = get_delivery_data(symbol)
    if delivery and delivery.get("delivery_pct") is not None:
        any_data = True
        dpct = delivery["delivery_pct"]
        if dpct >= 65:
            signal = "HIGH -- strong institutional/conviction buying"
        elif dpct >= 45:
            signal = "MODERATE -- mixed retail and institutional"
        else:
            signal = "LOW -- predominantly intraday/speculative"
        lines.append("\n[DELIVERY DATA]")
        lines.append(f"  Delivery %   : {dpct:.1f}%  ->  {signal}")

    # -- F&O buildup
    fno = get_fno_buildup(symbol)
    if fno:
        if fno.get("available"):
            any_data = True
            lines.append("\n[F&O BUILDUP -- Near Month Futures]")
            lines.append(f"  Signal       : {fno['buildup']}")
            lines.append(f"  Reading      : {fno['interpretation']}")
            lines.append(f"  Price chg    : {fno['price_change']:+.2f}%  |  OI chg: {fno['oi_change']:+.2f}%")
        else:
            lines.append(f"\n[F&O]  {fno.get('reason', 'Not in F&O segment')}")

    # -- Bulk & block deals
    deals = get_bulk_block_deals(symbol)
    bulk  = deals.get("bulk", [])
    block = deals.get("block", [])
    any_data = any_data or bool(bulk or block)
    lines.append("\n[BULK & BLOCK DEALS -- Today]")
    if bulk or block:
        for d in (bulk + block)[:8]:
            tag    = "BULK " if d in bulk else "BLOCK"
            client = str(d.get("clientName") or d.get("Client Name") or d.get("client_name") or "?")[:30]
            side   = str(d.get("buySell") or d.get("buy_sell") or "?")
            qty    = d.get("quantity") or d.get("Quantity") or "?"
            price  = d.get("tradePrice") or d.get("Trade Price") or "?"
            lines.append(f"  {tag} | {client:<30} | {side:<5} | Qty: {_fmt_num(qty)} | Rs{price}")
    else:
        lines.append("  No bulk/block deals today")

    # -- Promoter shareholding
    holding = get_promoter_shareholding(symbol)
    if holding:
        any_data = True
        lines.append("\n[SHAREHOLDING PATTERN -- Quarterly Trend]")
        lines.append(f"  {'Quarter':<14} {'Promoter':>10} {'FII':>8} {'DII':>8} {'Retail':>8}  Trend")
        prev_promo = None
        for q in holding:
            promo  = _fmt_pct(q.get("promoter_pct"))
            fii    = _fmt_pct(q.get("fii_pct"))
            dii    = _fmt_pct(q.get("dii_pct"))
            retail = _fmt_pct(q.get("retail_pct"))
            trend  = ""
            if prev_promo is not None and q.get("promoter_pct") is not None:
                try:
                    diff  = float(q["promoter_pct"]) - float(prev_promo)
                    arrow = "up" if diff > 0 else "down"
                    trend = f"  Promoter {arrow}{abs(diff):.2f}% QoQ"
                except Exception:
                    pass
            lines.append(f"  {str(q['quarter']):<14} {promo:>10} {fii:>8} {dii:>8} {retail:>8}{trend}")
            prev_promo = q.get("promoter_pct")

    # -- Stock-specific FII/DII flows in Rs Cr
    flows = get_fiidii_stock_flows(symbol)
    if flows:
        any_data = True
        lines.append("\n[FII / DII STOCK-LEVEL FLOWS -- Quarterly Delta Holding]")
        lines.append(f"  Market Cap    : Rs{flows['market_cap_cr']:,.0f} Cr")
        lines.append(f"  Period        : {flows['prev_quarter']}  ->  {flows['latest_quarter']}")
        lines.append(
            f"  FII/FPI       : {flows['fii_pct_prev']:.2f}%  ->  {flows['fii_pct_now']:.2f}%"
            f"  (Delta {flows['fii_chg_pct']:+.2f}%  approx  Rs{flows['fii_chg_cr']:+,.0f} Cr)"
        )
        lines.append(f"  Reading       : {flows['fii_interpretation']}")
        if flows.get("dii_pct_now") is not None:
            lines.append(
                f"  DII/MF        : {flows['dii_pct_prev']:.2f}%  ->  {flows['dii_pct_now']:.2f}%"
                f"  (Delta {flows['dii_chg_pct']:+.2f}%  approx  Rs{flows['dii_chg_cr']:+,.0f} Cr)"
            )
            lines.append(f"  Reading       : {flows['dii_interpretation']}")
        lines.append("  Note: Rs Cr figures are estimates (DeltaHolding% x Market Cap). Quarterly, not daily.")

    # -- Corporate announcements
    announcements = get_corporate_announcements(symbol)
    if announcements:
        any_data = True
        lines.append("\n[RECENT CORPORATE ANNOUNCEMENTS]")
        for a in announcements:
            lines.append(f"  {a['date']:<12} : {a['subject']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines) if any_data else ""


# ──────────────────────────────────────────────────────────────────────────────
# Bulk live screener — 1 API call for all stocks in a NSE index
# ──────────────────────────────────────────────────────────────────────────────

_NSE_INDEX_MAP = {
    "NIFTY 500":    "NIFTY%20500",
    "NIFTY 50":     "NIFTY%2050",
    "NIFTY NEXT 50":"NIFTY%20NEXT%2050",
    "NIFTY 100":    "NIFTY%20100",
    "NIFTY 200":    "NIFTY%20200",
    "NIFTY MIDCAP 100": "NIFTY%20MIDCAP%20100",
    "NIFTY SMALLCAP 100": "NIFTY%20SMALLCAP%20100",
}


def get_nse_index_live(index: str = "NIFTY 500") -> list[dict]:
    """
    Fetch all constituents of a NSE index with live prices and 52-week levels
    in a single API call.

    Returns a list of dicts with keys:
      symbol, ticker, company_name, sector,
      last_price, year_high, year_low,
      volume, pct_change,
      near_wkh  (% below 52W high -- 0 = at high),
      near_wkl  (% above 52W low  -- 0 = at low)

    Returns empty list on failure.
    """
    encoded = _NSE_INDEX_MAP.get(index.upper().strip(),
                                 index.replace(" ", "%20").upper())
    try:
        s = _session()
        r = s.get(
            f"https://www.nseindia.com/api/equity-stockIndices?index={encoded}",
            headers=_NSE_HDRS,
            timeout=20,
        )
        r.raise_for_status()
        raw_data = r.json().get("data", [])
    except Exception:
        return []

    result = []
    for stock in raw_data:
        sym = str(stock.get("symbol") or "").strip().upper()
        if not sym or sym in ("NIFTY 500", "NIFTY 50", "NIFTY 100",
                              "NIFTY 200", "NIFTY NEXT 50",
                              "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100"):
            continue   # skip the index row itself

        meta = stock.get("meta") or {}

        # nearWKH / nearWKL: NSE pre-computes "% away from 52W high/low"
        # Values are already percentages (e.g. 2.5 means 2.5% away)
        near_wkh = stock.get("nearWKH")
        near_wkl = stock.get("nearWKL")

        try:
            near_wkh = float(near_wkh) if near_wkh is not None else None
        except (TypeError, ValueError):
            near_wkh = None

        try:
            near_wkl = float(near_wkl) if near_wkl is not None else None
        except (TypeError, ValueError):
            near_wkl = None

        # Today's intraday high/low — NSE returns these as dayHigh/dayLow
        # Fall back to lastPrice if not present (e.g. pre-market or data gap)
        raw_day_high = (stock.get("dayHigh") or stock.get("high")
                        or stock.get("intradayHighPrice") or stock.get("lastPrice"))
        raw_day_low  = (stock.get("dayLow")  or stock.get("low")
                        or stock.get("intradayLowPrice")  or stock.get("lastPrice"))
        try:
            day_high = float(str(raw_day_high).replace(",", "")) if raw_day_high else None
        except (TypeError, ValueError):
            day_high = None
        try:
            day_low = float(str(raw_day_low).replace(",", "")) if raw_day_low else None
        except (TypeError, ValueError):
            day_low = None

        result.append({
            "symbol":       sym,
            "ticker":       f"{sym}.NS",
            "company_name": (meta.get("companyName") or sym),
            "sector":       (meta.get("industry") or ""),
            "last_price":   stock.get("lastPrice"),   # LTP — for display only
            "day_high":     day_high,                 # today's session high — for signal detection
            "day_low":      day_low,                  # today's session low  — for signal detection
            "year_high":    stock.get("yearHigh"),
            "year_low":     stock.get("yearLow"),
            "volume":       stock.get("totalTradedVolume"),
            "pct_change":   stock.get("pChange"),
            "near_wkh":     near_wkh,   # NSE pre-computed (LTP-based) — overridden in engine
            "near_wkl":     near_wkl,   # NSE pre-computed (LTP-based) — overridden in engine
        })

    return result
