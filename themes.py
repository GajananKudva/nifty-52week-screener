"""
themes.py — Thematic / basket breadth + sector-taxonomy normalisation
=====================================================================
The app's breadth check groups by the exact ``sector`` string from yfinance,
so real market themes that cut ACROSS yfinance sectors — defence, railways,
EV, solar/renewables, PSU re-rating, China+1, sugar, capital-markets — get
diluted or missed (e.g. a defence rally where HAL/BEL/BDL sit under different
GICS sectors).

This module provides:
  • THEME_MAP            — theme  -> set of clean NSE symbols (extend freely).
  • themes_for_ticker()  — which themes a symbol belongs to.
  • normalise_sector()   — collapse the two sector taxonomies (NSE feed vs
                           yfinance) into one canonical label.
  • theme_breadth()      — given the current screen's highs/lows DataFrame,
                           how many names from THIS stock's theme(s) also hit a
                           52-week extreme — the thematic analogue of
                           analysis_utils.sector_breadth().

Pure logic, zero network. Safe to import anywhere.
"""

from __future__ import annotations

from typing import Iterable, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Theme → constituent symbols (clean NSE symbols, no .NS/.BO suffix).
# This is a starter set covering the themes that most often drive synchronised
# 52-week moves on the NSE. Add/trim symbols as your universe evolves.
# ──────────────────────────────────────────────────────────────────────────────
THEME_MAP: dict[str, set[str]] = {
    "Defence": {
        "HAL", "BEL", "BDL", "BEML", "COCHINSHIP", "MAZDOCK", "GRSE",
        "DATAPATTNS", "PARAS", "ZENTEC", "MTARTECH", "SOLARINDS", "ASTRAMICRO",
        "IDEAFORGE", "DCXINDIA", "UNIMECH",
    },
    "Railways": {
        "IRFC", "RVNL", "IRCON", "RAILTEL", "TITAGARH", "TEXRAIL", "JWL",
        "IRCTC", "CONCOR", "BEML", "RITES", "JUPITER",
    },
    "EV & Mobility": {
        "TATAMOTORS", "M&M", "OLAELEC", "EXIDEIND", "ARE&M", "AMARAJABAT",
        "TATAPOWER", "HEROMOTOCO", "BAJAJ-AUTO", "TVSMOTOR", "SONACOMS",
        "UNOMINDA", "BOSCHLTD",
    },
    "Solar & Renewables": {
        "SUZLON", "INOXWIND", "WAAREEENER", "PREMIERENE", "BORORENEW",
        "ADANIGREEN", "NTPC", "TATAPOWER", "JSWENERGY", "KPIGREEN",
        "ORIENTGREEN", "SWSOLAR", "WEBELSOLAR", "GENSOL", "NHPC", "SJVN",
    },
    "PSU": {
        "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "COALINDIA",
        "NTPC", "POWERGRID", "ONGC", "IOC", "BPCL", "HPCL", "GAIL", "NHPC",
        "SJVN", "RECLTD", "PFC", "IRFC", "BHEL", "HINDPETRO", "NMDC",
        "NLCINDIA", "MOIL", "IRCON", "RVNL",
    },
    "China+1 / Specialty Chem": {
        "SRF", "NAVINFLUOR", "AARTIIND", "DEEPAKNTR", "PIIND", "ATUL",
        "VINATIORGA", "CLEAN", "TATACHEM", "FLUOROCHEM", "GNFC", "ALKYLAMINE",
        "BALAMINES", "LAXMIORG",
    },
    "Textiles & Apparel": {
        "PAGEIND", "KPRMILL", "TRIDENT", "WELSPUNLIV", "VARDHACRLC", "RAYMOND",
        "ARVIND", "GOKEX", "NITINSPIN", "VTL", "INDOCOUNT", "FAZE3",
        "SUTLEJTEX", "ALOKINDS", "GARFIBRES", "AMBIKCO", "RSWM", "SPLPETRO",
        "HIMATSEIDE", "DOLLAR", "RUPA", "LUXIND",
    },
    "Capital Markets": {
        "BSE", "MCX", "CDSL", "KFINTECH", "CAMS", "ANGELONE", "360ONE",
        "NUVAMA", "MOTILALOFS", "IEX", "ANANDRATHI", "IIFL",
    },
    "Sugar & Ethanol": {
        "BALRAMCHIN", "BAJAJHIND", "DALMIASUG", "TRIVENI", "DHAMPURSUG",
        "RENUKA", "DCMSHRIRAM", "AVADHSUGAR", "MAGADSUGAR", "UGARSUGAR",
        "KCPSUGIND",
    },
    "Real Estate": {
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "MACROTECH", "BRIGADE",
        "PHOENIXLTD", "SOBHA", "MAHLIFE", "SUNTECK", "ANANTRAJ", "NBCC",
    },
    "Capex & Infra": {
        "LT", "BHEL", "SIEMENS", "ABB", "CGPOWER", "THERMAX", "KEC",
        "KALPATPOWR", "HGINFRA", "PNCINFRA", "NCC", "GRINFRA", "IRB",
        "RVNL", "IRCON", "ENGINERSIN",
    },
}

# Canonical sector buckets that both the NSE feed and yfinance map into, so
# breadth grouping is stable regardless of which source labelled the stock.
_SECTOR_CANON = {
    "financial services": "Financials", "financial": "Financials",
    "bank": "Financials", "banks": "Financials", "nbfc": "Financials",
    "insurance": "Financials",
    "information technology": "IT", "it": "IT", "software": "IT",
    "technology": "IT", "tech": "IT",
    "healthcare": "Pharma & Healthcare", "pharmaceuticals": "Pharma & Healthcare",
    "pharma": "Pharma & Healthcare", "health": "Pharma & Healthcare",
    "consumer cyclical": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples", "fmcg": "Consumer Staples",
    "consumer staples": "Consumer Staples",
    "basic materials": "Materials", "materials": "Materials",
    "metals": "Materials", "metal": "Materials", "chemicals": "Materials",
    "industrials": "Industrials", "industrial": "Industrials",
    "capital goods": "Industrials",
    "energy": "Energy", "oil & gas": "Energy", "oil": "Energy", "power": "Energy",
    "utilities": "Energy",
    "real estate": "Real Estate", "realty": "Real Estate",
    "communication services": "Communication", "telecom": "Communication",
    "media": "Communication",
}


def normalise_sector(sector: str) -> str:
    """Collapse NSE-feed and yfinance sector strings into one canonical label."""
    if not sector:
        return ""
    s = str(sector).strip().lower()
    if s in _SECTOR_CANON:
        return _SECTOR_CANON[s]
    for key, canon in _SECTOR_CANON.items():
        if key in s:
            return canon
    return str(sector).strip().title()


def _clean(sym: str) -> str:
    return str(sym or "").upper().replace(".NS", "").replace(".BO", "").strip()


# Reverse index: symbol -> {themes}
_SYM_THEMES: dict[str, set[str]] = {}
for _theme, _syms in THEME_MAP.items():
    for _s in _syms:
        _SYM_THEMES.setdefault(_clean(_s), set()).add(_theme)


def themes_for_ticker(ticker: str) -> list[str]:
    """Return the list of themes a symbol belongs to (possibly empty)."""
    return sorted(_SYM_THEMES.get(_clean(ticker), set()))


def _df_symbols(df) -> set[str]:
    if df is None:
        return set()
    try:
        if "ticker" not in getattr(df, "columns", []):
            return set()
        return {_clean(t) for t in df["ticker"].tolist()}
    except Exception:
        return set()


def theme_breadth(ticker: str, highs_df=None, lows_df=None, is_hi: bool = True,
                  min_names: int = 3) -> Optional[dict]:
    """
    How many names from THIS ticker's theme(s) also hit a 52-week extreme in the
    same screen. Returns the strongest qualifying theme as
    ``{theme, n, names, line}`` or None when the stock is in no mapped theme or
    too few theme-peers moved together.
    """
    my_themes = themes_for_ticker(ticker)
    if not my_themes:
        return None
    df = highs_df if is_hi else lows_df
    hit = _df_symbols(df)
    if not hit:
        return None
    word = "high" if is_hi else "low"
    best = None
    for theme in my_themes:
        members = THEME_MAP.get(theme, set())
        moved = sorted({_clean(s) for s in members} & hit)
        if len(moved) >= min_names and (best is None or len(moved) > best["n"]):
            shown = ", ".join(moved[:6]) + ("…" if len(moved) > 6 else "")
            best = {
                "theme": theme,
                "n": len(moved),
                "names": moved,
                "line": (f"{len(moved)} {theme} names hit 52-week {word}s in "
                         f"today's screen ({shown}) — a thematic move that spans "
                         f"multiple sectors, not a single-company event."),
            }
    return best


if __name__ == "__main__":
    print("HAL themes:", themes_for_ticker("HAL.NS"))
    print("normalise 'Consumer Cyclical' ->", normalise_sector("Consumer Cyclical"))
