"""
app.py — Professional Streamlit Dashboard
==========================================
Nifty 500  |  52-Week High/Low Screener  |  AI-Powered Analysis

Run:
    streamlit run app.py

Architecture:
  - engine.py      → DataEngine  (screening, fundamentals, DCF)
  - mailer.py      → NIFTY_500_TICKERS, fetch_nifty500_live
  - app.py         → Streamlit UI  (this file)

All screener results are stored in st.session_state so that adjusting
sidebar controls never triggers a full API re-fetch.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser

# ── ReportLab PDF generation ──────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# Always load .env from the same folder as this script (local dev only)
_HERE = Path(__file__).parent
load_dotenv(dotenv_path=_HERE / ".env", override=True)


def _secret(key: str, default: str = "") -> str:
    """Read a secret from env (local .env) or st.secrets (Streamlit Cloud)."""
    val = os.getenv(key, "").strip()
    if not val:
        try:
            val = str(st.secrets.get(key, default)).strip()
        except Exception:
            pass
    return val or default


# ── AI client — OpenAI, Groq, or Perplexity (all share the OpenAI SDK) ───────
# Set LLM_PROVIDER=groq        → Groq free tier (llama-3.3-70b-versatile)
# Set LLM_PROVIDER=perplexity  → Perplexity Sonar (grounded web search built in)
# Set LLM_PROVIDER=openai (or leave blank) → OpenAI
_LLM_PROVIDER  = _secret("LLM_PROVIDER", "openai").lower()
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_PPLX_BASE_URL = "https://api.perplexity.ai"


def _llm_base_url() -> Optional[str]:
    """OpenAI-compatible base URL for the active provider (None = OpenAI native)."""
    if _LLM_PROVIDER == "groq":
        return _GROQ_BASE_URL
    if _LLM_PROVIDER == "perplexity":
        return _PPLX_BASE_URL
    return None


# Perplexity Sonar grounds its own web search. Whitelisting Indian + tier-1
# financial domains keeps retrieval on NSE/BSE/ET/Mint/MoneyControl/Reuters
# instead of skewing to general/US sources that miss small/mid-cap coverage.
_PPLX_DOMAINS = [
    d.strip() for d in _secret(
        "PERPLEXITY_SEARCH_DOMAINS",
        "nseindia.com,bseindia.com,moneycontrol.com,"
        "economictimes.indiatimes.com,livemint.com,reuters.com",
    ).split(",") if d.strip()
][:20]  # Perplexity caps the domain filter at 20 entries
_PPLX_RECENCY = _secret("PERPLEXITY_RECENCY", "month").strip()  # month|week|day|hour


def _pplx_extra_body(context_size: str = "low") -> dict:
    """Perplexity-only search controls. Empty dict for OpenAI/Groq (ignored)."""
    if _LLM_PROVIDER != "perplexity":
        return {}
    body: dict = {"web_search_options": {"search_context_size": context_size}}
    if _PPLX_DOMAINS:
        body["search_domain_filter"] = _PPLX_DOMAINS
    if _PPLX_RECENCY:
        body["search_recency_filter"] = _PPLX_RECENCY
    return body


def _pplx_citations(response) -> list:
    """Sonar's grounded citations/search_results — its own source-of-truth URLs."""
    if _LLM_PROVIDER != "perplexity":
        return []
    urls: list = []
    try:
        dump = response.model_dump()
    except Exception:
        dump = {}
    for u in (dump.get("citations") or []):
        if isinstance(u, str) and u.startswith("http"):
            urls.append(u)
    for sr in (dump.get("search_results") or []):
        u = (sr or {}).get("url", "")
        if isinstance(u, str) and u.startswith("http"):
            urls.append(u)
    # de-dupe, preserve order
    seen: set = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


# ── Cost guard — per-session call counter + rough running cost (Perplexity) ──
# USD per 1M tokens (input, output) and per-call request fee by search context.
_PPLX_PRICING = {
    "sonar":               {"in": 1.0, "out": 1.0,  "req": {"low": 0.005, "medium": 0.008, "high": 0.012}},
    "sonar-pro":           {"in": 3.0, "out": 15.0, "req": {"low": 0.006, "medium": 0.010, "high": 0.014}},
    "sonar-reasoning":     {"in": 1.0, "out": 5.0,  "req": {"low": 0.005, "medium": 0.008, "high": 0.012}},
    "sonar-reasoning-pro": {"in": 2.0, "out": 8.0,  "req": {"low": 0.006, "medium": 0.010, "high": 0.014}},
}


def _track_llm_call(model: str, response=None, context_size: str = "low") -> None:
    """Increment the session call counter and accumulate an estimated cost."""
    try:
        st.session_state["_llm_call_count"] = st.session_state.get("_llm_call_count", 0) + 1
        if _LLM_PROVIDER != "perplexity":
            return
        price = _PPLX_PRICING.get((model or "").lower())
        if not price:
            return
        usage = getattr(response, "usage", None)
        it = getattr(usage, "prompt_tokens", 0) or 0
        ot = getattr(usage, "completion_tokens", 0) or 0
        cost = (it / 1e6) * price["in"] + (ot / 1e6) * price["out"]
        cost += price["req"].get(context_size, price["req"]["low"])
        st.session_state["_llm_cost_usd"] = st.session_state.get("_llm_cost_usd", 0.0) + cost
    except Exception:
        pass


try:
    from openai import OpenAI as _OpenAI
    if _LLM_PROVIDER == "perplexity":
        _OPENAI_KEY   = _secret("PERPLEXITY_API_KEY") or _secret("OPENAI_API_KEY")
        _OPENAI_MODEL = _secret("OPENAI_MODEL", "sonar-pro")   # deep-dive default
    elif _LLM_PROVIDER == "groq":
        _OPENAI_KEY   = _secret("OPENAI_API_KEY") or _secret("GROQ_API_KEY")
        _OPENAI_MODEL = _secret("OPENAI_MODEL", "llama-3.3-70b-versatile")
    else:
        _OPENAI_KEY   = _secret("OPENAI_API_KEY")
        _OPENAI_MODEL = _secret("OPENAI_MODEL", "gpt-4o-mini")
    _AI_OK        = bool(_OPENAI_KEY)
except ImportError:
    _AI_OK        = False
    _OPENAI_KEY   = ""
    _OPENAI_MODEL = ""

# ── FMP client (optional — graceful fallback) ─────────────────────────────────
try:
    from fmp import build_fmp_context
    _FMP_KEY = _secret("FMP_API_KEY")
    _FMP_OK  = bool(_FMP_KEY)
except Exception:
    _FMP_OK  = False
    def build_fmp_context(_ticker: str) -> str:   # type: ignore[misc]
        return ""

# ── FRED macro (optional — needs FRED_API_KEY) ────────────────────────────────
try:
    from fred import build_fred_context, get_series_snapshot
    _FRED_KEY = _secret("FRED_API_KEY")
    _FRED_OK  = bool(_FRED_KEY)
except Exception:
    _FRED_OK  = False
    def build_fred_context() -> str:              # type: ignore[misc]
        return ""
    def get_series_snapshot() -> dict:            # type: ignore[misc]
        return {}

# ── India macro (NSE + RBI — no key needed) ───────────────────────────────────
try:
    from india_macro import build_india_macro_context, get_dashboard_snapshot
    _INDIA_MACRO_OK = True
except Exception:
    _INDIA_MACRO_OK = False

# ── NSE/BSE stock-level data (no key needed) ──────────────────────────────────
try:
    from nse_bse import build_nse_bse_context, get_nse_index_live
    _NSE_BSE_OK = True
except Exception:
    _NSE_BSE_OK = False
    def build_nse_bse_context(_sym: str) -> str:  # type: ignore[misc]
        return ""
    def get_nse_index_live(_index: str = "NIFTY 500") -> list:  # type: ignore[misc]
        return []
    def build_india_macro_context() -> str:       # type: ignore[misc]
        return ""
    def get_dashboard_snapshot() -> dict:         # type: ignore[misc]
        return {}

# ── Screener.in + yfinance deep fundamentals (no API key needed) ─────────────
try:
    from screener_alpha import build_screener_context, build_yfinance_context
    _SCREENER_OK = True
except Exception:
    _SCREENER_OK = False
    def build_screener_context(_sym: str) -> str:       # type: ignore[misc]
        return ""
    def build_yfinance_context(_sym: str) -> str:       # type: ignore[misc]
        return ""

_TAVILY_KEY   = _secret("TAVILY_API_KEY", "")
_NEWSAPI_KEY  = _secret("NEWSAPI_KEY", "")
_EXA_KEY      = _secret("EXA_API_KEY", "")

# Optional second LLM pass that re-checks entity isolation + dates. Default OFF
# (it costs an extra call and can hit rate limits). Set SELF_CRITIQUE=on to enable.
_SELF_CRITIQUE = _secret("SELF_CRITIQUE", "off").lower() in ("1", "true", "yes", "on")

# Rate-limit management for Groq's free tier:
#  • QUICK_MODEL    — small/fast model for the per-stock pre-screen (24+ calls).
#                     Using 8b-instant here keeps the big 70B token budget free
#                     for deep dives. Plenty for a one-line catalyst headline.
#  • FALLBACK_MODEL — used for a deep dive only when the primary model is 429'd.
_QUICK_MODEL    = _secret("QUICK_MODEL",
                          "sonar" if _LLM_PROVIDER == "perplexity"
                          else "llama-3.1-8b-instant" if _LLM_PROVIDER == "groq"
                          else _OPENAI_MODEL)
_FALLBACK_MODEL = _secret("FALLBACK_MODEL",
                          "sonar" if _LLM_PROVIDER == "perplexity"
                          else "llama-3.1-8b-instant" if _LLM_PROVIDER == "groq"
                          else "")

# ── Engine import (graceful fallback to mock data) ────────────────────────────
try:
    from engine import DataEngine, ScreenerConfig
    _ENGINE_OK = True
except ImportError:
    _ENGINE_OK = False

# ── Nifty 500 universe import ─────────────────────────────────────────────────
try:
    from mailer import (NIFTY_500_TICKERS, fetch_nifty500_live,
                        fetch_all_nse_live)
    _MAILER_OK = True
except ImportError:
    _MAILER_OK = False
    NIFTY_500_TICKERS: list[str] = []

    def fetch_nifty500_live() -> list[str]:
        return []

    def fetch_all_nse_live() -> list[str]:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PAGE CONFIG  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "Nifty 500 Screener",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>
/* ── Google Font ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Strip default Streamlit chrome ──────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hide sidebar close button (prevent accidental closure) ───────────── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 1.5rem !important;
    max-width: 100% !important;
}

/* ── App background ──────────────────────────────────────────────────────── */
.stApp { background-color: #080C14; }

/* ── Ticker Tape ─────────────────────────────────────────────────────────── */
.tape-wrap {
    background: #0D1117;
    border-bottom: 1px solid #1E2535;
    overflow: hidden;
    padding: 7px 0;
    position: relative;
}
.tape-track {
    display: inline-block;
    white-space: nowrap;
    animation: scroll-left 80s linear infinite;
}
.tape-track:hover { animation-play-state: paused; }
.tape-item {
    display: inline-block;
    padding: 0 20px;
    font-size: 12px;
    font-weight: 500;
    color: #C9D1D9;
}
.tape-label { color: #6E7681; margin-right: 5px; font-size: 11px; }
.tape-price { font-weight: 700; color: #E6EDF3; }
.tape-up    { color: #00c805; }
.tape-dn    { color: #ff3b3b; }
.tape-sep   { color: #21262D; margin: 0 6px; }
@keyframes scroll-left {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

/* ── App header ─────────────────────────────────────────────────────────── */
.app-hdr {
    background: linear-gradient(180deg, #0D1B2A 0%, #080C14 100%);
    border-bottom: 1px solid #1E2535;
    padding: 14px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-hdr .brand { font-size: 20px; font-weight: 700; color: #E6EDF3; letter-spacing: -0.5px; }
.app-hdr .accent { color: #00c805; }
.app-hdr .meta   { font-size: 11px; color: #484F58; }

/* ── Index cards ─────────────────────────────────────────────────────────── */
.idx-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 14px 18px 12px;
    height: 90px;
}
.idx-card .idx-lbl  { font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
                       text-transform: uppercase; color: #6E7681; }
.idx-card .idx-val  { font-size: 21px; font-weight: 700; color: #E6EDF3;
                       margin: 4px 0 2px; line-height: 1; }
.idx-card .idx-chg  { font-size: 12px; font-weight: 600; }
.c-up   { color: #00c805 !important; }
.c-dn   { color: #ff3b3b !important; }
.c-flat { color: #8B949E !important; }

/* ── Section headers ─────────────────────────────────────────────────────── */
.sec-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #8B949E;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262D;
    margin-bottom: 14px;
}

/* ── Signal chips ────────────────────────────────────────────────────────── */
.chip-hi { background:#1A3828; color:#00c805; border:1px solid #2EA043;
           border-radius:4px; padding:2px 8px; font-size:10px; font-weight:700; }
.chip-lo { background:#3B1219; color:#ff3b3b; border:1px solid #DA3633;
           border-radius:4px; padding:2px 8px; font-size:10px; font-weight:700; }

/* ── Signal cards (replaces dataframe table) ─────────────────────────────── */
.sig-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-bottom: 18px;
}
.sig-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
}
.sig-card:hover { border-color: #388BFD; background: #1C2230; }
.sig-card.hi-card { border-left: 3px solid #00c805; }
.sig-card.lo-card { border-left: 3px solid #ff3b3b; }
.sig-card .sc-name {
    font-size: 15px;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 2px;
}
.sig-card .sc-ticker {
    font-size: 11px;
    color: #6E7681;
    letter-spacing: 0.5px;
}
.sig-card .sc-sector {
    font-size: 11px;
    color: #8B949E;
    margin-top: 3px;
}
.sig-card .sc-right {
    text-align: right;
    min-width: 160px;
}
.sig-card .sc-price {
    font-size: 17px;
    font-weight: 700;
    color: #E6EDF3;
}
.sig-card .sc-status-hi {
    font-size: 12px;
    font-weight: 600;
    color: #00c805;
    margin-top: 3px;
}
.sig-card .sc-status-lo {
    font-size: 12px;
    font-weight: 600;
    color: #ff3b3b;
    margin-top: 3px;
}
.sig-card .sc-vol {
    font-size: 11px;
    color: #8B949E;
    margin-top: 2px;
}
/* ── Why/insight strip inside signal card ────────────────────────────────── */
.sc-why {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #21262D;
}
.sc-why-lbl {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #484F58;
    margin-bottom: 6px;
}
.sc-why ul {
    margin: 0;
    padding-left: 16px;
    list-style: disc;
}
.sc-why ul li {
    font-size: 12px;
    color: #8B949E;
    line-height: 1.6;
}
.sc-why ul li strong { color: #C9D1D9; font-weight: 600; }
.sc-news-pills { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 5px; }
.sc-news-pill {
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 10px;
    color: #6E7681;
}

/* ── st.metric overrides ────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #E6EDF3 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #6E7681 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0D1117;
    border-bottom: 1px solid #21262D;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #6E7681 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #E6EDF3 !important;
    border-bottom: 2px solid #00c805 !important;
    background: transparent !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D !important;
}
[data-testid="stSidebar"] label { color: #8B949E !important; font-size: 12px !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { margin-top: 4px; }

/* Run button */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] > button {
    background: linear-gradient(135deg, #238636, #1a6b29) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 2px 8px rgba(35,134,54,0.35) !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 8px !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] iframe { border: none !important; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: #21262D !important; margin: 12px 0 !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484F58; }

/* ── Signal card containers (st.container border=True override) ──────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #161B22 !important;
    border-color: #21262D !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
    padding: 2px 4px !important;
}
/* Button fills remaining height in right column */
[data-testid="stVerticalBlockBorderWrapper"] .stButton {
    display: flex;
    flex-direction: column;
    flex: 1;
}
[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
    flex: 1 !important;
    min-height: 52px !important;
    height: 100% !important;
    width: 100% !important;
}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONSTANTS & SECTOR LENS
# ══════════════════════════════════════════════════════════════════════════════
_GREEN  = "#00c805"
_RED    = "#ff3b3b"
_YELLOW = "#F0B429"
_BLUE   = "#58A6FF"
_MUTED  = "#8B949E"
_BG     = "#0D1117"

# Sector-specific analytical frameworks injected into the analyst prompt
_SECTOR_LENS: dict[str, str] = {
    "BANK": (
        "Key metrics: NIM trajectory, GNPA/NNPA %, PCR, CASA ratio, credit cost, loan growth by segment. "
        "52W HIGH → is it NIM expansion, loan growth re-rating, or asset quality improvement? "
        "52W LOW → asset quality deterioration, credit squeeze, or RBI regulatory action?"
    ),
    "NBFC": (
        "Key metrics: AUM growth, cost of funds vs yield on assets, NIM, stage-2/stage-3 assets, CAR. "
        "Watch for RBI circular impacts, co-lending arrangements, and ALM mismatches."
    ),
    "IT": (
        "Key metrics: deal TCV (total contract value), revenue guidance in USD, attrition rate, "
        "vertical mix (BFSI/Retail/Hi-tech), EBIT margin trajectory. "
        "52W HIGH → large deal ramp-up, demand recovery, or multiple expansion? "
        "52W LOW → demand slowdown in key verticals, margin pressure, or guidance cut?"
    ),
    "PHARMA": (
        "Key metrics: USFDA approvals/483 observations, ANDA pipeline filings, US generic price erosion %, "
        "domestic formulations growth, API vs formulations revenue mix. "
        "52W HIGH → new product launch, USFDA clearance, or domestic market share gain? "
        "52W LOW → import alert, warning letter, or US price erosion acceleration?"
    ),
    "AUTO": (
        "Key metrics: wholesale volumes by segment (2W/PV/CV/EV), ASP mix shift, dealer inventory days, "
        "commodity cost (steel/aluminium/rubber), EV transition traction. "
        "52W HIGH → volume surprise, EV re-rating, or ASP improvement? "
        "52W LOW → volume miss, inventory pile-up, or EV transition risk?"
    ),
    "FMCG": (
        "Key metrics: volume growth vs price growth (mix), gross margin (palm oil/wheat/crude derivatives), "
        "rural vs urban demand split, distribution reach, market share by category. "
        "52W HIGH → volume recovery, gross margin expansion, or rural demand uptick? "
        "52W LOW → volume slowdown, commodity cost spike, or competitive intensity?"
    ),
    "METAL": (
        "Key metrics: LME commodity prices, domestic vs export realisations, EBITDA/tonne, "
        "capacity utilisation, coking coal/iron ore input costs, China demand signals. "
        "52W HIGH → commodity cycle upturn, China stimulus, or cost reduction? "
        "52W LOW → commodity price collapse, China demand weakness, or input cost spike?"
    ),
    "INFRA": (
        "Key metrics: order book size (in x of revenue), L1 pipeline, execution rate, "
        "working capital cycle days, government capex budget utilisation. "
        "52W HIGH → large order win, government capex push, or execution improvement? "
        "52W LOW → order slowdown, payment delays, or balance sheet stress?"
    ),
    "REALTY": (
        "Key metrics: pre-sales value (Rs. Cr), collections, net debt, project launches, land bank value. "
        "52W HIGH → pre-sales surge, interest rate cut benefit, or new project launch? "
        "52W LOW → demand slowdown, unsold inventory build-up, or debt concerns?"
    ),
    "POWER": (
        "Key metrics: PLF (plant load factor), tariff revisions, capacity addition pipeline, "
        "T&D losses, renewable vs thermal mix, PPA coverage. "
        "52W HIGH → tariff hike approval, capacity addition, or renewable re-rating? "
        "52W LOW → PLF decline, fuel supply issues, or regulatory headwinds?"
    ),
    "CEMENT": (
        "Key metrics: volume growth (mn tonnes), realisation/tonne, fuel costs (pet coke/coal), "
        "capacity utilisation %, regional demand mix. "
        "52W HIGH → price hike realisation, cost decline, or demand surge? "
        "52W LOW → volume pressure, fuel cost spike, or price war?"
    ),
    "TELECOM": (
        "Key metrics: ARPU trend, subscriber net adds/churn, data usage/user, spectrum costs, "
        "5G rollout capex, revenue market share. "
        "52W HIGH → ARPU expansion, 5G monetisation, or tariff hike? "
        "52W LOW → subscriber churn, spectrum cost pressure, or capex drag?"
    ),
}


def _get_sector_lens(sector: str) -> str:
    """Return sector-specific analytical framework or a generic one."""
    sector_up = sector.upper()
    for key, lens in _SECTOR_LENS.items():
        if key in sector_up:
            return lens
    return (
        "Focus on: earnings quality, revenue growth sustainability, margin trajectory, "
        "management execution vs guidance, competitive positioning vs listed peers, "
        "and whether the move is driven by fundamentals or pure sentiment/flows."
    )
_CARD   = "#161B22"

_INDEX_MAP: dict[str, str] = {
    "NIFTY 50":   "^NSEI",
    "SENSEX":     "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "INDIA VIX":  "^INDIAVIX",
}

_TAPE_MAP: dict[str, str] = {
    "^NSEI":         "NIFTY 50",
    "^BSESN":        "SENSEX",
    "RELIANCE.NS":   "RELIANCE",
    "TCS.NS":        "TCS",
    "HDFCBANK.NS":   "HDFCBANK",
    "INFY.NS":       "INFOSYS",
    "ICICIBANK.NS":  "ICICIBANK",
    "BHARTIARTL.NS": "AIRTEL",
    "TMCV.NS":  "TATA MOTORS CV",
    "TMPV.NS":  "TATA MOTORS PV",
    "WIPRO.NS":      "WIPRO",
    "BAJFINANCE.NS": "BAJAJ FIN",
    "MARUTI.NS":     "MARUTI",
}

_NIFTY_50: list[str] = [
    "RELIANCE.NS", "TCS.NS",       "HDFCBANK.NS",  "BHARTIARTL.NS","ICICIBANK.NS",
    "INFY.NS",     "SBIN.NS",      "HINDUNILVR.NS","ITC.NS",       "LT.NS",
    "BAJFINANCE.NS","KOTAKBANK.NS","HCLTECH.NS",   "MARUTI.NS",    "AXISBANK.NS",
    "ASIANPAINT.NS","SUNPHARMA.NS","TITAN.NS",     "WIPRO.NS",     "ULTRACEMCO.NS",
    "ONGC.NS",     "NTPC.NS",      "JSWSTEEL.NS",  "POWERGRID.NS", "M&M.NS",
    "BAJAJFINSV.NS","TMCV.NS",  "TMPV.NS",   "ADANIENT.NS", "ADANIPORTS.NS","TATASTEEL.NS",
    "COALINDIA.NS","NESTLEIND.NS", "DIVISLAB.NS",  "CIPLA.NS",     "DRREDDY.NS",
    "HINDALCO.NS", "TECHM.NS",     "GRASIM.NS",    "APOLLOHOSP.NS","TATACONSUM.NS",
    "INDUSINDBK.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS","BRITANNIA.NS",
    "BEL.NS",      "SHRIRAMFIN.NS","BPCL.NS",      "TRENT.NS",     "SBILIFE.NS",
]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MOCK DATA  (fallback when API rate-limited or engine unavailable)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=120, show_spinner=False)
def _mock_index_data() -> dict:
    """Live fallback via yfinance — no hardcoded prices."""
    try:
        import yfinance as yf
        _map = {
            "NIFTY 50":   "^NSEI",
            "SENSEX":     "^BSESN",
            "NIFTY BANK": "^NSEBANK",
            "INDIA VIX":  "^INDIAVIX",
        }
        result = {}
        for label, sym in _map.items():
            try:
                fi = yf.Ticker(sym).fast_info
                price = getattr(fi, "last_price", None)
                prev  = getattr(fi, "previous_close", None)
                if price:
                    chg = ((price - prev) / prev * 100) if prev else 0.0
                    pts = (price - prev) if prev else 0.0
                    result[label] = {"price": price, "chg": round(chg, 2), "pts": round(pts, 2)}
            except Exception:
                pass
        if result:
            return result
    except Exception:
        pass
    # Hard floor — only reached if yfinance import itself fails
    return {
        "NIFTY 50":   {"price": 0, "chg": 0, "pts": 0},
        "SENSEX":     {"price": 0, "chg": 0, "pts": 0},
        "NIFTY BANK": {"price": 0, "chg": 0, "pts": 0},
        "INDIA VIX":  {"price": 0, "chg": 0, "pts": 0},
    }


@st.cache_data(ttl=120, show_spinner=False)
def _mock_tape_data() -> list[dict]:
    """Live fallback via yfinance — no hardcoded prices."""
    _tickers = [
        ("^NSEI",       "NIFTY 50"),
        ("^BSESN",      "SENSEX"),
        ("RELIANCE.NS", "RELIANCE"),
        ("TCS.NS",      "TCS"),
        ("HDFCBANK.NS", "HDFCBANK"),
        ("INFY.NS",     "INFOSYS"),
        ("ICICIBANK.NS","ICICIBANK"),
        ("BHARTIARTL.NS","AIRTEL"),
        ("TMCV.NS",  "TATA MOTORS CV"),
        ("TMPV.NS",  "TATA MOTORS PV"),
        ("WIPRO.NS",    "WIPRO"),
        ("BAJFINANCE.NS","BAJAJ FIN"),
        ("MARUTI.NS",   "MARUTI"),
    ]
    result = []
    try:
        import yfinance as yf
        for sym, label in _tickers:
            try:
                fi    = yf.Ticker(sym).fast_info
                price = getattr(fi, "last_price", None)
                prev  = getattr(fi, "previous_close", None)
                if price:
                    chg = round((price - prev) / prev * 100, 2) if prev else 0.0
                    result.append({"name": label, "price": price, "chg": chg})
            except Exception:
                pass
    except Exception:
        pass
    return result or [{"name": "NIFTY 50", "price": 0, "chg": 0}]


def _mock_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns empty DataFrames — no fake/dummy stocks are ever shown.
    The caller sets is_mock=True and the UI shows an engine-unavailable warning."""
    empty_cols = [
        "ticker", "company_name", "sector", "current_price",
        "week_52_high", "week_52_low", "pct_from_high", "pct_from_low",
        "volume_surge", "pe_ratio", "ret_1m", "ret_3m", "signal",
        "why_text", "news_headlines",
    ]
    return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CACHED DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def _fetch_index_data() -> dict:
    """Fetch live index data. Falls back to mock on any failure."""
    try:
        raw = yf.download(
            list(_INDEX_MAP.values()),
            period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        result: dict = {}
        for label, sym in _INDEX_MAP.items():
            try:
                s = close[sym].dropna()
                if len(s) >= 2:
                    p, v = float(s.iloc[-1]), float(s.iloc[-2])
                    result[label] = {"price": p, "chg": (p - v) / v * 100, "pts": p - v}
            except Exception:
                pass
        return result or _mock_index_data()
    except Exception:
        return _mock_index_data()


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_tape_data() -> list[dict]:
    """Fetch tape prices. Falls back to mock on any failure."""
    try:
        raw = yf.download(
            list(_TAPE_MAP.keys()),
            period="2d", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        result: list[dict] = []
        for sym, label in _TAPE_MAP.items():
            try:
                s = close[sym].dropna()
                if len(s) >= 2:
                    p, v = float(s.iloc[-1]), float(s.iloc[-2])
                    result.append({"name": label, "price": p, "chg": (p - v) / v * 100})
            except Exception:
                pass
        return result or _mock_tape_data()
    except Exception:
        return _mock_tape_data()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch OHLCV history for the spotlight candlestick chart."""
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df if not df.empty else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SCREENER  (session-state cached)
# ══════════════════════════════════════════════════════════════════════════════
def _run_screen(tickers: list[str], p: dict) -> dict:
    """
    Run DataEngine.screen_universe() and cache in st.session_state.
    Re-runs only when `tickers` or `p` (params) change.
    Falls back to mock data when engine unavailable.
    """
    # Drop placeholder tickers before they hit yfinance (e.g. DUMMYVEDL1-4.NS
    # from Vedanta demerger) — they have no price data and produce noisy errors.
    tickers = [t for t in tickers if not t.upper().startswith("DUMMY")]
    cache_key = f"{sorted(tickers)}|{p}"

    if (
        st.session_state.get("_screen_key") == cache_key
        and "screen_results" in st.session_state
    ):
        return st.session_state["screen_results"]

    # ── Mock data path ────────────────────────────────────────────────────────
    if not _ENGINE_OK:
        highs, lows = _mock_signals()
        out = {"highs": highs, "lows": lows, "errors": [], "is_mock": True}
        st.session_state["screen_results"] = out
        st.session_state["_screen_key"]    = cache_key
        st.session_state["screen_ts"]      = datetime.now()
        return out

    # ── Real engine path ──────────────────────────────────────────────────────
    # When volume filter is OFF, set threshold to 0 so every stock passes vol_confirmed check
    _vol_threshold = p["vol_surge"] if p.get("use_vol_filter", False) else 0.0
    cfg = ScreenerConfig(
        tickers                = tickers,
        high_low_window        = p["window"],
        breakout_threshold     = p["threshold"],
        volume_window          = 20,
        volume_surge_threshold = _vol_threshold,
        rate_limit_sleep       = 0.25,
        rate_limit_batch_size  = 50,
    )
    eng = DataEngine(cfg)

    # ── yfinance batch path (primary) ────────────────────────────────────────
    # Downloads all tickers in chunks of 100 via yf.download() — works on
    # Streamlit Cloud and locally without NSE geo-block dependency.
    raw = eng.screen_universe()

    highs = pd.DataFrame(eng.format_for_display(raw["highs"]))
    lows  = pd.DataFrame(eng.format_for_display(raw["lows"]))

    out = {"highs": highs, "lows": lows, "errors": raw["errors"], "is_mock": False}
    st.session_state["screen_results"] = out
    st.session_state["_screen_key"]    = cache_key
    st.session_state["screen_ts"]      = datetime.now()
    return out


# ── Pre-computed snapshot loader ──────────────────────────────────────────────
# build_snapshot.py (run on a schedule by GitHub Actions) writes this file. The
# dashboard reads it so users see fresh results INSTANTLY instead of waiting for
# a live scan on page load.
_SNAPSHOT_PATH = Path(__file__).parent / "data" / "latest_screen.json"


@st.cache_data(ttl=120, show_spinner=False)
def _load_snapshot() -> Optional[dict]:
    """Load the pre-computed screen snapshot, or None if unavailable/pending."""
    try:
        if not _SNAPSHOT_PATH.exists():
            return None
        raw = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        # A "pending" placeholder means no real scan has run yet — show the
        # normal landing page instead of an empty results view.
        if raw.get("pending"):
            return None
        return {
            "highs":            pd.DataFrame(raw.get("highs", [])),
            "lows":             pd.DataFrame(raw.get("lows", [])),
            "errors":           raw.get("errors", []),
            "is_mock":          False,
            "from_snapshot":    True,
            "generated_at_utc": raw.get("generated_at_utc"),
            "generated_at_ist": raw.get("generated_at_ist", ""),
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def _build_candle_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + Volume + MA20 + MA50."""
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=ticker,
        increasing=dict(line_color=_GREEN, fillcolor=_GREEN),
        decreasing=dict(line_color=_RED,   fillcolor=_RED),
        showlegend=False,
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA20"], name="MA 20",
        line=dict(color=_BLUE, width=1.2), opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA50"], name="MA 50",
        line=dict(color=_YELLOW, width=1.2), opacity=0.85,
    ), row=1, col=1)

    # Volume bars
    bar_colors = [
        _GREEN if c >= o else _RED
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=bar_colors, opacity=0.55,
        showlegend=False,
    ), row=2, col=1)

    _ax = dict(gridcolor="#1A2235", color=_MUTED, showgrid=True,
               zeroline=False, linecolor="#21262D")

    fig.update_layout(
        paper_bgcolor="#0D1117",
        plot_bgcolor ="#0D1117",
        font=dict(color=_MUTED, family="Inter", size=11),
        height=440,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right",  x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        xaxis  = {**_ax, "showgrid": False},
        xaxis2 = {**_ax, "showgrid": False},
        yaxis  = {**_ax, "side": "right", "tickprefix": "₹", "tickformat": ",.0f"},
        yaxis2 = {**_ax, "side": "right", "tickformat": ".2s"},
    )
    return fig


def _build_dcf_gauge(price: float, intrinsic: float) -> go.Figure:
    """Gauge chart: intrinsic value vs current price."""
    pct   = (intrinsic - price) / price * 100
    color = _GREEN if pct > 0 else _RED
    lo    = price * 0.40
    hi    = price * 1.80

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=intrinsic,
        delta={
            "reference": price,
            "valueformat": ".2f",
            "increasing": {"color": _GREEN},
            "decreasing": {"color": _RED},
        },
        number={"prefix": "₹", "valueformat": ",.2f",
                "font": {"size": 26, "color": "#E6EDF3"}},
        gauge={
            "axis": {
                "range": [lo, hi],
                "tickformat": ",.0f",
                "tickcolor": _MUTED,
                "tickprefix": "₹",
            },
            "bar":         {"color": color, "thickness": 0.28},
            "bgcolor":     _CARD,
            "bordercolor": "#21262D",
            "steps": [
                {"range": [lo,    price], "color": "#3B1219"},
                {"range": [price, hi],    "color": "#1A3828"},
            ],
            "threshold": {
                "line":      {"color": _YELLOW, "width": 2},
                "thickness": 0.75,
                "value":     price,
            },
        },
        title={"text": "Intrinsic Value (DCF)", "font": {"color": _MUTED, "size": 11}},
    ))
    fig.update_layout(
        paper_bgcolor=_CARD,
        font=dict(color=_MUTED, family="Inter"),
        height=230,
        margin=dict(l=16, r=16, t=32, b=8),
    )
    return fig


def _build_return_bar(row: dict) -> go.Figure:
    """Horizontal bar chart of trailing returns."""
    periods = ["1M", "3M", "6M"]
    values  = [row.get("ret_1m"), row.get("ret_3m"), row.get("ret_6m")]
    vals    = [v if v is not None else 0.0 for v in values]
    colors  = [_GREEN if v >= 0 else _RED for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=periods,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in vals],
        textposition="auto",
        textfont=dict(color="#E6EDF3", size=12),
    ))
    fig.update_layout(
        paper_bgcolor=_CARD,
        plot_bgcolor =_CARD,
        font=dict(color=_MUTED, family="Inter", size=11),
        height=160,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(gridcolor="#21262D", zeroline=True,
                   zerolinecolor="#30363D", ticksuffix="%"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8.  UI COMPONENT RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
def _render_tape():
    tape = _fetch_tape_data()
    items = ""
    for d in tape * 2:          # double for seamless loop
        cls   = "tape-up" if d["chg"] >= 0 else "tape-dn"
        arrow = "▲" if d["chg"] >= 0 else "▼"
        items += (
            f'<span class="tape-item">'
            f'<span class="tape-label">{d["name"]}</span>'
            f'<span class="tape-price">₹{d["price"]:,.2f}</span> '
            f'<span class="{cls}">{arrow}{abs(d["chg"]):.2f}%</span>'
            f'</span>'
            f'<span class="tape-sep">｜</span>'
        )
    st.markdown(
        f'<div class="tape-wrap"><div class="tape-track">{items}</div></div>',
        unsafe_allow_html=True,
    )


def _render_header():
    ts = st.session_state.get("screen_ts", datetime.now()).strftime("%d %b %Y · %H:%M")
    st.markdown(
        f'<div class="app-hdr">'
        f'  <div class="brand">&#9670; Nifty<span class="accent">500</span> Screener</div>'
        f'  <div class="meta">52-Week High / Low · Volume Confirmed &nbsp;|&nbsp; '
        f'Last screen: {ts}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_index_cards():
    data = _fetch_index_data()
    cols = st.columns(len(data))
    for col, (name, d) in zip(cols, data.items()):
        cls   = "c-up" if d["chg"] >= 0 else "c-dn"
        arrow = "▲" if d["chg"] >= 0 else "▼"
        col.markdown(
            f'<div class="idx-card">'
            f'  <div class="idx-lbl">{name}</div>'
            f'  <div class="idx-val">{"₹" if name != "INDIA VIX" else ""}'
            f'{d["price"]:,.2f}</div>'
            f'  <div class="idx-chg {cls}">'
            f'    {arrow} {abs(d["chg"]):.2f}%&nbsp;&nbsp;'
            f'    ({d["pts"]:+,.2f})'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _style_signals_df(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Build a styled pandas Styler for the signals table."""
    col_map = {
        "ticker":            "Ticker",
        "company_name":      "Company",
        "sector":            "Sector",
        "current_price":     "Price (₹)",
        "week_52_high":      "52W High",
        "week_52_low":       "52W Low",
        "pct_from_high":     "% From High",
        "pct_from_low":      "% From Low",
        "volume_surge":      "Vol Surge",
        "pe_ratio":          "P/E",
        "ret_1m":            "1M Ret%",
        "ret_3m":            "3M Ret%",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    out = df[list(available.keys())].copy()
    out.columns = list(available.values())

    def _c_ret(v):
        if pd.isna(v): return f"color:{_MUTED}"
        return f"color:{_GREEN}" if v > 0 else f"color:{_RED}"

    def _fmt_pe(v):
        return f"{v:.1f}×" if pd.notna(v) else "—"

    def _fmt_pct(v):
        return f"{v:+.2f}%" if pd.notna(v) else "—"

    styled = (
        out.style
        .map(_c_ret, subset=["1M Ret%", "3M Ret%"])
        .format({
            "Price (₹)":   "₹{:,.2f}",
            "52W High":    "₹{:,.2f}",
            "52W Low":     "₹{:,.2f}",
            "% From High": "{:.2f}%",
            "% From Low":  "{:.2f}%",
            "Vol Surge":   "{:.2f}×",
            "P/E":         _fmt_pe,
            "1M Ret%":     _fmt_pct,
            "3M Ret%":     _fmt_pct,
        }, na_rep="—")
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#161B22"),
                ("color",            "#6E7681"),
                ("font-size",        "11px"),
                ("font-weight",      "600"),
                ("text-transform",   "uppercase"),
                ("letter-spacing",   "0.5px"),
                ("padding",          "9px 12px"),
                ("border-bottom",    "1px solid #21262D"),
                ("white-space",      "nowrap"),
            ]},
            {"selector": "tbody td", "props": [
                ("background-color", "#0D1117"),
                ("color",            "#C9D1D9"),
                ("font-size",        "13px"),
                ("padding",          "9px 12px"),
                ("border-bottom",    "1px solid #161B22"),
            ]},
            {"selector": "tbody tr:hover td", "props": [
                ("background-color", "#161B22"),
                ("cursor",           "pointer"),
            ]},
        ])
        .hide(axis="index")
    )
    return styled


def _safe_float(v) -> Optional[float]:
    """Return float or None — handles NaN/None safely."""
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _build_why_bullets(row: dict, is_hi: bool) -> list[str]:
    """Generate 3-5 insight bullet strings from available row data."""
    bullets: list[str] = []

    # 1. LLM/news generated reason
    why = row.get("why_text", "")
    if why and str(why).strip() and str(why).strip() not in ("", "None", "nan"):
        bullets.append(str(why).strip())

    # 2. Proximity to 52W level
    pct_hi = _safe_float(row.get("pct_from_high"))
    pct_lo = _safe_float(row.get("pct_from_low"))
    if is_hi and pct_hi is not None:
        pv = abs(pct_hi)
        if pv < 0.5:
            bullets.append(f"<strong>At 52-week high</strong> — price has already reached the annual peak with minimal gap remaining.")
        else:
            bullets.append(f"<strong>{pv:.2f}% below 52-week high</strong> — price is in breakout territory, closing in on the annual ceiling.")
    elif not is_hi and pct_lo is not None:
        pv = abs(pct_lo)
        if pv < 0.5:
            bullets.append(f"<strong>At 52-week low</strong> — price has breached the annual floor, signalling heavy selling pressure.")
        else:
            bullets.append(f"<strong>{pv:.2f}% above 52-week low</strong> — price is testing a critical support zone near the yearly bottom.")

    # 3. Volume surge
    vsurge = _safe_float(row.get("volume_surge"))
    if vsurge is not None and vsurge >= 1.2:
        bullets.append(f"<strong>Volume surge {vsurge:.1f}×</strong> above 20-day average — institutional conviction behind the move.")

    # 4. P/E context
    pe = _safe_float(row.get("pe_ratio"))
    if pe is not None and pe > 0:
        if pe > 50:
            bullets.append(f"<strong>P/E of {pe:.1f}×</strong> — elevated multiple suggests high growth expectations priced in.")
        elif pe < 15:
            bullets.append(f"<strong>P/E of {pe:.1f}×</strong> — low multiple may attract value buyers near this level.")

    # 6. Recent returns
    ret1m = _safe_float(row.get("ret_1m"))
    ret3m = _safe_float(row.get("ret_3m"))
    if ret1m is not None and ret3m is not None:
        trend = "accelerating" if abs(ret1m) > abs(ret3m / 3) else "steady"
        dir_  = "uptrend" if ret1m > 0 else "downtrend"
        bullets.append(f"<strong>Returns:</strong> 1M {ret1m:+.1f}% / 3M {ret3m:+.1f}% — {trend} {dir_} in progress.")

    return bullets[:5]  # cap at 5


def _fmt_mcap(v) -> str:
    """Indian market-cap label, e.g. '₹15,230 Cr' or '₹2.14 L Cr'. '' if unknown."""
    m = _safe_float(v)
    if not m or m <= 0:
        return ""
    cr = m / 1e7  # rupees -> crore
    if cr >= 1e5:
        return f"₹{cr/1e5:.2f} L Cr"
    return f"₹{cr:,.0f} Cr"


def _render_signals_table(df: pd.DataFrame, key: str, params: dict | None = None) -> Optional[str]:
    """Render signal cards (one per stock, with insight bullets). Returns selected ticker or None."""
    if df is None or df.empty:
        st.info("No signals meeting the current criteria.")
        return None

    # Always show the largest companies first.
    if "market_cap" in df.columns:
        df = df.copy()
        df["_mc"] = pd.to_numeric(df["market_cap"], errors="coerce")
        df = df.sort_values("_mc", ascending=False, na_position="last").drop(columns=["_mc"])

    is_hi   = key.startswith("hi")
    card_cls   = "hi-card" if is_hi else "lo-card"
    status_cls = "sc-status-hi" if is_hi else "sc-status-lo"

    for _, row in df.iterrows():
        ticker  = row.get("ticker", "")
        name    = row.get("company_name", ticker)
        sector  = row.get("sector", "") or ""
        if str(sector).lower() in ("nan", "none", "n/a"):
            sector = ""
        price   = _safe_float(row.get("current_price"))
        pct_hi  = _safe_float(row.get("pct_from_high"))
        pct_lo  = _safe_float(row.get("pct_from_low"))
        vsurge  = _safe_float(row.get("volume_surge"))
        mcap_str = _fmt_mcap(row.get("market_cap"))

        price_str = f"&#8377;{price:,.2f}" if price else "&mdash;"

        if is_hi:
            pv = abs(pct_hi) if pct_hi is not None else None
            status = ("&#9650; AT 52-WEEK HIGH" if pv is not None and pv < 0.5
                      else f"&#9650; {pv:.2f}% FROM 52W HIGH" if pv is not None
                      else "&#9650; NEAR 52-WEEK HIGH")
        else:
            pv = abs(pct_lo) if pct_lo is not None else None
            status = ("&#9660; AT 52-WEEK LOW" if pv is not None and pv < 0.5
                      else f"&#9660; {pv:.2f}% FROM 52W LOW" if pv is not None
                      else "&#9660; NEAR 52-WEEK LOW")

        vol_confirmed = row.get("vol_confirmed")
        if vsurge:
            if vol_confirmed:
                vol_str = f"&#9889; Vol surge {vsurge:.1f}&times; confirmed"
            else:
                vol_str = f"Vol {vsurge:.1f}&times; (below surge threshold)"
        else:
            vol_str = ""

        # ── Business model — 2 full sentences, with data fallback ───────────
        biz_snippet = ""
        try:
            biz_ctx = _fetch_yf_company_context(ticker, row)
            if biz_ctx:
                desc_start = biz_ctx.find("Business Description:\n")
                if desc_start >= 0:
                    desc_text = biz_ctx[desc_start + len("Business Description:\n"):].strip()
                    sentences, remaining = [], desc_text
                    for _ in range(2):
                        dot = remaining.find(". ")
                        if dot == -1:
                            if remaining and len(remaining) < 400:
                                sentences.append(remaining.strip())
                            break
                        sentences.append(remaining[:dot + 1].strip())
                        remaining = remaining[dot + 2:].strip()
                    biz_snippet = " ".join(sentences)
        except Exception:
            pass
        # Fallback: build a short description from what we already have in the row
        if not biz_snippet:
            _parts = []
            _ind = str(row.get("industry") or "").strip()
            if _ind and _ind.lower() not in ("nan", "none"):
                _parts.append(f"Operates in {_ind}")
            elif sector:
                _parts.append(f"{sector} sector company listed on NSE")
            _mc = _safe_float(row.get("market_cap"))
            if _mc and _mc > 0:
                _parts.append(f"market cap Rs{_mc/1e7:,.0f} Cr")
            _pe2 = _safe_float(row.get("pe_ratio"))
            if _pe2 and _pe2 > 0:
                _parts.append(f"P/E {_pe2:.1f}×")
            if _parts:
                biz_snippet = " · ".join(_parts)

        # ── Catalyst cache ────────────────────────────────────────────────────
        catalyst_color  = "#3FB950" if is_hi else "#F85149"
        catalyst_text   = ""
        catalyst_impact = ""
        signal          = row.get("signal", "BREAKOUT_HIGH" if is_hi else "BREAKDOWN_LOW")
        ai_cache_key    = f"quick_{ticker}_{signal}"

        cached_quick     = st.session_state.get(ai_cache_key, {})
        catalyst_date    = ""
        catalyst_stale   = False
        catalyst_breadth = ""
        catalyst_origin  = None
        catalyst_nocat   = False
        if cached_quick.get("headline"):
            catalyst_text    = cached_quick["headline"]
            catalyst_impact  = cached_quick.get("impact_pct", "")
            catalyst_date    = cached_quick.get("catalyst_date", "")
            catalyst_stale   = bool(cached_quick.get("is_stale", False))
            catalyst_breadth = cached_quick.get("move_breadth", "") or ""
            catalyst_origin  = cached_quick.get("momentum_origin")
            catalyst_nocat   = bool(cached_quick.get("no_catalyst", False))

        # ── Card: 80% details left, 20% button right ──────────────────────────
        left_border = "#00c805" if is_hi else "#ff3b3b"

        impact_chip = (
            f'<span style="font-size:10px;font-weight:700;margin-left:6px;'
            f'background:{"#1a3828" if is_hi else "#3b1219"};'
            f'color:{catalyst_color};border-radius:3px;padding:1px 6px;">'
            f'{catalyst_impact}</span>'
        ) if catalyst_impact and catalyst_impact not in ("N/A", "") else ""

        # Reframed: an older catalyst means the move is momentum continuation,
        # not "stale" — present it as the run's origin rather than a warning.
        stale_badge = (
            ' <span style="font-size:10px;font-weight:700;background:#16261a;'
            'color:#7FB950;border-radius:3px;padding:1px 6px;">↗ Momentum</span>'
        ) if catalyst_stale else ""

        date_chip = (
            f' <span style="font-size:10px;color:#6E7681;margin-left:6px;">'
            f'{catalyst_date}</span>'
        ) if catalyst_date and catalyst_date != "Not found" else ""

        if catalyst_text and catalyst_stale:
            _breadth_line = (
                f'<div style="font-size:11px;color:#8B949E;line-height:1.5;'
                f'margin-top:5px;">📊 {catalyst_breadth}</div>'
            ) if catalyst_breadth else ""
            _has_origin = (isinstance(catalyst_origin, dict)
                           and catalyst_origin.get("title") and not catalyst_nocat)
            if _has_origin:
                # A genuine earlier company catalyst is driving the continuation.
                _origin_url = catalyst_origin.get("url", "")
                if _origin_url.startswith("http"):
                    _origin_txt = (
                        f'<a href="{_origin_url}" target="_blank" rel="noopener noreferrer" '
                        f'style="color:{catalyst_color};text-decoration:none;">{catalyst_text}</a>'
                        f'{date_chip} <span style="color:#388bfd;font-size:10px;">↗ source</span>'
                    )
                else:
                    _origin_txt = f'{catalyst_text}{date_chip}'
                catalyst_html = (
                    f'<div style="margin-top:8px;">'
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
                    f'text-transform:uppercase;color:#6E7681;margin-bottom:3px;">'
                    f'Major Catalyst{stale_badge}</div>'
                    f'<div style="font-size:12px;color:#8B949E;line-height:1.5;">'
                    f'No fresh catalyst recently — earlier driver:</div>'
                    f'<div style="font-size:13px;color:{catalyst_color};font-weight:600;'
                    f'line-height:1.5;margin-top:3px;">{_origin_txt}</div>'
                    f'{_breadth_line}'
                    f'</div>'
                )
            else:
                # Honest: no recent company-specific catalyst — state it plainly,
                # no misleading "riding momentum from" or Momentum badge.
                catalyst_html = (
                    f'<div style="margin-top:8px;">'
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
                    f'text-transform:uppercase;color:#6E7681;margin-bottom:3px;">'
                    f'Major Catalyst</div>'
                    f'<div style="font-size:13px;color:#8B949E;font-weight:600;'
                    f'line-height:1.5;">{catalyst_text}</div>'
                    f'{_breadth_line}'
                    f'</div>'
                )
        elif catalyst_text:
            catalyst_html = (
                f'<div style="margin-top:8px;">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
                f'text-transform:uppercase;color:#6E7681;margin-bottom:3px;">'
                f'Major Catalyst{stale_badge}</div>'
                f'<div style="font-size:13px;color:{catalyst_color};font-weight:600;'
                f'line-height:1.5;">{catalyst_text}{impact_chip}{date_chip}</div>'
                f'</div>'
            )
        else:
            catalyst_html = ""

        with st.container(border=True):
            left_col, right_col = st.columns([3, 1])

            # ── Left: accent bar + name + biz + catalyst ──────────────────────
            with left_col:
                st.markdown(
                    f'<div style="display:flex;gap:14px;padding:6px 0 4px;">'
                    f'<div style="background:{left_border};width:3px;border-radius:3px;'
                    f'min-height:70px;flex-shrink:0;"></div>'
                    f'<div style="flex:1;">'
                    f'<div style="font-size:15px;font-weight:700;color:#E6EDF3;">{name}</div>'
                    f'<div style="font-size:11px;color:#6E7681;margin-top:2px;">'
                    f'{ticker}{"  ·  " + sector if sector else ""}</div>'
                    + (f'<div style="font-size:12px;color:#8B949E;margin-top:7px;'
                       f'line-height:1.6;font-style:italic;">{biz_snippet}</div>'
                       if biz_snippet else "")
                    + catalyst_html
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

            # ── Right: price + status + vol, then button fills rest ───────────
            with right_col:
                st.markdown(
                    f'<div style="text-align:right;padding:6px 0 10px;">'
                    f'<div style="font-size:17px;font-weight:700;color:#E6EDF3;">{price_str}</div>'
                    f'<div style="font-size:12px;font-weight:600;color:{left_border};'
                    f'margin-top:4px;">{status}</div>'
                    f'<div style="font-size:11px;color:#8B949E;margin-top:3px;">{vol_str}</div>'
                    + (f'<div style="font-size:11px;color:#C9D1D9;font-weight:600;'
                       f'margin-top:3px;">MCap {mcap_str}</div>' if mcap_str else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )
                if not catalyst_text and _AI_OK:
                    if st.button(
                        "⚡ Major Catalyst",
                        key=f"mc_{ticker}_{key}",
                        help=f"Get AI-powered major catalyst for {name}",
                        width="stretch",
                    ):
                        with st.spinner(f"Searching news for {name}…"):
                            _ret1m = _safe_float(row.get("ret_1m")) or 0
                            _ret3m = _safe_float(row.get("ret_3m")) or 0
                            _pe    = _safe_float(row.get("pe_ratio")) or 0
                            _px    = _safe_float(row.get("current_price")) or 0
                            result = _quick_catalyst_groq(
                                ticker, name, sector or "", signal,
                                _px, _ret1m, _ret3m, vsurge or 0, _pe,
                            )
                            if result.get("headline"):
                                st.session_state[ai_cache_key] = result
                                st.rerun()

                # Per-card detailed-analysis trigger (toggles the inline report).
                if st.button("🔍 Detailed Analysis", key=f"deep_btn_{key}_{ticker}",
                             help=f"Full AI analyst report for {name}",
                             width="stretch"):
                    _cur = st.session_state.get(f"_deepsel_{key}")
                    st.session_state[f"_deepsel_{key}"] = None if _cur == ticker else ticker
                    st.rerun()

        # If this card's button was clicked, render the deep-dive right here.
        if params is not None and st.session_state.get(f"_deepsel_{key}") == ticker:
            st.markdown("<hr/>", unsafe_allow_html=True)
            _render_spotlight(ticker, row.to_dict(), params)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    return st.session_state.get(f"_deepsel_{key}")


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_all_news_items(ticker: str, company: str, cutoff_days: int = 14) -> list[dict]:
    """
    Fetch + dedup news from ALL sources ONCE (cached 15 min) so the
    pre-analysis quick screen and the detailed deep-dive share a single
    network round-trip instead of fetching the same articles twice.
    Sources (merged, deduped, sorted newest-first):
      1. yfinance headlines   — free, fast, company-specific
      2. Tavily advanced      — web search with date filter (primary)
      3. NewsAPI              — date-sorted articles (NEWSAPI_KEY)
      4. Exa neural search    — semantic web search (EXA_KEY)
    Returns a list of {date, source, title, snippet} dicts (newest first).
    """
    import requests as _req
    from datetime import datetime, date, timedelta
    from concurrent.futures import ThreadPoolExecutor, as_completed

    clean        = ticker.replace(".NS", "").replace(".BO", "")
    today_str    = date.today().strftime("%d %b %Y")
    # cutoff_days is a parameter: 14d for fresh-news screening, ~90d when the
    # deep-dive looks back to find the event that STARTED a momentum run (A).
    all_items: list[dict] = []   # {date_str, source, title, snippet, url}

    # ── 1. yfinance headlines ─────────────────────────────────────────────────
    def _yf_news():
        items = []
        try:
            import yfinance as _yf
            for item in (_yf.Ticker(ticker).news or [])[:8]:
                content = item.get("content", {}) if isinstance(item, dict) else {}
                title   = (content.get("title") or item.get("title") or "")[:150]
                pub     = content.get("pubDate") or item.get("providerPublishTime") or ""
                if not title:
                    continue
                try:
                    dt = datetime.fromtimestamp(pub).strftime("%Y-%m-%d") \
                         if isinstance(pub, (int, float)) else str(pub)[:10]
                except Exception:
                    dt = ""
                _url = ((content.get("canonicalUrl") or {}).get("url")
                        or (content.get("clickThroughUrl") or {}).get("url")
                        or item.get("link") or "")
                items.append({"date": dt, "source": "yfinance", "title": title,
                              "snippet": "", "url": _url})
        except Exception:
            pass
        return items

    # ── 2. Tavily advanced search ─────────────────────────────────────────────
    def _tavily_news():
        items = []
        if not _TAVILY_KEY:
            return items
        try:
            _yr = datetime.now().year
            query = (f"{company} {clean} India NSE stock earnings results "
                     f"order win acquisition announcement filing {_yr-1} {_yr}")
            resp  = _req.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        _TAVILY_KEY,
                    "query":          query,
                    "search_depth":   "advanced",
                    "include_answer": True,
                    "max_results":    8,
                    "days":           cutoff_days,
                    "include_domains": [
                        "economictimes.indiatimes.com", "moneycontrol.com",
                        "livemint.com", "business-standard.com",
                        "bseindia.com", "nseindia.com",
                        "financialexpress.com", "ndtvprofit.com",
                        "reuters.com", "thehindubuisessline.com",
                        "cnbctv18.com", "zeebiz.com",
                    ],
                },
                timeout=12,
            )
            if resp.status_code == 200:
                data    = resp.json()
                answer  = data.get("answer", "")
                if answer:
                    items.append({"date": today_str, "source": "Tavily-summary",
                                  "title": "Web Summary", "snippet": answer[:400],
                                  "url": ""})
                for r in data.get("results", []):
                    title   = (r.get("title") or "")[:150]
                    snippet = (r.get("content") or "")[:200].replace("\n", " ")
                    pub     = (r.get("published_date") or "")[:10]
                    src     = r.get("url", "").split("/")[2].replace("www.", "") \
                              if r.get("url") else "web"
                    if title:
                        items.append({"date": pub, "source": src, "title": title,
                                      "snippet": snippet, "url": r.get("url", "")})
        except Exception:
            pass
        return items

    # ── 3. NewsAPI ────────────────────────────────────────────────────────────
    def _newsapi_news():
        items = []
        if not _NEWSAPI_KEY:
            return items
        try:
            from_date = (date.today() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
            resp = _req.get(
                "https://newsapi.org/v2/everything",
                params={
                    "apiKey":   _NEWSAPI_KEY,
                    "q":        f'"{company}" OR "{clean}" NSE India',
                    "from":     from_date,
                    "sortBy":   "publishedAt",
                    "language": "en",
                    "pageSize": 8,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                for art in resp.json().get("articles", []):
                    title   = (art.get("title") or "")[:150]
                    snippet = (art.get("description") or "")[:200]
                    pub     = (art.get("publishedAt") or "")[:10]
                    src     = (art.get("source", {}).get("name") or "NewsAPI")
                    if title and "[Removed]" not in title:
                        items.append({"date": pub, "source": src, "title": title,
                                      "snippet": snippet, "url": art.get("url", "")})
        except Exception:
            pass
        return items

    # ── 4. Exa neural search ──────────────────────────────────────────────────
    def _exa_news():
        items = []
        if not _EXA_KEY:
            return items
        try:
            from datetime import timezone as _tz, timedelta as _td
            since = (datetime.now(_tz.utc) - timedelta(days=cutoff_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            payload = {
                "query":              f"{company} {clean} NSE India stock earnings results news",
                "numResults":         8,
                "type":               "neural",
                "startPublishedDate": since,
                "contents": {
                    "highlights": {"numSentences": 2, "highlightsPerUrl": 1},
                },
            }
            r = _req.post("https://api.exa.ai/search", json=payload,
                          headers={"x-api-key": _EXA_KEY, "Content-Type": "application/json"},
                          timeout=12)
            if r.status_code == 200:
                for result in r.json().get("results", []):
                    title  = (result.get("title") or "")[:150]
                    pub    = (result.get("publishedDate") or "")[:10]
                    url    = result.get("url", "")
                    src    = url.split("/")[2].replace("www.", "") if url else "Exa"
                    hi     = result.get("highlights") or []
                    snip   = " … ".join(hi[:1]) if hi else ""
                    if title:
                        items.append({"date": pub, "source": f"Exa/{src}",
                                      "title": title, "snippet": snip, "url": url})
        except Exception:
            pass
        return items

    # ── Run all 4 in parallel ─────────────────────────────────────────────────
    # ── 5. NSE/BSE corporate announcements (often the earliest source) ────────
    def _filings_news():
        try:
            import exchange_filings as _ef
            return _ef.fetch_filings(ticker, company, days=cutoff_days)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_yf_news):      "yfinance",
            pool.submit(_tavily_news):  "tavily",
            pool.submit(_newsapi_news): "newsapi",
            pool.submit(_exa_news):     "exa",
            pool.submit(_filings_news): "filings",
        }
        for f in as_completed(futures, timeout=15):
            try:
                all_items.extend(f.result())
            except Exception:
                pass

    if not all_items:
        return []

    # ── Deduplicate by title prefix, sort newest-first ────────────────────────
    seen: set[str] = set()
    unique: list[dict] = []
    for item in sorted(all_items, key=lambda x: x["date"], reverse=True):
        key = item["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # ── Hard date floor: drop anything older than the cutoff window so stale
    # (months-old) articles can never leak into catalyst / momentum-origin
    # selection. Undated items are kept (we can't prove they're old). ──────────
    try:
        import analysis_utils as _au_dates
        _floor = date.today() - timedelta(days=cutoff_days)
        def _fresh_enough(it):
            _d = _au_dates._parse_date(it.get("date", ""))
            return _d is None or _d >= _floor
        unique = [it for it in unique if _fresh_enough(it)]
    except Exception:
        pass

    return unique


def _fetch_all_news_context(ticker: str, company: str,
                            max_items: int = 12, max_snippet: int = 400,
                            cutoff_days: int = 14) -> str:
    """
    Format the cached news items into a prompt block.

    The deep-dive uses the full slice (12 items / 200-char snippets); the
    pre-analysis quick screen passes a smaller slice (fewer items, shorter
    snippets) to keep its token footprint minimal. Both still share ONE
    cached network fetch via _fetch_all_news_items().
    """
    from datetime import date as _date
    unique = _fetch_all_news_items(ticker, company, cutoff_days)
    if not unique:
        return ""

    today_str = _date.today().strftime("%d %b %Y")
    lines = [f"[News context as of {today_str} — {len(unique)} articles from "
             f"Tavily / NewsAPI / yfinance / Exa, newest first]"]
    for item in unique[:max_items]:
        date_tag = f"[{item['date']}]" if item["date"] else ""
        src_tag  = f"[{item['source']}]"
        snippet  = f"  — {item['snippet'][:max_snippet]}" if item["snippet"] else ""
        lines.append(f"{src_tag} {date_tag} {item['title']}{snippet}")

    return "\n".join(lines)


def _momentum_origin_and_breadth(ticker: str, company: str,
                                 sector: str, signal: str) -> tuple:
    """
    Deterministic enrichment for the pre-analysis card (zero LLM tokens):
      • momentum origin — the actual news event that likely STARTED the run
        (prefers a real catalyst-type headline, else the oldest dated item).
      • breadth         — whether the move is sector-wide or company-specific.
    Reuses the cached 75-day news items + the current screen's sector data.
    """
    origin = None
    try:
        import analysis_utils as _au
        is_hi = "HIGH" in (signal or "").upper()
        clean = ticker.replace(".NS", "").replace(".BO", "")
        items = _fetch_all_news_items(ticker, company, cutoff_days=45) or []
        # Most-recent company-specific, direction-consistent catalyst — or None.
        origin = _au.pick_momentum_origin(items, company, clean, is_hi,
                                          max_age_days=45)
    except Exception:
        origin = None
    breadth = ""
    try:
        import analysis_utils as _au
        breadth = _au.sector_breadth(
            sector, ticker,
            st.session_state.get("_last_highs"),
            st.session_state.get("_last_lows"),
            "HIGH" in (signal or "").upper(),
        )
    except Exception:
        breadth = ""
    return origin, breadth


@st.cache_data(ttl=10800, show_spinner=False)   # 3h — macro/sector moves slowly
def _macro_sector_ctx(sector: str = "") -> str:
    """Cached macro + sector backdrop (World Bank + FRED + FMP + Finnhub)."""
    try:
        import macro_sector as _ms
        return _ms.macro_sector_context(sector or "")
    except Exception:
        return ""


def _quick_catalyst_groq(ticker: str, company: str, sector: str,
                          signal: str, price: float,
                          ret1m: float, ret3m: float,
                          vsurge: float, pe: float) -> dict:
    """
    Major Catalyst — single Groq call with pre-aggregated multi-source context.

    Flow:
      1. _fetch_all_news_context pulls Tavily + NewsAPI + yfinance in parallel.
      2. llama-3.3-70b-versatile receives all dated articles + the improved prompt.
      3. Returns structured JSON with headline, date, staleness flag, source.
    """
    if not _AI_OK:
        return {}
    try:
        from datetime import date as _date
        today        = _date.today().strftime("%d %B %Y")
        is_hi        = "HIGH" in signal.upper()
        direction    = "52-week high" if is_hi else "52-week low"
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

        # ── Step 1: aggregate + materiality-rank the shared news pool ─────────
        # Pull the FULL cached pool (the same data the deep-dive sees), then rank
        # by materiality and drop junk/listicles ("5 stocks to watch", brokerage
        # "buy" notes, pure price/technical pieces, circular "hits 52-week high")
        # so the model only ever picks from genuine, company-named, price-moving
        # events. Wider slice than before (top 10 / 220-char snippets) so the real
        # catalyst is actually in view.
        import analysis_utils as _au_rank
        _all_items = _fetch_all_news_items(ticker, company, cutoff_days=75) or []
        _ranked = _au_rank.rank_catalysts(
            _all_items, company, clean_ticker, is_hi, max_age_days=75, min_score=1)
        if not _ranked:
            # Nothing material/company-specific — skip the LLM and fall back to
            # the honest momentum-origin / no-catalyst path below.
            raise ValueError("no_material")
        from datetime import date as _date2
        _today_s = _date2.today().strftime("%d %b %Y")
        _lines = [f"[Ranked candidate catalysts as of {_today_s} — most material first]"]
        for _it in _ranked[:10]:
            _dt  = f"[{_it.get('date','')}]" if _it.get("date") else ""
            _src = f"[{_it.get('source','')}]"
            _sn  = f" — {_it.get('snippet','')[:220]}" if _it.get("snippet") else ""
            _lines.append(f"{_src} {_dt} {_it.get('title','')}{_sn}")
        news_context = "\n".join(_lines)

        _macro = _macro_sector_ctx(sector)
        _macro_block = (
            f"MACRO & SECTOR BACKDROP (judge sector-wide vs company-specific and "
            f"global risk-on/off — NOT the company's own catalyst):\n{_macro}\n\n"
        ) if _macro else ""

        # ── Step 2: single LLM call with full context ─────────────────────────
        client = _OpenAI(api_key=_OPENAI_KEY, base_url=_llm_base_url())

        system_msg = f"""You are a senior equity analyst at a top-tier Indian brokerage. Today is {today}.

EVIDENCE PROTOCOL
Use ONLY the NEWS CONTEXT in the user message as evidence — you cannot browse. Within it, prioritize BSE/NSE exchange filings and tier-1 outlets (Reuters/ET/Mint/MoneyControl) over weaker sources. If the freshest item in the context is older than 20 days, set is_stale=true and do NOT present it as current. A MACRO & SECTOR BACKDROP block may follow the news — use it ONLY to judge sector-wide vs company-specific moves and global risk-on/off tone, never as the company's own catalyst; confirm nothing material broke in the last 24-48 hours.

ENTITY ISOLATION RULES (non-negotiable)
- Report ONLY catalysts that explicitly name "{company}" or "{clean_ticker}" in the source text.
- Do NOT attribute news from the parent group, subsidiaries, JVs, associate companies, or sector peers — even if they share a brand, promoter, or conglomerate name.
- If a headline refers to the group/holding entity rather than {clean_ticker} specifically → exclude it entirely.
- If you cannot directly verify a catalyst names {company} → exclude it.
- Do NOT describe price movement or volume as the catalyst — find the ROOT CAUSE business event.
- Do NOT use the ticker symbol in the headline — use the company name.
- DATE = EVENT DATE: use the date the business event actually occurred (the earnings/board-meeting/order date), NOT the date a news article was published or re-published. If only a recent re-report exists for an older event, use the original event date.
- Sibling companies of the same business house (same promoter/brand, different listed entity and ticker) are DIFFERENT companies — exclude them.

OUTPUT
Return ONLY valid JSON — no markdown, no extra text:
{{"headline": "DD Mon YYYY: specific ROOT CAUSE catalyst under 90 chars — must include a figure", "impact_pct": "+X% or N/A", "catalyst_date": "DD Mon YYYY or Not found", "is_stale": false, "source": "publication name"}}
Set is_stale=true if the most recent article you can find is older than 20 days from today ({today}). If no verified catalyst names {company} directly, set headline to "{company} — no verified catalyst found" and is_stale=true."""

        ret1m_str = f"{ret1m:+.1f}%" if ret1m else "N/A"
        ret3m_str = f"{ret3m:+.1f}%" if ret3m else "N/A"
        vsurge_str = f"{vsurge:.1f}x" if vsurge else "N/A"
        pe_str = f"{pe:.1f}x" if pe else "N/A"

        user_msg = (
            f"{company} (NSE: {clean_ticker}) has just hit its {direction} today ({today}). "
            f"Price: Rs{price:,.0f} | 1M: {ret1m_str} | 3M: {ret3m_str} | "
            f"VolSurge: {vsurge_str} | P/E: {pe_str}\n"
            f"Sector: {sector}\n\n"
            f"NEWS CONTEXT (Tavily + NewsAPI + yfinance — newest first):\n{news_context}\n\n"
            f"{_macro_block}"
            f"Task: Identify the SPECIFIC, DATED ROOT CAUSE business event that triggered this {direction}.\n"
            f"- Date the catalyst by when the EVENT actually happened (e.g. the earnings/board-meeting "
            f"date), NOT when an article about it was published. A recent article re-reporting an older "
            f"result must carry the older EVENT date.\n"
            f"- If the most recent genuinely NEW event is older than 20 days, set is_stale=true.\n"
            f"- If conflicting reports exist, pick the most credible and note the conflict in the headline.\n"
            f"- If no news explicitly names {company}, set headline to "
            f"\"{company} — no verified catalyst found\" and is_stale=true."
        )

        _quick_kwargs: dict = {"temperature": 0.1, "max_tokens": 300,
                               "extra_body": _pplx_extra_body("low")}
        # Perplexity supports json_schema only (not json_object); the prompt already
        # mandates raw JSON and we extract it defensively below.
        if _LLM_PROVIDER != "perplexity":
            _quick_kwargs["response_format"] = {"type": "json_object"}
        resp   = client.chat.completions.create(
            model=_QUICK_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            **_quick_kwargs,
        )
        _track_llm_call(_QUICK_MODEL, resp, "low")
        _raw_quick = resp.choices[0].message.content.strip()
        # Sonar sometimes wraps JSON in prose/citation markers — extract the object.
        if not _raw_quick.startswith("{"):
            import re as _re_q
            _mq = _re_q.search(r"\{[\s\S]*\}", _raw_quick)
            if _mq:
                _raw_quick = _mq.group(0)
        result = json.loads(_raw_quick)

        # ── Validate — reject bad fallback headlines ──────────────────────────
        headline = result.get("headline", "")
        _bad = (
            not headline
            or clean_ticker.lower() in headline.lower()
            or ticker.lower() in headline.lower()
            or "52-week" in headline.lower()
            or "52 week" in headline.lower()
            or headline.lower().strip().endswith("high")
            or headline.lower().strip().endswith("low")
            or _au_rank.is_junk_headline(headline)   # reject listicles/price-only
        )
        if _bad:
            raise ValueError("bad_headline")

        # Snap the displayed date to the ACTUAL event date (earliest report /
        # exchange-filing date), not the article's publish/re-report date.
        try:
            _ev = _au_rank.resolve_event_date(result.get("headline", ""), _ranked)
            if _ev:
                result["catalyst_date"] = _ev
        except Exception:
            pass

        # Deterministic enrichment: sector-vs-individual + the news that started
        # the run. If the model couldn't name a catalyst, use the origin headline.
        _o, _b = _momentum_origin_and_breadth(ticker, company, sector, signal)
        result["momentum_origin"] = _o
        result["move_breadth"]    = _b
        if ("no verified catalyst" in headline.lower()
                or "no recent news" in headline.lower()):
            if _o:
                result["headline"]      = f"{_o['date']}: {_o['title']}"[:120]
                result["catalyst_date"] = _o.get("date", result.get("catalyst_date", ""))
            else:
                # Honest: no recent company-specific catalyst exists.
                import analysis_utils as _au2
                result["headline"]    = _au2.no_catalyst_summary(
                    sector, "HIGH" in signal.upper(), _b)
                result["catalyst_date"] = ""
                result["no_catalyst"]   = True
            result["is_stale"] = True
        return result

    except Exception:
        # ── Heuristic fallback — data-driven, never hallucinated ──────────────
        _parts = []
        if ret3m and abs(ret3m) >= 5:
            _parts.append(f"{ret3m:+.1f}% over 3 months")
        elif ret1m and abs(ret1m) >= 3:
            _parts.append(f"{ret1m:+.1f}% over 1 month")
        if vsurge and vsurge >= 1.5:
            _parts.append(f"{vsurge:.1f}× volume surge")
        if pe and pe > 0:
            _parts.append(f"P/E {pe:.0f}×")
        if sector:
            _parts.append(f"{sector} sector momentum")
        _o, _b = _momentum_origin_and_breadth(ticker, company, sector, signal)
        if _o:
            # A genuine company-specific catalyst started the run — show it.
            return {
                "headline":        f"{_o['date']}: {_o['title']}"[:120],
                "impact_pct":      "N/A",
                "catalyst_date":   _o.get("date", "Not found"),
                "is_stale":        True,
                "source":          "Momentum origin",
                "momentum_origin": _o,
                "move_breadth":    _b,
            }
        try:
            import analysis_utils as _au3
            _honest = _au3.no_catalyst_summary(sector, "HIGH" in signal.upper(), _b)
        except Exception:
            _honest = (f"{company}: " + " · ".join(_parts)) if _parts \
                      else f"{company} — no recent company-specific catalyst found"
        return {
            "headline":        _honest[:120],
            "impact_pct":      "N/A",
            "catalyst_date":   "Not found",
            "is_stale":        True,
            "source":          "No company-specific catalyst",
            "momentum_origin": None,
            "move_breadth":    _b,
            "no_catalyst":     True,
        }


def _ai_deep_dive(ticker: str, company: str, sector: str, signal: str,
                  price: float, high52: float, low52: float,
                  pct_from_high: float, pct_from_low: float,
                  vsurge: float, pe: float,
                  news_headlines: str,
                  ret_1m: Optional[float] = None,
                  ret_3m: Optional[float] = None,
                  ret_6m: Optional[float] = None,
                  ret_1y: Optional[float] = None,
                  price_series: str = "",
                  fmp_context: str = "",
                  fred_context: str = "",
                  india_macro_context: str = "",
                  nse_bse_context: str = "",
                  google_news_context: str = "",
                  screener_context: str = "",
                  alpha_vantage_context: str = "",
                  tavily_context: str = "",
                  exa_context: str = "",
                  upgrades_context: str = "",
                  earnings_surprise_context: str = "",
                  bse_announcements_context: str = "",
                  calendar_context: str = "",
                  peer_context: str = "",
                  pre_catalyst_context: str = "",
                  fresh_news_context: str = "") -> dict:
    """
    Call AI with a senior equity analyst prompt focused on WHY a stock hit its 52W extreme.
    Results cached in st.session_state — only successful calls are cached.
    Failed calls are never cached so retry always hits the API fresh.
    """
    cache_key = f"ai_{ticker}_{signal}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if not _AI_OK:
        return _fallback_deep_dive(ticker, company, sector, signal,
                                   price, high52, low52, pct_from_high,
                                   pct_from_low, vsurge, pe)

    is_hi   = "HIGH" in signal.upper()
    extreme = "HIGH" if is_hi else "LOW"
    level   = "52-week high" if is_hi else "52-week low"

    def _trim(text: str, max_chars: int) -> str:
        return text[:max_chars] + "\n[...truncated]" if len(text) > max_chars else text

    # Exa is PRIMARY (neural search, highest quality) — Tavily is supplementary fallback only
    combined_news = ""
    if exa_context:
        combined_news = _trim(exa_context, 5500)          # Exa gets the lion's share
        if tavily_context:
            combined_news += "\n\n--- Supplementary (Tavily) ---\n" + _trim(tavily_context, 1000)
        if google_news_context:
            combined_news += "\n\n--- Google News ---\n" + _trim(google_news_context, 400)
    elif tavily_context:                                  # Tavily only if Exa unavailable
        combined_news = _trim(tavily_context, 3500)
        if google_news_context:
            combined_news += "\n\n--- Google News ---\n" + _trim(google_news_context, 800)
    elif google_news_context:
        combined_news = _trim(google_news_context, 3000)

    # fresh_news_context = same parallel pipeline (Exa+Tavily+NewsAPI+yfinance) as
    # the pre-analysis quick screen — always today's data, no cache lag.
    # It goes FIRST so the AI sees the most recent events before anything else.
    fresh_block      = f"\n\n=== TODAY'S LIVE NEWS (same source as initial screen) ===\n{_trim(fresh_news_context, 3000)}" \
                       if fresh_news_context else ""
    fmp_block        = f"\n\n{_trim(fmp_context, 2000)}"                   if fmp_context               else ""
    fred_block       = f"\n\n{_trim(fred_context, 800)}"                   if fred_context              else ""
    india_block      = f"\n\n{_trim(india_macro_context, 800)}"            if india_macro_context       else ""
    nse_bse_block    = f"\n\n{_trim(nse_bse_context, 1500)}"               if nse_bse_context           else ""
    screener_block   = f"\n\n{_trim(screener_context, 2000)}"              if screener_context          else ""
    av_block         = f"\n\n{_trim(alpha_vantage_context, 1200)}"         if alpha_vantage_context     else ""
    news_block       = f"\n\n{combined_news}"                              if combined_news             else ""
    upgrades_block   = f"\n\n{_trim(upgrades_context, 800)}"               if upgrades_context          else ""
    surprise_block   = f"\n\n{_trim(earnings_surprise_context, 600)}"      if earnings_surprise_context else ""
    bse_block        = f"\n\n{_trim(bse_announcements_context, 800)}"      if bse_announcements_context else ""
    calendar_block   = f"\n\n{_trim(calendar_context, 200)}"               if calendar_context          else ""
    peer_block       = f"\n\n{_trim(peer_context, 600)}"                   if peer_context              else ""
    all_context      = (fresh_block + news_block + screener_block + av_block + fmp_block
                        + upgrades_block + surprise_block + bse_block
                        + fred_block + india_block + nse_bse_block
                        + calendar_block + peer_block)

    _action_word = "high" if is_hi else "low"
    _risk_label  = "rally" if is_hi else "recovery"

    _today_str = datetime.now().strftime("%B %d, %Y")

    _clean_ticker = ticker.replace(".NS", "").replace(".BO", "")

    macro_ctx = _macro_sector_ctx(sector)
    system_prompt = f"""You are a senior equity analyst at a top-tier Indian brokerage. Today is {_today_str}.

GLOBAL ENTITY ISOLATION RULES (apply to every sentence you write)
- Analyze ONLY {company} (NSE: {_clean_ticker}).
- Every catalyst, data point, and claim must explicitly name "{company}" or "{_clean_ticker}" in its source.
- Do NOT attribute news from the parent group, subsidiaries, JVs, associate companies, or sector peers — even if they share a brand, promoter, or conglomerate name.
- Sibling companies of the same business house are DIFFERENT entities with different tickers — exclude them entirely (e.g. for an AMC, exclude the group's fashion, capital, cement or metals arms). Same promoter/brand ≠ same company.
- Group-level items must be excluded entirely.
- EVENT DATE, NOT PUBLISH DATE: date every catalyst, timeline entry and the primary driver by when the event actually OCCURRED. For earnings/results, use the official report date from the VERIFIED FUNDAMENTAL DATA / earnings-surprises block above — NEVER the publication date of a later article that merely references it. A June article about April results is an April event.
- Do NOT describe price movement or volume as a catalyst — only ROOT CAUSE business events.
- COMPANY NAME, NOT TICKER: in EVERY prose field (business_model, summary, catalyst headlines/details, timeline events) refer to the company as "{company}". NEVER write the ticker ("{ticker}" or "{_clean_ticker}") as if it were the company's name (e.g. write "AIA Engineering", never "AIAENG.NS designs…").
- MOMENTUM CONTINUATION: if the root-cause catalyst occurred more than ~2 weeks ago, the move TODAY is the CONTINUED re-rating / momentum from it — not a fresh event. Say so explicitly: name the original event and its date, then explain that sustained follow-through buying (for a high) or selling (for a low), and the absence of offsetting news, is carrying {company} to this 52-week {_action_word} now. Never imply a months-old event happened today.
- Write with conviction and specific numbers. No hedging language: "may", "could", "might".
- DIRECTION MATCH: This stock hit a 52-week {_action_word}. For a HIGH the PRIMARY catalyst MUST be a positive/supportive event (beat, order win, upgrade, capex); for a LOW it MUST be a negative event. An event of the opposite sign can only appear as a secondary or conflicting catalyst — NEVER as the primary driver. A profit decline can never be the primary driver of a record high.
- SOURCE URLs: Every article in the context ends with 'URL: <link>'. When you cite a catalyst, copy that exact URL verbatim into its "url" field. If you have no URL from the context for a claim, set "url" to "". NEVER invent, guess, or build a URL.
- Return ONLY valid JSON. No markdown fences, no preamble, no trailing text."""

    user_prompt = f"""Write a deep-dive Stock Catalyst Analysis for {company} ({ticker}). Today is {_today_str}.

MARKET DATA
  Current Price : Rs{price:,.2f}
  52W High      : Rs{high52:,.2f}  ({abs(pct_from_high):.1f}% away)
  52W Low       : Rs{low52:,.2f}   ({abs(pct_from_low):.1f}% away)
  Volume Surge  : {vsurge:.1f}x 20-day average
  P/E           : {pe if pe else "N/A"}x
  Returns       : 1M={f"{ret_1m:+.1f}%" if ret_1m is not None else "N/A"}  3M={f"{ret_3m:+.1f}%" if ret_3m is not None else "N/A"}  6M={f"{ret_6m:+.1f}%" if ret_6m is not None else "N/A"}  1Y={f"{ret_1y:+.1f}%" if ret_1y is not None else "N/A"}
  Sector        : {sector}
{f"  Recent closes (oldest→newest): {price_series}" if price_series else ""}

{f"══ CONFIRMED PRIMARY CATALYST (from initial screen — must be catalyst #1) ══{chr(10)}{pre_catalyst_context}{chr(10)}═══════════════════════════════════════════════════════════════════{chr(10)}" if pre_catalyst_context else ""}
{f"== MACRO & SECTOR BACKDROP (frame sector-wide vs company-specific and global risk-on/off — background only, NOT a company catalyst) =={chr(10)}{macro_ctx}{chr(10)}" if macro_ctx else ""}
NEWS & RESEARCH CONTEXT (Exa neural search + Tavily + NewsAPI + BSE filings — use these as primary evidence):
{all_context if all_context.strip() else "  No additional context available — use your training knowledge."}

══ YOUR TASK ══
STEP 1 — FRESHNESS CHECK State the date of the most recent news item in the context above. If it is older than 20 days from today ({_today_str}), note this in the `summary` field — do NOT present stale news as current.
STEP 2 — PRIMARY ROOT CAUSE Identify the single business event most directly responsible for {company} reaching a 52-week {_action_word}. This must be a specific, dated event with figures (earnings beat, order win, regulatory ruling, capex announcement, etc.) — NOT a market-wide or sector move unless it names {company} directly. {"The confirmed primary catalyst above IS catalyst #1. Expand it with deeper evidence from the news context." if pre_catalyst_context else "Build the primary_catalyst field around the root cause you found."} If that root-cause event is older than ~2 weeks, treat it as the MOMENTUM ORIGIN: keep it as primary_catalyst, but make BOTH primary_catalyst.detail AND summary explicitly say that {company}'s 52-week {_action_word} TODAY is the continued momentum / ongoing re-rating from that event (sustained follow-through, no offsetting news) — quantify the run since then (e.g. "up X% since the DD Mon result") and do NOT imply it happened today.
STEP 3 — SUPPORTING CATALYSTS List 4–6 supporting catalysts. Mix positive/negative/neutral. Each must: (a) name {company} or {_clean_ticker} explicitly, (b) have a dated headline with a specific figure, (c) explain WHY it moved the stock. If fewer than 4 verified catalysts exist, list what you found and add a "neutral" catalyst noting thin news flow.
STEP 4 — CONFLICTING SIGNALS If any contradictory reports, analyst downgrades, or unverified rumors exist → include them as a "neutral" catalyst type with detail starting "⚠️ CONFLICTING: ".
STEP 5 — CHRONOLOGICAL TIMELINE 3–5 events in strict date order. Every event must involve {company} directly. Never include events from other companies, peers, or macro unless they explicitly name {company}.
STEP 6 — WATCH NEXT Only future events after {_today_str}. Include earnings date, AGM, regulatory decisions, analyst days if known.

IMPORTANT: Return ONLY valid JSON. No markdown fences, no preamble.

{{
  "sentiment": "Bullish | Bearish | Neutral",
  "confidence": "High | Medium | Low",
  "business_model": "2-3 sentences: what {company} does, core revenue streams, competitive moat.",
  "primary_catalyst": {{
    "headline": "One punchy sentence naming the ROOT CAUSE event with a specific figure and date — e.g. 'DD Mon YYYY: [company] Q4 revenue surges 20% to Rs X,XXX Cr'",
    "impact_pct": "Estimated % of total price move driven by this single catalyst (e.g. '+18%')",
    "detail": "3-4 sentences of evidence: exact figures, dates, and WHY this moved the stock. Be specific."
  }},
  "summary": "3-4 sentence analyst note: root cause verdict, key risk, and what triggered this extreme. Mention the primary catalyst explicitly.",
  "catalyst_timeline": [
    {{"date": "Mon YYYY", "event": "Specific one-line event naming {company} directly — NO other companies", "impact": "price/sentiment effect"}}
  ],
  "catalysts": [
    {{
      "type": "positive | negative | neutral",
      "headline": "Dated headline with a specific figure — e.g. 'DD Mon YYYY: [event with specific number]'",
      "detail": "2-3 sentences: what happened, exact figures (Rs Cr, %, dates), and why it moved the stock.",
      "impact_pct": "e.g. '+8%' or '-5%'",
      "date": "Month DD, YYYY",
      "source": "Publication name",
      "url": "Exact source URL copied verbatim from the context 'URL:' line, or '' if none — never invent one"
    }}
  ],
  "watch_next": [
    {{"event": "Future event name", "date": "Date after {_today_str}", "implication": "Bull/bear implication"}}
  ],
  "peer_context": "1-2 sentences: is this move company-specific or sector-wide? How does {company} compare to closest peers?",
  "news_freshness": "FRESH (most recent: DD Mon YYYY) | STALE (most recent: DD Mon YYYY, older than 20 days) | THIN (fewer than 3 verified sources)",
  "conflicting_signals": "One sentence summarizing any contradictory reports, or 'None found' if clean.",
  "sources": ["Publication Name, Month DD YYYY"]
}}

RULES (violations will make the output useless):
- Every catalyst must name {company} or {_clean_ticker} — no group-level or peer news
- Do NOT use price/volume movement as a catalyst — only ROOT CAUSE business events
- The primary catalyst's sign MUST match the move: positive event for a 52-week high, negative event for a low
- Dates are EVENT dates, not article publish dates — anchor earnings dates to the VERIFIED FUNDAMENTAL DATA block
- Each catalyst's "url" must be copied verbatim from a context 'URL:' line, or left "" — never fabricated
- primary_catalyst headline must contain a specific date and figure
- impact_pct values across all catalysts should roughly sum to the total move from the opposite 52W extreme
- catalyst_timeline: 3-5 events in strict chronological order — EVERY event must involve {company} directly; never include events from other companies, peers, or macro events unless they directly name {company}
- watch_next: ONLY events after {_today_str} — no past events
- Write with conviction and specific numbers — no hedging language like "may", "could", "might"
- sources: list only unique publications actually referenced"""

    last_err = ""
    import time as _time
    client = _OpenAI(
        api_key=_OPENAI_KEY,
        base_url=_llm_base_url(),
    )
    kwargs: dict = {"temperature": 0.1, "max_tokens": 2000}
    # Perplexity supports json_schema only (not json_object), so skip response_format
    # for it entirely and rely on the prompt + JSON extraction below. Same for local
    # reasoning models that emit JSON inside prose.
    _NO_JSON_MODE_MODELS = ("deepseek", "qwq", "r1", "sonar")
    if (_LLM_PROVIDER != "perplexity"
            and not any(x in _OPENAI_MODEL.lower() for x in _NO_JSON_MODE_MODELS)):
        kwargs["response_format"] = {"type": "json_object"}
    # Deep dive gets a richer (medium) search context; no-op for OpenAI/Groq.
    _deep_extra = _pplx_extra_body("medium")
    if _deep_extra:
        kwargs["extra_body"] = _deep_extra

    # Retry with backoff on 429; try the primary model first, then a
    # higher-rate-limit fallback model before giving up.
    import re as _re
    _try_models = [_OPENAI_MODEL]
    if _FALLBACK_MODEL and _FALLBACK_MODEL != _OPENAI_MODEL:
        _try_models.append(_FALLBACK_MODEL)
    _plan = [(m, r) for m in _try_models for r in range(2)]   # 2 rounds per model
    for _attempt, (_use_model, _round) in enumerate(_plan):
        try:
            response = client.chat.completions.create(
                model=_use_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                **kwargs,
            )
            _track_llm_call(_use_model, response, "medium")
            _sonar_cites = _pplx_citations(response)   # grounded source-of-truth URLs
            raw = response.choices[0].message.content.strip()
            # Strip DeepSeek R1 thinking tags
            if "<think>" in raw:
                raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            # Strip markdown code fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rstrip("`")
            # Extract JSON object if surrounded by text
            if not raw.strip().startswith("{"):
                m = _re.search(r"\{[\s\S]*\}", raw)
                if m:
                    raw = m.group(0)
            result = json.loads(raw.strip())

            # ── Anti-hallucination: keep a catalyst URL only if it appears
            #    verbatim in the context we fed the model — OR, for Perplexity,
            #    if Sonar returned it as a grounded citation. Sonar's citations
            #    are retrieved+verified server-side, so they're trustworthy even
            #    though they were never in `all_context`. (The regex check exists
            #    because Groq can't verify its own URLs; Sonar can.) ────────────
            _valid_urls = set(_re.findall(r'https?://[^\s\]\)"<>]+', all_context))
            _valid_urls.update(_sonar_cites)
            def _clean_url(u: str) -> str:
                u = (u or "").strip()
                if u.startswith("http") and (u in all_context or u in _valid_urls):
                    return u
                return ""
            # Expose Sonar's grounded citations on the result for display/sourcing.
            if _sonar_cites:
                result["citations"] = _sonar_cites
            for _cat in result.get("catalysts", []) or []:
                if isinstance(_cat, dict):
                    _cat["url"] = _clean_url(_cat.get("url", ""))
            if isinstance(result.get("primary_catalyst"), dict):
                _pc = result["primary_catalyst"]
                _pc["url"] = _clean_url(_pc.get("url", ""))

            # ── Deterministic post-processing: verify dates vs FMP filings,
            #    drop foreign-entity catalysts, recompute confidence ──────────
            try:
                import analysis_utils as _au
                result = _au.post_process(result, company, _clean_ticker, fmp_context)

                # ── Conditional self-critique pass (#6) — only when the
                #    deterministic checks found problems AND it's enabled.
                #    Default OFF to avoid extra latency / 429 rate limits. ────
                if _SELF_CRITIQUE and _au.needs_critique(result):
                    import json as _json2
                    _crit_user = (
                        "The draft JSON below FAILED automated checks (wrong dates "
                        "and/or unrelated companies). Fix ONLY those problems:\n"
                        f"- Use the real EVENT dates from the verified data for {company}.\n"
                        f"- Remove any catalyst/timeline item not about {company} "
                        f"(NSE: {_clean_ticker}).\n"
                        "- Keep the exact same JSON schema and all other content.\n\n"
                        "VERIFIED DATA:\n" + (fmp_context or "")[:1500] + "\n\n"
                        "DRAFT JSON:\n" + _json2.dumps(result)[:6000] + "\n\n"
                        "Return ONLY the corrected JSON."
                    )
                    try:
                        _cr = client.chat.completions.create(
                            model=_OPENAI_MODEL,
                            messages=[{"role": "system", "content": system_prompt},
                                      {"role": "user",   "content": _crit_user}],
                            **kwargs,
                        )
                        _track_llm_call(_OPENAI_MODEL, _cr, "medium")
                        _raw2 = _cr.choices[0].message.content.strip()
                        _m2 = _re.search(r"\{[\s\S]*\}", _raw2)
                        if _m2:
                            _fixed = _json2.loads(_m2.group(0))
                            if isinstance(_fixed, dict) and _fixed.get("catalysts") is not None:
                                for _c in _fixed.get("catalysts", []) or []:
                                    if isinstance(_c, dict):
                                        _c["url"] = _clean_url(_c.get("url", ""))
                                result = _au.post_process(_fixed, company,
                                                          _clean_ticker, fmp_context)
                    except Exception:
                        pass
            except Exception:
                pass

            st.session_state[cache_key] = result
            return result
        except Exception as e:
            last_err = str(e)
            if "429" in last_err or "rate limit" in last_err.lower():
                # Honor Groq's "Please try again in 12.3s" hint when present.
                _m = _re.search(r"try again in ([\d.]+)\s*s", last_err, _re.I)
                wait = min(float(_m.group(1)) + 1.0, 25.0) if _m else 4 * (2 ** _round)
                _time.sleep(wait)
            else:
                break

    return _fallback_deep_dive(ticker, company, sector, signal,
                               price, high52, low52, pct_from_high,
                               pct_from_low, vsurge, pe,
                               error=last_err)


def _fallback_deep_dive(ticker, company, sector, signal,
                        price, high52, low52, pct_from_high, pct_from_low,
                        vsurge, pe, error="") -> dict:
    is_hi = "HIGH" in signal.upper()
    level = "52-week high" if is_hi else "52-week low"
    _is_rate = bool(error) and ("429" in error or "rate limit" in error.lower())
    _sector_txt = f"the {sector} sector" if sector else "an undisclosed sector"
    if _is_rate:
        err_note = f"AI provider rate limit reached ({_LLM_PROVIDER})."
        action   = "Wait ~30–60s, then click Retry Analysis. Tip: set FALLBACK_MODEL, screen a smaller universe, or raise your provider tier."
    elif error:
        err_note = f"AI error: {error[:300]}"
        action   = "Click Retry Analysis to try again."
    else:
        err_note = "AI API key not configured"
        action   = "Set LLM_PROVIDER and the matching key (OPENAI_API_KEY / GROQ_API_KEY / PERPLEXITY_API_KEY) in secrets."
    return {
        "sentiment":     "Neutral",
        "confidence":    "Low",
        "ai_error":      (error or "")[:600],   # raw provider error for the UI
        "ai_provider":   _LLM_PROVIDER,
        "ai_model":      _OPENAI_MODEL,
        "business_model": f"{company} operates in {_sector_txt}. Detailed description unavailable — {err_note}",
        "primary_catalyst": {
            "headline": f"{company} is at its {level}",
            "impact_pct": "N/A",
            "detail": f"AI ANALYSIS UNAVAILABLE — {err_note} {action}",
        },
        "summary": f"AI ANALYSIS UNAVAILABLE — {company} ({ticker}) is near its {level} at Rs{price:,.2f} (volume surge {vsurge:.1f}x). {err_note} {action}",
        "catalyst_timeline": [],
        "catalysts": [
            {
                "type":       "neutral",
                "headline":   f"{company} is near its {level} — AI analysis unavailable",
                "detail":     f"{err_note} {action}",
                "impact_pct": "N/A",
                "date":       "",
                "source":     "System",
            }
        ],
        "watch_next": [],
        "peer_context": "",
        "news_freshness": "STALE",
        "conflicting_signals": "None found",
        "sources": [],
    }


def _parse_rss_feed(url: str, company: str, ticker: str,
                    source_name: str, max_items: int = 8) -> list[dict]:
    """
    Parse an RSS feed and return articles mentioning the company/ticker.
    Filters by company name or ticker symbol appearing in title or summary.
    """
    # Generic words that appear in virtually every Indian financial headline —
    # using them as a match criterion produces false positives for any stock.
    _STOPWORDS = {
        "india", "indian", "market", "stock", "share", "equity", "fund",
        "limited", "ltd", "corp", "company", "group", "finance", "bank",
        "financial", "nifty", "sensex", "bse", "nse", "sebi",
    }

    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").replace("-", "").lower()

    # Keep words that are ≥4 chars AND not generic stopwords.
    # Also keep the raw ticker stem as a match token.
    company_words = [
        w.lower() for w in company.split()
        if len(w) >= 4 and w.lower() not in _STOPWORDS
    ]
    # If all words were filtered (e.g. "NAM India"), fall back to the ticker stem.
    if not company_words:
        company_words = [clean_ticker] if len(clean_ticker) >= 3 else []

    def _mentions(text: str) -> bool:
        t = text.lower()
        # Ticker stem must appear, OR a non-generic company word must appear in the TITLE.
        return clean_ticker in t or any(w in t for w in company_words)

    from datetime import timezone as _tz, timedelta as _td
    _cutoff = datetime.now(_tz.utc) - _td(days=90)   # ignore articles older than 90 days

    results = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip()
            if not _mentions(title) and not _mentions(summary):
                continue
            pub = ""
            pub_dt_raw = entry.get("published", "")
            article_dt = None
            try:
                article_dt = parsedate_to_datetime(pub_dt_raw)
                # Make timezone-aware if naive
                if article_dt.tzinfo is None:
                    article_dt = article_dt.replace(tzinfo=_tz.utc)
                # Skip stale articles
                if article_dt < _cutoff:
                    continue
                pub = article_dt.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                pub = pub_dt_raw[:16]
            results.append({
                "title":        title,
                "link":         entry.get("link", "#"),
                "source":       source_name,
                "published":    pub,
                "published_dt": pub_dt_raw,
                "summary":      summary[:220] + "…" if len(summary) > 220 else summary,
            })
            if len(results) >= max_items:
                break
    except Exception:
        pass
    return results


@st.cache_data(ttl=300, show_spinner=False)   # 5 min — live Indian financial RSS
def _fetch_indian_rss(company: str, ticker: str) -> list[dict]:
    """
    Fetch live news from ET Markets, Moneycontrol, and Business Standard RSS feeds.
    Filters articles that mention the company name or ticker.
    Updates every 5-15 minutes — far fresher than Google News RSS.
    """
    feeds = [
        ("https://economictimes.indiatimes.com/markets/stocks/rss.cms",
         "Economic Times"),
        ("https://economictimes.indiatimes.com/markets/rss.cms",
         "ET Markets"),
        ("https://www.moneycontrol.com/rss/marketsnews.xml",
         "Moneycontrol"),
        ("https://www.moneycontrol.com/rss/MCtopnews.xml",
         "Moneycontrol"),
        ("https://www.business-standard.com/rss/markets-106.rss",
         "Business Standard"),
        ("https://www.business-standard.com/rss/companies-101.rss",
         "Business Standard"),
        ("https://www.livemint.com/rss/markets",
         "Livemint"),
    ]
    all_articles: list[dict] = []
    seen_titles: set[str] = set()
    for url, source in feeds:
        arts = _parse_rss_feed(url, company, ticker, source, max_items=5)
        for a in arts:
            key = a["title"][:60].lower()
            if key not in seen_titles:
                seen_titles.add(key)
                all_articles.append(a)
    # Sort newest first (best-effort — RSS published strings vary)
    all_articles.sort(key=lambda x: x.get("published_dt", ""), reverse=True)
    return all_articles[:15]


@st.cache_data(ttl=300, show_spinner=False)   # 5 min — stay fresh
def _fetch_newsapi(company: str, ticker: str, api_key: str) -> list[dict]:
    """
    Search NewsAPI.org for the company across Indian financial outlets.
    Returns structured, date-sorted articles — the most accurate company-specific feed.
    Free tier: 100 req/day.
    """
    if not api_key:
        return []
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").replace("-", "")
    # Build a precise query: exact company name OR ticker. For short/generic
    # names (e.g. "NAM India"), also try the ticker stem alone so we don't
    # get swamped by every article mentioning "India".
    q_parts = [f'"{company}"']
    if clean_ticker.lower() not in company.lower():
        q_parts.append(f'"{clean_ticker}"')
    newsapi_query = " OR ".join(q_parts)

    try:
        import requests as _req
        resp = _req.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        newsapi_query,
                "apiKey":   api_key,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 12,
                "domains":  (
                    "economictimes.indiatimes.com,moneycontrol.com,"
                    "livemint.com,business-standard.com,"
                    "financialexpress.com,ndtvprofit.com,"
                    "reuters.com,bloomberg.com,thehindu.com"
                ),
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        articles = data.get("articles", [])
        results = []
        for art in articles:
            # Parse ISO date → readable
            pub_raw = art.get("publishedAt", "")
            try:
                from datetime import timezone
                dt  = datetime.strptime(pub_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                pub = dt.strftime("%d %b %Y, %I:%M %p") + " UTC"
            except Exception:
                pub = pub_raw[:10]
            source = art.get("source", {}).get("name", "")
            desc   = (art.get("description") or art.get("content") or "").strip()
            desc   = re.sub(r"\[\+\d+ chars\]$", "", desc).strip()
            results.append({
                "title":        art.get("title", ""),
                "link":         art.get("url", "#"),
                "source":       source,
                "published":    pub,
                "published_dt": pub_raw,
                "summary":      desc[:220] + "…" if len(desc) > 220 else desc,
            })
        return results
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_news(company: str, ticker: str) -> list[dict]:
    """
    Master news fetcher — merges NewsAPI + Indian RSS (ET/MC/BS/Mint).
    NewsAPI results go first (most accurate company match).
    Indian RSS fills in additional coverage.
    Deduplicates by title. Sorted newest first.
    """
    results: list[dict] = []
    seen: set[str] = set()

    # 1. NewsAPI — best quality, company-searched
    for art in _fetch_newsapi(company, ticker, _NEWSAPI_KEY):
        key = art["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            results.append(art)

    # 2. Indian RSS feeds — live market feeds filtered by company name
    for art in _fetch_indian_rss(company, ticker):
        key = art["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            results.append(art)

    # Sort newest first
    results.sort(key=lambda x: x.get("published_dt", ""), reverse=True)
    return results[:20]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_tavily_context(company: str, ticker: str, api_key: str) -> str:
    """
    Search Tavily for recent news + analyst commentary on a stock.
    Returns a rich context string for the AI prompt.
    Tavily returns full article snippets — much richer than Google News RSS.
    """
    if not api_key:
        return ""

    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    _yr = datetime.now().year
    query = f"{company} {clean_ticker} stock India latest news analyst target earnings results {_yr-1} {_yr}"

    try:
        import requests as _req
        resp = _req.post(
            "https://api.tavily.com/search",
            json={
                "api_key":        api_key,
                "query":          query,
                "search_depth":   "advanced",
                "include_answer": True,
                "include_domains": [
                    "economictimes.indiatimes.com",
                    "moneycontrol.com",
                    "livemint.com",
                    "business-standard.com",
                    "bseindia.com",
                    "nseindia.com",
                    "screener.in",
                    "reuters.com",
                    "bloomberg.com",
                    "financialexpress.com",
                    "thehindubuisessline.com",
                    "ndtvprofit.com",
                ],
                "max_results": 8,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        lines: list[str] = []

        # Tavily top-level answer (synthesised summary)
        answer = data.get("answer", "")
        if answer:
            lines.append(f"[Tavily Summary]\n{answer[:600]}")

        # Individual search results
        results = data.get("results", [])
        for r in results[:6]:
            title    = r.get("title", "")
            url      = r.get("url", "")
            content  = r.get("content", "")
            pub_date = r.get("published_date", "")
            source   = url.split("/")[2].replace("www.", "") if url else "unknown"
            snippet  = content[:350].replace("\n", " ").strip() if content else ""
            date_str = f" ({pub_date})" if pub_date else ""
            lines.append(
                f"[{source}]{date_str} {title}\n  {snippet}"
            )

        return "\n\n".join(lines)

    except Exception:
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_exa_context(company: str, ticker: str, api_key: str) -> str:
    """
    Exa AI neural search — PRIMARY web research source for the detailed analysis.
    Runs THREE queries per call:
      1. Recent news:     company-specific news from the last 45 days (Indian outlets first)
      2. Analyst research: price targets, ratings, earnings estimates (last 90 days)
      3. Corporate events: earnings calls, results, announcements, filings (last 60 days)
    Returns a rich context block for the AI prompt. Tavily is supplementary only.
    """
    if not api_key:
        return ""

    import requests as _req
    from datetime import timezone as _tz, timedelta as _td

    clean = ticker.replace(".NS", "").replace(".BO", "")
    headers = {"x-api-key": api_key, "Content-Type": "application/json"}
    # Tightened windows — a 52-week extreme is a recent event; old articles are context, not catalysts.
    cutoff_news     = (datetime.now(_tz.utc) - _td(days=14)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff_events   = (datetime.now(_tz.utc) - _td(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff_research = (datetime.now(_tz.utc) - _td(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")

    _INDIAN_DOMAINS = [
        "economictimes.indiatimes.com", "moneycontrol.com",
        "livemint.com", "business-standard.com",
        "financialexpress.com", "ndtvprofit.com",
        "cnbctv18.com", "zeebiz.com", "thehindubuisessline.com",
        "bseindia.com", "nseindia.com",
        "reuters.com", "bloomberg.com",
    ]

    lines: list[str] = []

    def _exa_search(query: str, since: str, num: int = 8, domains: list | None = None) -> list[dict]:
        payload: dict = {
            "query":              query,
            "numResults":         num,
            "type":               "neural",
            "startPublishedDate": since,
            "contents": {
                "text":       {"maxCharacters": 600},
                "highlights": {"numSentences": 3, "highlightsPerUrl": 3},
            },
        }
        if domains:
            payload["includeDomains"] = domains
        try:
            r = _req.post("https://api.exa.ai/search", json=payload,
                          headers=headers, timeout=15)
            if r.status_code == 200:
                return r.json().get("results", [])
        except Exception:
            pass
        return []

    def _norm(results: list[dict], category: str) -> list[dict]:
        """Flatten raw Exa hits into a uniform shape (keeping the real URL)."""
        out = []
        for r in results:
            title = (r.get("title") or "").strip()
            url   = r.get("url", "") or ""
            if not title:
                continue
            pub        = (r.get("publishedDate") or "")[:10]
            source     = url.split("/")[2].replace("www.", "") if url else "web"
            highlights = r.get("highlights") or []
            snippet    = " … ".join(highlights[:3]) if highlights else (r.get("text") or "")[:400]
            snippet    = snippet.replace("\n", " ").strip()
            out.append({"title": title, "url": url, "pub": pub,
                        "source": source, "snippet": snippet, "category": category})
        return out

    # ── Query 1: Recent company news (Indian outlets, 14 days) ────────────────
    news_q = (f"{company} {clean} NSE India latest news update "
              f"earnings revenue profit quarterly results")
    # ── Query 2: Analyst research + price targets (60 days, open web) ─────────
    research_q = (f"{company} {clean} India stock analyst buy sell hold "
                  f"price target upgrade downgrade rating EPS estimate forecast")
    # ── Query 3: Corporate events — earnings calls, filings, announcements ─────
    events_q = (f"{company} {clean} India quarterly earnings call results "
                f"dividend board meeting order win acquisition expansion {datetime.now().year-1} {datetime.now().year}")

    pooled  = (_norm(_exa_search(news_q,     cutoff_news,     num=8, domains=_INDIAN_DOMAINS), "News")
             + _norm(_exa_search(events_q,   cutoff_events,   num=5),                          "Corporate Event")
             + _norm(_exa_search(research_q, cutoff_research, num=6),                          "Analyst Research"))

    # ── Dedupe (by URL, then by title prefix) and rank newest-first ───────────
    seen_url: set[str] = set()
    seen_ttl: set[str] = set()
    deduped: list[dict] = []
    for it in pooled:
        ukey = it["url"].split("?")[0].rstrip("/").lower()
        tkey = it["title"][:60].lower()
        if (ukey and ukey in seen_url) or tkey in seen_ttl:
            continue
        if ukey:
            seen_url.add(ukey)
        seen_ttl.add(tkey)
        deduped.append(it)
    deduped.sort(key=lambda x: x["pub"] or "", reverse=True)   # newest-first

    if deduped:
        lines.append(f"=== Exa: {company} — Verified Sources (newest first) ===")
        lines.append("Each item ends with 'URL: <link>' — copy that exact link when citing it.")
        for it in deduped:
            lines.append(
                f"[{it['source']}] ({it['pub'] or 'date n/a'}) [{it['category']}] {it['title']}\n"
                f"  {it['snippet']}\n"
                f"  URL: {it['url']}"
            )

    return "\n\n".join(lines) if lines else ""


@st.cache_data(ttl=3600, show_spinner=False)   # 1h — NSE CSV changes rarely
def _fetch_index_tickers(universe: str) -> list[str]:
    """Fetch current constituent list for any NSE index from the official NSE CSV."""
    _NSE_CSV = {
        "Nifty 500":          "ind_nifty500list.csv",
        "Nifty 50":           "ind_nifty50list.csv",
        "Nifty 100":          "ind_nifty100list.csv",
        "Nifty Midcap 100":   "ind_niftymidcap100list.csv",
        "Nifty Smallcap 100": "ind_niftysmallcap100list.csv",
    }
    if universe.startswith("All NSE"):
        return fetch_all_nse_live()
    filename = _NSE_CSV.get(universe)
    if not filename:
        return []
    url = f"https://nsearchives.nseindia.com/content/indices/{filename}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer":    "https://www.nseindia.com/",
        "Accept":     "text/html,*/*;q=0.9",
    }
    try:
        import requests as _req
        resp = _req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df["Symbol"].dropna().str.strip().tolist()
        # Filter out NSE placeholder/dummy tickers (e.g. DUMMYVEDL1-4 created
        # for Vedanta demerger entities — they have no real price data).
        tickers = [
            s + ".NS" for s in symbols
            if s and not s.upper().startswith("DUMMY")
        ]
        return tickers
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)    # 5 min — allows fixes to take effect promptly
def _cached_yfinance_context(ticker: str) -> str:
    """Cached wrapper for build_yfinance_context. Prevents repeated calls on spotlight re-renders."""
    return build_yfinance_context(ticker) if _SCREENER_OK else ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yf_company_context(ticker: str, row: dict | None = None) -> str:
    """
    Company profile for the prompt + sidebar snippet.

    FMP profile is PRIMARY because it works everywhere (API-key based), giving
    the SAME description locally and on Streamlit Cloud — where Yahoo Finance
    geo-blocks datacenter IPs and yfinance returns nothing. yfinance is used to
    supplement extra ratios and as a description fallback when FMP has none.
    """
    parts: list[str] = []
    desc  = ""

    # ── Primary: FMP profile (cloud-safe, environment-consistent) ─────────────
    try:
        from fmp import get_company_profile
        prof = get_company_profile(ticker)
        if prof:
            desc = (prof.get("description") or "").strip()
            if desc:
                parts.append(f"Business Description:\n{desc[:800]}")
            if prof.get("mktCap"):
                try:
                    parts.append(f"Market Cap: Rs {float(prof['mktCap'])/1e7:,.0f} Cr")
                except Exception:
                    pass
            if prof.get("fullTimeEmployees"):
                try:
                    parts.append(f"Employees: {int(prof['fullTimeEmployees']):,}")
                except Exception:
                    pass
            if prof.get("beta"):
                try:
                    parts.append(f"Beta: {float(prof['beta']):.2f}")
                except Exception:
                    pass
    except Exception:
        pass

    # ── Supplement / fallback: yfinance (rich ratios; description if FMP empty) ─
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        if not desc and info.get("longBusinessSummary"):
            desc = info["longBusinessSummary"]
            parts.insert(0, f"Business Description:\n{desc[:800]}")
        if not any(p.startswith("Market Cap") for p in parts) and info.get("marketCap"):
            parts.append(f"Market Cap: Rs {info['marketCap']/1e7:,.0f} Cr")
        if info.get("dividendYield"):
            parts.append(f"Dividend Yield: {info['dividendYield']*100:.2f}%")
        if info.get("returnOnEquity"):
            parts.append(f"ROE: {info['returnOnEquity']*100:.1f}%")
        if info.get("debtToEquity"):
            parts.append(f"Debt/Equity: {info['debtToEquity']:.2f}")
        if info.get("revenueGrowth"):
            parts.append(f"Revenue Growth (YoY): {info['revenueGrowth']*100:.1f}%")
    except Exception:
        pass

    # ── Cloud-safe fallback: Screener.in (works where FMP/yfinance don't) ──────
    # On Streamlit Cloud, FMP's free tier frequently has no profile for mid/small
    # NSE names AND yfinance is geo-blocked, so the whole card came back empty.
    # Screener.in works in both environments (no key), so use it for BOTH the
    # description and key ratios whenever the profile is still empty/sparse.
    if (not desc) or len(parts) < 2:
        try:
            if _SCREENER_OK:
                _sc = build_screener_context(ticker) or ""
                if _sc:
                    if not desc:
                        _i = _sc.find("About:")
                        if _i >= 0:
                            _about = _sc[_i + len("About:"):].split("\n")[0].strip()
                            if len(_about) > 40:
                                desc = _about
                                parts.insert(0, f"Business Description:\n{_about[:800]}")
                    if not any(p.startswith("Key Ratios") for p in parts):
                        _j = _sc.find("Key Ratios:")
                        if _j >= 0:
                            _kr = _sc[_j + len("Key Ratios:"):].split("\n")[0].strip()
                            if _kr:
                                parts.append(f"Key Ratios (Screener.in): {_kr[:300]}")
        except Exception:
            pass

    # ── Guaranteed fallback: synthesize from the screen row so the profile is
    # NEVER empty. FMP's free tier lacks many NSE names, yfinance is geo-blocked
    # on Streamlit Cloud, and Screener can miss SME "About" blurbs — but the row
    # always carries name / sector / industry / market cap / P/E. ──────────────
    if not parts and row is not None and hasattr(row, "get"):
        _cn   = str(row.get("company_name") or "").strip()
        _ind  = str(row.get("industry") or "").strip()
        _sec  = str(row.get("sector") or "").strip()
        _dom  = _ind if _ind and _ind.lower() not in ("nan", "none", "") else _sec
        if _cn and _dom:
            parts.append(f"Business Description:\n{_cn} operates in the {_dom} "
                         f"space and is listed on NSE.")
        elif _dom:
            parts.append(f"Business Description:\nA {_dom} company listed on NSE.")
        try:
            _mc = row.get("market_cap")
            if _mc and float(_mc) > 0:
                parts.append(f"Market Cap: Rs {float(_mc)/1e7:,.0f} Cr")
        except Exception:
            pass
        try:
            _pe = row.get("pe_ratio")
            if _pe and float(_pe) > 0:
                parts.append(f"P/E: {float(_pe):.1f}x")
        except Exception:
            pass
        if _sec:
            parts.append(f"Sector: {_sec}")

    return "\n".join(parts) if parts else ""


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_yf_upgrades_context(ticker: str) -> str:
    """Analyst consensus + price targets.
    Primary: yfinance info fields (fast, free).
    Fallback: FMP price-target + analyst-estimates endpoints (Indian brokers
    often publish there even when Yahoo Finance has no coverage)."""
    lines: list[str] = []

    # ── Primary: yfinance ────────────────────────────────────────────────────
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info

        rec_key    = info.get("recommendationKey", "")
        n_analysts = info.get("numberOfAnalystOpinions")
        if rec_key or n_analysts:
            label = rec_key.replace("_", " ").title() if rec_key else "N/A"
            lines.append(f"Analyst Consensus: {label}"
                         + (f" ({n_analysts} analysts)" if n_analysts else ""))

        target_mean = info.get("targetMeanPrice")
        target_hi   = info.get("targetHighPrice")
        target_lo   = info.get("targetLowPrice")
        curr        = info.get("currentPrice") or info.get("regularMarketPrice")
        if target_mean and curr:
            upside = ((target_mean - curr) / curr) * 100
            lines.append(f"Consensus Target: Rs{target_mean:,.0f} "
                         f"({upside:+.1f}% vs current)  "
                         f"[Range Rs{target_lo:,.0f}–Rs{target_hi:,.0f}]"
                         if target_hi and target_lo else
                         f"Consensus Target: Rs{target_mean:,.0f} ({upside:+.1f}%)")

        from datetime import datetime, timedelta
        try:
            ud = yf.Ticker(ticker).upgrades_downgrades
            if ud is not None and not ud.empty:
                cutoff = datetime.now() - timedelta(days=90)
                recent = ud[ud.index >= cutoff].head(5)
                if not recent.empty:
                    lines.append("Recent Rating Actions (last 90 days):")
                    for date, r in recent.iterrows():
                        firm   = r.get("Firm", "Unknown")
                        action = r.get("ToGrade", "") or r.get("Action", "")
                        frm    = r.get("FromGrade", "")
                        line   = f"  {date.strftime('%b %d, %Y')}: {firm} — {action}"
                        if frm:
                            line += f" (from {frm})"
                        lines.append(line)
        except Exception:
            pass
    except Exception:
        pass

    if lines:
        return "\n".join(lines)

    # ── Fallback: FMP price targets + analyst estimates ───────────────────────
    try:
        from fmp import get_price_targets, get_analyst_estimates
        fmp_lines: list[str] = []

        targets = get_price_targets(ticker, limit=5)
        if targets:
            fmp_lines.append("Analyst Price Targets (FMP):")
            for t in targets[:5]:
                analyst  = t.get("analystName") or t.get("analyst") or "?"
                company  = t.get("analystCompany") or ""
                pt       = t.get("priceTarget")
                date     = (t.get("publishedDate") or "")[:10]
                pt_str   = f"Rs{float(pt):,.0f}" if pt else "N/A"
                fmp_lines.append(f"  {date} | {analyst} ({company}) → {pt_str}")

        estimates = get_analyst_estimates(ticker, limit=2)
        if estimates:
            e = estimates[0]
            rev_lo  = e.get("revenueEstimatedLow")
            rev_hi  = e.get("revenueEstimatedHigh")
            rev_avg = e.get("revenueEstimatedAvg")
            eps_avg = e.get("estimatedEpsAvg")
            period  = e.get("date", "")
            parts   = []
            if rev_avg:
                parts.append(f"Rev est Rs{float(rev_avg)/1e7:,.0f} Cr"
                             + (f" [Rs{float(rev_lo)/1e7:,.0f}–Rs{float(rev_hi)/1e7:,.0f} Cr]"
                                if rev_lo and rev_hi else ""))
            if eps_avg:
                parts.append(f"EPS est Rs{float(eps_avg):,.2f}")
            if parts:
                fmp_lines.append(f"Analyst Estimates ({period}): " + " | ".join(parts))

        if fmp_lines:
            return "\n".join(fmp_lines)
    except Exception:
        pass

    # ── Last resort: Screener.in pros/cons as qualitative analyst proxy ────────
    # Indian brokers rarely publish to Yahoo/FMP, but Screener.in has
    # community-curated pros/cons that serve as a qualitative analyst signal.
    try:
        if _SCREENER_OK:
            from screener_alpha import build_screener_context
            sc = build_screener_context(ticker)
            if sc:
                pros, cons = [], []
                for line in sc.splitlines():
                    if line.startswith("Pros:"):
                        pros = [p.strip() for p in line[5:].split(";") if p.strip()]
                    elif line.startswith("Cons:"):
                        cons = [c.strip() for c in line[5:].split(";") if c.strip()]
                sc_lines: list[str] = []
                if pros:
                    sc_lines.append("Analyst / Screener Positives:")
                    sc_lines.extend(f"  + {p}" for p in pros[:4])
                if cons:
                    sc_lines.append("Analyst / Screener Risks:")
                    sc_lines.extend(f"  - {c}" for c in cons[:4])
                if sc_lines:
                    return "\n".join(sc_lines)
    except Exception:
        pass

    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_fmp_earnings_surprise_context(ticker: str) -> str:
    """Earnings beat/miss history — yfinance quarterly earnings (primary),
    FMP earnings-surprises (fallback)."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Try yfinance earnings_history first (most reliable for NSE)
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty:
                lines = ["Earnings Surprise History (last 4 quarters):"]
                for _, row in eh.tail(4).iterrows():
                    date    = str(row.get("quarter", ""))[:10]
                    actual  = row.get("epsActual")
                    est     = row.get("epsEstimate")
                    surp    = row.get("surprisePercent")
                    if actual is not None and est is not None:
                        tag = "BEAT" if (surp or 0) > 0 else "MISS"
                        lines.append(
                            f"  {date}: EPS {tag} {f'{surp:+.1f}%' if surp else ''} "
                            f"(actual Rs{actual:.2f}, est Rs{est:.2f})"
                        )
                if len(lines) > 1:
                    return "\n".join(lines)
        except Exception:
            pass

        # Fallback: quarterly income for QoQ EPS trend
        try:
            qi = t.quarterly_income_stmt
            if qi is not None and not qi.empty and "Net Income" in qi.index:
                ni = qi.loc["Net Income"].dropna().head(4)
                if len(ni) >= 2:
                    lines = ["Quarterly Net Income Trend (Rs Cr):"]
                    for col, val in ni.items():
                        lines.append(f"  {str(col)[:10]}: Rs{val/1e7:,.0f} Cr")
                    return "\n".join(lines)
        except Exception:
            pass

        # Final fallback: FMP
        import requests, os
        fmp_key = os.getenv("FMP_KEY", "")
        if fmp_key:
            clean = ticker.replace(".NS","").replace(".BO","")
            r = requests.get(
                f"https://financialmodelingprep.com/api/v3/earnings-surprises/{clean}"
                f"?limit=4&apikey={fmp_key}", timeout=10)
            if r.status_code == 200 and r.json():
                lines = ["Earnings Surprise History (FMP):"]
                for q in r.json()[:4]:
                    actual = q.get("actualEarningResult")
                    est    = q.get("estimatedEarning")
                    if actual is not None and est and est != 0:
                        surp = ((actual - est) / abs(est)) * 100
                        tag  = "BEAT" if surp > 0 else "MISS"
                        lines.append(f"  {q.get('date','')}: EPS {tag} by {abs(surp):.1f}%")
                if len(lines) > 1:
                    return "\n".join(lines)
        return ""
    except Exception:
        return ""


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_bse_announcements_context(ticker: str) -> str:
    """Recent corporate announcements — tries BSE API with scrip lookup,
    falls back to yfinance news headlines."""
    import requests
    clean = ticker.replace(".NS", "").replace(".BO", "")

    # Step 1: Look up BSE scrip code from NSE ticker
    scrip_code = ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.bseindia.com/",
        "Origin": "https://www.bseindia.com",
    }

    # Step 1: Use a session to get BSE cookies first, then look up scrip code
    try:
        sess = requests.Session()
        sess.get("https://www.bseindia.com", headers=headers, timeout=8)  # get cookies
        lookup_url = (f"https://api.bseindia.com/BseIndiaAPI/api/GetCompanySearch/w"
                      f"?strScrip={clean}&strType=L")
        lr = sess.get(lookup_url, headers=headers, timeout=8)
        if lr.status_code == 200:
            data = lr.json()
            items = data if isinstance(data, list) else data.get("Table", [])
            if items:
                scrip_code = str(items[0].get("SECURITY_CODE") or
                                 items[0].get("scripCode") or "")
    except Exception:
        pass

    # Step 2: Fetch announcements with session cookies
    if scrip_code:
        try:
            ann_url = (f"https://api.bseindia.com/BseIndiaAPI/api/AnnGetAnnouncementDt/w"
                       f"?scripcd={scrip_code}&strCat=-1&strPrevDate=&strScrip="
                       f"&strSearch=P&strToDate=&strType=C&subcategory=-1")
            ar = sess.get(ann_url, headers=headers, timeout=10)
            if ar.status_code == 200:
                rows = ar.json().get("Table", [])[:6]
                if rows:
                    lines = ["Recent BSE Corporate Announcements:"]
                    for ann in rows:
                        date = (ann.get("NewsDate") or ann.get("DissemDt") or "")[:10]
                        cat  = ann.get("Categoryname") or ann.get("CATEGORYNAME") or ""
                        hdl  = (ann.get("HEADLINE") or ann.get("Headline") or "")[:100]
                        lines.append(f"  {date}: [{cat}] {hdl}")
                    if len(lines) > 1:
                        return "\n".join(lines)
        except Exception:
            pass

    # Step 3: yfinance news (reliable for all major Indian stocks)
    try:
        import yfinance as yf
        from datetime import datetime
        t = yf.Ticker(ticker)
        news = t.news or []
        # yfinance 0.2.x returns list of dicts; newer versions may differ
        if not news:
            # try alternate attribute
            try:
                news = list(t.get_news() or [])
            except Exception:
                pass
        if news:
            lines = ["Recent News:"]
            for item in news[:6]:
                # Handle both old and new yfinance news formats
                content = item.get("content", {}) if isinstance(item, dict) else {}
                title = (content.get("title") or item.get("title") or "")[:120]
                pub   = (content.get("pubDate") or
                         item.get("providerPublishTime") or
                         item.get("pubDate") or "")
                src   = (content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict)
                         else item.get("publisher") or "")
                if title:
                    try:
                        if isinstance(pub, (int, float)):
                            dt = datetime.fromtimestamp(pub).strftime("%b %d")
                        else:
                            dt = str(pub)[:10]
                    except Exception:
                        dt = ""
                    lines.append(f"  {dt}: {title}" + (f" [{src}]" if src else ""))
            if len(lines) > 1:
                return "\n".join(lines)
    except Exception:
        pass

    return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yf_calendar_context(ticker: str) -> str:
    """Next earnings date and ex-dividend date from yfinance."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        if not cal:
            return ""
        lines = []
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                if hasattr(ed, "__iter__") and not isinstance(ed, str):
                    ed = list(ed)[0] if list(ed) else None
                if ed:
                    lines.append(f"Next Earnings Date: {ed}")
            xd = cal.get("Ex-Dividend Date")
            if xd:
                lines.append(f"Ex-Dividend Date: {xd}")
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


_SECTOR_PEERS: dict = {
    "ITC.NS":         ["HINDUNILVR.NS", "NESTLEIND.NS", "DABUR.NS", "MARICO.NS"],
    "HINDUNILVR.NS":  ["ITC.NS", "NESTLEIND.NS", "DABUR.NS", "BRITANNIA.NS"],
    "HDFCBANK.NS":    ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "ICICIBANK.NS":   ["HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    "TCS.NS":         ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "INFY.NS":        ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "RELIANCE.NS":    ["ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS"],
    "TMCV.NS":  ["TMPV.NS", "M&M.NS", "ASHOKLEY.NS", "EICHERMOT.NS"],
    "TMPV.NS":  ["TMCV.NS", "MARUTI.NS", "M&M.NS", "EICHERMOT.NS"],
    "MARUTI.NS":["TMPV.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "SUNPHARMA.NS":   ["DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "DRREDDY.NS":     ["SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "WIPRO.NS":       ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS"],
    "HCLTECH.NS":     ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS"],
    "BAJFINANCE.NS":  ["BAJAJFINSV.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "ADANIENT.NS":    ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS"],
    "TATASTEEL.NS":   ["JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "ADANIENT.NS"],
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_peer_comparison_context(ticker: str) -> str:
    """Compare stock 1M/3M performance against sector peers."""
    try:
        import yfinance as yf
        peers = _SECTOR_PEERS.get(ticker, [])
        if not peers:
            return ""
        all_t = [ticker] + peers[:4]
        raw = yf.download(all_t, period="3mo", auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            return ""
        close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        results = []
        for t in all_t:
            try:
                col = close[t] if t in close.columns else close.iloc[:, 0]
                col = col.dropna()
                if len(col) < 2:
                    continue
                ret_1m = (col.iloc[-1] / col.iloc[max(0, len(col)-21)] - 1) * 100
                ret_3m = (col.iloc[-1] / col.iloc[0] - 1) * 100
                tag = " ← THIS STOCK" if t == ticker else ""
                results.append(f"  {t.replace('.NS',''):15s}: 1M {ret_1m:+.1f}%  3M {ret_3m:+.1f}%{tag}")
            except Exception:
                continue
        if not results:
            return ""
        return "Peer Performance Comparison (1M / 3M returns):\n" + "\n".join(results)
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sector_index_move(sector: str) -> tuple:
    """
    Best-effort ~1-month % move of the matching NSE sector index (option E).
    Returns (pct, index_name) or (None, "") when the sector can't be mapped or
    the fetch fails. One cached yfinance call — no LLM tokens.
    """
    if not sector:
        return (None, "")
    s = sector.lower()
    _MAP = [
        (("bank",),                              "^NSEBANK",    "Nifty Bank"),
        (("financ", "nbfc", "insur"),            "NIFTY_FIN_SERVICE.NS", "Nifty Financial Services"),
        (("it", "software", "tech"),             "^CNXIT",      "Nifty IT"),
        (("auto", "motor", "vehicle"),           "^CNXAUTO",    "Nifty Auto"),
        (("pharma", "health", "drug", "medic"),  "^CNXPHARMA",  "Nifty Pharma"),
        (("fmcg", "staple", "consumer defens"),  "^CNXFMCG",    "Nifty FMCG"),
        (("metal", "steel", "mining"),           "^CNXMETAL",   "Nifty Metal"),
        (("realty", "real estate", "property"),  "^CNXREALTY",  "Nifty Realty"),
        (("energy", "oil", "gas", "power", "utilit"), "^CNXENERGY", "Nifty Energy"),
        (("infra",),                             "^CNXINFRA",   "Nifty Infra"),
    ]
    sym = name = ""
    for keys, symbol, nm in _MAP:
        if any(k in s for k in keys):
            sym, name = symbol, nm
            break
    if not sym:
        return (None, "")
    try:
        import yfinance as _yf
        h = _yf.Ticker(sym).history(period="1mo")
        if h is not None and len(h) > 1:
            first = float(h["Close"].iloc[0])
            last  = float(h["Close"].iloc[-1])
            if first > 0:
                return (round((last - first) / first * 100, 1), name)
    except Exception:
        pass
    return (None, "")


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_fmp_pe(ticker: str):
    """
    P/E fallback for when yfinance .info is rate-limited/blocked (common on
    cloud for NSE) so the screen returns no pe_ratio. Uses FMP — which already
    powers the verified-fundamentals block — so it works wherever FMP works.
    """
    try:
        from fmp import get_key_metrics
        km = get_key_metrics(ticker, limit=1)
        if km and isinstance(km, list) and isinstance(km[0], dict):
            pe = km[0].get("peRatio")
            if pe is not None:
                pe = float(pe)
                if 0 < pe < 100000:
                    return round(pe, 1)
    except Exception:
        pass
    return None


def _match_article_url(headline: str, *pools) -> str:
    """
    Best-effort deterministic resolver: map an AI-written catalyst headline to a
    REAL fetched-article URL by token overlap, so cards link to the actual story
    instead of a search page. `pools` are lists of dicts that carry a title and a
    link/url. Returns '' when no confident match exists.
    """
    import re as _re

    _STOP = {"the", "and", "for", "with", "from", "that", "this", "after",
             "amid", "into", "over", "near", "says", "report", "stock",
             "shares", "share", "nse", "bse", "india", "indian", "ltd",
             "limited", "rs", "cr", "crore", "year", "week", "high", "low"}

    def _toks(s: str) -> set[str]:
        return {w for w in _re.findall(r"[a-z0-9]{3,}", (s or "").lower())
                if w not in _STOP}

    h = _toks(headline)
    if not h:
        return ""

    best_url, best_score = "", 0.0
    for pool in pools:
        for a in pool or []:
            if not isinstance(a, dict):
                continue
            link = (a.get("link") or a.get("url") or "").strip()
            if not link.startswith("http"):
                continue
            t = _toks(a.get("title", ""))
            if not t:
                continue
            # Coverage of the headline's distinctive tokens by the article title.
            score = len(h & t) / len(h)
            if score > best_score:
                best_score, best_url = score, link

    # Require a meaningful overlap so we never link to an unrelated story.
    return best_url if best_score >= 0.34 else ""


def _render_news_feed(company: str, ticker: str, accent: str):
    """Render the Google News feed section below the analyst report."""
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;color:{_MUTED};padding:10px 0 10px;">'
        f'&#128240;&nbsp;&nbsp;Live News — {company}'
        f'<span style="font-weight:400;letter-spacing:0;margin-left:10px;'
        f'color:#484F58;font-size:9px;">NewsAPI · ET Markets · Moneycontrol · Business Standard</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching live news from ET, Moneycontrol, BSE, NewsAPI…"):
        articles = _fetch_live_news(company, ticker)

    if not articles:
        # Fallback to the Exa/Tavily/NewsAPI pool the analysis itself used — so we
        # never show "no news" while the report above cites recent articles.
        try:
            _pool = _fetch_all_news_items(ticker, company, cutoff_days=75) or []
            articles = [
                {"source":    str(it.get("source", "")).replace("Exa/", ""),
                 "title":     it.get("title", ""),
                 "link":      it.get("url", "") or "#",
                 "published": it.get("date", ""),
                 "summary":   it.get("snippet", "")}
                for it in _pool
                if it.get("title")
                and "summary" not in str(it.get("source", "")).lower()
            ]
        except Exception:
            articles = []

    if not articles:
        st.info("No recent news found for this company.")
        return

    for art in articles:
        pub_color = _MUTED
        source_badge = (
            f'<span style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:10px;padding:1px 8px;font-size:10px;'
            f'color:#8B949E;margin-right:8px;">{art["source"]}</span>'
            if art["source"] else ""
        )
        st.markdown(
            f'<div style="background:{_CARD};border:1px solid #21262D;'
            f'border-radius:8px;padding:14px 18px;margin-bottom:8px;">'
            # Title row
            f'<a href="{art["link"]}" target="_blank" style="font-size:14px;'
            f'font-weight:600;color:#E6EDF3;text-decoration:none;line-height:1.4;'
            f'display:block;margin-bottom:6px;">{art["title"]}</a>'
            # Meta row
            f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
            f'{source_badge}'
            f'<span style="font-size:11px;color:{pub_color};">&#128337; {art["published"]}</span>'
            f'</div>'
            # Headlines only — article bodies omitted by design (the title links out).
            + f'</div>',
            unsafe_allow_html=True,
        )


def _build_pdf_report(
    ticker: str, company: str, sector: str, signal: str,
    price: float, high52: float, low52: float,
    pfh: float, pfl: float, vsurge: float, pe: float,
    analysis: dict, articles: list[dict],
) -> bytes:
    """
    Build a professional equity research PDF and return as bytes.
    Styled like a clean broker note — suitable for fund manager distribution.
    """
    buf = io.BytesIO()

    # ── Colour palette ────────────────────────────────────────────────────────
    _INK       = colors.HexColor("#0D1117")
    _DARK_GREY = colors.HexColor("#24292F")
    _MID_GREY  = colors.HexColor("#57606A")
    _LIGHT_BG  = colors.HexColor("#F6F8FA")
    _BORDER    = colors.HexColor("#D0D7DE")
    _GREEN_C   = colors.HexColor("#1A7F37")
    _RED_C     = colors.HexColor("#CF222E")
    _BLUE_C    = colors.HexColor("#0969DA")
    _YELLOW_C  = colors.HexColor("#9A6700")
    _PURPLE_C  = colors.HexColor("#6639BA")
    is_hi      = "HIGH" in signal.upper()
    SIGNAL_COL = _GREEN_C if is_hi else _RED_C
    signal_lbl = "BREAKOUT HIGH" if is_hi else "BREAKDOWN LOW"

    # ── Document setup ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm,  bottomMargin=2*cm,
        title=f"{company} — Equity Research Note",
        author="Nifty 500 Screener",
    )

    # ── Paragraph styles ──────────────────────────────────────────────────────
    S = getSampleStyleSheet()

    def ps(name, parent="Normal", **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=S[parent], **kw)

    sty_report_label = ps("ReportLabel", fontSize=8,  textColor=_MID_GREY,
                           spaceAfter=0, fontName="Helvetica")
    sty_company      = ps("Company",     fontSize=22, textColor=_INK,
                           fontName="Helvetica-Bold", spaceAfter=2)
    sty_ticker       = ps("Ticker",      fontSize=11, textColor=_MID_GREY,
                           fontName="Helvetica", spaceAfter=6)
    sty_signal       = ps("Signal",      fontSize=10, textColor=SIGNAL_COL,
                           fontName="Helvetica-Bold", spaceAfter=4)
    sty_summary      = ps("Summary",     fontSize=12, textColor=_DARK_GREY,
                           fontName="Helvetica-Oblique", leading=18,
                           spaceAfter=10, spaceBefore=6)
    sty_sec_hdr      = ps("SecHdr",      fontSize=8,  textColor=SIGNAL_COL,
                           fontName="Helvetica-Bold", spaceBefore=14,
                           spaceAfter=4, leading=10)
    sty_body         = ps("Body",        fontSize=10, textColor=_DARK_GREY,
                           leading=15, spaceAfter=4, fontName="Helvetica")
    sty_news_title   = ps("NewsTitle",   fontSize=10, textColor=_BLUE_C,
                           fontName="Helvetica-Bold", spaceAfter=2,
                           leading=14)
    sty_news_meta    = ps("NewsMeta",    fontSize=8,  textColor=_MID_GREY,
                           fontName="Helvetica", spaceAfter=2)
    sty_news_body    = ps("NewsBody",    fontSize=9,  textColor=_DARK_GREY,
                           leading=13, spaceAfter=6, fontName="Helvetica")
    sty_disclaimer   = ps("Disclaimer",  fontSize=7,  textColor=_MID_GREY,
                           fontName="Helvetica-Oblique", leading=10)
    sty_footer_right = ps("FootRight",   fontSize=8,  textColor=_MID_GREY,
                           alignment=TA_RIGHT, fontName="Helvetica")

    story = []

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ═══════════════════════════════════════════════════════════════════════════

    # Top rule
    story.append(HRFlowable(width="100%", thickness=3,
                             color=SIGNAL_COL, spaceAfter=10))

    # Label row
    story.append(Paragraph("EQUITY RESEARCH  |  NIFTY 500 SCREENER  |  "
                            f"NSE INDIA", sty_report_label))
    story.append(Spacer(1, 4))
    story.append(Paragraph(company, sty_company))
    story.append(Paragraph(f"{ticker}  •  {sector}", sty_ticker))
    story.append(Paragraph(f"52-WEEK {signal_lbl}  •  "
                            f"Report Date: {datetime.now().strftime('%d %B %Y')}",
                            sty_signal))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=_BORDER, spaceBefore=8, spaceAfter=10))

    # ── Key Metrics Table ──────────────────────────────────────────────────────
    def fmt(v, prefix="", suffix="", decimals=2):
        if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
        return f"{prefix}{v:,.{decimals}f}{suffix}"

    metrics_data = [
        ["METRIC", "VALUE", "METRIC", "VALUE"],
        ["Current Price",  fmt(price,  "Rs.", ""),
         "52-Week High",   fmt(high52, "Rs.", "")],
        ["52-Week Low",    fmt(low52,  "Rs.", ""),
         "% From High",    fmt(abs(pfh) if pfh else None, "", "%")],
        ["% From Low",     fmt(abs(pfl) if pfl else None, "", "%"),
         "Volume Surge",   fmt(vsurge, "", "x")],
        ["P/E Ratio",      fmt(pe,     "", "x"),
         "Volume Surge",   fmt(vsurge, "", "x")],
    ]
    col_w = [3.8*cm, 4.2*cm, 3.8*cm, 4.2*cm]
    mt = Table(metrics_data, colWidths=col_w)
    mt.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",   (0,0), (-1,0), _LIGHT_BG),
        ("TEXTCOLOR",    (0,0), (-1,0), _MID_GREY),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0), 7),
        ("TOPPADDING",   (0,0), (-1,0), 5),
        ("BOTTOMPADDING",(0,0), (-1,0), 5),
        # Data rows
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,1), (-1,-1), 10),
        ("TEXTCOLOR",    (0,1), (-1,-1), _INK),
        ("FONTNAME",     (1,1), (1,-1),  "Helvetica-Bold"),
        ("FONTNAME",     (3,1), (3,-1),  "Helvetica-Bold"),
        ("TOPPADDING",   (0,1), (-1,-1), 6),
        ("BOTTOMPADDING",(0,1), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        # Labels col background
        ("BACKGROUND",   (0,1), (0,-1), _LIGHT_BG),
        ("BACKGROUND",   (2,1), (2,-1), _LIGHT_BG),
        ("TEXTCOLOR",    (0,1), (0,-1), _MID_GREY),
        ("TEXTCOLOR",    (2,1), (2,-1), _MID_GREY),
        ("FONTSIZE",     (0,1), (0,-1), 8),
        ("FONTSIZE",     (2,1), (2,-1), 8),
        # Grid
        ("GRID",         (0,0), (-1,-1), 0.5, _BORDER),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, _LIGHT_BG]),
    ]))
    story.append(mt)
    story.append(Spacer(1, 12))

    # ── Analyst Verdict (summary) ─────────────────────────────────────────────
    summary = analysis.get("summary", "")
    if summary:
        verdict_data = [[Paragraph(
            f"<b>ANALYST VERDICT:</b>  {summary}", sty_body)]]
        vt = Table(verdict_data, colWidths=[16*cm])
        vt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1),
             colors.HexColor("#EAF5EC" if is_hi else "#FFEBE9")),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ("RIGHTPADDING",  (0,0), (-1,-1), 12),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("BOX",           (0,0), (-1,-1), 1.5, SIGNAL_COL),
            ("ROUNDEDCORNERS",(0,0), (-1,-1), [4,4,4,4]),
        ]))
        story.append(vt)
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 2+ — ANALYST SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    sections = [
        ("1. RECENT PRICE ACTION",         "recent_price_action",   _BLUE_C),
        ("2. FUNDAMENTAL CATALYSTS",        "fundamental_catalysts", SIGNAL_COL),
        ("3. SECTORAL & MACRO TAILWINDS",   "sectoral_macro",        _YELLOW_C),
        ("4. CORPORATE ACTIONS & EVENTS",   "corporate_actions",     _PURPLE_C),
        ("5. NARRATIVE SHIFT",              "narrative_shift",       _BLUE_C),
        ("6. TECHNICAL & FLOW FACTORS",     "flow_factors",          _GREEN_C),
        ("7. RISKS",                        "risks",                 _RED_C),
    ]

    for title, key, col in sections:
        body_text = analysis.get(key, "—")
        block = KeepTogether([
            HRFlowable(width="100%", thickness=0.5, color=_BORDER,
                       spaceBefore=6, spaceAfter=6),
            Paragraph(title, ps(f"sh_{key}", parent="Normal",
                                fontSize=8, textColor=col,
                                fontName="Helvetica-Bold",
                                spaceBefore=2, spaceAfter=5, leading=10)),
            Paragraph(body_text, sty_body),
        ])
        story.append(block)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════════════
    # PAGE 3+ — NEWS FEED
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=2,
                             color=_BLUE_C, spaceAfter=8))
    story.append(Paragraph(
        f"RECENT NEWS — {company.upper()}",
        ps("NewsHdr", parent="Normal", fontSize=11, textColor=_INK,
           fontName="Helvetica-Bold", spaceAfter=10)
    ))

    if articles:
        for i, art in enumerate(articles, 1):
            art_title   = art.get("title",     "Untitled")
            art_source  = art.get("source",    "")
            art_pub     = art.get("published", "")
            art_summary = art.get("summary",   "")
            art_link    = art.get("link",      "")

            meta = " ".join(filter(None, [art_source, art_pub]))
            block = KeepTogether([
                Paragraph(f"{i}. {art_title}", sty_news_title),
                Paragraph(meta, sty_news_meta),
                *(([Paragraph(art_summary, sty_news_body)] if art_summary else [])),
                Paragraph(f'<link href="{art_link}">{art_link[:80]}</link>',
                          sty_news_meta),
                Spacer(1, 4),
            ])
            story.append(block)
    else:
        story.append(Paragraph("No recent news articles found.", sty_body))

    # ── Disclaimer footer ─────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=_BORDER, spaceAfter=6))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-assisted screening tool "
        "for informational purposes only. It does not constitute investment advice. "
        "All data sourced from publicly available information via Yahoo Finance and "
        "Google News. Verify all facts independently before making investment "
        "decisions. Past performance is not indicative of future results.",
        sty_disclaimer,
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}  |  "
        f"Nifty 500 Screener  |  For internal use only",
        sty_footer_right,
    ))

    doc.build(story)
    return buf.getvalue()


def _section_card(icon: str, label: str, body: str, accent: str):
    st.markdown(
        f'<div style="background:{_CARD};border:1px solid #21262D;'
        f'border-left:3px solid {accent};border-radius:0 8px 8px 0;'
        f'padding:16px 20px;margin-bottom:10px;">'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;color:{accent};margin-bottom:8px;">'
        f'{icon}&nbsp;&nbsp;{label}</div>'
        f'<div style="font-size:13px;line-height:1.7;color:#C9D1D9;">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_spotlight(ticker: str, row: dict, params: dict):
    """Stock spotlight: price strip + 7-section AI analyst report."""
    name   = row.get("company_name", ticker)
    sig    = row.get("signal", "")
    is_hi  = "HIGH" in sig.upper()
    chip   = (f'<span class="chip-hi">&#8679; BREAKOUT HIGH</span>'
              if is_hi else f'<span class="chip-lo">&#8681; BREAKDOWN LOW</span>')
    accent = _GREEN if is_hi else _RED

    st.markdown(
        f'<div class="sec-hdr">&#9670; Analyst Report &nbsp; {name} &nbsp;'
        f'<span style="color:{_MUTED}">({ticker})</span> &nbsp; {chip}</div>',
        unsafe_allow_html=True,
    )

    # ── Price strip ───────────────────────────────────────────────────────────
    price  = _safe_float(row.get("current_price"))
    high52 = _safe_float(row.get("week_52_high"))
    low52  = _safe_float(row.get("week_52_low"))
    vsurge = _safe_float(row.get("volume_surge"))
    pe     = _safe_float(row.get("pe_ratio"))
    if not pe:                       # yfinance fundamentals often blocked → FMP fallback
        pe = _fetch_fmp_pe(ticker)
    pfh    = _safe_float(row.get("pct_from_high")) or 0.0
    pfl    = _safe_float(row.get("pct_from_low"))  or 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price",  f"₹{price:,.2f}"   if price  else "—")
    c2.metric("52W High",       f"₹{high52:,.2f}"  if high52 else "—",
              f"{abs(pfh):.2f}% from high")
    c3.metric("52W Low",        f"₹{low52:,.2f}"   if low52  else "—",
              f"{abs(pfl):.2f}% from low")
    c4.metric("P/E Ratio",      f"{pe:.1f}×"       if pe     else "—")
    c5.metric("Volume Surge",   f"{vsurge:.2f}×"   if vsurge else "—")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Trailing returns from screener row ────────────────────────────────────
    ret_1m = _safe_float(row.get("ret_1m"))
    ret_3m = _safe_float(row.get("ret_3m"))
    ret_6m = _safe_float(row.get("ret_6m"))
    ret_1y = _safe_float(row.get("ret_1y"))

    # ── 30-day price series from cached OHLCV ─────────────────────────────────
    price_series = ""
    ohlcv = _fetch_ohlcv(ticker, period="3mo")
    if ohlcv is not None and not ohlcv.empty:
        last30 = ohlcv["Close"].tail(30)
        price_series = "  ".join(
            f"₹{v:,.0f}" for v in last30.values
        )

    # ── AI call ───────────────────────────────────────────────────────────────
    headlines_raw = row.get("news_headlines") or []
    headlines_str = (" | ".join(str(h) for h in headlines_raw[:5])
                     if headlines_raw else "No recent headlines available.")

    # ── Fetch all data sources ────────────────────────────────────────────────
    src_col1, src_col2, src_col3, src_col4 = st.columns(4)

    with src_col1:
        with st.spinner("FMP: fetching fundamentals…"):
            fmp_ctx = build_fmp_context(ticker) if _FMP_OK else ""
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if fmp_ctx else "#F0B429"};">'
            f'{"✅ FMP fundamentals loaded" if fmp_ctx else "⚠️ FMP unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col2:
        with st.spinner("FRED: fetching global macro…"):
            fred_ctx = build_fred_context() if _FRED_OK else ""
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if fred_ctx else "#F0B429"};">'
            f'{"✅ FRED macro loaded" if fred_ctx else "⚠️ FRED unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col3:
        with st.spinner("NSE/RBI: fetching India macro…"):
            india_ctx = build_india_macro_context() if _INDIA_MACRO_OK else ""
            # Append India trade/policy watch (FTAs, tariffs, PLI, GST, bans).
            try:
                import policy_news as _pol_ctx
                _policy_blk = _pol_ctx.policy_context_block()
                if _policy_blk:
                    india_ctx = (india_ctx + "\n\n" + _policy_blk) if india_ctx else _policy_blk
            except Exception:
                pass
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if india_ctx else "#F0B429"};">'
            f'{"✅ India macro + FII/DII loaded" if india_ctx else "⚠️ India macro unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col4:
        with st.spinner("yfinance: fetching company profile…"):
            nse_bse_ctx = _fetch_yf_company_context(ticker, row)
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if nse_bse_ctx else "#F0B429"};">'
            f'{"✅ Company profile loaded" if nse_bse_ctx else "⚠️ Company profile unavailable"}</div>',
            unsafe_allow_html=True,
        )

    # ── Row 2: Indian-specific data sources ───────────────────────────────────
    src_col5, src_col6, src_col7, _ = st.columns(4)

    with src_col5:
        with st.spinner("Screener.in: fetching India fundamentals…"):
            screener_ctx = build_screener_context(ticker) if _SCREENER_OK else ""
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if screener_ctx else "#F0B429"};">'
            f'{"✅ Screener.in loaded (10yr financials)" if screener_ctx else "⚠️ Screener.in unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col6:
        with st.spinner("yfinance: fetching analyst targets & EPS…"):
            av_ctx = _cached_yfinance_context(ticker) if _SCREENER_OK else ""
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if av_ctx else "#8B949E"};">'
            f'{"✅ yfinance fundamentals loaded" if av_ctx else "ℹ️ Fundamentals via FMP / Screener"}</div>',
            unsafe_allow_html=True,
        )

    with src_col7:
        with st.spinner("Tavily / Exa: searching latest news…"):
            tavily_ctx = _fetch_tavily_context(name, ticker, _TAVILY_KEY) if _TAVILY_KEY else ""
            exa_ctx    = _fetch_exa_context(name, ticker, _EXA_KEY) if _EXA_KEY else ""
        web_ctx_ok = bool(tavily_ctx or exa_ctx)
        web_label  = ("✅ Exa + Tavily search loaded" if (exa_ctx and tavily_ctx)
                      else "✅ Exa neural search loaded" if exa_ctx
                      else "✅ Tavily web search loaded" if tavily_ctx
                      else "⚠️ Web search unavailable")
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if web_ctx_ok else "#F0B429"};">'
            f'{web_label}</div>',
            unsafe_allow_html=True,
        )

    # ── Row 3: analyst actions, earnings history, exchange filings ────────────
    src_col8, src_col9, src_col10, _ = st.columns(4)

    with src_col8:
        with st.spinner("yfinance / FMP: analyst targets…"):
            upgrades_ctx = _fetch_yf_upgrades_context(ticker)
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if upgrades_ctx else "#8B949E"};">'
            f'{"✅ Analyst targets loaded" if upgrades_ctx else "ℹ️ Analyst view via news flow"}</div>',
            unsafe_allow_html=True,
        )

    with src_col9:
        with st.spinner("FMP: earnings surprises…"):
            earnings_surprise_ctx = _fetch_fmp_earnings_surprise_context(ticker)
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if earnings_surprise_ctx else "#F0B429"};">'
            f'{"✅ Earnings history loaded" if earnings_surprise_ctx else "⚠️ Earnings data unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col10:
        with st.spinner("BSE: corporate announcements…"):
            bse_ctx = _fetch_bse_announcements_context(ticker)
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if bse_ctx else "#F0B429"};">'
            f'{"✅ Announcements / news loaded" if bse_ctx else "⚠️ Announcements unavailable"}</div>',
            unsafe_allow_html=True,
        )

    # ── Fetch calendar + peer context (background, no spinner) ────────────────
    calendar_ctx = _fetch_yf_calendar_context(ticker)
    peer_ctx     = _fetch_peer_comparison_context(ticker)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Fetch the same multi-source fresh news used by the pre-analysis ──────────
    # _fetch_all_news_context runs Exa + Tavily + NewsAPI + yfinance in parallel
    # with today's date as the cutoff — same pipeline as the quick catalyst screen.
    with st.spinner("Fetching today's news (Exa + Tavily + NewsAPI in parallel)…"):
        fresh_news_ctx = _fetch_all_news_context(ticker, name)
    # Structured items (with real URLs) from the same cached fetch — used to
    # resolve AI catalyst headlines to actual article links in the cards below.
    fresh_news_pool = _fetch_all_news_items(ticker, name)

    # ── Fetch live news (NewsAPI + ET/MC/BS RSS — also used for news feed below) ─
    with st.spinner("Fetching live news (NewsAPI + ET · MC · BS)…"):
        gn_articles = _fetch_live_news(name, ticker)
    gn_context = ""
    if gn_articles:
        lines = []
        for a in gn_articles:
            lines.append(
                f"[{a['source']}] {a['title']} ({a['published']})\n"
                f"  {a['summary']}"
            )
        gn_context = "\n\n".join(lines)

    # Pull primary catalyst from pre-analysis (stored during screen run)
    _pre_cache = st.session_state.get(f"ai_{ticker}_{sig}", {})
    _pre_primary = _pre_cache.get("primary_catalyst", {})
    _pre_catalyst_ctx = ""
    if _pre_primary and isinstance(_pre_primary, dict):
        _pc_head = _pre_primary.get("headline", "")
        _pc_det  = _pre_primary.get("detail", "")
        if _pc_head:
            _pre_catalyst_ctx = (
                f"PRE-ANALYSIS PRIMARY CATALYST (from initial screen — must be addressed):\n"
                f"  Headline: {_pc_head}\n"
                f"  Detail: {_pc_det}\n"
                f"This was identified as the single most impactful driver. Your detailed analysis "
                f"MUST explain and expand on this catalyst with deeper evidence."
            )
    # If no prior deep-dive exists, seed from the CARD's quick catalyst so the
    # card and the detailed report ALWAYS describe the SAME main catalyst
    # (fixes the card-vs-detailed mismatch on both highs and lows).
    if not _pre_catalyst_ctx:
        _quick = st.session_state.get(f"quick_{ticker}_{sig}", {})
        if isinstance(_quick, dict):
            _qh = _quick.get("headline", "")
            if _qh and not _quick.get("no_catalyst"):
                _qd = _quick.get("catalyst_date", "")
                _pre_catalyst_ctx = (
                    f"PRE-ANALYSIS PRIMARY CATALYST (from the screening card — must be addressed):\n"
                    f"  Headline: {_qh}\n"
                    f"  Event date: {_qd}\n"
                    f"This is the SAME catalyst shown on the card. Your detailed analysis MUST "
                    f"build the primary_catalyst around THIS event — do not switch to a different one."
                )

    with st.spinner(f"AI is writing the analyst report for {name}…"):
        analysis = _ai_deep_dive(
            ticker=ticker, company=name, sector=row.get("sector", ""),
            signal=sig, price=price or 0, high52=high52 or 0,
            low52=low52 or 0, pct_from_high=pfh, pct_from_low=pfl,
            vsurge=vsurge or 0, pe=pe or 0,
            news_headlines=headlines_str,
            ret_1m=ret_1m, ret_3m=ret_3m, ret_6m=ret_6m, ret_1y=ret_1y,
            price_series=price_series,
            fmp_context=fmp_ctx,
            fred_context=fred_ctx,
            india_macro_context=india_ctx,
            nse_bse_context=nse_bse_ctx,
            google_news_context=gn_context,
            screener_context=screener_ctx,
            alpha_vantage_context=av_ctx,
            tavily_context=tavily_ctx,
            exa_context=exa_ctx,
            upgrades_context=upgrades_ctx,
            earnings_surprise_context=earnings_surprise_ctx,
            bse_announcements_context=bse_ctx,
            calendar_context=calendar_ctx,
            peer_context=peer_ctx,
            pre_catalyst_context=_pre_catalyst_ctx,
            fresh_news_context=fresh_news_ctx,
        )

    # ── Detect AI failure (rate limit / key / error) and show actionable help ──
    summary    = analysis.get("summary", "")
    is_api_fail = ("AI ANALYSIS UNAVAILABLE" in summary
                   or "AI API key not configured" in summary
                   or "OpenAI error" in summary)
    is_rate_limited = "rate limit" in summary.lower()

    if is_api_fail:
        if is_rate_limited:
            _box = (
                '<div style="font-size:13px;font-weight:700;color:#F0B429;'
                'margin-bottom:8px;">⏳ AI Rate Limit Reached</div>'
                '<div style="font-size:13px;color:#C9D1D9;line-height:1.7;">'
                f'The AI provider (<code>{_LLM_PROVIDER}</code>) is temporarily throttled. '
                'Wait ~30–60 seconds and click Retry.<br>'
                '<b>To avoid this:</b> set a lighter <code>FALLBACK_MODEL</code>, '
                'screen a smaller universe, or raise your provider tier.'
                '</div>'
            )
        else:
            import html as _html_err
            _raw_err = str(analysis.get("ai_error", "")).strip()
            _prov    = analysis.get("ai_provider", _LLM_PROVIDER)
            _mdl     = analysis.get("ai_model", _OPENAI_MODEL)
            if _raw_err:
                # A real provider error came back — show it verbatim so the
                # actual cause (bad param, model name, credits, auth) is visible.
                _detail_html = (
                    f'The AI call to <code>{_html_err.escape(str(_prov))}</code> '
                    f'(model <code>{_html_err.escape(str(_mdl))}</code>) failed:<br>'
                    f'<span style="color:#F0B429;font-family:monospace;font-size:12px;">'
                    f'{_html_err.escape(_raw_err)}</span>'
                )
            else:
                _detail_html = (
                    'Could not connect to the AI model — no API key detected. '
                    'Check your key in Streamlit secrets.<br>'
                    '<b>Set:</b> LLM_PROVIDER and OPENAI_API_KEY / GROQ_API_KEY / PERPLEXITY_API_KEY'
                )
            _box = (
                '<div style="font-size:13px;font-weight:700;color:#F0B429;'
                'margin-bottom:8px;">⚠️ AI Analysis Unavailable</div>'
                '<div style="font-size:13px;color:#C9D1D9;line-height:1.7;">'
                + _detail_html +
                '</div>'
            )
        st.markdown(
            '<div style="background:#2a1a00;border:1px solid #F0B429;'
            'border-radius:8px;padding:16px 20px;margin-bottom:16px;">' + _box + '</div>',
            unsafe_allow_html=True,
        )
        cache_key = f"ai_{ticker}_{sig}"
        if st.button("🔄 Retry Analysis", key=f"retry_{ticker}",
                     width='stretch'):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()
        return

    # ── Sentiment + confidence header ─────────────────────────────────────────
    sentiment  = analysis.get("sentiment", "")
    confidence = analysis.get("confidence", "")
    _sent_color = {"Bullish": "#3FB950", "Bearish": "#F85149", "Neutral": "#8B949E"}.get(sentiment, _MUTED)
    _sent_dot   = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}.get(sentiment, "⚪")
    _conf_color = {"High": "#3FB950", "Medium": "#F0B429", "Low": "#F85149"}.get(confidence, _MUTED)

    if sentiment or confidence:
        st.markdown(
            f'<div style="background:{_CARD};border:1px solid #30363D;'
            f'border-radius:8px;padding:12px 20px;margin-bottom:14px;'
            f'display:flex;align-items:center;gap:16px;">'
            f'<span style="font-size:20px">{_sent_dot}</span>'
            f'<span style="font-size:15px;font-weight:700;color:{_sent_color};">'
            f'Sentiment: {sentiment}</span>'
            f'<span style="color:{_MUTED};font-size:14px;margin:0 8px;">|</span>'
            f'<span style="font-size:14px;color:{_conf_color};">'
            f'Confidence: <b>{confidence}</b></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Data-quality row: news freshness badge + conflicting-signals callout ───
    _freshness = str(analysis.get("news_freshness", "")).strip()
    _fresh_key = _freshness.split()[0].upper() if _freshness else ""
    _is_stale_news = _fresh_key in ("STALE", "THIN")
    if _freshness:
        _fb_color = {"FRESH": "#3FB950", "STALE": "#F0B429", "THIN": "#8B949E"}.get(_fresh_key, "#8B949E")
        _fb_bg    = {"FRESH": "#0d2618", "STALE": "#2d2208", "THIN": "#161B22"}.get(_fresh_key, "#161B22")
        st.markdown(
            f'<div style="display:inline-block;background:{_fb_bg};'
            f'border:1px solid {_fb_color};border-radius:6px;padding:5px 12px;margin-bottom:12px;">'
            f'<span style="font-size:11px;font-weight:700;letter-spacing:1px;color:{_fb_color};">'
            f'📰 NEWS: {_freshness}</span></div>',
            unsafe_allow_html=True,
        )

    _conflict = str(analysis.get("conflicting_signals", "")).strip()
    if _conflict and _conflict.lower() not in ("none found", "none", "n/a", ""):
        st.markdown(
            f'<div style="background:#2d2208;border:1px solid #F0B429;'
            f'border-radius:8px;padding:12px 18px;margin-bottom:14px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#F0B429;margin-bottom:5px;">⚠ Conflicting Reports</div>'
            f'<div style="font-size:13px;color:#C9D1D9;line-height:1.6;">{_conflict}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Move Diagnosis (deterministic, ALWAYS shown) ──────────────────────────
    # Grounds the move in data the LLM doesn't have: how many sector peers also
    # hit the extreme (sector-wide vs company-specific), earnings trajectory,
    # analyst actions, sector-index tailwind and the momentum origin. 0 tokens.
    import html as _html
    import analysis_utils as _au
    try:
        _dx_origin = None
        try:
            _wide  = _fetch_all_news_items(ticker, name, cutoff_days=60) or []
            # Optional LLM sentiment+entity guard (#6b): enable with
            # POLICY_LLM_GUARD=1 in .env. Default OFF to protect Groq free-tier
            # limits; the hardened keyword guard is used otherwise.
            _llm_client = _llm_model = None
            try:
                if _AI_OK and str(_secret("POLICY_LLM_GUARD", "")).strip() in ("1", "true", "yes"):
                    _llm_client = _OpenAI(
                        api_key=_OPENAI_KEY,
                        base_url=_llm_base_url(),
                    )
                    _llm_model = _QUICK_MODEL
            except Exception:
                _llm_client = _llm_model = None
            _dx_origin = _au.pick_momentum_origin(
                _wide, name, ticker.replace(".NS", "").replace(".BO", ""),
                is_hi, max_age_days=60,
                llm_client=_llm_client, llm_model=_llm_model or "",
            )
        except Exception:
            _dx_origin = None
        try:
            _dx_idx_pct, _dx_idx_name = _fetch_sector_index_move(row.get("sector", ""))
        except Exception:
            _dx_idx_pct, _dx_idx_name = (None, "")

        # ── Thematic breadth (cross-sector basket: defence, railways, EV…) ────
        _hi, _lo = st.session_state.get("_last_highs"), st.session_state.get("_last_lows")
        _dx_theme, _dx_policy, _theme_name = None, [], ""
        try:
            import themes as _themes
            _dx_theme = _themes.theme_breadth(ticker, _hi, _lo, is_hi)
            _theme_name = (_dx_theme or {}).get("theme", "")
            _canon_sector = _themes.normalise_sector(row.get("sector", ""))
        except Exception:
            _canon_sector = row.get("sector", "")
        # ── Policy / trade catalyst (FTA, tariff, PLI, GST, ban) ──────────────
        try:
            import policy_news as _pol
            _dx_policy = _pol.policy_drivers_for(_canon_sector, _theme_name, is_hi)
        except Exception:
            _dx_policy = []

        _diag = _au.diagnose_move(
            sector=row.get("sector", ""), ticker=ticker, is_hi=is_hi,
            highs_df=_hi, lows_df=_lo,
            vsurge=vsurge, ret_1m=ret_1m, ret_3m=ret_3m, ret_6m=ret_6m, ret_1y=ret_1y,
            earnings_surprise_context=locals().get("earnings_surprise_ctx", ""),
            upgrades_context=locals().get("upgrades_ctx", ""),
            index_move_pct=_dx_idx_pct, index_name=_dx_idx_name,
            momentum_origin=_dx_origin,
            theme_breadth=_dx_theme, policy_drivers=_dx_policy,
        )
    except Exception:
        _diag = {"classification": "", "headline": "", "drivers": []}

    if _diag.get("drivers"):
        _dx_html = ""
        for _d in _diag["drivers"]:
            _txt = _html.escape(_d.get("text", ""))
            _lbl = _html.escape(_d.get("label", ""))
            _u   = (_d.get("url") or "").strip()
            _lk  = (f' <a href="{_u}" target="_blank" rel="noopener noreferrer" '
                    f'style="color:#388bfd;font-size:11px;text-decoration:none;">↗ source</a>'
                    ) if _u.startswith("http") else ""
            _dx_html += (
                f'<div style="display:flex;gap:10px;margin-top:9px;">'
                f'<span style="font-size:14px;line-height:1.3;">{_d.get("icon", "•")}</span>'
                f'<div style="flex:1;">'
                f'<span style="font-size:12px;font-weight:700;color:#C9D1D9;">{_lbl}.</span> '
                f'<span style="font-size:13px;color:#8B949E;line-height:1.6;">{_txt}{_lk}</span>'
                f'</div></div>'
            )
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:10px;padding:14px 20px;margin:4px 0 14px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:#8B949E;margin-bottom:4px;">'
            f'&#128202; Move Diagnosis &nbsp;·&nbsp; {_html.escape(_diag.get("classification", ""))}</div>'
            f'<div style="font-size:13px;color:#C9D1D9;line-height:1.5;">'
            f'{_html.escape(_diag.get("headline", ""))}</div>'
            f'{_dx_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Primary Catalyst (Idea 1) ─────────────────────────────────────────────
    primary = analysis.get("primary_catalyst", {})
    if _is_stale_news:
        # Drivers already appear in the Move Diagnosis panel above — here we just
        # state the classification + headline as the primary driver.
        st.markdown(
            f'<div style="background:#161B22;border:1px solid {accent};'
            f'border-radius:10px;padding:16px 22px;margin:8px 0 16px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:{accent};margin-bottom:6px;">'
            f'&#9889; Primary Driver &nbsp;·&nbsp; {_html.escape(_diag.get("classification", "") or "Momentum-driven")}</div>'
            f'<div style="font-size:15px;font-weight:600;color:#E6EDF3;line-height:1.5;">'
            f'{_html.escape(_diag.get("headline", "") or "No single fresh catalyst — momentum / sector-driven.")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif primary and primary.get("headline"):
        p_headline  = primary.get("headline", "")
        p_detail    = primary.get("detail", "")
        p_impact    = primary.get("impact_pct", "")
        impact_html = (f'<span style="font-size:13px;font-weight:700;color:{accent};'
                       f'margin-left:10px;">{p_impact}</span>') if p_impact and p_impact != "N/A" else ""
        st.markdown(
            f'<div style="background:{"#0d2618" if is_hi else "#2a0d12"};'
            f'border:2px solid {accent};border-radius:10px;'
            f'padding:18px 22px;margin:8px 0 16px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:{accent};margin-bottom:10px;">'
            f'&#9889; Primary Driver{impact_html}</div>'
            f'<div style="font-size:16px;font-weight:700;color:#E6EDF3;'
            f'line-height:1.4;margin-bottom:8px;">{p_headline}</div>'
            f'<div style="font-size:13px;color:#C9D1D9;line-height:1.6;">{p_detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary:
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:8px;padding:16px 20px;margin:8px 0 14px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:#8B949E;margin-bottom:8px;">&#9670; Analyst Summary</div>'
            f'<div style="font-size:14px;color:#C9D1D9;line-height:1.7;">{summary}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Business Model ────────────────────────────────────────────────────────
    business_model = analysis.get("business_model", "")
    if business_model:
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:8px;padding:16px 20px;margin:8px 0 14px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:#8B949E;margin-bottom:8px;">&#9670; Business Model</div>'
            f'<div style="font-size:14px;color:#C9D1D9;line-height:1.7;">{business_model}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Reported Quarterly Results (FMP ground truth) ─────────────────────────
    try:
        from fmp import get_quarterly_summary
        _qs = get_quarterly_summary(ticker, limit=5) if _FMP_OK else []
    except Exception:
        _qs = []
    if _qs:
        def _cr(v):
            try:
                return f"Rs {float(v)/1e7:,.0f} Cr"
            except Exception:
                return "—"
        _rows = ""
        for q in _qs:
            beat = q.get("beat_pct")
            if beat is None:
                beat_html = '<span style="color:#6E7681;">—</span>'
            else:
                _bc = "#3FB950" if beat >= 0 else "#F85149"
                beat_html = f'<span style="color:{_bc};">{beat:+.1f}%</span>'
            eps_a = q.get("eps_actual")
            eps_txt = f"Rs {float(eps_a):,.2f}" if eps_a not in (None, "") else "—"
            _rows += (
                f'<tr>'
                f'<td style="padding:6px 10px;color:#C9D1D9;">{q.get("date","")}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:#C9D1D9;">{_cr(q.get("revenue"))}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:#C9D1D9;">{_cr(q.get("net_income"))}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:#C9D1D9;">{eps_txt}</td>'
                f'<td style="padding:6px 10px;text-align:right;">{beat_html}</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:8px;padding:14px 18px;margin:8px 0 14px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:#8B949E;margin-bottom:10px;">'
            f'&#9670; Reported Quarterly Results <span style="color:#6E7681;font-weight:400;'
            f'text-transform:none;letter-spacing:0;">(verified · Financial Modeling Prep)</span></div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            f'<tr style="color:#6E7681;border-bottom:1px solid #30363D;">'
            f'<th style="padding:6px 10px;text-align:left;font-weight:600;">Quarter end</th>'
            f'<th style="padding:6px 10px;text-align:right;font-weight:600;">Revenue</th>'
            f'<th style="padding:6px 10px;text-align:right;font-weight:600;">Net income</th>'
            f'<th style="padding:6px 10px;text-align:right;font-weight:600;">EPS</th>'
            f'<th style="padding:6px 10px;text-align:right;font-weight:600;">vs est</th>'
            f'</tr>{_rows}</table></div>',
            unsafe_allow_html=True,
        )

    # ── Catalyst Timeline (Idea 2) ────────────────────────────────────────────
    timeline = analysis.get("catalyst_timeline", [])
    if timeline:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            'text-transform:uppercase;color:#8B949E;margin:16px 0 10px;">&#9670; How We Got Here</div>',
            unsafe_allow_html=True,
        )
        tl_html = ""
        for i, evt in enumerate(timeline):
            is_last = i == len(timeline) - 1
            tl_html += (
                f'<div style="display:flex;gap:12px;margin-bottom:{"4px" if not is_last else "0"};">'
                f'<div style="display:flex;flex-direction:column;align-items:center;">'
                f'<div style="width:10px;height:10px;border-radius:50%;background:{accent};'
                f'flex-shrink:0;margin-top:4px;"></div>'
                + (f'<div style="width:2px;flex:1;background:#30363D;margin-top:2px;"></div>'
                   if not is_last else "")
                + f'</div>'
                f'<div style="padding-bottom:14px;">'
                f'<div style="font-size:11px;color:{accent};font-weight:600;">{evt.get("date","")}</div>'
                f'<div style="font-size:13px;color:#E6EDF3;font-weight:500;">{evt.get("event","")}</div>'
                f'<div style="font-size:12px;color:#8B949E;">{evt.get("impact","")}</div>'
                f'</div></div>'
            )
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:8px;padding:16px 18px;margin-bottom:14px;">{tl_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Catalyst cards (Idea 6 — with impact%) ───────────────────────────────
    catalysts = analysis.get("catalysts", [])
    if catalysts:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            'text-transform:uppercase;color:#8B949E;margin:4px 0 10px;">Key Catalysts</div>',
            unsafe_allow_html=True,
        )
        import html as _html
        import urllib.parse as _up
        for cat in catalysts:
            cat_type   = str(cat.get("type", "neutral")).lower()
            headline   = _html.escape(cat.get("headline", ""))
            detail     = _html.escape(cat.get("detail", ""))
            cat_date   = _html.escape(cat.get("date", ""))
            cat_source = _html.escape(cat.get("source", ""))
            cat_impact = cat.get("impact_pct", "")

            # Link resolution priority:
            #   1. The LLM's URL, but only if validated against context upstream.
            #   2. Deterministic match of the headline to a real fetched article
            #      (live news + fresh news pool) — fixes the common case where the
            #      model couldn't copy a URL verbatim.
            #   3. Last resort: Google NEWS results (not a plain web search).
            _real_url = (cat.get("url") or "").strip()
            if not _real_url.startswith("http"):
                _real_url = _match_article_url(cat.get("headline", ""),
                                               gn_articles, fresh_news_pool)
            if _real_url.startswith("http"):
                cat_url   = _real_url
                _link_txt = "↗ Read article"
            else:
                _q = _up.quote_plus(f"{cat.get('headline', '')} {name}")
                cat_url   = f"https://news.google.com/search?q={_q}"
                _link_txt = "↗ Find article"

            _icon  = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(cat_type, "➖")
            _cbg   = {"positive": "#0d2618", "negative": "#2a0d12", "neutral": "#161B22"}.get(cat_type, "#161B22")
            _cbdr  = {"positive": "#238636", "negative": "#8B1A1A", "neutral": "#30363D"}.get(cat_type, "#30363D")
            _chover = {"positive": "#0f3020", "negative": "#331018", "neutral": "#1c2128"}.get(cat_type, "#1c2128")
            _cmeta = f"{cat_date}  ·  {cat_source}" if cat_date or cat_source else ""
            _impact_chip = (
                f'<span style="font-size:11px;font-weight:700;'
                f'background:{"#1a3828" if cat_type=="positive" else "#3b1219" if cat_type=="negative" else "#21262D"};'
                f'color:{"#3FB950" if cat_type=="positive" else "#F85149" if cat_type=="negative" else "#8B949E"};'
                f'border-radius:4px;padding:2px 7px;margin-left:8px;">{cat_impact}</span>'
            ) if cat_impact and cat_impact != "N/A" else ""

            st.markdown(
                f'<a href="{cat_url}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">'
                f'<div style="background:{_cbg};border:1px solid {_cbdr};'
                f'border-radius:8px;padding:14px 18px;margin-bottom:10px;'
                f'cursor:pointer;transition:background 0.15s;"'
                f' onmouseover="this.style.background=\'{_chover}\'"'
                f' onmouseout="this.style.background=\'{_cbg}\'">'
                f'<div style="display:flex;align-items:flex-start;gap:10px;">'
                f'<span style="font-size:18px;line-height:1.3">{_icon}</span>'
                f'<div style="flex:1">'
                f'<div style="font-size:14px;font-weight:600;color:#E6EDF3;'
                f'line-height:1.4;margin-bottom:6px;">{headline}{_impact_chip}</div>'
                f'<div style="font-size:13px;color:#C9D1D9;line-height:1.6;">{detail}</div>'
                + (f'<div style="font-size:11px;color:#6E7681;margin-top:8px;">'
                   f'{_cmeta}'
                   f'<span style="margin-left:8px;color:#388bfd;font-size:10px;">{_link_txt}</span>'
                   f'</div>'
                   if _cmeta else
                   f'<div style="font-size:11px;color:#388bfd;margin-top:8px;">{_link_txt}</div>')
                + f'</div></div></div></a>',
                unsafe_allow_html=True,
            )

    # ── Peer Context (Idea 4) ───────────────────────────────────────────────
    peer_context_txt = analysis.get("peer_context", "")
    if peer_context_txt:
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:8px;padding:14px 18px;margin:4px 0 14px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:#8B949E;margin-bottom:8px;">&#9670; Sector Context</div>'
            f'<div style="font-size:13px;color:#C9D1D9;line-height:1.6;">{peer_context_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── What to Watch Next (Idea 5) ───────────────────────────────────────────
    watch_next_raw = analysis.get("watch_next", [])

    # Drop items whose date is clearly in the past.
    def _is_future_event(w: dict) -> bool:
        date_str = w.get("date", "").strip()
        if not date_str:
            return True  # no date → keep
        try:
            from dateutil import parser as _dp
            parsed = _dp.parse(date_str, default=datetime(datetime.now().year, 12, 31))
            return parsed.date() >= datetime.now().date()
        except Exception:
            return True  # unparseable → keep to be safe

    watch_next = [w for w in watch_next_raw if _is_future_event(w)]

    if watch_next:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:2px;'
            'text-transform:uppercase;color:#8B949E;margin:4px 0 10px;">&#9670; What to Watch Next</div>',
            unsafe_allow_html=True,
        )
        wn_html = ""
        for w in watch_next:
            wn_html += (
                f'<div style="display:flex;gap:14px;align-items:flex-start;'
                f'padding:10px 0;border-bottom:1px solid #21262D;">'
                f'<div style="min-width:110px;font-size:11px;color:{accent};'
                f'font-weight:600;padding-top:2px;">{w.get("date","")}</div>'
                f'<div>'
                f'<div style="font-size:13px;color:#E6EDF3;font-weight:600;">{w.get("event","")}</div>'
                f'<div style="font-size:12px;color:#8B949E;margin-top:2px;">{w.get("implication","")}</div>'
                f'</div></div>'
            )
        st.markdown(
            f'<div style="background:#161B22;border:1px solid #30363D;'
            f'border-radius:8px;padding:4px 18px;margin-bottom:14px;">{wn_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Sources ─────────────────────────────────────────────────────────────────────
    sources = analysis.get("sources", [])
    if sources:
        src_html = "  &nbsp;·&nbsp;  ".join(
            f'<span style="color:#8B949E">{s}</span>' for s in sources
        )
        st.markdown(
            f'<div style="font-size:11px;color:#6E7681;padding:4px 0 8px;">'
            f'<b style="color:#8B949E">Sources:</b>&nbsp;&nbsp;{src_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Full Google News feed ─────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#21262D;margin:20px 0 4px;'>",
                unsafe_allow_html=True)
    _render_news_feed(name, ticker, accent)

    # ── Download PDF report ─────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#21262D;margin:0 0 12px;'>",
                unsafe_allow_html=True)

    with st.spinner("Building PDF report…"):
        articles   = _fetch_live_news(name, ticker)   # already cached
        pdf_bytes  = _build_pdf_report(
            ticker   = ticker,
            company  = name,
            sector   = row.get("sector", ""),
            signal   = sig,
            price    = price or 0,
            high52   = high52 or 0,
            low52    = low52 or 0,
            pfh      = pfh,
            pfl      = pfl,
            vsurge   = vsurge or 0,
            pe       = pe or 0,
            analysis = analysis,
            articles = articles,
        )

    fname = (f"{ticker.replace('.NS','')}_research_"
             f"{datetime.now().strftime('%Y%m%d')}.pdf")
    st.download_button(
        label     = "⬇  Download Full Research Report (PDF)",
        data      = pdf_bytes,
        file_name = fname,
        mime      = "application/pdf",
        use_container_width = True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def _render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#E6EDF3;'
            'padding:12px 0 4px;letter-spacing:-0.3px;">&#9881; Screener Controls</div>',
            unsafe_allow_html=True,
        )

        # ── Universe ────────────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#6E7681;'
            'letter-spacing:2px;text-transform:uppercase;margin:12px 0 6px;">Universe</div>',
            unsafe_allow_html=True,
        )
        universe = st.radio(
            "Universe",
            options=["Nifty 500", "All NSE (~2000)", "Nifty 50", "Nifty 100",
                     "Nifty Midcap 100", "Nifty Smallcap 100", "Custom"],
            index=0,
            label_visibility="collapsed",
        )
        custom_txt = ""
        if universe == "Custom":
            custom_txt = st.text_area(
                "Custom tickers (one per line, e.g. RELIANCE.NS):",
                height=120,
                placeholder="RELIANCE.NS\nINFY.NS\nHDFCBANK.NS",
            )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Signal filters ──────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#6E7681;'
            'letter-spacing:2px;text-transform:uppercase;margin:4px 0 6px;">Signal Filters</div>',
            unsafe_allow_html=True,
        )
        threshold   = 0.0   # strict: only stocks that actually HIT the 52W extreme
        window      = 252   # 52-week lookback
        vol_surge   = st.slider("Min volume surge multiple (x)", 1.0, 5.0, 1.5, 0.5)
        use_vol_filter = st.toggle("Volume Surge filter", value=False,
                                   help="Only show stocks with unusual volume")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Auto-refresh ────────────────────────────────────────────────────
        auto_refresh = st.toggle("Auto-refresh index prices (60s)", value=False)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Run button ──────────────────────────────────────────────────────
        run = st.button("▶  Run Screen", width='stretch', type="primary")

        # ── Clear AI cache ──────────────────────────────────────────────────
        ai_keys = [k for k in st.session_state if k.startswith("ai_")]
        if ai_keys:
            if st.button(
                f"🗑  Clear AI Cache ({len(ai_keys)} entries)",
                width='stretch',
                help="Clears cached primary catalysts so they are re-fetched fresh on next screen run.",
            ):
                for k in ai_keys:
                    del st.session_state[k]
                st.success("AI cache cleared — run the screen again to refresh catalysts.")
                st.rerun()

        # ── LLM usage / cost guard ──────────────────────────────────────────
        # Perplexity Sonar calls are billed per token + per request, so surface a
        # running tally to avoid a surprise on the bill after a heavy screen day.
        _calls = st.session_state.get("_llm_call_count", 0)
        if _calls:
            if _LLM_PROVIDER == "perplexity":
                _cost = st.session_state.get("_llm_cost_usd", 0.0)
                _usage_txt = f"{_calls} calls · ≈${_cost:,.3f} est."
                _usage_hint = ("Estimated Perplexity spend this session "
                               "(tokens + per-request search fee).")
            else:
                _usage_txt = f"{_calls} LLM calls this session"
                _usage_hint = "Provider is not billed per call on this plan."
            st.markdown(
                '<div style="font-size:10px;font-weight:700;color:#6E7681;'
                'letter-spacing:2px;text-transform:uppercase;margin:10px 0 4px;">LLM Usage</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:13px;color:#8B949E;" title="{_usage_hint}">'
                f'💸 {_usage_txt}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Downloads (only when results exist) ─────────────────────────────
        if "screen_results" in st.session_state:
            results = st.session_state["screen_results"]
            highs_df = results.get("highs", pd.DataFrame())
            lows_df  = results.get("lows",  pd.DataFrame())
            if not highs_df.empty or not lows_df.empty:
                all_df = pd.concat([highs_df, lows_df], ignore_index=True)
                csv = all_df.to_csv(index=False).encode()
                st.download_button(
                    "⬇  Download CSV",
                    data=csv,
                    file_name=f"nifty52w_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width='stretch',
                )

    return {
        "threshold":      threshold,
        "vol_surge":      vol_surge,
        "use_vol_filter": use_vol_filter,
        "window":         window,
        "universe":       universe,
        "custom_txt":     custom_txt,
        "run":            run,
        "auto_refresh":   auto_refresh,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def _signals_with_spotlight(df, key: str, default_sig: str, params: dict):
    """Render the signal cards; each card's 'Detailed Analysis' button shows the
    full deep-dive report inline beneath that card."""
    _render_signals_table(df, key=key, params=params)


def main():
    # ── Inject styles ──────────────────────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Ticker tape ────────────────────────────────────────────────────────
    _render_tape()

    # ── App header ─────────────────────────────────────────────────────────
    _render_header()

    # ── Sidebar ────────────────────────────────────────────────────────────
    p = _render_sidebar()

    # ── Spacing ────────────────────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Index cards ────────────────────────────────────────────────────────
    _render_index_cards()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Auto-refresh clears cached data ───────────────────────────────────
    if p["auto_refresh"]:
        _fetch_index_data.clear()
        _fetch_tape_data.clear()

        # ── Build ticker list ──────────────────────
    if p["universe"] == "Custom" and p["custom_txt"].strip():
        tickers = [t.strip() for t in p["custom_txt"].splitlines() if t.strip()]
    else:
        live = _fetch_index_tickers(p["universe"])
        if live:
            tickers = live
        elif p["universe"] == "Nifty 50":
            tickers = _NIFTY_50
        else:
            tickers = fetch_nifty500_live() or NIFTY_500_TICKERS or _NIFTY_50


    # ── Run screen (only on explicit button click) ─────────────────
    if p["run"]:
        with st.status("Running screen…", expanded=True) as status:
            st.write(f"⬇ Downloading price history for {len(tickers)} stocks via yfinance…")
            results = _run_screen(tickers, p)
            n_hi = len(results.get("highs", pd.DataFrame()))
            n_lo = len(results.get("lows",  pd.DataFrame()))
            status.update(
                label=f"✅ Screen complete — {n_hi} breakout highs · {n_lo} breakdown lows",
                state="complete", expanded=False,
            )

        # Auto AI enrichment removed — analysis is now on-demand per stock (Major Catalyst button)

    elif "screen_results" in st.session_state:
        results = st.session_state["screen_results"]
    else:
        # Fall back to the pre-computed snapshot (scheduled background job) so the
        # dashboard shows fresh results instantly without waiting for a live scan.
        snap = _load_snapshot()
        if snap:
            results = snap
            st.session_state["screen_results"] = snap
            st.session_state["_screen_key"]    = "__snapshot__"
            try:
                _gen = snap.get("generated_at_utc")
                st.session_state["screen_ts"] = (
                    datetime.fromisoformat(_gen).astimezone() if _gen else datetime.now()
                )
            except Exception:
                st.session_state["screen_ts"] = datetime.now()
        else:
            results = None

    if not results:
        st.markdown(
            "<div style='text-align:center;padding:80px;color:#484F58;'>"
            "<div style='font-size:48px;'>📊</div>"
            "<div style='font-size:18px;font-weight:600;margin-top:16px;'>Nifty 52-Week Screener</div>"
            "<div style='font-size:14px;margin-top:8px;color:#6B7280;'>"
            "Select a universe in the sidebar, then press <b>▶ Run Screen</b> to find today's breakouts."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    highs_df = results.get("highs", pd.DataFrame())
    lows_df  = results.get("lows",  pd.DataFrame())
    errors   = results.get("errors", [])
    is_mock  = results.get("is_mock", False)

    # Stash for sector-breadth analysis inside the deep-dive spotlight
    st.session_state["_last_highs"] = highs_df
    st.session_state["_last_lows"]  = lows_df

    if is_mock:
        st.warning(
            "⚠ **Mock data active** — `engine.py` not found or all API calls failed. "
            "Install dependencies and ensure `engine.py` is in the same folder.",
            icon="⚠",
        )

    if highs_df.empty and lows_df.empty and not is_mock:
        ts = st.session_state.get("screen_ts", datetime.now()).strftime("%H:%M:%S")
        st.info(
            f"**No stocks hit a 52-week extreme today** (screen run at {ts}).\n\n"
            "The screener only flags stocks whose **session high** touched or exceeded "
            "the 52-week high, or whose **session low** touched or breached the 52-week low. "
            "On most trading days only a handful of stocks qualify — on quiet days, none.\n\n"
            f"✅ Data feed is working — {len(tickers)} stocks were screened via yfinance."
        )
        if errors:
            with st.expander(f"⚠ {len(errors)} fetch errors"):
                st.dataframe(pd.DataFrame(errors), width='stretch')
        return

    if results.get("from_snapshot"):
        st.caption(
            f"\U0001F7E2 Auto-updated snapshot \u00b7 last scan {results.get('generated_at_ist', '')} "
            f"\u00b7 refreshes automatically every 30 min \u00b7 press \u25b6 Run Screen for a live scan"
        )

    st.markdown(
        f"<div class='sec-hdr' style='margin-top:8px;'>"
        f"&#9670; Today's Signals &nbsp;"
        f"<span style='color:{_GREEN}'>&#8679; {len(highs_df)} Breakout Highs</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:{_RED}'>&#8681; {len(lows_df)} Breakdown Lows</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    t_hi, t_lo, t_err = st.tabs([
        f"▲  Breakout Highs  ({len(highs_df)})",
        f"▼  Breakdown Lows  ({len(lows_df)})",
        f"⚠  Errors  ({len(errors)})",
    ])

    with t_hi:
        if isinstance(highs_df, pd.DataFrame) and "high_type" in highs_df.columns:
            _new_df  = highs_df[highs_df["high_type"] == "NEW"]
            _cont_df = highs_df[highs_df["high_type"] != "NEW"]
        else:
            _empty = highs_df.iloc[0:0] if isinstance(highs_df, pd.DataFrame) else pd.DataFrame()
            _new_df, _cont_df = _empty, highs_df
        st.caption(
            "**New** = first 52-week high in ≥ 40 days  ·  "
            "**Continuation** = also hit a 52-week high within the last 40 days"
        )
        st_new, st_cont = st.tabs([
            f"🆕  New 52W High  ({len(_new_df)})",
            f"🔁  Continuation  ({len(_cont_df)})",
        ])
        with st_new:
            _signals_with_spotlight(_new_df, "hi_new", "BREAKOUT_HIGH", p)
        with st_cont:
            _signals_with_spotlight(_cont_df, "hi_cont", "BREAKOUT_HIGH", p)

    with t_lo:
        _signals_with_spotlight(lows_df, "lo", "BREAKDOWN_LOW", p)

    with t_err:
        if not errors:
            st.success("✅ No errors in the last run — all tickers processed cleanly.")
        else:
            err_df = pd.DataFrame(errors)
            st.dataframe(err_df, width='stretch')

    if p["auto_refresh"]:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
