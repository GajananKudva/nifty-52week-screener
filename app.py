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


# ── AI client — supports OpenAI and Groq (same SDK, different base_url) ──────
# Set LLM_PROVIDER=groq in .env to use Groq's free tier (llama-3.3-70b-versatile)
# Set LLM_PROVIDER=openai (or leave blank) for OpenAI
_LLM_PROVIDER = _secret("LLM_PROVIDER", "openai").lower()
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_KEY   = _secret("OPENAI_API_KEY")
    _OPENAI_MODEL = _secret("OPENAI_MODEL",
                            "llama-3.3-70b-versatile" if _LLM_PROVIDER == "groq"
                            else "gpt-4o-mini")
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

# ── Engine import (graceful fallback to mock data) ────────────────────────────
try:
    from engine import DataEngine, ScreenerConfig
    _ENGINE_OK = True
except ImportError:
    _ENGINE_OK = False

# ── Nifty 500 universe import ─────────────────────────────────────────────────
try:
    from mailer import NIFTY_500_TICKERS, fetch_nifty500_live
    _MAILER_OK = True
except ImportError:
    _MAILER_OK = False
    NIFTY_500_TICKERS: list[str] = []

    def fetch_nifty500_live() -> list[str]:
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
    "TATAMOTORS.NS": "TATA MOTORS",
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
    "BAJAJFINSV.NS","TATAMOTORS.NS","ADANIENT.NS", "ADANIPORTS.NS","TATASTEEL.NS",
    "COALINDIA.NS","NESTLEIND.NS", "DIVISLAB.NS",  "CIPLA.NS",     "DRREDDY.NS",
    "HINDALCO.NS", "TECHM.NS",     "GRASIM.NS",    "APOLLOHOSP.NS","TATACONSUM.NS",
    "INDUSINDBK.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","EICHERMOT.NS","BRITANNIA.NS",
    "BEL.NS",      "SHRIRAMFIN.NS","BPCL.NS",      "TRENT.NS",     "SBILIFE.NS",
]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MOCK DATA  (fallback when API rate-limited or engine unavailable)
# ══════════════════════════════════════════════════════════════════════════════
def _mock_index_data() -> dict:
    return {
        "NIFTY 50":   {"price": 22_450.55, "chg": +0.45, "pts": +100.55},
        "SENSEX":     {"price": 73_821.30, "chg": +0.32, "pts": +236.50},
        "NIFTY BANK": {"price": 48_112.75, "chg": -0.18, "pts":  -87.25},
        "INDIA VIX":  {"price":    14.22,  "chg": -1.10, "pts":   -0.16},
    }


def _mock_tape_data() -> list[dict]:
    return [
        {"name": "NIFTY 50",    "price": 22450.55, "chg": +0.45},
        {"name": "SENSEX",      "price": 73821.30, "chg": +0.32},
        {"name": "RELIANCE",    "price": 2941.80,  "chg": +1.20},
        {"name": "TCS",         "price": 3812.55,  "chg": -0.38},
        {"name": "HDFCBANK",    "price": 1721.45,  "chg": +0.55},
        {"name": "INFOSYS",     "price": 1542.90,  "chg": -0.22},
        {"name": "ICICIBANK",   "price": 1198.70,  "chg": +0.88},
        {"name": "AIRTEL",      "price": 1650.30,  "chg": +1.42},
        {"name": "TATA MOTORS", "price":  812.45,  "chg": -1.15},
        {"name": "WIPRO",       "price":  462.80,  "chg": +0.30},
        {"name": "BAJAJ FIN",   "price": 7120.50,  "chg": +0.68},
        {"name": "MARUTI",      "price": 12310.00, "chg": +0.12},
    ]


def _mock_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    highs = pd.DataFrame([
        {
            "ticker": "RELIANCE.NS", "company_name": "Reliance Industries",
            "sector": "Energy",      "current_price": 2941.80,
            "week_52_high": 2960.00, "week_52_low": 2220.00,
            "pct_from_high": 0.61,   "pct_from_low": 32.52,
            "volume_surge": 1.85,    "pe_ratio": 24.3,
            "intrinsic_value": 3180.00, "upside_downside_pct": 8.10,
            "ret_1m": 4.20, "ret_3m": 11.80, "signal": "BREAKOUT_HIGH",
            "why_text": "New energy investments and Jio subscriber growth exceeding analyst expectations drove the breakout.",
            "news_headlines": [
                "Reliance Q4 profit beats estimates on retail, Jio strength [ET]",
                "Reliance New Energy unit secures $2B green financing [Reuters]",
            ],
        },
        {
            "ticker": "HCLTECH.NS",  "company_name": "HCL Technologies",
            "sector": "IT",          "current_price": 1592.40,
            "week_52_high": 1601.00, "week_52_low": 1235.00,
            "pct_from_high": 0.54,   "pct_from_low": 28.90,
            "volume_surge": 2.10,    "pe_ratio": 28.7,
            "intrinsic_value": 1720.00, "upside_downside_pct": 8.02,
            "ret_1m": 6.10, "ret_3m": 14.20, "signal": "BREAKOUT_HIGH",
            "why_text": "Strong Q4 deal wins in AI services and cloud migration mandates accelerated above consensus.",
            "news_headlines": [
                "HCL Tech bags $500M AI services deal with European bank [Mint]",
                "IT sector re-rating as AI spending cycle begins [Bloomberg]",
            ],
        },
        {
            "ticker": "BHARTIARTL.NS","company_name": "Bharti Airtel",
            "sector": "Telecom",      "current_price": 1648.90,
            "week_52_high": 1660.00,  "week_52_low": 1090.00,
            "pct_from_high": 0.67,    "pct_from_low": 51.28,
            "volume_surge": 1.42,     "pe_ratio": 45.2,
            "intrinsic_value": 1580.00,"upside_downside_pct": -4.18,
            "ret_1m": 3.80, "ret_3m": 18.40, "signal": "BREAKOUT_HIGH",
            "why_text": "ARPU expansion from 5G monetisation and Jio price-hike spillover reducing competitive intensity.",
            "news_headlines": [
                "Airtel raises mobile tariffs by 10-25% across plans [Economic Times]",
                "5G rollout reaches 500 cities; ARPU to hit ₹250 in FY26 [ICICI Securities]",
            ],
        },
    ])
    lows = pd.DataFrame([
        {
            "ticker": "NYKAA.NS",    "company_name": "FSN E-Commerce (Nykaa)",
            "sector": "Retail",      "current_price": 142.55,
            "week_52_high": 218.00,  "week_52_low": 139.00,
            "pct_from_high": 34.61,  "pct_from_low": 2.55,
            "volume_surge": 2.35,    "pe_ratio": 98.4,
            "intrinsic_value": 110.00,"upside_downside_pct": -22.84,
            "ret_1m": -12.30, "ret_3m": -28.50, "signal": "BREAKDOWN_LOW",
            "why_text": "Margin pressure from intensified beauty brand competition and elevated customer acquisition costs.",
            "news_headlines": [
                "Nykaa Q3 EBITDA margins disappoint at 4.2% vs 6.1% expected [Jefferies]",
                "Reliance Tira Beauty expansion threatens Nykaa market share [Mint]",
            ],
        },
        {
            "ticker": "PAYTM.NS",    "company_name": "One97 Communications",
            "sector": "Fintech",     "current_price": 412.80,
            "week_52_high": 698.00,  "week_52_low": 402.00,
            "pct_from_high": 40.86,  "pct_from_low": 2.69,
            "volume_surge": 3.12,    "pe_ratio": None,
            "intrinsic_value": None, "upside_downside_pct": None,
            "ret_1m": -18.40, "ret_3m": -35.20, "signal": "BREAKDOWN_LOW",
            "why_text": "RBI action on Paytm Payments Bank continues to erode GMV and loan disbursement volumes.",
            "news_headlines": [
                "Paytm GMV falls 45% YoY after Payments Bank shutdown [HDFC Sec]",
                "Paytm losses mount as loan portfolio shrinks rapidly [ET Markets]",
            ],
        },
        {
            "ticker": "ZOMATO.NS",   "company_name": "Zomato",
            "sector": "Food Delivery","current_price": 192.40,
            "week_52_high": 289.00,  "week_52_low": 188.00,
            "pct_from_high": 33.43,  "pct_from_low": 2.34,
            "volume_surge": 1.68,    "pe_ratio": 310.5,
            "intrinsic_value": 148.00,"upside_downside_pct": -23.08,
            "ret_1m": -9.80, "ret_3m": -21.40, "signal": "BREAKDOWN_LOW",
            "why_text": "Food delivery growth deceleration and Blinkit unit economics concerns after aggressive dark-store expansion.",
            "news_headlines": [
                "Zomato food delivery order growth slows to 8% QoQ in Q3 [Kotak]",
                "Blinkit reaches 1,000 stores but losses widen on infra costs [Bloomberg Quint]",
            ],
        },
    ])
    return highs, lows


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


def _render_signals_table(df: pd.DataFrame, key: str) -> Optional[str]:
    """Render signal cards (one per stock, with insight bullets). Returns selected ticker or None."""
    if df is None or df.empty:
        st.info("No signals meeting the current criteria.")
        return None

    is_hi   = key == "hi"
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

        # ── Business model (1 sentence from yfinance, cached) ───────────────
        biz_line = ""
        try:
            biz_ctx = _fetch_yf_company_context(ticker)
            if biz_ctx:
                # Extract first sentence of Business Description
                desc_start = biz_ctx.find("Business Description:\n")
                if desc_start >= 0:
                    desc_text = biz_ctx[desc_start + len("Business Description:\n"):]
                    # Take first sentence (up to first period + space or 140 chars)
                    dot = desc_text.find(". ")
                    snippet = desc_text[:dot + 1] if 0 < dot < 140 else desc_text[:140]
                    biz_line = (
                        f'<div style="font-size:12px;color:#8B949E;margin:3px 0 6px;'
                        f'line-height:1.5;font-style:italic;">{snippet}</div>'
                    )
        except Exception:
            pass

        # ── Primary catalyst: AI cache if analyzed, else best heuristic ──────
        catalyst_label = "Primary Catalyst"
        catalyst_color = "#3FB950" if is_hi else "#F85149"
        catalyst_text  = ""
        catalyst_impact = ""

        # Check AI cache first
        for ck in [f"ai_{ticker}_52W_HIGH", f"ai_{ticker}_52W_LOW"]:
            cached_ai = st.session_state.get(ck, {})
            pc = cached_ai.get("primary_catalyst", {})
            if pc and pc.get("headline"):
                _h = pc["headline"]
                catalyst_text  = (_h[:197] + "…") if len(_h) > 200 else _h
                catalyst_impact = pc.get("impact_pct", "")
                break

        # Fallback: derive single best catalyst from screener data
        if not catalyst_text:
            ret1m  = _safe_float(row.get("ret_1m"))  or 0
            ret3m  = _safe_float(row.get("ret_3m"))  or 0
            vs     = vsurge or 0
            if is_hi:
                if vs >= 3.0:
                    catalyst_text = f"Institutional accumulation — {vs:.1f}× volume surge above 20-day avg"
                elif ret1m >= 15:
                    catalyst_text = f"Strong momentum — +{ret1m:.1f}% in 1 month, +{ret3m:.1f}% over 3 months"
                elif ret3m >= 30:
                    catalyst_text = f"Multi-month breakout — +{ret3m:.1f}% over 3 months with sustained buying"
                else:
                    catalyst_text = f"52-week breakout — price hit annual peak with {vs:.1f}× volume confirmation"
            else:
                if vs >= 3.0:
                    catalyst_text = f"Institutional distribution — {vs:.1f}× volume surge driving the sell-off"
                elif ret1m <= -15:
                    catalyst_text = f"Sharp selloff — {ret1m:.1f}% in 1 month, {ret3m:.1f}% over 3 months"
                elif ret3m <= -25:
                    catalyst_text = f"Extended downtrend — {ret3m:.1f}% over 3 months, hitting annual low"
                else:
                    catalyst_text = f"52-week breakdown — price breached annual floor with {vs:.1f}× volume"

        impact_chip = (
            f' <span style="font-size:10px;font-weight:700;background:{"#1a3828" if is_hi else "#3b1219"};'
            f'color:{catalyst_color};border-radius:3px;padding:1px 6px;">{catalyst_impact}</span>'
        ) if catalyst_impact and catalyst_impact not in ("N/A", "") else ""

        catalyst_block = (
            f'<div style="margin-top:8px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:1.5px;'
            f'text-transform:uppercase;color:#6E7681;margin-bottom:4px;">Primary Catalyst</div>'
            f'<div style="font-size:13px;color:{catalyst_color};font-weight:600;'
            f'line-height:1.5;">{catalyst_text}{impact_chip}</div>'
            f'</div>'
        )

        # Render each card as its own st.markdown call — avoids markdown code-block mis-parse
        card = (
            f'<div class="sig-card {card_cls}">'
            f'<div style="flex:1">'
            f'<div class="sc-name">{name}</div>'
            f'<div class="sc-ticker">{ticker}</div>'
            f'<div class="sc-sector">{sector}</div>'
            f'{biz_line}'
            f'{catalyst_block}'
            f'</div>'
            f'<div class="sc-right">'
            f'<div class="sc-price">{price_str}</div>'
            f'<div class="{status_cls}">{status}</div>'
            f'<div class="sc-vol">{vol_str}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card, unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    tickers = df["ticker"].tolist()
    selected = st.selectbox(
        "🔍 Select a stock for deep-dive analysis:",
        options=["— Select —"] + tickers,
        key=f"sel_{key}",
    )
    return None if selected.startswith("—") else selected


def _fetch_quick_news_context(ticker: str, company: str) -> str:
    """
    Lightweight news fetch for the parallel enrichment pass.
    Combines yfinance headlines + a fast Tavily search.
    Runs inside a worker thread — must be self-contained.
    """
    import requests as _req
    from datetime import datetime
    lines: list[str] = []

    # ── yfinance news (fast, no quota cost) ──────────────────────────────────
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news or []
        for item in news[:5]:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = (content.get("title") or item.get("title") or "")[:120]
            pub   = (content.get("pubDate") or item.get("providerPublishTime") or "")
            if title:
                try:
                    dt = datetime.fromtimestamp(pub).strftime("%b %d") if isinstance(pub, (int, float)) else str(pub)[:10]
                except Exception:
                    dt = ""
                lines.append(f"[{dt}] {title}")
    except Exception:
        pass

    # ── Tavily quick search (3 results, basic depth, ~1s) ────────────────────
    if _TAVILY_KEY:
        try:
            clean = ticker.replace(".NS", "").replace(".BO", "")
            query = f"{company} {clean} India stock news earnings results announcement 2025 2026"
            resp = _req.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        _TAVILY_KEY,
                    "query":          query,
                    "search_depth":   "basic",
                    "include_answer": True,
                    "max_results":    5,
                    "include_domains": [
                        "economictimes.indiatimes.com", "moneycontrol.com",
                        "livemint.com", "business-standard.com",
                        "financialexpress.com", "ndtvprofit.com",
                        "bseindia.com",
                    ],
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                if answer:
                    lines.insert(0, f"[Tavily summary] {answer[:300]}")
                for r in data.get("results", [])[:4]:
                    title   = (r.get("title") or "")[:100]
                    snippet = (r.get("content") or "")[:150]
                    src     = r.get("url", "").split("/")[2] if r.get("url") else ""
                    if title:
                        lines.append(f"[{src}] {title} — {snippet}")
        except Exception:
            pass

    return "\n".join(lines[:10]) if lines else ""


def _quick_catalyst_groq(ticker: str, company: str, sector: str,
                          signal: str, price: float,
                          ret1m: float, ret3m: float,
                          vsurge: float, pe: float,
                          news_context: str = "",
                          yfinance_context: str = "") -> dict:
    """
    Fast primary-catalyst call using llama-3.1-8b-instant on Groq.
    news_context: pre-fetched yfinance headlines + Tavily results.
    yfinance_context: analyst targets, EPS, margins from build_yfinance_context.
    Combining both gives the model real quantitative grounding instead of
    hallucinating from training knowledge.
    Returns {headline, impact_pct} or {} on failure.
    """
    if not _AI_OK:
        return {}
    try:
        client = _OpenAI(api_key=_OPENAI_KEY, base_url=_GROQ_BASE_URL)
        is_hi     = "HIGH" in signal.upper()
        direction = "52-week high" if is_hi else "52-week low"

        news_block = (
            f"\nRECENT NEWS & ANNOUNCEMENTS (use these as your primary source):\n"
            f"{news_context}\n"
        ) if news_context else "\n(No live news available — use training knowledge.)\n"

        # Trim yfinance block to the most useful lines for a quick call
        yf_block = ""
        if yfinance_context:
            # Keep analyst consensus, valuation, earnings, profitability lines only
            yf_lines = [
                ln for ln in yfinance_context.splitlines()
                if any(kw in ln for kw in ("Analyst:", "Valuation:", "Earnings:", "Profitability:", "Market:"))
            ]
            if yf_lines:
                yf_block = "\nFUNDAMENTALS (yfinance):\n" + "\n".join(yf_lines[:6]) + "\n"

        has_news = bool(news_context and news_context.strip())

        if not has_news:
            # No real news → return a safe heuristic answer, never hallucinate
            direction_word = "52-week high" if is_hi else "52-week low"
            pct_str = f"{ret3m:+.1f}% over 3 months" if ret3m else (f"{ret1m:+.1f}% over 1 month" if ret1m else "sustained move")
            vs_str  = f"{vsurge:.1f}× volume surge" if vsurge and vsurge >= 1.2 else "volume-backed move"
            headline = f"{pct_str} with {vs_str} drives stock to {direction_word}"
            return {"headline": headline[:90], "impact_pct": "N/A"}

        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
        prompt = (
            f"You are a senior equity analyst. Analyze ONLY {company} (NSE: {clean_ticker}), "
            f"a {sector} company. It has just hit its {direction}.\n"
            f"CRITICAL: Do NOT confuse {company} with any related group entity, subsidiary, or peer. "
            f"Only report catalysts that explicitly name '{company}' or '{clean_ticker}'.\n\n"
            f"Price Data: current=Rs{price:,.0f}, "
            f"1M={f'{ret1m:+.1f}%' if ret1m else 'N/A'}, "
            f"3M={f'{ret3m:+.1f}%' if ret3m else 'N/A'}, "
            f"VolSurge={f'{vsurge:.1f}x' if vsurge else 'N/A'}, "
            f"P/E={f'{pe:.1f}x' if pe else 'N/A'}\n"
            f"{yf_block}"
            f"{news_block}\n"
            f"Task: From the news above, identify the SPECIFIC event causing this move.\n"
            f"STRICT RULES:\n"
            f"- Only cite Rs amounts, contract values, or dates that appear VERBATIM in the news above.\n"
            f"- Do NOT invent, guess, or extrapolate figures not present in the news.\n"
            f"- Do NOT use the ticker symbol in the headline — use the company name.\n"
            f"- Do NOT describe volume or price movement — find the ROOT CAUSE.\n"
            f"- If the news does not mention a specific catalyst, write a general fundamental reason.\n"
            f"- Headline must be under 90 chars.\n\n"
            f'JSON only: {{"headline": "specific catalyst under 90 chars", '
            f'"impact_pct": "e.g. +18% or -22%"}}'
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=180,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception:
        return {}


def _enrich_results_parallel(highs_df: pd.DataFrame, lows_df: pd.DataFrame) -> int:
    """
    Post-screen enrichment: for each hit stock, fetch yfinance news + Tavily
    in parallel, then fire a llama-3.1-8b-instant call with that real context.
    Stores {primary_catalyst} in session_state. Returns enriched count.
    """
    if not _AI_OK:
        return 0

    # Build task list — skip stocks already fully analysed
    tasks = []
    for sig, df in [("52W_HIGH", highs_df), ("52W_LOW", lows_df)]:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            ticker = row.get("ticker", "")
            cache_key = f"ai_{ticker}_{sig}"
            if cache_key in st.session_state:
                continue
            sector_val = row.get("sector", "") or ""
            if str(sector_val).lower() in ("nan", "none"):
                sector_val = ""
            tasks.append({
                "ticker":    ticker,
                "company":   row.get("company_name", ticker),
                "sector":    sector_val,
                "signal":    sig,
                "price":     _safe_float(row.get("current_price")) or 0,
                "ret1m":     _safe_float(row.get("ret_1m"))         or 0,
                "ret3m":     _safe_float(row.get("ret_3m"))         or 0,
                "vsurge":    _safe_float(row.get("volume_surge"))   or 0,
                "pe":        _safe_float(row.get("pe_ratio"))       or 0,
                "cache_key": cache_key,
            })

    if not tasks:
        return 0

    # Stagger tasks: Groq free tier allows ~30 req/min for llama-3.1-8b-instant.
    # With 2 workers, space each task 2s apart so we stay well under the limit.
    for i, task in enumerate(tasks):
        task["_delay"] = i * 2.0   # 0s, 2s, 4s, 6s … — absorbed by the worker thread

    _results: dict = {}

    def _call_one(task: dict) -> None:
        import time as _time
        # Step 1a: fetch real news context (yfinance headlines + Tavily) — inside this thread
        news_ctx = _fetch_quick_news_context(task["ticker"], task["company"])

        # Step 1b: fetch yfinance fundamentals (analyst targets, EPS, margins) in parallel
        yf_ctx = ""
        try:
            if _SCREENER_OK:
                yf_ctx = build_yfinance_context(task["ticker"])
        except Exception:
            pass

        # Small stagger before Groq call — prevents all threads hitting the API simultaneously
        # Groq free tier: ~30 req/min for llama-3.1-8b-instant → need ≥2s between batches
        _time.sleep(task.get("_delay", 0))

        # Step 2: Groq call with both news AND fundamentals for grounded analysis
        pc = _quick_catalyst_groq(
            task["ticker"], task["company"], task["sector"], task["signal"],
            task["price"], task["ret1m"], task["ret3m"], task["vsurge"], task["pe"],
            news_context=news_ctx,
            yfinance_context=yf_ctx,
        )
        if pc and pc.get("headline"):
            _results[task["cache_key"]] = {
                "primary_catalyst": pc,
                "sentiment": "",
                "confidence": "Low",
                "business_model": "",
                "summary": "",
                "catalyst_timeline": [],
                "catalysts": [],
                "watch_next": [],
                "peer_context": "",
                "sources": [],
            }

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 2 workers max: each thread does Tavily + yfinance + Groq calls.
    # Staggered delays above keep us under Groq's 30 req/min free-tier limit.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_call_one, t): t for t in tasks}
        try:
            # Timeout = 15s base + 3s per task (accounts for stagger delays)
            _timeout = 15 + len(tasks) * 3
            for _ in as_completed(futures, timeout=_timeout):
                pass
        except Exception:
            pass

    # Write to session_state in the main Streamlit thread (thread-safe)
    for cache_key, data in _results.items():
        if cache_key not in st.session_state:   # still don't overwrite a full analysis
            st.session_state[cache_key] = data

    return len(_results)


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
                  upgrades_context: str = "",
                  earnings_surprise_context: str = "",
                  bse_announcements_context: str = "",
                  calendar_context: str = "",
                  peer_context: str = "") -> dict:
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

    # Tavily takes priority over Google News (richer content); merge both
    combined_news = ""
    if tavily_context:
        combined_news = _trim(tavily_context, 3500)
        if google_news_context:
            combined_news += "\n\n--- Google News ---\n" + _trim(google_news_context, 800)
    elif google_news_context:
        combined_news = _trim(google_news_context, 3000)

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
    all_context      = (news_block + screener_block + av_block + fmp_block
                        + upgrades_block + surprise_block + bse_block
                        + fred_block + india_block + nse_bse_block
                        + calendar_block + peer_block)

    _action_word = "high" if is_hi else "low"
    _risk_label  = "rally" if is_hi else "recovery"

    prompt = f"""You are a senior equity research analyst at a top institutional fund.
Write a Stock Catalyst Analysis for {company} ({ticker}), which is near its 52-week {_action_word}.

MARKET DATA
  Current Price : Rs{price:,.2f}
  52W High      : Rs{high52:,.2f}  ({abs(pct_from_high):.1f}% away)
  52W Low       : Rs{low52:,.2f}   ({abs(pct_from_low):.1f}% away)
  Volume Surge  : {vsurge:.1f}x 20-day average
  P/E           : {pe if pe else "N/A"}x
  Returns       : 1M={f"{ret_1m:+.1f}%" if ret_1m is not None else "N/A"}  3M={f"{ret_3m:+.1f}%" if ret_3m is not None else "N/A"}  6M={f"{ret_6m:+.1f}%" if ret_6m is not None else "N/A"}  1Y={f"{ret_1y:+.1f}%" if ret_1y is not None else "N/A"}
  Sector        : {sector}
{f"  Recent closes (oldest to newest): {price_series}" if price_series else ""}

RECENT NEWS AND FILINGS (use these as your primary source for catalyst dates and figures):
{all_context if all_context.strip() else "  No additional context available — use your training knowledge."}

TASK:
Identify 4 to 7 specific catalysts that explain why {company} is at a 52-week {_action_word}.
For each catalyst, provide a dated headline, a 2-3 sentence explanation with specific figures, and the source.
Include a mix of positive, negative, and neutral catalysts where relevant.
Classify overall sentiment and confidence.

IMPORTANT: Return ONLY valid JSON. No markdown fences, no preamble.

{{
  "sentiment": "Bullish",
  "confidence": "Medium",
  "business_model": "2-3 sentences: what {company} does, its core revenue streams, competitive moat.",
  "primary_catalyst": {{
    "headline": "Single most important reason {company} is at a 52-week {_action_word} — one punchy sentence with a specific figure",
    "impact_pct": "Estimated % of total price move driven by this catalyst (e.g. '-18%' or '+24%')",
    "detail": "2-3 sentences of supporting evidence with specific data points"
  }},
  "summary": "3-4 sentence analyst note: overall verdict, key risk, and what triggered this extreme.",
  "catalyst_timeline": [
    {{"date": "Month YYYY", "event": "One-line event description", "impact": "brief price/sentiment effect"}}
  ],
  "catalysts": [
    {{
      "type": "positive",
      "headline": "Catalyst headline with specific figure or date",
      "detail": "2-3 sentences with Rs crore amounts, %, dates. Why does this move the stock?",
      "impact_pct": "Estimated % of move attributable to this catalyst (e.g. '+8%' or '-12%')",
      "date": "Month DD, YYYY",
      "source": "Source name"
    }}
  ],
  "watch_next": [
    {{"event": "Upcoming event name", "date": "Approximate date or quarter", "implication": "What it means for the stock if positive/negative"}}
  ],
  "peer_context": "1-2 sentences: is this move company-specific or sector-wide? How does {company} compare to peers?",
  "sources": ["Source Name, Month DD YYYY"]
}}

Rules:
- catalyst type must be exactly: "positive", "negative", or "neutral"
- impact_pct on each catalyst must sum to roughly the total move from the opposite extreme
- primary_catalyst must be the single most impactful item — not a generic statement
- catalyst_timeline: 3-5 events in chronological order showing the causal chain
- watch_next: 2-3 specific upcoming events (earnings date, policy meeting, product launch)
- Each headline must include at least one specific figure or date
- Write with conviction — avoid hedging language
- sources array should list unique sources referenced"""

    last_err = ""
    try:
        client = _OpenAI(
            api_key=_OPENAI_KEY,
            base_url=_GROQ_BASE_URL if _LLM_PROVIDER == "groq" else None,
        )
        kwargs: dict = {"temperature": 0.2}
        # Reasoning/thinking models (DeepSeek R1, QwQ) do NOT support
        # response_format JSON mode — only enable it for standard models
        _NO_JSON_MODE_MODELS = ("deepseek", "qwq", "r1")
        if not any(x in _OPENAI_MODEL.lower() for x in _NO_JSON_MODE_MODELS):
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        raw = response.choices[0].message.content.strip()
        # Strip DeepSeek R1 internal thinking tags
        if "<think>" in raw:
            import re as _re
            raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rstrip("`")
        # Find JSON object in response even if there's surrounding text
        if not raw.strip().startswith("{"):
            import re as _re
            m = _re.search(r"\{[\s\S]*\}", raw)
            if m:
                raw = m.group(0)
        result = json.loads(raw.strip())
        st.session_state[cache_key] = result
        return result
    except Exception as e:
        last_err = str(e)

    return _fallback_deep_dive(ticker, company, sector, signal,
                               price, high52, low52, pct_from_high,
                               pct_from_low, vsurge, pe,
                               error=last_err)


def _fallback_deep_dive(ticker, company, sector, signal,
                        price, high52, low52, pct_from_high, pct_from_low,
                        vsurge, pe, error="") -> dict:
    is_hi = "HIGH" in signal.upper()
    level = "52-week high" if is_hi else "52-week low"
    err_note = f"OpenAI error: {error[:120]}" if error else "AI API key not configured"
    return {
        "sentiment":     "Neutral",
        "confidence":    "Low",
        "business_model": f"{company} operates in the {sector} sector. Configure an AI API key for a detailed business description.",
        "primary_catalyst": {
            "headline": f"{company} is at its {level}",
            "impact_pct": "N/A",
            "detail": f"Configure an AI API key for full catalyst analysis. {err_note}",
        },
        "summary": f"{company} ({ticker}) is near its {level} at Rs{price:,.2f}. Volume surge: {vsurge:.1f}x. Configure an AI API key for full catalyst analysis.",
        "catalyst_timeline": [],
        "catalysts": [
            {
                "type":       "neutral",
                "headline":   f"{company} is near its {level} — AI analysis unavailable",
                "detail":     f"Configure an AI API key to get detailed catalyst analysis. {err_note}",
                "impact_pct": "N/A",
                "date":       "",
                "source":     "System",
            }
        ],
        "watch_next": [],
        "peer_context": "",
        "sources": [],
    }


def _parse_rss_feed(url: str, company: str, ticker: str,
                    source_name: str, max_items: int = 8) -> list[dict]:
    """
    Parse an RSS feed and return articles mentioning the company/ticker.
    Filters by company name or ticker symbol appearing in title or summary.
    """
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "").lower()
    company_words = [w.lower() for w in company.split() if len(w) > 3]

    def _mentions(text: str) -> bool:
        t = text.lower()
        return clean_ticker in t or any(w in t for w in company_words[:3])

    results = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip()
            if not _mentions(title) and not _mentions(summary):
                continue
            pub = ""
            try:
                dt  = parsedate_to_datetime(entry.get("published", ""))
                pub = dt.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                pub = entry.get("published", "")[:16]
            results.append({
                "title":     title,
                "link":      entry.get("link", "#"),
                "source":    source_name,
                "published": pub,
                "published_dt": entry.get("published", ""),
                "summary":   summary[:220] + "…" if len(summary) > 220 else summary,
            })
            if len(results) >= max_items:
                break
    except Exception:
        pass
    return results


@st.cache_data(ttl=600, show_spinner=False)   # 10 min — live Indian financial RSS
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


@st.cache_data(ttl=900, show_spinner=False)   # 15 min — conserves 100 req/day quota
def _fetch_newsapi(company: str, ticker: str, api_key: str) -> list[dict]:
    """
    Search NewsAPI.org for the company across Indian financial outlets.
    Returns structured, date-sorted articles — the most accurate company-specific feed.
    Free tier: 100 req/day.
    """
    if not api_key:
        return []
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    try:
        import requests as _req
        resp = _req.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        f'"{company}" OR "{clean_ticker}"',
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


@st.cache_data(ttl=900, show_spinner=False)
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


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_tavily_context(company: str, ticker: str, api_key: str) -> str:
    """
    Search Tavily for recent news + analyst commentary on a stock.
    Returns a rich context string for the AI prompt.
    Tavily returns full article snippets — much richer than Google News RSS.
    """
    if not api_key:
        return ""

    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query = f"{company} {clean_ticker} stock India latest news analyst target earnings results 2025 2026"

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
        tickers = [s + ".NS" for s in symbols if s]
        return tickers
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)    # 1h cache — yfinance is free but rate-sensitive
def _cached_yfinance_context(ticker: str) -> str:
    """Cached wrapper for build_yfinance_context. Prevents repeated calls on spotlight re-renders."""
    return build_yfinance_context(ticker) if _SCREENER_OK else ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yf_company_context(ticker: str) -> str:
    """Fetch company profile from yfinance as replacement for geo-blocked NSE stock API."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        parts = []
        if info.get("longBusinessSummary"):
            parts.append(f"Business Description:\n{info['longBusinessSummary'][:800]}")
        mktcap = info.get("marketCap")
        if mktcap:
            parts.append(f"Market Cap: Rs {mktcap/1e7:,.0f} Cr")
        if info.get("fullTimeEmployees"):
            parts.append(f"Employees: {info['fullTimeEmployees']:,}")
        if info.get("dividendYield"):
            parts.append(f"Dividend Yield: {info['dividendYield']*100:.2f}%")
        if info.get("beta"):
            parts.append(f"Beta: {info['beta']:.2f}")
        if info.get("returnOnEquity"):
            parts.append(f"ROE: {info['returnOnEquity']*100:.1f}%")
        if info.get("debtToEquity"):
            parts.append(f"Debt/Equity: {info['debtToEquity']:.2f}")
        if info.get("revenueGrowth"):
            parts.append(f"Revenue Growth (YoY): {info['revenueGrowth']*100:.1f}%")
        return "\n".join(parts) if parts else ""
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yf_upgrades_context(ticker: str) -> str:
    """Analyst consensus + price targets from yfinance info.
    More reliable than upgrades_downgrades for Indian NSE stocks."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        lines = []

        # Analyst consensus (almost always available)
        rec_key  = info.get("recommendationKey", "")
        n_analysts = info.get("numberOfAnalystOpinions")
        if rec_key or n_analysts:
            label = rec_key.replace("_", " ").title() if rec_key else "N/A"
            lines.append(f"Analyst Consensus: {label}"
                         + (f" ({n_analysts} analysts)" if n_analysts else ""))

        # Price targets
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

        # Recent upgrades/downgrades (when available)
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

        return "\n".join(lines) if lines else ""
    except Exception:
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
    "TATAMOTORS.NS":  ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "MARUTI.NS":      ["TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
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
            # Summary
            + (f'<div style="font-size:12px;color:#8B949E;line-height:1.6;">'
               f'{art["summary"]}</div>' if art["summary"] else "")
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
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if india_ctx else "#F0B429"};">'
            f'{"✅ India macro + FII/DII loaded" if india_ctx else "⚠️ India macro unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col4:
        with st.spinner("yfinance: fetching company profile…"):
            nse_bse_ctx = _fetch_yf_company_context(ticker)
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
            f'<div style="font-size:11px;color:{"#3FB950" if av_ctx else "#F0B429"};">'
            f'{"✅ yfinance analyst data loaded" if av_ctx else "⚠️ yfinance fundamentals unavailable"}</div>',
            unsafe_allow_html=True,
        )

    with src_col7:
        with st.spinner("Tavily: searching latest news…"):
            tavily_ctx = _fetch_tavily_context(name, ticker, _TAVILY_KEY) if _TAVILY_KEY else ""
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if tavily_ctx else "#F0B429"};">'
            f'{"✅ Tavily web search loaded" if tavily_ctx else "⚠️ Tavily unavailable"}</div>',
            unsafe_allow_html=True,
        )

    # ── Row 3: analyst actions, earnings history, exchange filings ────────────
    src_col8, src_col9, src_col10, _ = st.columns(4)

    with src_col8:
        with st.spinner("yfinance: broker actions…"):
            upgrades_ctx = _fetch_yf_upgrades_context(ticker)
        st.markdown(
            f'<div style="font-size:11px;color:{"#3FB950" if upgrades_ctx else "#F0B429"};">'
            f'{"✅ Analyst consensus + targets loaded" if upgrades_ctx else "⚠️ Analyst data unavailable"}</div>',
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
            upgrades_context=upgrades_ctx,
            earnings_surprise_context=earnings_surprise_ctx,
            bse_announcements_context=bse_ctx,
            calendar_context=calendar_ctx,
            peer_context=peer_ctx,
        )

    # ── Detect quota error and show actionable help ───────────────────────────
    summary    = analysis.get("summary", "")
    is_api_fail = ("AI API key not configured" in summary or "OpenAI error" in summary)

    if is_api_fail:
        st.markdown(
            '<div style="background:#2a1a00;border:1px solid #F0B429;'
            'border-radius:8px;padding:16px 20px;margin-bottom:16px;">'
            '<div style="font-size:13px;font-weight:700;color:#F0B429;'
            'margin-bottom:8px;">⚠️ AI Analysis Unavailable</div>'
            '<div style="font-size:13px;color:#C9D1D9;line-height:1.7;">'
            'Could not connect to the AI model. Check your API key in Streamlit secrets.<br>'
            '<b>Set:</b> LLM_PROVIDER and OPENAI_API_KEY / GROQ_API_KEY'
            '</div></div>',
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

    # ── Primary Catalyst (Idea 1) ─────────────────────────────────────────────
    primary = analysis.get("primary_catalyst", {})
    if primary and primary.get("headline"):
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
        for cat in catalysts:
            cat_type   = str(cat.get("type", "neutral")).lower()
            headline   = cat.get("headline", "")
            detail     = cat.get("detail", "")
            cat_date   = cat.get("date", "")
            cat_source = cat.get("source", "")
            cat_impact = cat.get("impact_pct", "")

            _icon  = {"positive": "✅", "negative": "❌", "neutral": "➖"}.get(cat_type, "➖")
            _cbg   = {"positive": "#0d2618", "negative": "#2a0d12", "neutral": "#161B22"}.get(cat_type, "#161B22")
            _cbdr  = {"positive": "#238636", "negative": "#8B1A1A", "neutral": "#30363D"}.get(cat_type, "#30363D")
            _cmeta = f"{cat_date}  ·  {cat_source}" if cat_date or cat_source else ""
            _impact_chip = (
                f'<span style="font-size:11px;font-weight:700;'
                f'background:{"#1a3828" if cat_type=="positive" else "#3b1219" if cat_type=="negative" else "#21262D"};'
                f'color:{"#3FB950" if cat_type=="positive" else "#F85149" if cat_type=="negative" else "#8B949E"};'
                f'border-radius:4px;padding:2px 7px;margin-left:8px;">{cat_impact}</span>'
            ) if cat_impact and cat_impact != "N/A" else ""

            st.markdown(
                f'<div style="background:{_cbg};border:1px solid {_cbdr};'
                f'border-radius:8px;padding:14px 18px;margin-bottom:10px;">'
                f'<div style="display:flex;align-items:flex-start;gap:10px;">'
                f'<span style="font-size:18px;line-height:1.3">{_icon}</span>'
                f'<div style="flex:1">'
                f'<div style="font-size:14px;font-weight:600;color:#E6EDF3;'
                f'line-height:1.4;margin-bottom:6px;">{headline}{_impact_chip}</div>'
                f'<div style="font-size:13px;color:#C9D1D9;line-height:1.6;">{detail}</div>'
                + (f'<div style="font-size:11px;color:#6E7681;margin-top:8px;">{_cmeta}</div>'
                   if _cmeta else "")
                + f'</div></div></div>',
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
    watch_next = analysis.get("watch_next", [])
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
            options=["Nifty 500", "Nifty 50", "Nifty 100",
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

        # ── Parallel AI enrichment (post-screen, llama-3.1-8b-instant) ───────
        _h = results.get("highs", pd.DataFrame())
        _l = results.get("lows",  pd.DataFrame())
        _total = len(_h) + len(_l)
        if _total > 0 and _AI_OK:
            with st.status(
                f"🤖 AI pre-analysing {_total} stocks in parallel…",
                expanded=False,
            ) as enrich_status:
                _n = _enrich_results_parallel(_h, _l)
                enrich_status.update(
                    label=f"✅ Primary catalysts ready for {_n} stocks",
                    state="complete",
                )

    elif "screen_results" in st.session_state:
        results = st.session_state["screen_results"]
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
        selected_hi = _render_signals_table(highs_df, key="hi")
        if selected_hi and not highs_df.empty:
            mask = highs_df["ticker"] == selected_hi
            if mask.any():
                row = highs_df[mask].iloc[0].to_dict()
                st.markdown("<hr/>", unsafe_allow_html=True)
                _render_spotlight(selected_hi, row, p)

    with t_lo:
        selected_lo = _render_signals_table(lows_df, key="lo")
        if selected_lo and not lows_df.empty:
            mask = lows_df["ticker"] == selected_lo
            if mask.any():
                row = lows_df[mask].iloc[0].to_dict()
                st.markdown("<hr/>", unsafe_allow_html=True)
                _render_spotlight(selected_lo, row, p)

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
