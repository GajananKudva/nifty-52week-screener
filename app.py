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


# ── OpenAI client (optional — graceful fallback) ───────────────────────────────
try:
    from openai import OpenAI as _OpenAI
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
except ImportError:
    _FMP_OK  = False
    def build_fmp_context(_ticker: str) -> str:   # type: ignore[misc]
        return ""

# ── FRED macro (optional — needs FRED_API_KEY) ────────────────────────────────
try:
    from fred import build_fred_context, get_series_snapshot
    _FRED_KEY = _secret("FRED_API_KEY")
    _FRED_OK  = bool(_FRED_KEY)
except ImportError:
    _FRED_OK  = False
    def build_fred_context() -> str:              # type: ignore[misc]
        return ""
    def get_series_snapshot() -> dict:            # type: ignore[misc]
        return {}

# ── India macro (NSE + RBI — no key needed) ───────────────────────────────────
try:
    from india_macro import build_india_macro_context, get_dashboard_snapshot
    _INDIA_MACRO_OK = True
except ImportError:
    _INDIA_MACRO_OK = False
    def build_india_macro_context() -> str:       # type: ignore[misc]
        return ""
    def get_dashboard_snapshot() -> dict:         # type: ignore[misc]
        return {}

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
# 3.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
_GREEN  = "#00c805"
_RED    = "#ff3b3b"
_YELLOW = "#F0B429"
_BLUE   = "#58A6FF"
_MUTED  = "#8B949E"
_BG     = "#0D1117"
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
    cfg = ScreenerConfig(
        tickers                = tickers,
        high_low_window        = p["window"],
        breakout_threshold     = p["threshold"],
        volume_window          = 20,
        volume_surge_threshold = p["vol_surge"],
        dcf_discount_rate      = p["wacc"],
        dcf_terminal_growth    = p["term_growth"],
        dcf_stage1_years       = 5,
        dcf_stage2_years       = 5,
        rate_limit_sleep       = 0.25,
        rate_limit_batch_size  = 50,
    )
    eng     = DataEngine(cfg)
    raw     = eng.screen_universe()
    highs   = pd.DataFrame(eng.format_for_display(raw["highs"]))
    lows    = pd.DataFrame(eng.format_for_display(raw["lows"]))

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
        f'  <div class="meta">52-Week High / Low · Volume Confirmed · 2-Stage DCF &nbsp;|&nbsp; '
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
        "upside_downside_pct": "DCF Upside%",
        "ret_1m":            "1M Ret%",
        "ret_3m":            "3M Ret%",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    out = df[list(available.keys())].copy()
    out.columns = list(available.values())

    def _c_dcf(v):
        if pd.isna(v): return f"color:{_MUTED}"
        return f"color:{_GREEN}" if v > 0 else f"color:{_RED}"

    def _c_ret(v):
        if pd.isna(v): return f"color:{_MUTED}"
        return f"color:{_GREEN}" if v > 0 else f"color:{_RED}"

    def _fmt_pe(v):
        return f"{v:.1f}×" if pd.notna(v) else "—"

    def _fmt_pct(v):
        return f"{v:+.2f}%" if pd.notna(v) else "—"

    styled = (
        out.style
        .map(_c_dcf, subset=["DCF Upside%"])
        .map(_c_ret, subset=["1M Ret%", "3M Ret%"])
        .format({
            "Price (₹)":   "₹{:,.2f}",
            "52W High":    "₹{:,.2f}",
            "52W Low":     "₹{:,.2f}",
            "% From High": "{:.2f}%",
            "% From Low":  "{:.2f}%",
            "Vol Surge":   "{:.2f}×",
            "P/E":         _fmt_pe,
            "DCF Upside%": _fmt_pct,
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

    # 4. DCF valuation
    dcf = _safe_float(row.get("upside_downside_pct"))
    if dcf is not None:
        if dcf > 15:
            bullets.append(f"<strong>DCF shows {dcf:.1f}% upside</strong> — stock appears undervalued relative to intrinsic value.")
        elif dcf < -15:
            bullets.append(f"<strong>DCF shows {abs(dcf):.1f}% downside</strong> — price may be stretched beyond intrinsic value.")
        else:
            bullets.append(f"<strong>Fairly valued</strong> by DCF ({dcf:+.1f}%) — momentum-driven rather than valuation-driven move.")

    # 5. P/E context
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
        sector  = row.get("sector", "")
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

        vol_str = f"Vol surge {vsurge:.1f}&times;" if vsurge else ""

        # Build insight bullets
        bullets = _build_why_bullets(dict(row), is_hi)
        bullet_html = "".join(f"<li>{b}</li>" for b in bullets)
        why_block = (
            f'<div class="sc-why">'
            f'<div class="sc-why-lbl">Why this level</div>'
            f'<ul>{bullet_html}</ul>'
            f'</div>'
        ) if bullets else ""

        # Render each card as its own st.markdown call — avoids markdown code-block mis-parse
        card = (
            f'<div class="sig-card {card_cls}">'
            f'<div style="flex:1">'
            f'<div class="sc-name">{name}</div>'
            f'<div class="sc-ticker">{ticker}</div>'
            f'<div class="sc-sector">{sector}</div>'
            f'{why_block}'
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


def _ai_deep_dive(ticker: str, company: str, sector: str, signal: str,
                  price: float, high52: float, low52: float,
                  pct_from_high: float, pct_from_low: float,
                  vsurge: float, pe: float, dcf_upside: float,
                      news_headlines: str,
                      fmp_context: str = "",
                      fred_context: str = "",
                      india_macro_context: str = "") -> dict:
    """
    Call OpenAI with the senior equity analyst prompt.
    Results cached in st.session_state — only successful calls are cached.
    Failed calls are never cached so retry always hits the API fresh.
    """
    # ── Session-state cache (only stores successes) ───────────────────────────
    cache_key = f"ai_{ticker}_{signal}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if not _AI_OK:
        return _fallback_deep_dive(ticker, company, sector, signal,
                                   price, high52, low52, pct_from_high,
                                   pct_from_low, vsurge, pe, dcf_upside)

    is_hi  = "HIGH" in signal.upper()
    level  = "52-WEEK HIGH" if is_hi else "52-WEEK LOW"
    action = "rally" if is_hi else "sell-off"

    # Trim contexts to keep total prompt under ~3,000 tokens
    def _trim(text: str, max_chars: int) -> str:
        return text[:max_chars] + "\n[...truncated]" if len(text) > max_chars else text

    fmp_block         = f"\n\n{_trim(fmp_context, 2000)}"         if fmp_context         else ""
    fred_block        = f"\n\n{_trim(fred_context, 800)}"         if fred_context        else ""
    india_macro_block = f"\n\n{_trim(india_macro_context, 800)}"  if india_macro_context else ""
    all_context       = fmp_block + fred_block + india_macro_block

    if is_hi:
        prompt = f"""Act as a senior equity analyst. {company} ({ticker}, NSE India) has hit a 52-week high.
Identify and explain the specific catalysts driving this {action}.

Stock data:
- Sector       : {sector}
- Current Price: Rs.{price:,.2f}
- 52W High     : Rs.{high52:,.2f}  ({abs(pct_from_high):.2f}% from high)
- 52W Low      : Rs.{low52:,.2f}  ({abs(pct_from_low):.2f}% above low)
- Volume Surge : {vsurge:.1f}x vs 20-day avg
- P/E Ratio    : {pe if pe else "N/A"}
- DCF Upside   : {f"{dcf_upside:+.1f}%" if dcf_upside else "N/A"}
- Recent Headlines: {news_headlines}
{all_context}

IMPORTANT: You have been provided with verified financial data above (FMP fundamentals, FRED global macro, India macro/FII-DII flows).
Use the actual earnings numbers, margin figures, growth rates, and transcript quotes wherever possible.
Cite specific figures (e.g. "beat EPS by 8.2% in Q3 FY25", "gross margin expanded 340bps").
Label any claim not supported by the above data as [SPECULATIVE].

Return ONLY a valid JSON object with exactly these 8 keys (no markdown, no code fences):
{{
  "price_action": "Describe the % move over 1M, 3M, 6M, 1Y and when the breakout began. Cite actual price levels.",
  "fundamental_catalysts": "Use the FMP earnings data above. Quote actual EPS beats, margin trends, revenue growth rates. Be specific with numbers.",
  "sectoral_macro": "Cover industry cycle turns, policy changes, government incentives (PLI, tariffs, regulation), commodity moves, or peer re-rating relevant to {sector}.",
  "corporate_actions": "Cover management changes, M&A, demergers, fundraises, buybacks, promoter buying, block deals, or index inclusion in the last 6 months.",
  "narrative_shift": "What is the new story the market is pricing in vs 6-12 months ago? Has the company moved from one bucket (commodity/cyclical) to another (structural/growth)?",
  "flow_factors": "Cover FII/DII buying, mutual fund accumulation, breakout from long consolidation, or short covering.",
  "risks": "What could break this story? Address valuation, execution risk, dependency on a single driver, and any other material risks.",
  "summary": "One sentence completing: The stock is at a 52-week high primarily because _."
}}"""
    else:
        prompt = f"""Act as a senior equity analyst. {company} ({ticker}, NSE India) has hit a 52-week low.
Identify and explain the specific catalysts driving this {action}.

Stock data:
- Sector       : {sector}
- Current Price: Rs.{price:,.2f}
- 52W High     : Rs.{high52:,.2f}  ({abs(pct_from_high):.2f}% below high)
- 52W Low      : Rs.{low52:,.2f}  ({abs(pct_from_low):.2f}% from low)
- Volume Surge : {vsurge:.1f}x vs 20-day avg
- P/E Ratio    : {pe if pe else "N/A"}
- DCF Upside   : {f"{dcf_upside:+.1f}%" if dcf_upside else "N/A"}
- Recent Headlines: {news_headlines}
{all_context}

IMPORTANT: You have been provided with verified financial data above (FMP fundamentals, FRED global macro, India macro/FII-DII flows).
Use the actual earnings numbers, margin figures, growth rates, and transcript quotes wherever possible.
Cite specific figures. Label any claim not supported by the above data as [SPECULATIVE].

Return ONLY a valid JSON object with exactly these 8 keys (no markdown, no code fences):
{{
  "price_action": "Describe the % decline over 1M, 3M, 6M, 1Y and when the breakdown began. Cite actual price levels.",
  "fundamental_catalysts": "Use the FMP earnings data above. Quote actual EPS misses, margin compression, revenue declines. Be specific with numbers.",
  "sectoral_macro": "Cover industry cycle deterioration, adverse policy changes, regulatory headwinds, commodity pressure, or peer de-rating relevant to {sector}.",
  "corporate_actions": "Cover management exits, stake sales, promoter pledging or selling, fundraise failures, or index exclusion in the last 6 months.",
  "narrative_shift": "What story broke down vs 6-12 months ago? Has the company been re-rated from growth/structural to cyclical/value-trap?",
  "flow_factors": "Cover FII/DII selling, mutual fund redemptions, breakdown from key support, or long unwinding.",
  "risks": "What could reverse or deepen this fall? Address valuation support, potential catalysts for recovery, and risks of further downside.",
  "summary": "One sentence completing: The stock is at a 52-week low primarily because _."
}}"""

    last_err = ""
    try:
        client   = _OpenAI(api_key=_OPENAI_KEY)
        response = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        raw    = response.choices[0].message.content.strip()
        result = json.loads(raw)
        # ✅ Only cache on success — errors are never stored
        st.session_state[cache_key] = result
        return result
    except Exception as e:
        last_err = str(e)

    # Return fallback but do NOT cache it — next click will retry fresh
    return _fallback_deep_dive(ticker, company, sector, signal,
                               price, high52, low52, pct_from_high,
                               pct_from_low, vsurge, pe, dcf_upside,
                               error=last_err)


def _fallback_deep_dive(ticker, company, sector, signal,
                        price, high52, low52, pct_from_high, pct_from_low,
                        vsurge, pe, dcf_upside, error="") -> dict:
    is_hi = "HIGH" in signal.upper()
    level = "52-week high" if is_hi else "52-week low"
    err_note = f" (OpenAI error: {error[:120]})" if error else " (OpenAI API key not configured)"
    return {
        "price_action":          f"Price data available above. AI analysis unavailable.{err_note}",
        "fundamental_catalysts": f"{company} is near its {level}. Fundamental catalyst research requires OpenAI API.{err_note}",
        "sectoral_macro":        f"Sector: {sector}. Macro analysis requires OpenAI API.{err_note}",
        "corporate_actions":     f"Corporate action research requires OpenAI API.{err_note}",
        "narrative_shift":       f"Narrative analysis requires OpenAI API.{err_note}",
        "flow_factors":          f"Volume surge: {vsurge:.1f}×. FII/DII flow analysis requires OpenAI API.{err_note}",
        "risks":                 f"Risk analysis requires OpenAI API.{err_note}",
        "summary":               f"The stock is at a {level} — enable OpenAI API for full analysis.",
    }


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_google_news(company: str, ticker: str) -> list[dict]:
    """
    Fetch recent news for a company via Google News RSS (no API key needed).
    Returns list of dicts: {title, link, source, published, summary}
    """
    # Build two queries — one by company name, one by ticker (without .NS)
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query        = f'"{company}" OR "{clean_ticker}" stock India'
    encoded      = urllib.parse.quote(query)
    url          = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    try:
        feed    = feedparser.parse(url)
        results = []
        for entry in feed.entries[:15]:   # cap at 15 articles
            # Parse publish date
            pub = ""
            try:
                dt  = parsedate_to_datetime(entry.get("published", ""))
                pub = dt.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                pub = entry.get("published", "")

            # Strip HTML tags from summary
            raw_summary = entry.get("summary", "")
            clean_summary = re.sub(r"<[^>]+>", "", raw_summary).strip()

            # Source is usually inside the title as "Headline - Source"
            title  = entry.get("title", "")
            source = ""
            if " - " in title:
                parts  = title.rsplit(" - ", 1)
                title  = parts[0].strip()
                source = parts[1].strip()

            results.append({
                "title":     title,
                "link":      entry.get("link", "#"),
                "source":    source,
                "published": pub,
                "summary":   clean_summary[:220] + "…" if len(clean_summary) > 220 else clean_summary,
            })
        return results
    except Exception:
        return []


def _render_news_feed(company: str, ticker: str, accent: str):
    """Render the Google News feed section below the analyst report."""
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;color:{_MUTED};padding:10px 0 10px;">'
        f'&#128240;&nbsp;&nbsp;Latest News — {company}</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching latest news…"):
        articles = _fetch_google_news(company, ticker)

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
    upside: float, analysis: dict, articles: list[dict],
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
         "DCF Upside",     fmt(upside, "", "%")],
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
        ("1. RECENT PRICE ACTION",         "price_action",          _BLUE_C),
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
    upside = _safe_float(row.get("upside_downside_pct"))
    vsurge = _safe_float(row.get("volume_surge"))
    pe     = _safe_float(row.get("pe_ratio"))
    pfh    = _safe_float(row.get("pct_from_high")) or 0.0
    pfl    = _safe_float(row.get("pct_from_low"))  or 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price",  f"₹{price:,.2f}"   if price  else "—")
    c2.metric("52W High",       f"₹{high52:,.2f}"  if high52 else "—",
              f"{abs(pfh):.2f}% from high")
    c3.metric("52W Low",        f"₹{low52:,.2f}"   if low52  else "—",
              f"{abs(pfl):.2f}% from low")
    c4.metric("DCF Upside",
              f"{upside:+.2f}%" if upside is not None else "—",
              "undervalued" if upside and upside > 0 else "overvalued" if upside else "—")
    c5.metric("P/E Ratio",      f"{pe:.1f}×"       if pe     else "—")
    c6.metric("Volume Surge",   f"{vsurge:.2f}×"   if vsurge else "—")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── AI call ───────────────────────────────────────────────────────────────
    headlines_raw = row.get("news_headlines") or []
    headlines_str = (" | ".join(str(h) for h in headlines_raw[:5])
                     if headlines_raw else "No recent headlines available.")

    # ── Fetch all data sources in parallel display ────────────────────────────
    src_col1, src_col2, src_col3 = st.columns(3)

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
            f'{"✅ FRED macro loaded" if fred_ctx else "⚠️ FRED unavailable (add FRED_API_KEY)"}</div>',
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

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    with st.spinner(f"AI is writing the analyst report for {name}…"):
        analysis = _ai_deep_dive(
            ticker=ticker, company=name, sector=row.get("sector", ""),
            signal=sig, price=price or 0, high52=high52 or 0,
            low52=low52 or 0, pct_from_high=pfh, pct_from_low=pfl,
            vsurge=vsurge or 0, pe=pe or 0, dcf_upside=upside or 0,
            news_headlines=headlines_str,
            fmp_context=fmp_ctx,
            fred_context=fred_ctx,
            india_macro_context=india_ctx,
        )

    # ── Detect quota error and show actionable help ───────────────────────────
    summary       = analysis.get("summary", "")
    is_api_fail = "enable OpenAI API" in summary or "OpenAI error" in summary

    if is_api_fail:
        st.markdown(
            '<div style="background:#2a1a00;border:1px solid #F0B429;'
            'border-radius:8px;padding:16px 20px;margin-bottom:16px;">'
            '<div style="font-size:13px;font-weight:700;color:#F0B429;'
            'margin-bottom:8px;">⚠️ AI Analysis Unavailable</div>'
            '<div style="font-size:13px;color:#C9D1D9;line-height:1.7;">'
            'Could not connect to OpenAI. Check your API key in the .env file.<br>'
            '<b>To fix:</b> Ensure OPENAI_API_KEY is set correctly and your account has credits.<br>'
            '<a href="https://platform.openai.com/account/api-keys" target="_blank" '
            'style="color:#58A6FF;">→ platform.openai.com/account/api-keys</a>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        cache_key = f"ai_{ticker}_{sig}"
        if st.button("🔄 Retry Analysis", key=f"retry_{ticker}",
                     use_container_width=True):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()
        return   # Don't render empty cards

    # ── Summary banner ────────────────────────────────────────────────────────
    if summary:
        st.markdown(
            f'<div style="background:{"#0d2618" if is_hi else "#2a0d12"};'
            f'border:1px solid {accent};border-radius:8px;'
            f'padding:14px 20px;margin-bottom:16px;">'
            f'<span style="font-size:11px;font-weight:700;letter-spacing:2px;'
            f'text-transform:uppercase;color:{accent};">&#9670; Analyst Verdict</span>'
            f'<div style="font-size:14px;font-weight:600;color:#E6EDF3;'
            f'margin-top:6px;line-height:1.5;">{summary}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 7 sections in 2-column grid ───────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        _section_card("📊", "1. Recent Price Action",
                      analysis.get("price_action", "—"), _BLUE)
        _section_card("⚡", "2. Fundamental Catalysts",
                      analysis.get("fundamental_catalysts", "—"), accent)
        _section_card("🏛", "3. Sectoral & Macro Tailwinds",
                      analysis.get("sectoral_macro", "—"), _YELLOW)
        _section_card("🔔", "4. Corporate Actions & Events",
                      analysis.get("corporate_actions", "—"), "#A371F7")

    with right:
        _section_card("💬", "5. Narrative Shift",
                      analysis.get("narrative_shift", "—"), "#58A6FF")
        _section_card("🌊", "6. Technical & Flow Factors",
                      analysis.get("flow_factors", "—"), "#3FB950")
        _section_card("⚠", "7. Risks to the Rally" if is_hi else "7. Risks & Recovery Triggers",
                      analysis.get("risks", "—"), _RED)

    # ── Full Google News feed ─────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#21262D;margin:20px 0 4px;'>",
                unsafe_allow_html=True)
    _render_news_feed(name, ticker, accent)

    # ── Download PDF report ───────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#21262D;margin:0 0 12px;'>",
                unsafe_allow_html=True)

    with st.spinner("Building PDF report…"):
        articles   = _fetch_google_news(name, ticker)   # already cached
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
            upside   = upside or 0,
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
    params: dict = {}

    with st.sidebar:
        st.markdown(
            '<div style="font-size:15px;font-weight:700;color:#E6EDF3;'
            'padding:12px 0 4px;letter-spacing:-0.3px;">&#9881; Screener Controls</div>',
            unsafe_allow_html=True,
        )

        # ── Universe ──────────────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#6E7681;'
            'letter-spacing:2px;text-transform:uppercase;margin:12px 0 6px;">Universe</div>',
            unsafe_allow_html=True,
        )
        universe = st.radio(
            "Universe",
            options=["Nifty 500 (live)", "Nifty 50 only", "Custom"],
            index=0,
            label_visibility="collapsed",
        )
        custom_txt = ""
        if universe == "Custom":
            custom_txt = st.text_area(
                "Tickers (one per line, .NS suffix required)",
                placeholder="RELIANCE.NS\nTCS.NS\nHDFCBANK.NS",
                height=90,
            )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Technical params ──────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#6E7681;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">'
            'Technical Parameters</div>',
            unsafe_allow_html=True,
        )
        threshold = st.slider("52W Proximity Band (%)", 0.5, 10.0, 2.5, 0.5,
                               help="Flag stocks within this % of their 52W extreme") / 100
        vol_surge = st.slider("Min Volume Surge (×)",    1.0,  5.0, 1.2, 0.1,
                               help="Current volume ÷ 20-day average volume")
        window    = st.slider("Lookback Window (days)", 100,  365, 252, 10)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── DCF params ────────────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#6E7681;'
            'letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">'
            'DCF Model</div>',
            unsafe_allow_html=True,
        )
        wacc        = st.slider("WACC / Discount Rate (%)", 6.0, 18.0, 10.0, 0.5) / 100
        term_growth = st.slider("Terminal Growth Rate (%)", 1.0,  6.0,  3.0, 0.5) / 100

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Run button ────────────────────────────────────────────────────────
        run = st.button("▶  Run Screen", use_container_width=True, type="primary")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # ── Downloads (only when data exists) ────────────────────────────────
        if "screen_results" in st.session_state:
            res   = st.session_state["screen_results"]
            highs = res.get("highs", pd.DataFrame())
            lows  = res.get("lows",  pd.DataFrame())

            if not highs.empty or not lows.empty:
                combined  = pd.concat([highs, lows], ignore_index=True)
                # Drop list-type columns that break CSV export
                drop_cols = [c for c in ["news_headlines"] if c in combined.columns]
                combined  = combined.drop(columns=drop_cols, errors="ignore")

                st.download_button(
                    label="⬇  Download CSV",
                    data=combined.to_csv(index=False).encode("utf-8"),
                    file_name=f"nifty500_screen_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.caption("To save as PDF: open the CSV in Excel, or use your browser's Print → Save as PDF on the report page.")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── Auto-refresh toggle ───────────────────────────────────────────────
        auto_refresh = st.toggle("Auto-refresh index prices (60s)", value=False)

        # ── Run stats ─────────────────────────────────────────────────────────
        if "screen_results" in st.session_state:
            res = st.session_state["screen_results"]
            n_h = len(res.get("highs", pd.DataFrame()))
            n_l = len(res.get("lows",  pd.DataFrame()))
            n_e = len(res.get("errors", []))
            is_mock = res.get("is_mock", False)
            ts = st.session_state.get("screen_ts", datetime.now()).strftime("%H:%M:%S")

            st.markdown(
                f'<div style="background:{_CARD};border:1px solid #21262D;'
                f'border-radius:6px;padding:12px 14px;font-size:12px;">'
                f'<div style="font-size:10px;font-weight:700;color:{_MUTED};'
                f'letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">Last Run</div>'
                f'<div style="color:{_GREEN}">&#8679; {n_h} Breakout High</div>'
                f'<div style="color:{_RED}">&#8681; {n_l} Breakdown Low</div>'
                f'<div style="color:{_MUTED}">&#9888; {n_e} Ticker Errors</div>'
                f'<div style="color:{_MUTED};margin-top:4px;">&#9201; {ts}</div>'
                f'{"<div style=\'color:#D29922;font-size:11px;margin-top:4px;\'>★ Mock data</div>" if is_mock else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    params = {
        "threshold":   threshold,
        "vol_surge":   vol_surge,
        "window":      window,
        "wacc":        wacc,
        "term_growth": term_growth,
        "universe":    universe,
        "custom_txt":  custom_txt,
        "run":         run,
        "auto_refresh":auto_refresh,
    }
    return params


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Inject styles ─────────────────────────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Ticker tape ───────────────────────────────────────────────────────────
    _render_tape()

    # ── App header ────────────────────────────────────────────────────────────
    _render_header()

    # ── Sidebar (returns param dict) ──────────────────────────────────────────
    p = _render_sidebar()

    # ── Spacing ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Index cards ───────────────────────────────────────────────────────────
    _render_index_cards()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Auto-refresh (clears cached index/tape data, re-runs every 60 s) ─────
    if p["auto_refresh"]:
        _fetch_index_data.clear()
        _fetch_tape_data.clear()

    # ── Build ticker list ─────────────────────────────────────────────────────
    if p["universe"] == "Nifty 50 only":
        tickers = _NIFTY_50
    elif p["universe"] == "Custom" and p["custom_txt"].strip():
        tickers = [t.strip() for t in p["custom_txt"].splitlines() if t.strip()]
    else:
        # Nifty 500: try live NSE fetch, fall back to hardcoded
        tickers = fetch_nifty500_live() or NIFTY_500_TICKERS or _NIFTY_50

    # ── Run screen (on button click OR first page load) ───────────────────────
    if p["run"] or "screen_results" not in st.session_state:
        with st.spinner(
            f"Screening **{len(tickers)} stocks**… "
            "This takes 2–10 minutes for Nifty 500 to respect API rate limits."
        ):
            results = _run_screen(tickers, p)
    else:
        results = st.session_state["screen_results"]

    if not results:
        st.markdown(
            '<div style="text-align:center;padding:60px;color:#484F58;">'
            '<div style="font-size:40px;">📊</div>'
            '<div style="font-size:16px;margin-top:12px;">Press <b>▶ Run Screen</b> in the sidebar to start.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    highs_df = results.get("highs", pd.DataFrame())
    lows_df  = results.get("lows",  pd.DataFrame())
    errors   = results.get("errors", [])
    is_mock  = results.get("is_mock", False)

    # Mock data banner
    if is_mock:
        st.warning(
            "⚠ **Mock data active** — `engine.py` not found or all API calls failed. "
            "Install dependencies and ensure `engine.py` is in the same folder.",
            icon="⚠",
        )

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sec-hdr" style="margin-top:8px;">'
        f'&#9670; Today\'s Signals &nbsp;'
        f'<span style="color:{_GREEN}">&#8679; {len(highs_df)} Breakout Highs</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:{_RED}">&#8681; {len(lows_df)} Breakdown Lows</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t_hi, t_lo, t_err = st.tabs([
        f"▲  Breakout Highs  ({len(highs_df)})",
        f"▼  Breakdown Lows  ({len(lows_df)})",
        f"⚠  Errors  ({len(errors)})",
    ])

    # ── HIGHS TAB ─────────────────────────────────────────────────────────────
    with t_hi:
        selected_hi = _render_signals_table(highs_df, key="hi")
        if selected_hi and not highs_df.empty:
            mask = highs_df["ticker"] == selected_hi
            if mask.any():
                row = highs_df[mask].iloc[0].to_dict()
                st.markdown("<hr/>", unsafe_allow_html=True)
                _render_spotlight(selected_hi, row, p)

    # ── LOWS TAB ──────────────────────────────────────────────────────────────
    with t_lo:
        selected_lo = _render_signals_table(lows_df, key="lo")
        if selected_lo and not lows_df.empty:
            mask = lows_df["ticker"] == selected_lo
            if mask.any():
                row = lows_df[mask].iloc[0].to_dict()
                st.markdown("<hr/>", unsafe_allow_html=True)
                _render_spotlight(selected_lo, row, p)

    # ── ERRORS TAB ────────────────────────────────────────────────────────────
    with t_err:
        if not errors:
            st.success("✅ No errors in the last run — all tickers processed cleanly.")
        else:
            err_df = pd.DataFrame(errors)
            st.dataframe(err_df, use_container_width=True)

    # ── Auto-refresh sleep + rerun ────────────────────────────────────────────
    if p["auto_refresh"]:
        time.sleep(60)
        st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
