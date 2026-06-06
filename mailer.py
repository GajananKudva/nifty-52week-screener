"""
mailer.py — AI Narrative Generator & Email Delivery
=====================================================
Orchestrates the full daily pipeline:
  1. Loads config from environment variables
  2. Runs the DataEngine screen
  3. Sends the flagged data to an LLM (OpenAI or Google Gemini)
     to generate a CIO-style executive narrative
  4. Renders the Jinja2 HTML report
  5. Delivers via SMTP (or saves a local fallback if SMTP not configured)

Usage
-----
  python mailer.py                   # run once immediately
  python mailer.py --save-only       # render & save HTML but skip email

Environment variables required — see .env.example
"""

import argparse
import io
import logging
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import requests

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from engine import DataEngine, ScreenerConfig

# Load .env from same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# ── Logging (mirrors engine.py config) ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("screener.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mailer")

# ── Optional LLM imports ──────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

try:
    from google import genai as genai
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# NIFTY 500 UNIVERSE
# Primary  : fetched live from NSE India at runtime (always up-to-date)
# Fallback : hardcoded list below (used when NSE is unreachable)
# Override : set TICKER_UNIVERSE in .env (comma-separated, e.g. "RELIANCE.NS,TCS.NS")
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nifty500_live() -> list[str]:
    """
    Download the current Nifty 500 constituent list from NSE India and
    return Yahoo Finance ticker symbols (suffixed with '.NS').

    Returns an empty list on any network / parse failure — caller falls
    back to NIFTY_500_TICKERS.
    """
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept":          "text/html,application/xhtml+xml,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer":         "https://www.nseindia.com/",
    }
    try:
        import pandas as pd
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        df   = pd.read_csv(io.StringIO(resp.text))
        # NSE CSV column is "Symbol"
        symbols = df["Symbol"].dropna().str.strip().tolist()
        tickers = [s + ".NS" for s in symbols if s]
        logger.info(f"Nifty 500 live fetch: {len(tickers)} tickers from NSE ✓")
        return tickers
    except Exception as exc:
        logger.warning(f"Nifty 500 live fetch failed ({exc}); using hardcoded fallback.")
        return []


# ── Hardcoded Nifty 500 fallback (as of May 2025) ────────────────────────────
# All tickers use the Yahoo Finance '.NS' (NSE) suffix.
NIFTY_500_TICKERS: list[str] = [

    # ── NIFTY 50 ─────────────────────────────────────────────────────────────
    "RELIANCE.NS", "TCS.NS",        "HDFCBANK.NS",   "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS",     "SBIN.NS",       "HINDUNILVR.NS", "ITC.NS",        "LT.NS",
    "BAJFINANCE.NS","KOTAKBANK.NS", "HCLTECH.NS",    "MARUTI.NS",     "AXISBANK.NS",
    "ASIANPAINT.NS","SUNPHARMA.NS", "TITAN.NS",      "WIPRO.NS",      "ULTRACEMCO.NS",
    "ONGC.NS",     "NTPC.NS",       "JSWSTEEL.NS",   "POWERGRID.NS",  "M&M.NS",
    "BAJAJFINSV.NS","TMCV.NS","TMPV.NS",       "ADANIENT.NS",   "ADANIPORTS.NS", "TATASTEEL.NS",
    "COALINDIA.NS","NESTLEIND.NS",  "DIVISLAB.NS",   "CIPLA.NS",      "DRREDDY.NS",
    "HINDALCO.NS", "TECHM.NS",      "GRASIM.NS",     "APOLLOHOSP.NS", "TATACONSUM.NS",
    "INDUSINDBK.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS", "EICHERMOT.NS",  "BRITANNIA.NS",
    "BEL.NS",      "SHRIRAMFIN.NS", "BPCL.NS",       "TRENT.NS",      "SBILIFE.NS",

    # ── NIFTY NEXT 50 ────────────────────────────────────────────────────────
    "ADANIGREEN.NS","ADANIPOWER.NS","ATGL.NS",       "AMBUJACEM.NS",  "ABB.NS",
    "ABCAPITAL.NS","ABFRL.NS",      "ACC.NS",        "AUROPHARMA.NS", "BANKBARODA.NS",
    "BERGEPAINT.NS","BOSCHLTD.NS",  "CANBK.NS",      "CHOLAFIN.NS",   "COLPAL.NS",
    "DABUR.NS",    "DLF.NS",        "GODREJCP.NS",   "GODREJPROP.NS", "HAVELLS.NS",
    "HDFCAMC.NS",  "HDFCLIFE.NS",   "HINDZINC.NS",   "ICICIPRULI.NS", "ICICIGI.NS",
    "INDIGO.NS",   "IOC.NS",        "IRCTC.NS",      "JINDALSTEL.NS", "LICI.NS",
    "LODHA.NS",    "LTIM.NS",       "LTTS.NS",       "MARICO.NS",     "MCDOWELL-N.NS",
    "MOTHERSON.NS","MUTHOOTFIN.NS", "NAUKRI.NS",     "OBEROIRLTY.NS", "OFSS.NS",
    "PAGEIND.NS",  "PERSISTENT.NS", "PIDILITIND.NS", "PIIND.NS",      "PNB.NS",
    "POLYCAB.NS",  "SBICARD.NS",    "SIEMENS.NS",    "TORNTPHARM.NS", "TVSMOTOR.NS",

    # ── NIFTY MIDCAP 150 ─────────────────────────────────────────────────────
    # Banking & Finance
    "IDFCFIRSTB.NS","FEDERALBNK.NS","BANDHANBNK.NS", "RBLBANK.NS",    "YESBANK.NS",
    "KARURVYSYA.NS","KTKBANK.NS",   "DCBBANK.NS",    "EQUITASBNK.NS", "UJJIVANSFB.NS",
    "CANFINHOME.NS","AAVAS.NS",     "HOMEFIRST.NS",  "APTUS.NS",      "REPCO.NS",
    "MANAPPURAM.NS","SUNDARMFIN.NS","LICHSGFIN.NS",  "POONAWALLA.NS", "IIFL.NS",
    "CAMS.NS",     "CDSL.NS",       "BSE.NS",        "RECLTD.NS",     "PFC.NS",
    "IRFC.NS",     "HUDCO.NS",      "IREDA.NS",

    # IT & Technology
    "MPHASIS.NS",  "COFORGE.NS",    "KPITTECH.NS",   "TATAELXSI.NS",  "ZENSAR.NS",
    "CYIENT.NS",   "BIRLASOFT.NS",  "HAPPSTMNDS.NS", "BSOFT.NS",      "MASTEK.NS",
    "ROUTE.NS",    "TANLA.NS",      "INTELLECT.NS",  "NEWGEN.NS",     "RATEGAIN.NS",
    "REDINGTON.NS",

    # Pharma & Healthcare
    "LUPIN.NS",    "ALKEM.NS",      "BIOCON.NS",     "LAURUSLABS.NS", "GLAND.NS",
    "IPCALAB.NS",  "GRANULES.NS",   "ABBOTINDIA.NS", "PFIZER.NS",     "GLAXO.NS",
    "AJANTPHARM.NS","ERIS.NS",      "WOCKPHARMA.NS", "NATCOPHARMA.NS","STRIDES.NS",
    "INDOCO.NS",   "ZYDUSLIFE.NS",  "CAPLIPOINT.NS", "SEQUENT.NS",    "SUVEN.NS",
    "JB.NS",       "METROPOLIS.NS", "LALPATHLAB.NS", "FORTIS.NS",     "MAXHEALTH.NS",
    "ASTER.NS",    "RAINBOW.NS",    "MEDPLUS.NS",

    # Energy & Power
    "TATAPOWER.NS","GAIL.NS",       "HINDPETRO.NS",  "NHPC.NS",       "SJVN.NS",
    "JSWENERGY.NS","CESC.NS",       "TORNTPOWER.NS", "ADANIENSOL.NS", "SUZLON.NS",
    "INOXWIND.NS", "WAAREEENER.NS", "IGL.NS",        "MGL.NS",        "GUJGASLTD.NS",
    "AEGISCHEM.NS",

    # Metals & Mining
    "NMDC.NS",     "SAIL.NS",       "NATIONALUM.NS", "WELCORP.NS",    "RATNAMANI.NS",
    "APARINDS.NS", "MIDHANI.NS",    "HINDZINC.NS",   "VEDL.NS",

    # Capital Goods & Industrials
    "BHEL.NS",     "CUMMINSIND.NS", "THERMAX.NS",    "KEC.NS",        "KPTL.NS",
    "GRINDWELL.NS","TIMKEN.NS",     "SKF.NS",        "ELGI.NS",       "ISGEC.NS",
    "ELECON.NS",   "POLYCAB.NS",    "DIXON.NS",      "AMBER.NS",      "BLUESTARCO.NS",
    "VOLTAS.NS",   "HAVELLS.NS",    "HONAUT.NS",     "SIEMENS.NS",

    # Automobiles & Auto Ancillaries
    "ASHOKLEY.NS", "ESCORTS.NS",    "ENDURANCE.NS",  "BHARATFORG.NS", "TIINDIA.NS",
    "SUNDRMFAST.NS","MRF.NS",       "BALKRISIND.NS", "CEATLTD.NS",    "APOLLOTYRE.NS",
    "EXIDEIND.NS", "BOSCHLTD.NS",   "SUBROS.NS",     "GABRIEL.NS",    "SUPRAJIT.NS",
    "CRAFTSMAN.NS","MAHINDCIE.NS",  "FORCEMOT.NS",

    # Cement
    "SHREECEM.NS", "RAMCOCEM.NS",   "JKCEMENT.NS",   "DALMIACEM.NS",  "BIRLACORPN.NS",
    "JKLAKSHMI.NS","STARCEMENT.NS",

    # Chemicals
    "SRF.NS",      "ATUL.NS",       "DEEPAKNTR.NS",  "VINATIORGA.NS", "NAVIN.NS",
    "TATACHEM.NS", "GSFC.NS",       "GNFC.NS",       "COROMANDEL.NS", "CHAMBAL.NS",
    "DEEPAKFERT.NS","NFL.NS",       "BALAMINES.NS",  "FINEORG.NS",    "ALKYLAMINE.NS",
    "GALAXYSURF.NS","ROSSARI.NS",   "CLEAN.NS",      "NOCIL.NS",      "PCBL.NS",
    "RALLIS.NS",   "SUMICHEM.NS",   "ASAHIINDIA.NS",

    # Real Estate
    "PHOENIXLTD.NS","PRESTIGE.NS",  "BRIGADE.NS",    "SOBHA.NS",      "MAHLIFE.NS",
    "KOLTEPATIL.NS","SUNTECK.NS",   "CHALET.NS",     "NESCO.NS",      "ASHIANA.NS",

    # Consumer Goods / FMCG
    "EMAMILTD.NS", "JYOTHYLAB.NS",  "RADICO.NS",     "VSTIND.NS",     "GODFRYPHLP.NS",
    "ZYDUSWELL.NS","UBL.NS",        "VBL.NS",

    # Logistics & Transport
    "CONCOR.NS",   "BLUEDART.NS",   "GMRINFRA.NS",   "IRB.NS",

    # Infrastructure / Construction
    "NBCC.NS",     "NCC.NS",        "KNRCON.NS",     "PNCINFRA.NS",   "AHLUCONT.NS",

    # Pipes, Plastics & Building Materials
    "ASTRAL.NS",   "PRINCEPIPES.NS","SUPREMEIND.NS", "FINOLEXIND.NS",

    # Textiles & Apparel
    "KPR.NS",      "TRIDENT.NS",    "WELSPUNIND.NS", "VARDHMAN.NS",   "RAYMOND.NS",
    "ARVIND.NS",

    # Media & Entertainment
    "SUNTVNET.NS", "ZEEL.NS",       "PVRINOX.NS",

    # Hospitality & Tourism
    "INDHOTEL.NS", "EIHOTEL.NS",    "LEMONTREE.NS",

    # Telecom & Tech Infra
    "TATACOMM.NS", "HFCL.NS",

    # Defence
    "HAL.NS",      "BDL.NS",        "COCHINSHIP.NS", "GRSE.NS",       "MAZAGON.NS",
    "DATAPATTNS.NS","MTAR.NS",      "BEML.NS",

    # ── NIFTY SMALLCAP 250 (selection) ───────────────────────────────────────
    # Banking & Finance
    "SOUTHBANK.NS","IDFCFIRSTB.NS", "MANAPPURAM.NS",

    # IT
    "SONATSOFTW.NS","MASTECH.NS",   "SAKSOFT.NS",    "INTELLECT.NS",

    # Pharma
    "ALEMBICLTD.NS","SOLARA.NS",    "WINDLAS.NS",    "LAURUS.NS",

    # Energy
    "ORIENTELEC.NS","INOXWIND.NS",  "WAAREEENER.NS", "PTCIL.NS",

    # Chemicals
    "AAPL.NS",  # Astec LifeSciences  ← ASTEC.NS actually
    "ASTEC.NS", "LXCHEM.NS",

    # Auto
    "LUMAXTECH.NS","SETCO.NS",      "AUTOAXLES.NS",

    # Sugar
    "BALRAMCHIN.NS","RENUKA.NS",    "BAJAJHIND.NS",  "TRIVENIENG.NS",

    # Paper
    "TNPL.NS",     "JKPAPER.NS",

    # Retail
    "SHOPERSTOP.NS","VMART.NS",

    # Real Estate
    "GODREJPROP.NS","OBEROIRLTY.NS",

    # Diversified / Conglomerate
    "ADANIENT.NS", "BAJAJHLDNG.NS", "ESCORTS.NS",    "WABCOINDIA.NS",

    # FMCG / Consumer small
    "PATANJALI.NS","BIKAJI.NS",     "PRATAAP.NS",    "DEVYANI.NS",    "SAPPHIRE.NS",

    # New-age / Digital
    "ZOMATO.NS",   "NYKAA.NS",      "POLICYBZR.NS",  "PAYTM.NS",

    # Miscellaneous
    "TATACOMM.NS", "ORIENTCEM.NS",  "JKIL.NS",       "NESCO.NS",
]

# De-duplicate while preserving order
_seen: set = set()
NIFTY_500_TICKERS = [
    t for t in NIFTY_500_TICKERS
    if t not in _seen and not _seen.add(t)  # type: ignore[func-returns-value]
]


# ══════════════════════════════════════════════════════════════════════════════
# NARRATIVE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
class NarrativeGenerator:
    """
    Generates a CIO-style 3-paragraph executive summary via LLM.

    Supports:
      - OpenAI  (set LLM_PROVIDER=openai  + OPENAI_API_KEY)
      - Gemini  (set LLM_PROVIDER=gemini  + GEMINI_API_KEY)
    Falls back to a structured placeholder if no LLM key is present.
    """

    def __init__(self) -> None:
        self.provider   = os.getenv("LLM_PROVIDER", "openai").lower()
        self._client    = None
        self._setup_client()

    # ── Client initialisation ─────────────────────────────────────────────────
    def _setup_client(self) -> None:
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                logger.warning("OPENAI_API_KEY not set — narrative disabled.")
                return
            if not _OPENAI_OK:
                logger.warning("openai package not installed — narrative disabled.")
                return
            self._client = OpenAI(api_key=key)
            logger.info("LLM client: OpenAI ✓")

        elif self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY", "")
            if not key:
                logger.warning("GEMINI_API_KEY not set — narrative disabled.")
                return
            if not _GEMINI_OK:
                logger.warning("google-generativeai not installed — narrative disabled.")
                return
            self._client = genai.Client(api_key=key)
            self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            logger.info("LLM client: Google Gemini ✓")

        else:
            logger.warning(f"Unknown LLM_PROVIDER='{self.provider}'. Use 'openai' or 'gemini'.")

    # ── Prompt builder ────────────────────────────────────────────────────────
    @staticmethod
    def _build_prompt(highs: list[dict], lows: list[dict], run_date: str) -> str:
        def _row(r: dict) -> str:
            name    = r.get("company_name") or r.get("ticker", "N/A")
            ticker  = r.get("ticker", "N/A")
            sector  = r.get("sector")  or "N/A"
            price   = r.get("current_price", "N/A")
            upside  = r.get("upside_downside_pct")
            pe      = r.get("pe_ratio")
            vsurge  = r.get("volume_surge", "N/A")

            upside_str = f"{upside:+.1f}%" if upside is not None else "N/A"
            pe_str     = f"{pe:.1f}x"      if pe     is not None else "N/A"

            return (
                f"  • {name} ({ticker}) | Sector: {sector} | Price: ${price} | "
                f"DCF Upside: {upside_str} | P/E: {pe_str} | Vol Surge: {vsurge}x"
            )

        highs_block = "\n".join(_row(r) for r in highs[:12]) if highs else "  (none)"
        lows_block  = "\n".join(_row(r) for r in lows[:12])  if lows  else "  (none)"

        return f"""You are the Chief Investment Officer at a $5 billion long/short equity hedge fund.
Today is {run_date}. Your quantitative screening system has completed its nightly EOD scan.

STOCKS BREAKING OUT TO 52-WEEK HIGHS (volume-surge confirmed ≥ 1.2x):
{highs_block}

STOCKS BREAKING DOWN TO 52-WEEK LOWS (volume-surge confirmed ≥ 1.2x):
{lows_block}

Write a precise, institutional-quality executive summary using EXACTLY the following three paragraphs
(label each with its heading):

**Paragraph 1 — Sector Trends & Macro Context**
Identify sector patterns in today's breakouts and breakdowns. What does the clustering of moves in
specific sectors signal about macro forces, sector rotation, or a regime change?

**Paragraph 2 — Risk Warnings**
Flag the 2–3 most significant risks visible in today's data: overbought readings, high-P/E
names with limited DCF upside (momentum-driven vs. value-driven), and breakdown names that warrant
immediate risk-reduction or short consideration.

**Paragraph 3 — Top Conviction Picks**
Based on the confluence of (a) technical breakout/breakdown, (b) DCF upside/downside, and
(c) volume confirmation, state your top 2–3 long ideas and 1–2 short ideas. Be explicit: name the
ticker, the reason, and the key risk to the thesis.

Tone: Direct, institutional, no hedging phrases ("it seems", "perhaps", "could potentially").
Be decisive. Maximum 380 words total across all three paragraphs."""

    # ── Generate ──────────────────────────────────────────────────────────────
    def generate(self, highs: list[dict], lows: list[dict]) -> str:
        """Return the AI narrative string, or a structured placeholder."""
        if not highs and not lows:
            return (
                "Today's screen returned no volume-confirmed breakouts or breakdowns. "
                "Markets appear range-bound — no high-conviction signals meet the combined "
                "technical and volume threshold. Hold current positions; no new entries triggered."
            )

        prompt = self._build_prompt(highs, lows, datetime.now().strftime("%B %d, %Y"))

        if self._client is None:
            return self._placeholder(highs, lows)

        try:
            if self.provider == "openai":
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                resp  = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an elite institutional portfolio manager with 20+ years of "
                                "equity investing experience. Write with conviction and precision."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=700,
                    temperature=0.65,
                )
                narrative = resp.choices[0].message.content.strip()

            elif self.provider == "gemini":
                resp      = self._client.models.generate_content(
                    model=self._gemini_model,
                    contents=prompt,
                )
                narrative = resp.text.strip()

            else:
                return self._placeholder(highs, lows)

            logger.info("AI narrative generated ✓")
            return narrative

        except Exception as exc:
            logger.error(f"LLM narrative generation failed: {exc}")
            return self._placeholder(highs, lows)

    # ── Per-stock "Why" explanations ─────────────────────────────────────────
    def generate_stock_reasons(self, all_stocks: list[dict]) -> dict[str, str]:
        """
        For each flagged stock, ask the LLM for a 1-2 sentence explanation of
        WHY it reached its 52-week extreme, using sector context, news headlines,
        and valuation data as evidence.

        Returns {ticker: reason_string}. Falls back to a headline-based summary
        if no LLM client is available.
        """
        if not all_stocks:
            return {}

        # ── Build concise data block for the prompt ───────────────────────────
        lines = []
        for s in all_stocks:
            ticker    = s.get("ticker", "")
            name      = s.get("company_name") or ticker
            sector    = s.get("sector") or "N/A"
            signal    = s.get("signal", "")
            vol       = s.get("volume_surge")
            pe        = s.get("pe_ratio")
            upside    = s.get("upside_downside_pct")
            headlines = s.get("news_headlines") or []

            if signal == "BREAKOUT_HIGH":
                pos = f"52-week HIGH breakout (within {s.get('pct_from_high')}% of high)"
            else:
                pos = f"52-week LOW breakdown (within {s.get('pct_from_low')}% of low)"

            news_str = " | ".join(headlines[:3]) if headlines else "No recent headlines"
            pe_str   = f"{pe:.1f}x" if pe else "N/A"
            up_str   = f"{upside:+.1f}%" if upside is not None else "N/A"

            lines.append(
                f"• {name} ({ticker}) | Sector: {sector} | Position: {pos} | "
                f"Vol Surge: {vol}x | P/E: {pe_str} | DCF Upside: {up_str}\n"
                f"  Recent News: {news_str}"
            )

        stocks_block = "\n".join(lines)

        prompt = (
            "You are a senior equity research analyst. For each stock below, write "
            "ONE concise sentence (max 35 words) explaining the most likely reason "
            "this stock has reached its current 52-week extreme. Use the sector, "
            "recent news headlines, and valuation data as evidence. Be specific — "
            "name the actual driver (e.g. earnings miss, macro headwind, sector rotation, "
            "product cycle, regulatory event). Do NOT use vague phrases like 'market conditions'.\n\n"
            f"Stocks:\n{stocks_block}\n\n"
            "Return valid JSON only — no markdown, no extra text:\n"
            '{"TICKER1": "reason sentence", "TICKER2": "reason sentence"}'
        )

        if self._client is None:
            return self._headline_fallback(all_stocks)

        try:
            import json

            if self.provider == "openai":
                resp = self._client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()

            elif self.provider == "gemini":
                resp = self._client.models.generate_content(
                    model=self._gemini_model,
                    contents=prompt,
                )
                raw = resp.text.strip()
                # Strip markdown code fences if Gemini wraps in ```json
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
            else:
                return self._headline_fallback(all_stocks)

            reasons = json.loads(raw)
            logger.info(f"Per-stock reasons generated for {len(reasons)} tickers ✓")
            return reasons

        except Exception as exc:
            logger.error(f"generate_stock_reasons failed: {exc}")
            return self._headline_fallback(all_stocks)

    @staticmethod
    def _headline_fallback(stocks: list[dict]) -> dict[str, str]:
        """Use the top news headline as the 'why' when no LLM is available."""
        result = {}
        for s in stocks:
            ticker    = s.get("ticker", "")
            headlines = s.get("news_headlines") or []
            if headlines:
                result[ticker] = headlines[0]  # top headline as-is
            else:
                signal = s.get("signal", "")
                result[ticker] = (
                    "No recent news available — signal driven by price/volume action alone."
                    if signal else ""
                )
        return result

    # ── Placeholder narrative ─────────────────────────────────────────────────
    @staticmethod
    def _placeholder(highs: list[dict], lows: list[dict]) -> str:
        n_h = len(highs)
        n_l = len(lows)
        s_h = sorted(
            {r.get("sector") for r in highs if r.get("sector")},
            key=lambda s: sum(1 for r in highs if r.get("sector") == s),
            reverse=True,
        )
        s_l = sorted(
            {r.get("sector") for r in lows if r.get("sector")},
            key=lambda s: sum(1 for r in lows if r.get("sector") == s),
            reverse=True,
        )
        top_h = ", ".join(s_h[:3]) if s_h else "broad market"
        top_l = ", ".join(s_l[:3]) if s_l else "select names"

        return (
            f"**Paragraph 1 — Sector Trends & Macro Context**\n"
            f"Today's screen flagged {n_h} breakout(s) to 52-week highs concentrated in "
            f"{top_h}, alongside {n_l} breakdown(s) near 52-week lows in {top_l}. "
            f"The divergence suggests active sector rotation rather than broad market moves. "
            f"Volume confirmation on all signals reduces false-positive risk.\n\n"
            f"**Paragraph 2 — Risk Warnings**\n"
            f"Monitor any high-P/E breakout names where DCF upside is below 10% — these are "
            f"momentum-driven and susceptible to sharp reversals on any macro surprise. "
            f"Breakdown names with negative DCF upside and elevated volume may accelerate lower; "
            f"tight stops or hedge positions are warranted.\n\n"
            f"**Paragraph 3 — Top Conviction Picks**\n"
            f"Review the data tables below for specific conviction rankings sorted by DCF upside. "
            f"[NOTE: Configure OPENAI_API_KEY or GEMINI_API_KEY in your .env file to enable "
            f"AI-generated CIO narratives tailored to today's specific data.]"
        )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT RENDERER & EMAILER
# ══════════════════════════════════════════════════════════════════════════════
class ReportMailer:
    """Renders the Jinja2 HTML template and delivers via SMTP."""

    def __init__(self) -> None:
        self.smtp_host  = os.getenv("SMTP_HOST",     "smtp.gmail.com")
        self.smtp_port  = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user  = os.getenv("SMTP_USER",     "")
        self.smtp_pass  = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL",    self.smtp_user)
        _to_raw         = os.getenv("TO_EMAILS",     "")
        self.to_emails  = [e.strip() for e in _to_raw.split(",") if e.strip()]

        self.jinja = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent)),
            autoescape=True,
        )

    # ── Render ────────────────────────────────────────────────────────────────
    def render(
        self,
        highs: list[dict],
        lows:  list[dict],
        narrative: str,
        errors: list[dict],
        run_date: str,
    ) -> str:
        try:
            tmpl = self.jinja.get_template("template.html")
        except TemplateNotFound:
            logger.error("template.html not found in script directory.")
            raise

        return tmpl.render(
            run_date        = run_date,
            generation_time = datetime.now().strftime("%H:%M UTC"),
            narrative       = narrative,
            highs           = highs,
            lows            = lows,
            total_highs     = len(highs),
            total_lows      = len(lows),
            errors          = errors[:15],   # cap error display
            has_errors      = len(errors) > 0,
        )

    # ── Send ──────────────────────────────────────────────────────────────────
    def send(self, html: str, run_date: str) -> bool:
        """Send HTML email. Returns True on success, False if saving to file instead."""
        if not self.smtp_user or not self.smtp_pass:
            logger.warning("SMTP credentials not configured — saving report locally.")
            self._save_local(html)
            return False

        if not self.to_emails:
            logger.warning("TO_EMAILS not configured — saving report locally.")
            self._save_local(html)
            return False

        msg              = MIMEMultipart("alternative")
        msg["Subject"]   = f"📊 52W High/Low Daily Screen | {run_date}"
        msg["From"]      = self.from_email
        msg["To"]        = ", ".join(self.to_emails)
        msg["X-Mailer"]  = "52W-Screener v1.0"

        plain = (
            f"Daily 52-Week High/Low Screening Report — {run_date}\n"
            "This report is best viewed in an HTML-capable email client.\n\n"
            f"Breakouts to 52W High: see HTML attachment.\n"
            f"Breakdowns to 52W Low: see HTML attachment."
        )
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html,  "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as srv:
                srv.ehlo()
                srv.starttls()
                srv.ehlo()
                srv.login(self.smtp_user, self.smtp_pass)
                srv.sendmail(self.from_email, self.to_emails, msg.as_string())

            logger.info(f"Email sent to: {self.to_emails}")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "SMTP authentication failed. For Gmail, use an App Password "
                "(https://myaccount.google.com/apppasswords)."
            )
            raise
        except smtplib.SMTPException as exc:
            logger.error(f"SMTP delivery error: {exc}")
            raise
        except OSError as exc:
            logger.error(f"Network error sending email: {exc}")
            raise

    # ── Local save ────────────────────────────────────────────────────────────
    @staticmethod
    def _save_local(html: str) -> Path:
        fname = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.html")
        fname.write_text(html, encoding="utf-8")
        logger.info(f"Report saved locally → {fname.resolve()}")
        return fname


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def run_daily_screen(save_only: bool = False) -> dict:
    """
    Full pipeline:
      DataEngine.screen_universe()
        → NarrativeGenerator.generate()
          → ReportMailer.render() + .send()

    Parameters
    ----------
    save_only : bool
        If True, render and save HTML but skip SMTP delivery.
    """
    run_date = datetime.now().strftime("%B %d, %Y")
    logger.info(f"{'='*60}")
    logger.info(f"  Daily 52W Screen  |  {run_date}")
    logger.info(f"{'='*60}")

    # ── Build ticker universe ─────────────────────────────────────────────────
    tickers_env = os.getenv("TICKER_UNIVERSE", "")
    if tickers_env:
        # Manual override from .env
        tickers = [t.strip().upper() for t in tickers_env.split(",") if t.strip()]
        logger.info(f"Universe: {len(tickers)} tickers (from TICKER_UNIVERSE env)")
    else:
        # Try live NSE fetch first; fall back to hardcoded Nifty 500
        tickers = fetch_nifty500_live()
        if not tickers:
            tickers = NIFTY_500_TICKERS
            logger.info(f"Universe: {len(tickers)} tickers (hardcoded Nifty 500 fallback)")
        else:
            logger.info(f"Universe: {len(tickers)} tickers (live NSE Nifty 500)")

    # ── Screener configuration ────────────────────────────────────────────────
    config = ScreenerConfig(
        tickers               = tickers,
        high_low_window       = int(os.getenv("HIGH_LOW_WINDOW",        "252")),
        breakout_threshold    = float(os.getenv("BREAKOUT_THRESHOLD",   "0.025")),
        volume_window         = int(os.getenv("VOLUME_WINDOW",          "20")),
        volume_surge_threshold= float(os.getenv("VOLUME_SURGE_THRESH",  "1.2")),
        dcf_stage1_years      = int(os.getenv("DCF_STAGE1_YEARS",       "5")),
        dcf_stage2_years      = int(os.getenv("DCF_STAGE2_YEARS",       "5")),
        dcf_terminal_growth   = float(os.getenv("DCF_TERMINAL_GROWTH",  "0.03")),
        dcf_discount_rate     = float(os.getenv("DCF_DISCOUNT_RATE",    "0.10")),
        rate_limit_sleep      = float(os.getenv("RATE_LIMIT_SLEEP",     "0.35")),
        rate_limit_batch_size = int(os.getenv("RATE_LIMIT_BATCH",       "50")),
    )

    # ── Run screen ────────────────────────────────────────────────────────────
    engine  = DataEngine(config)
    results = engine.screen_universe()

    highs_raw  = results["highs"]
    lows_raw   = results["lows"]
    errors     = results["errors"]

    highs_data = engine.format_for_display(highs_raw)
    lows_data  = engine.format_for_display(lows_raw)

    logger.info(
        f"Flagged stocks — Highs: {len(highs_data)} | Lows: {len(lows_data)} | "
        f"Errors: {len(errors)}"
    )

    # ── AI Narrative + Per-stock "Why" ───────────────────────────────────────
    narrator  = NarrativeGenerator()
    narrative = narrator.generate(highs_data, lows_data)

    # Generate per-stock reasons and merge into each stock dict
    all_flagged = highs_data + lows_data
    reasons     = narrator.generate_stock_reasons(all_flagged)
    for stock in highs_data:
        stock["why_text"] = reasons.get(stock["ticker"], "")
    for stock in lows_data:
        stock["why_text"] = reasons.get(stock["ticker"], "")

    # ── Render report ─────────────────────────────────────────────────────────
    mailer     = ReportMailer()
    html_report = mailer.render(highs_data, lows_data, narrative, errors, run_date)

    # Always persist a local copy
    local_path = ReportMailer._save_local(html_report)

    # ── Email delivery ────────────────────────────────────────────────────────
    if not save_only:
        try:
            mailer.send(html_report, run_date)
        except Exception as exc:
            logger.error(f"Email delivery failed (report still saved locally): {exc}")
    else:
        logger.info("--save-only flag set; skipping email delivery.")

    logger.info(f"{'='*60}")
    logger.info(f"  Run complete  |  local copy: {local_path.name}")
    logger.info(f"{'='*60}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily 52-Week High/Low Stock Screener with AI Narrative"
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Render report and save HTML locally; skip email delivery.",
    )
    args = parser.parse_args()

    try:
        run_daily_screen(save_only=args.save_only)
    except KeyboardInterrupt:
        logger.info("Run interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.critical(f"Fatal error in main run: {exc}", exc_info=True)
        sys.exit(1)
