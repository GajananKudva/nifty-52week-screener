"""
engine.py — Data & Valuation Engine
====================================
Handles all market data fetching, technical screening, fundamental overlays,
and 2-stage DCF valuation. Designed for daily EOD batch runs.

Author  : Senior Quant Dev
Version : 1.0.0
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("screener.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("engine")


# ── Configuration Dataclass ───────────────────────────────────────────────────
@dataclass
class ScreenerConfig:
    """All tunable parameters for the screening engine."""

    tickers: list

    # Technical
    high_low_window: int = 252
    breakout_threshold: float = 0.0   # 0 = only stocks that actually HIT the 52W extreme today
    volume_window: int = 20
    volume_surge_threshold: float = 1.2

    # DCF
    dcf_stage1_years: int = 5
    dcf_stage2_years: int = 5
    dcf_terminal_growth: float = 0.03
    dcf_discount_rate: float = 0.10

    # Operational
    rate_limit_sleep: float = 0.35
    rate_limit_batch_size: int = 50
    history_buffer_days: int = 420


# ── Data Engine ───────────────────────────────────────────────────────────────
class DataEngine:
    def __init__(self, config: ScreenerConfig):
        self.config = config
        self.errors: list[dict] = []

    def fetch_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        try:
            end_date   = datetime.now() + timedelta(days=1)
            start_date = datetime.now() - timedelta(days=self.config.history_buffer_days)
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True, actions=False,
            )
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).strip().title() for c in df.columns]
            required = {"Open", "High", "Low", "Close", "Volume"}
            if required - set(df.columns):
                return None
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df = df[df["Close"].notna() & (df["Close"] > 0)]
            if len(df) < self.config.volume_window + 5:
                return None
            return df
        except Exception as exc:
            self._log_error(ticker, "price_history", exc)
            return None

    def calculate_technical_signals(self, df: pd.DataFrame, ticker: str) -> Optional[dict]:
        try:
            n = self.config.high_low_window
            if len(df) < n:
                return None

            window_df = df.tail(n)
            try:
                live = float(stock.fast_info.last_price or 0)
            except Exception:
                live = 0.0
            last_close    = float(df["Close"].iloc[-1])
            current_price = live if live > 0 else last_close   # LTP — shown in UI, not used for signal
            today_high    = float(df["High"].iloc[-1])          # session high — used for BREAKOUT signal
            today_low     = float(df["Low"].iloc[-1])           # session low  — used for BREAKDOWN signal
            current_vol   = float(df["Volume"].iloc[-1])

            # 52W range always includes today's extremes
            week52_high = max(float(window_df["High"].max()), today_high)
            week52_low  = min(float(window_df["Low"].min()),  today_low)

            baseline_vol_series = df["Volume"].iloc[-(self.config.volume_window + 1):-1]
            avg_vol_20d = float(baseline_vol_series.mean())
            if avg_vol_20d <= 0 or np.isnan(avg_vol_20d):
                return None

            volume_surge = current_vol / avg_vol_20d

            # ── Signal detection: day HIGH/LOW vs 52W extreme ─────────────────
            # "Did the stock TOUCH the 52W high/low zone during today's session?"
            # This catches stocks that hit the extreme intraday even if LTP has pulled back.
            pct_from_high = (week52_high - today_high) / week52_high   # day HIGH vs 52W high
            pct_from_low  = (today_low   - week52_low) / week52_low    # day LOW  vs 52W low

            near_high     = pct_from_high <= self.config.breakout_threshold
            near_low      = pct_from_low  <= self.config.breakout_threshold
            vol_confirmed = volume_surge  >= self.config.volume_surge_threshold

            if near_high:
                signal = "BREAKOUT_HIGH"
            elif near_low:
                signal = "BREAKDOWN_LOW"
            else:
                signal = None

            def trailing_return(n_days: int) -> Optional[float]:
                if len(df) > n_days:
                    base = float(df["Close"].iloc[-n_days - 1])
                    return round((current_price / base - 1) * 100, 2) if base > 0 else None
                return None

            # New vs continuation: days since the stock last sat at its prior
            # 52-week-high level (the prior peak, excluding today's bar).
            days_since_prior_high = None
            high_type = None
            if signal == "BREAKOUT_HIGH":
                try:
                    prior_high = window_df["High"].iloc[:-1]
                    if len(prior_high) > 0:
                        prior_peak_dt = prior_high.idxmax()
                        days_since_prior_high = int((df.index[-1] - prior_peak_dt).days)
                        high_type = "NEW" if days_since_prior_high >= 40 else "CONTINUATION"
                except Exception:
                    days_since_prior_high = None
                    high_type = None

            return {
                "ticker":         ticker,
                "current_price":  round(current_price, 2),   # LTP — display only
                "day_high":       round(today_high, 2),       # session high — used for signal
                "day_low":        round(today_low,  2),       # session low  — used for signal
                "week_52_high":   round(week52_high, 2),
                "week_52_low":    round(week52_low,  2),
                "pct_from_high":  round(pct_from_high * 100, 2),  # day HIGH vs 52W high
                "pct_from_low":   round(pct_from_low  * 100, 2),  # day LOW  vs 52W low
                "volume_surge":   round(volume_surge,  2),
                "vol_confirmed":  vol_confirmed,
                "avg_vol_20d":    int(avg_vol_20d),
                "current_vol":    int(current_vol),
                "signal":         signal,
                "days_since_prior_high": days_since_prior_high,
                "high_type":      high_type,
                "ret_1m":         trailing_return(21),
                "ret_3m":         trailing_return(63),
                "ret_6m":         trailing_return(126),
                "ret_1y":         trailing_return(252),
            }
        except Exception as exc:
            self._log_error(ticker, "technical_signals", exc)
            return None

    def fetch_fundamentals(self, ticker: str) -> dict:
        defaults = {
            "company_name": ticker, "sector": None, "industry": None,
            "pe_ratio": None, "forward_pe": None, "fcf": None,
            "revenue": None, "market_cap": None, "beta": None,
            "shares_outstanding": None, "dividend_yield": None,
            "profit_margin": None, "debt_to_equity": None, "return_on_equity": None,
        }
        try:
            # Retry once on rate-limit with a short sleep
            for _attempt in range(2):
                try:
                    info = yf.Ticker(ticker).info
                    break
                except Exception as _e:
                    if "429" in str(_e) or "rate" in str(_e).lower():
                        time.sleep(3)
                    else:
                        raise
            else:
                return defaults
            if not info:
                return defaults
            defaults["company_name"] = info.get("longName") or info.get("shortName") or ticker
            defaults["sector"]   = info.get("sector")
            defaults["industry"] = info.get("industry")
            numeric_map = {
                "pe_ratio": "trailingPE", "forward_pe": "forwardPE",
                "fcf": "freeCashflow", "revenue": "totalRevenue",
                "market_cap": "marketCap", "beta": "beta",
                "shares_outstanding": "sharesOutstanding",
                "dividend_yield": "dividendYield", "profit_margin": "profitMargins",
                "debt_to_equity": "debtToEquity", "return_on_equity": "returnOnEquity",
            }
            for our_key, yf_key in numeric_map.items():
                raw = info.get(yf_key)
                if raw is not None:
                    try:
                        val = float(raw)
                        if our_key in ("pe_ratio", "forward_pe", "beta", "debt_to_equity"):
                            val = round(val, 2)
                        defaults[our_key] = val
                    except (ValueError, TypeError):
                        pass
        except Exception as exc:
            self._log_error(ticker, "fundamentals", exc)
        return defaults

    def calculate_dcf(self, ticker: str, current_price: float, fundamentals: dict) -> dict:
        result = {"intrinsic_value": None, "upside_downside_pct": None, "dcf_stage1_growth": None}
        try:
            fcf    = fundamentals.get("fcf")
            shares = fundamentals.get("shares_outstanding")
            if not fcf or not shares:
                return result
            fcf = float(fcf); shares = float(shares)
            if fcf <= 0 or shares <= 0:
                return result
            mktcap = fundamentals.get("market_cap") or 0.0
            g1 = 0.08 if mktcap > 100e9 else (0.12 if mktcap > 10e9 else 0.15)
            result["dcf_stage1_growth"] = round(g1 * 100, 1)
            r = self.config.dcf_discount_rate
            gt = self.config.dcf_terminal_growth
            n1 = self.config.dcf_stage1_years
            n2 = self.config.dcf_stage2_years
            g2 = (g1 + gt) / 2
            pv_s1 = sum(fcf * ((1+g1)**yr) / ((1+r)**yr) for yr in range(1, n1+1))
            fcf_end_s1 = fcf * ((1+g1)**n1)
            pv_s2 = sum(fcf_end_s1 * ((1+g2)**yr) / ((1+r)**(n1+yr)) for yr in range(1, n2+1))
            fcf_end_s2 = fcf_end_s1 * ((1+g2)**n2)
            tv    = fcf_end_s2 * (1+gt) / (r-gt)
            pv_tv = tv / ((1+r)**(n1+n2))
            intrinsic = (pv_s1 + pv_s2 + pv_tv) / shares
            upside    = (intrinsic - current_price) / current_price * 100
            result["intrinsic_value"]     = round(float(intrinsic), 2)
            result["upside_downside_pct"] = round(float(upside), 2)
        except Exception as exc:
            self._log_error(ticker, "dcf", exc)
        return result

    def screen_universe(self) -> dict:
        """
        Fallback path when NSE live API is unavailable (e.g. geo-blocked on cloud).
        Uses yf.download() in chunks of 100 — single HTTP call per chunk instead of
        one call per ticker. 500 stocks: ~4 chunks × ~5s = ~20s total vs ~8 minutes serial.
        """
        # Strip blanks and filter placeholder/dummy tickers (e.g. DUMMYVEDL1.NS
        # created by NSE for Vedanta demerger entities — no real price data exists).
        tickers = [
            t.strip().upper() for t in self.config.tickers
            if t.strip() and not t.strip().upper().startswith("DUMMY")
        ]
        logger.info(f"Starting batch universe screen | {len(tickers)} tickers")

        # ── Step 1: Batch download history for all tickers ────────────────────
        CHUNK     = 100
        end_date  = datetime.now() + timedelta(days=1)
        start_date = datetime.now() - timedelta(days=self.config.history_buffer_days)
        hist_map: dict[str, pd.DataFrame] = {}

        for chunk_start in range(0, len(tickers), CHUNK):
            chunk = tickers[chunk_start: chunk_start + CHUNK]
            chunk_num = chunk_start // CHUNK + 1
            total_chunks = (len(tickers) + CHUNK - 1) // CHUNK
            logger.info(f"Batch download chunk {chunk_num}/{total_chunks} ({len(chunk)} tickers)")
            try:
                raw = yf.download(
                    chunk,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if len(chunk) == 1:
                    t    = chunk[0]
                    df_c = raw.copy()
                    if isinstance(df_c.columns, pd.MultiIndex):
                        df_c.columns = df_c.columns.get_level_values(0)
                    df_c.columns = [str(c).strip().title() for c in df_c.columns]
                    df_c.index   = pd.to_datetime(df_c.index)
                    df_c.sort_index(inplace=True)
                    df_c = df_c[df_c["Close"].notna() & (df_c["Close"] > 0)]
                    if not df_c.empty:
                        hist_map[t] = df_c
                else:
                    for t in chunk:
                        try:
                            df_c = raw[t].copy()
                            if isinstance(df_c.columns, pd.MultiIndex):
                                df_c.columns = df_c.columns.get_level_values(0)
                            df_c.columns = [str(c).strip().title() for c in df_c.columns]
                            df_c.index   = pd.to_datetime(df_c.index)
                            df_c.sort_index(inplace=True)
                            df_c = df_c[df_c["Close"].notna() & (df_c["Close"] > 0)]
                            if not df_c.empty:
                                hist_map[t] = df_c
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning(f"Chunk {chunk_num} batch download failed: {exc} — skipping chunk")

        logger.info(f"Batch download done: {len(hist_map)}/{len(tickers)} tickers fetched")

        # ── Step 2: Compute signals from downloaded data ──────────────────────
        all_records: list[dict] = []
        for ticker in tickers:
            df = hist_map.get(ticker)
            if df is None:
                continue
            signals = self.calculate_technical_signals(df, ticker)
            if signals is None:
                continue

            if signals["signal"] is not None:
                fundamentals   = self.fetch_fundamentals(ticker)
                news_headlines = self.fetch_news(ticker)
                dcf = self.calculate_dcf(ticker, signals["current_price"], fundamentals)
                record = {**signals, **fundamentals, **dcf, "news_headlines": news_headlines}
            else:
                record = {
                    **signals,
                    "company_name": ticker, "sector": None, "industry": None,
                    "pe_ratio": None, "forward_pe": None, "fcf": None,
                    "market_cap": None, "beta": None,
                    "intrinsic_value": None, "upside_downside_pct": None,
                    "dcf_stage1_growth": None, "news_headlines": [],
                }
            all_records.append(record)

        if not all_records:
            return {"highs": pd.DataFrame(), "lows": pd.DataFrame(),
                    "all_flagged": pd.DataFrame(), "errors": self.errors}

        all_df   = pd.DataFrame(all_records)
        flagged  = all_df[all_df["signal"].notna()].copy()
        highs_df = flagged[flagged["signal"] == "BREAKOUT_HIGH"].copy()
        lows_df  = flagged[flagged["signal"] == "BREAKDOWN_LOW"].copy()
        _sort_col = "market_cap"
        for _sdf in (highs_df, lows_df):
            if not _sdf.empty and _sort_col in _sdf.columns:
                _sdf["_mc"] = pd.to_numeric(_sdf[_sort_col], errors="coerce")
                _sdf.sort_values("_mc", ascending=False, na_position="last", inplace=True)
                _sdf.drop(columns=["_mc"], inplace=True)
        logger.info(f"Screen complete | {len(highs_df)} BREAKOUT_HIGH | {len(lows_df)} BREAKDOWN_LOW")
        return {"highs": highs_df, "lows": lows_df, "all_flagged": flagged, "errors": self.errors}

    def format_for_display(self, df: pd.DataFrame) -> list[dict]:
        if df is None or df.empty:
            return []
        ordered_cols = [
            "ticker", "company_name", "sector", "industry",
            "current_price", "week_52_high", "week_52_low",
            "pct_from_high", "pct_from_low",
            "volume_surge", "vol_confirmed", "avg_vol_20d", "current_vol",
            "pe_ratio", "forward_pe", "beta",
            "intrinsic_value", "upside_downside_pct", "dcf_stage1_growth",
            "ret_1m", "ret_3m", "ret_6m", "ret_1y",
            "market_cap", "days_since_prior_high", "high_type",
            "news_headlines", "why_text", "signal",
        ]
        available = [c for c in ordered_cols if c in df.columns]
        out = df[available].copy()
        numeric_cols = [
            "current_price", "week_52_high", "week_52_low",
            "pct_from_high", "pct_from_low", "volume_surge",
            "pe_ratio", "forward_pe", "beta",
            "intrinsic_value", "upside_downside_pct",
            "ret_1m", "ret_3m", "ret_6m", "ret_1y",
            "market_cap",
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
        out = out.where(pd.notnull(out), None)
        return out.to_dict(orient="records")

    def fetch_news(self, ticker: str, max_items: int = 5) -> list[str]:
        try:
            raw_news = yf.Ticker(ticker).news or []
            headlines = []
            for item in raw_news[:max_items]:
                if isinstance(item, dict) and "content" in item:
                    item = item["content"]
                title     = item.get("title", "").strip()
                publisher = item.get("publisher", "").strip()
                if title:
                    headlines.append(f"{title} [{publisher}]" if publisher else title)
            return headlines
        except Exception:
            return []

    def screen_from_nse_live(self, nse_stocks: list[dict]) -> dict:
        """
        Hybrid NSE-first screener.
        Phase 1: filter by proximity using TODAY'S SESSION HIGH/LOW vs 52W extreme.
                 Catches stocks that touched the zone intraday even if LTP has pulled back.
                 Falls back to LTP if day_high/day_low not in NSE payload.
        Phase 2: yfinance 30-day history for volume surge + confirm day high/low.
                 Signal is set on PROXIMITY alone; vol_confirmed is metadata.
        """
        threshold_pct = self.config.breakout_threshold * 100
        sleep_short   = self.config.rate_limit_sleep
        logger.info(f"NSE live screen | {len(nse_stocks)} stocks | threshold={threshold_pct:.1f}%")

        def _f(v) -> Optional[float]:
            """Safe float conversion."""
            try:
                return float(str(v).replace(",", "")) if v is not None else None
            except (TypeError, ValueError):
                return None

        # Phase 1
        candidates: list[dict] = []
        for s in nse_stocks:
            lp_f = _f(s.get("last_price"))    # LTP  — fallback only
            dh_f = _f(s.get("day_high")) or lp_f   # session HIGH — primary for BREAKOUT
            dl_f = _f(s.get("day_low"))  or lp_f   # session LOW  — primary for BREAKDOWN
            yh_f = _f(s.get("year_high"))
            yl_f = _f(s.get("year_low"))

            # Always recompute using day high/low (overrides NSE's LTP-based nearWKH/nearWKL)
            if dh_f and yh_f and yh_f > 0:
                wkh = (yh_f - dh_f) / yh_f * 100   # how far session HIGH is from 52W high
                s["near_wkh"] = wkh
            else:
                wkh = s.get("near_wkh")   # use NSE pre-computed as last resort

            if dl_f and yl_f and yl_f > 0:
                wkl = (dl_f - yl_f) / yl_f * 100   # how far session LOW is from 52W low
                s["near_wkl"] = wkl
            else:
                wkl = s.get("near_wkl")

            is_near_high = wkh is not None and wkh <= threshold_pct
            is_near_low  = wkl is not None and wkl <= threshold_pct
            if is_near_high or is_near_low:
                s["_candidate_high"] = is_near_high
                s["_candidate_low"]  = is_near_low
                candidates.append(s)

        logger.info(f"Phase 1 done: {len(candidates)} candidates within {threshold_pct:.1f}%")

        # Phase 2 — yfinance only needed when volume surge filter is ON
        needs_yf     = self.config.volume_surge_threshold > 0
        cand_tickers = [s["ticker"] for s in candidates]
        batch_df: dict[str, pd.DataFrame] = {}

        if needs_yf and cand_tickers:
            end_date   = datetime.now() + timedelta(days=1)
            start_date = datetime.now() - timedelta(days=45)
            try:
                raw = yf.download(
                    cand_tickers,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                )
                if len(cand_tickers) == 1:
                    t   = cand_tickers[0]
                    df_c = raw.copy()
                    if isinstance(df_c.columns, pd.MultiIndex):
                        df_c.columns = df_c.columns.get_level_values(0)
                    df_c.columns = [str(c).strip().title() for c in df_c.columns]
                    df_c = df_c[df_c["Close"].notna() & (df_c["Close"] > 0)]
                    if not df_c.empty:
                        batch_df[t] = df_c
                else:
                    for t in cand_tickers:
                        try:
                            df_c = raw[t].copy()
                            if isinstance(df_c.columns, pd.MultiIndex):
                                df_c.columns = df_c.columns.get_level_values(0)
                            df_c.columns = [str(c).strip().title() for c in df_c.columns]
                            df_c = df_c[df_c["Close"].notna() & (df_c["Close"] > 0)]
                            if not df_c.empty:
                                batch_df[t] = df_c
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning(f"Batch yfinance failed ({exc}), falling back to serial")
                for t in cand_tickers:
                    try:
                        df_c = yf.Ticker(t).history(
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            auto_adjust=True, actions=False,
                        )
                        if isinstance(df_c.columns, pd.MultiIndex):
                            df_c.columns = df_c.columns.get_level_values(0)
                        df_c.columns = [str(c).strip().title() for c in df_c.columns]
                        df_c = df_c[df_c["Close"].notna() & (df_c["Close"] > 0)]
                        if not df_c.empty:
                            batch_df[t] = df_c
                    except Exception as e2:
                        self._log_error(t, "nse_short_history_serial", e2)
            logger.info(f"Phase 2 batch fetch: {len(batch_df)}/{len(cand_tickers)} tickers OK")
        else:
            logger.info(f"Phase 2: yfinance skipped (vol filter off) — {len(candidates)} candidates from NSE only")

        all_records: list[dict] = []
        for s in candidates:
            ticker   = s["ticker"]
            df_short = batch_df.get(ticker)  # None when vol filter off

            # Volume stats — from yfinance when available, else N/A
            if df_short is not None and len(df_short) >= self.config.volume_window + 2:
                current_vol   = float(df_short["Volume"].iloc[-1])
                baseline_vols = df_short["Volume"].iloc[-(self.config.volume_window + 1):-1]
                avg_vol_20d   = float(baseline_vols.mean())
                if avg_vol_20d <= 0 or np.isnan(avg_vol_20d):
                    avg_vol_20d = 0.0
                volume_surge  = (current_vol / avg_vol_20d) if avg_vol_20d > 0 else None
            else:
                # Vol filter is off — NSE volume field used for display only
                current_vol  = float(_f(s.get("volume")) or 0)
                avg_vol_20d  = 0.0
                volume_surge = None

            near_wkh      = s.get("near_wkh", 999.0)
            near_wkl      = s.get("near_wkl", 999.0)
            vol_confirmed = (
                volume_surge is not None and
                volume_surge >= self.config.volume_surge_threshold
            ) if needs_yf else False

            # Signal on proximity only
            if s["_candidate_high"]:
                signal = "BREAKOUT_HIGH"
            elif s["_candidate_low"]:
                signal = "BREAKDOWN_LOW"
            else:
                signal = None

            # Price — LTP from NSE (yfinance close as fallback if available)
            try:
                current_price = float(s["last_price"] or 0)
                if current_price <= 0 and df_short is not None:
                    current_price = float(df_short["Close"].iloc[-1])
            except (TypeError, ValueError):
                current_price = float(df_short["Close"].iloc[-1]) if df_short is not None else 0.0

            try:
                year_high = float(s["year_high"])
                year_low  = float(s["year_low"])
            except (TypeError, ValueError):
                year_high = float(df_short["High"].max()) if df_short is not None else 0.0
                year_low  = float(df_short["Low"].min())  if df_short is not None else 0.0

            pct_from_high = (near_wkh or 0.0) / 100.0
            pct_from_low  = (near_wkl or 0.0) / 100.0

            def trailing_ret(n_days: int) -> Optional[float]:
                if df_short is not None and len(df_short) > n_days:
                    base = float(df_short["Close"].iloc[-n_days - 1])
                    return round((current_price / base - 1) * 100, 2) if base > 0 else None
                return None

            record = {
                "ticker":        ticker,
                "company_name":  s.get("company_name") or ticker,
                "sector":        s.get("sector") or None,
                "industry":      s.get("sector") or None,
                "current_price": round(current_price, 2),
                "day_high":      round(float(s.get("day_high") or current_price), 2),
                "day_low":       round(float(s.get("day_low")  or current_price), 2),
                "week_52_high":  round(year_high, 2),
                "week_52_low":   round(year_low, 2),
                "pct_from_high": round(pct_from_high * 100, 2),
                "pct_from_low":  round(pct_from_low  * 100, 2),
                "volume_surge":  round(volume_surge, 2),
                "vol_confirmed": vol_confirmed,
                "avg_vol_20d":   int(avg_vol_20d),
                "current_vol":   int(current_vol),
                "signal":        signal,
                "ret_1m":        trailing_ret(21),
                "ret_3m":        trailing_ret(63) if len(df_short) > 63 else None,
                "ret_6m":        trailing_ret(126) if len(df_short) > 126 else None,
            }

            if signal is not None:
                time.sleep(sleep_short)
                fundamentals   = self.fetch_fundamentals(ticker)
                news_headlines = self.fetch_news(ticker)
                dcf            = self.calculate_dcf(ticker, current_price, fundamentals)
                if fundamentals.get("company_name") == ticker:
                    fundamentals["company_name"] = s.get("company_name") or ticker
                if not fundamentals.get("sector") and s.get("sector"):
                    fundamentals["sector"]   = s["sector"]
                    fundamentals["industry"] = s["sector"]
                record.update({**fundamentals, **dcf, "news_headlines": news_headlines})
                logger.info(
                    f"  signal={signal} | VolSurge={volume_surge:.2f}x confirmed={vol_confirmed}"
                    f" | nearWKH={near_wkh:.2f}%"
                )
            else:
                record.update({
                    "pe_ratio": None, "forward_pe": None, "fcf": None,
                    "market_cap": None, "beta": None, "intrinsic_value": None,
                    "upside_downside_pct": None, "dcf_stage1_growth": None,
                    "news_headlines": [],
                })

            all_records.append(record)

        if not all_records:
            return {"highs": pd.DataFrame(), "lows": pd.DataFrame(),
                    "all_flagged": pd.DataFrame(), "errors": self.errors}

        all_df  = pd.DataFrame(all_records)
        flagged = all_df[all_df["signal"].notna()].copy()
        highs_df = flagged[flagged["signal"] == "BREAKOUT_HIGH"].copy()
        lows_df  = flagged[flagged["signal"] == "BREAKDOWN_LOW"].copy()
        _sort_col = "market_cap"
        for _sdf in (highs_df, lows_df):
            if not _sdf.empty and _sort_col in _sdf.columns:
                _sdf["_mc"] = pd.to_numeric(_sdf[_sort_col], errors="coerce")
                _sdf.sort_values("_mc", ascending=False, na_position="last", inplace=True)
                _sdf.drop(columns=["_mc"], inplace=True)
        logger.info(
            f"NSE live screen complete | {len(highs_df)} BREAKOUT_HIGH | {len(lows_df)} BREAKDOWN_LOW"
        )
        return {"highs": highs_df, "lows": lows_df, "all_flagged": flagged, "errors": self.errors}

    def _log_error(self, ticker: str, step: str, exc: Exception) -> None:
        self.errors.append({
            "ticker": ticker,
            "step":   step,
            "error":  str(exc),
            "time":   datetime.now().isoformat(timespec="seconds"),
        })
