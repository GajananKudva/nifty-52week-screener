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
    high_low_window: int = 252          # Trading-day lookback for 52W high/low
    breakout_threshold: float = 0.025   # 2.5% proximity band
    volume_window: int = 20             # Days for average-volume baseline
    volume_surge_threshold: float = 1.2 # Minimum surge multiplier

    # DCF
    dcf_stage1_years: int = 5
    dcf_stage2_years: int = 5
    dcf_terminal_growth: float = 0.03   # 3% perpetuity growth
    dcf_discount_rate: float = 0.10     # 10% WACC

    # Operational
    rate_limit_sleep: float = 0.35      # Seconds between individual ticker calls
    rate_limit_batch_size: int = 50     # Longer pause after this many tickers
    history_buffer_days: int = 420      # Calendar days fetched (> 252 trading days)


# ── Data Engine ───────────────────────────────────────────────────────────────
class DataEngine:
    """
    Core engine that orchestrates:
      1. Price history fetch + validation
      2. Technical signal computation (52W high/low + volume)
      3. Fundamental data overlay (P/E, FCF, market cap, sector)
      4. 2-Stage DCF intrinsic value + upside/downside %
    """

    def __init__(self, config: ScreenerConfig):
        self.config = config
        self.errors: list[dict] = []

    # ── 1. Price History ──────────────────────────────────────────────────────
    def fetch_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download OHLCV history via yfinance.

        Resilience:
          - Wraps everything in try/except; logs and returns None on failure.
          - Flattens any MultiIndex columns that yfinance can produce in batch
            download mode, preventing 'Ambiguous Truth Value' errors.
          - Validates required columns before returning.
        """
        try:
            end_date   = datetime.now() + timedelta(days=1)   # +1 so yfinance includes today
            start_date = datetime.now() - timedelta(days=self.config.history_buffer_days)

            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                actions=False,
            )

            if df is None or df.empty:
                logger.warning(f"{ticker}: Empty price history — skipping.")
                return None

            # ── Flatten MultiIndex columns (critical for yfinance ≥ 0.2) ─────
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalise column names (yfinance casing can vary)
            df.columns = [str(c).strip().title() for c in df.columns]

            required = {"Open", "High", "Low", "Close", "Volume"}
            missing = required - set(df.columns)
            if missing:
                logger.warning(f"{ticker}: Missing columns {missing} — skipping.")
                return None

            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Drop rows where Close is NaN or zero (data quality guard)
            df = df[df["Close"].notna() & (df["Close"] > 0)]

            if len(df) < self.config.volume_window + 5:
                logger.warning(f"{ticker}: Insufficient rows after cleaning ({len(df)}) — skipping.")
                return None

            return df

        except Exception as exc:
            logger.error(f"{ticker}: fetch_price_history failed — {exc}")
            self._log_error(ticker, "price_history", exc)
            return None

    # ── 2. Technical Signals ──────────────────────────────────────────────────
    def calculate_technical_signals(
        self, df: pd.DataFrame, ticker: str
    ) -> Optional[dict]:
        """
        Compute 52-week high/low proximity and volume-surge signal.

        Returns a dict, or None if data is inadequate.

        Signal logic:
          - BREAKOUT_HIGH : current price within breakout_threshold of 252-day high
                            AND volume_surge >= volume_surge_threshold
          - BREAKDOWN_LOW  : current price within breakout_threshold of 252-day low
                            AND volume_surge >= volume_surge_threshold
          - None           : no significant signal today
        """
        try:
            n = self.config.high_low_window

            if len(df) < n:
                logger.info(
                    f"{ticker}: Only {len(df)} rows; need {n} for 52W window — skipping."
                )
                return None

            window_df = df.tail(n)

            # Use live price from fast_info — more current than last Close row
            try:
                live = float(stock.fast_info.last_price or 0)
            except Exception:
                live = 0.0
            last_close    = float(df["Close"].iloc[-1])
            current_price = live if live > 0 else last_close

            # Use today's intraday high if available (last row = today's partial candle)
            today_high    = float(df["High"].iloc[-1])
            current_vol   = float(df["Volume"].iloc[-1])

            week52_high = max(float(window_df["High"].max()), today_high)
            week52_low  = float(window_df["Low"].min())

            # ── Volume Surge Multiplier ───────────────────────────────────────
            # Exclude today from the baseline to avoid look-ahead bias
            baseline_vol_series = df["Volume"].iloc[-(self.config.volume_window + 1):-1]
            avg_vol_20d = float(baseline_vol_series.mean())

            if avg_vol_20d <= 0 or np.isnan(avg_vol_20d):
                logger.warning(f"{ticker}: Zero/NaN 20-day avg volume — skipping.")
                return None

            volume_surge = current_vol / avg_vol_20d

            # ── Proximity calculations ────────────────────────────────────────
            pct_from_high = (week52_high - current_price) / week52_high  # 0 = at high
            pct_from_low  = (current_price - week52_low)  / week52_low   # 0 = at low

            near_high = pct_from_high <= self.config.breakout_threshold
            near_low  = pct_from_low  <= self.config.breakout_threshold
            vol_ok    = volume_surge  >= self.config.volume_surge_threshold

            if near_high and vol_ok:
                signal = "BREAKOUT_HIGH"
            elif near_low and vol_ok:
                signal = "BREAKDOWN_LOW"
            else:
                signal = None

            # ── Trailing return helpers ───────────────────────────────────────
            def trailing_return(n_days: int) -> Optional[float]:
                if len(df) > n_days:
                    base = float(df["Close"].iloc[-n_days - 1])
                    return round((current_price / base - 1) * 100, 2) if base > 0 else None
                return None

            return {
                "ticker":         ticker,
                "current_price":  round(current_price, 2),
                "week_52_high":   round(week52_high, 2),
                "week_52_low":    round(week52_low, 2),
                "pct_from_high":  round(pct_from_high * 100, 2),
                "pct_from_low":   round(pct_from_low  * 100, 2),
                "volume_surge":   round(volume_surge,  2),
                "avg_vol_20d":    int(avg_vol_20d),
                "current_vol":    int(current_vol),
                "signal":         signal,
                "ret_1m":         trailing_return(21),
                "ret_3m":         trailing_return(63),
                "ret_6m":         trailing_return(126),
            }

        except Exception as exc:
            logger.error(f"{ticker}: calculate_technical_signals failed — {exc}")
            self._log_error(ticker, "technical_signals", exc)
            return None

    # ── 3. Fundamentals ───────────────────────────────────────────────────────
    def fetch_fundamentals(self, ticker: str) -> dict:
        """
        Pull fundamental metrics from yfinance .info dict.

        All fields default to None; numeric fields are coerced to float
        to prevent type errors downstream. Never raises — returns defaults
        on any failure.
        """
        defaults = {
            "company_name":      ticker,
            "sector":            None,
            "industry":          None,
            "pe_ratio":          None,
            "forward_pe":        None,
            "fcf":               None,
            "revenue":           None,
            "market_cap":        None,
            "beta":              None,
            "shares_outstanding": None,
            "dividend_yield":    None,
            "profit_margin":     None,
            "debt_to_equity":    None,
            "return_on_equity":  None,
        }

        try:
            info = yf.Ticker(ticker).info

            if not info:
                logger.warning(f"{ticker}: Empty .info dict.")
                return defaults

            # String fields
            defaults["company_name"] = (
                info.get("longName") or info.get("shortName") or ticker
            )
            defaults["sector"]   = info.get("sector")
            defaults["industry"] = info.get("industry")

            # Numeric fields — safe coerce
            numeric_map = {
                "pe_ratio":           "trailingPE",
                "forward_pe":         "forwardPE",
                "fcf":                "freeCashflow",
                "revenue":            "totalRevenue",
                "market_cap":         "marketCap",
                "beta":               "beta",
                "shares_outstanding": "sharesOutstanding",
                "dividend_yield":     "dividendYield",
                "profit_margin":      "profitMargins",
                "debt_to_equity":     "debtToEquity",
                "return_on_equity":   "returnOnEquity",
            }
            for our_key, yf_key in numeric_map.items():
                raw = info.get(yf_key)
                if raw is not None:
                    try:
                        val = float(raw)
                        # Round display-facing ratios
                        if our_key in ("pe_ratio", "forward_pe", "beta",
                                       "debt_to_equity"):
                            val = round(val, 2)
                        defaults[our_key] = val
                    except (ValueError, TypeError):
                        pass  # leave as None

        except Exception as exc:
            logger.error(f"{ticker}: fetch_fundamentals failed — {exc}")
            self._log_error(ticker, "fundamentals", exc)

        return defaults

    # ── 4. 2-Stage DCF ───────────────────────────────────────────────────────
    def calculate_dcf(
        self,
        ticker: str,
        current_price: float,
        fundamentals: dict,
    ) -> dict:
        """
        2-Stage Discounted Cash Flow model.

          Stage 1 : Explicit FCF growth for dcf_stage1_years
          Stage 2 : Transition period (avg of Stage-1 rate and terminal rate)
          Terminal : Gordon Growth Model perpetuity

        Growth rate is estimated conservatively by market-cap tier:
          Large-cap  (>$100B)  →  8%
          Mid-cap   ($10–100B) → 12%
          Small-cap  (<$10B)   → 15%

        Returns a dict with intrinsic_value, upside_downside_pct, and
        the assumed Stage-1 growth rate. All None if FCF/shares unavailable.
        """
        result = {
            "intrinsic_value":      None,
            "upside_downside_pct":  None,
            "dcf_stage1_growth":    None,
        }

        try:
            fcf    = fundamentals.get("fcf")
            shares = fundamentals.get("shares_outstanding")

            if not fcf or not shares:
                return result
            fcf    = float(fcf)
            shares = float(shares)

            if fcf <= 0 or shares <= 0:
                logger.info(f"{ticker}: Negative or zero FCF ({fcf:.0f}) — DCF skipped.")
                return result

            mktcap = fundamentals.get("market_cap") or 0.0
            if mktcap > 100e9:
                g1 = 0.08
            elif mktcap > 10e9:
                g1 = 0.12
            else:
                g1 = 0.15

            result["dcf_stage1_growth"] = round(g1 * 100, 1)

            r   = self.config.dcf_discount_rate
            gt  = self.config.dcf_terminal_growth
            n1  = self.config.dcf_stage1_years
            n2  = self.config.dcf_stage2_years
            g2  = (g1 + gt) / 2  # Blended transition growth

            # ── Stage 1: PV of explicit FCF ──────────────────────────────────
            pv_s1 = sum(
                fcf * ((1 + g1) ** yr) / ((1 + r) ** yr)
                for yr in range(1, n1 + 1)
            )

            # ── Stage 2: PV of transition FCF ────────────────────────────────
            fcf_end_s1 = fcf * ((1 + g1) ** n1)
            pv_s2 = sum(
                fcf_end_s1 * ((1 + g2) ** yr) / ((1 + r) ** (n1 + yr))
                for yr in range(1, n2 + 1)
            )

            # ── Terminal Value ────────────────────────────────────────────────
            fcf_end_s2 = fcf_end_s1 * ((1 + g2) ** n2)
            tv         = fcf_end_s2 * (1 + gt) / (r - gt)
            pv_tv      = tv / ((1 + r) ** (n1 + n2))

            # ── Intrinsic value per share ─────────────────────────────────────
            intrinsic = (pv_s1 + pv_s2 + pv_tv) / shares
            upside    = (intrinsic - current_price) / current_price * 100

            result["intrinsic_value"]     = round(float(intrinsic), 2)
            result["upside_downside_pct"] = round(float(upside), 2)

        except Exception as exc:
            logger.error(f"{ticker}: calculate_dcf failed — {exc}")
            self._log_error(ticker, "dcf", exc)

        return result

    # ── 5. Main Screen Loop ───────────────────────────────────────────────────
    def screen_universe(self) -> dict:
        """
        Iterate over the ticker universe and return screening results.

        Returns
        -------
        dict with keys:
          'highs'       – DataFrame of BREAKOUT_HIGH stocks (volume confirmed)
          'lows'        – DataFrame of BREAKDOWN_LOW  stocks (volume confirmed)
          'all_flagged' – Combined flagged DataFrame
          'errors'      – List of error dicts for the run log
        """
        tickers = self.config.tickers
        logger.info(f"Starting universe screen | {len(tickers)} tickers")

        all_records: list[dict] = []
        batch_size  = self.config.rate_limit_batch_size
        sleep_short = self.config.rate_limit_sleep

        for idx, ticker in enumerate(tickers):
            ticker = ticker.strip().upper()
            if not ticker:
                continue

            logger.info(f"[{idx+1:>3}/{len(tickers)}] {ticker}")

            # ── Rate limiting ─────────────────────────────────────────────────
            if idx > 0 and idx % batch_size == 0:
                logger.info(f"  Batch pause after {idx} tickers (2 s) …")
                time.sleep(2.0)
            else:
                time.sleep(sleep_short)

            # ── Step 1 : Price history ────────────────────────────────────────
            df = self.fetch_price_history(ticker)
            if df is None:
                continue

            # ── Step 2 : Technical signals ────────────────────────────────────
            signals = self.calculate_technical_signals(df, ticker)
            if signals is None:
                continue

            # ── Step 3 : Fundamentals + News + DCF (flagged stocks only) ────────
            if signals["signal"] is not None:
                time.sleep(sleep_short)
                fundamentals = self.fetch_fundamentals(ticker)

                # ── Step 4 : News headlines ───────────────────────────────────
                news_headlines = self.fetch_news(ticker)

                # ── Step 5 : DCF ──────────────────────────────────────────────
                dcf = self.calculate_dcf(
                    ticker, signals["current_price"], fundamentals
                )
                record = {**signals, **fundamentals, **dcf,
                          "news_headlines": news_headlines}
            else:
                # Lightweight record — no fundamentals for unflagged stocks
                record = {
                    **signals,
                    "company_name": ticker,
                    "sector": None,
                    "industry": None,
                    "pe_ratio": None,
                    "forward_pe": None,
                    "fcf": None,
                    "market_cap": None,
                    "beta": None,
                    "intrinsic_value": None,
                    "upside_downside_pct": None,
                    "dcf_stage1_growth": None,
                    "news_headlines": [],
                }

            all_records.append(record)

            if signals["signal"]:
                logger.info(
                    f"  ✓ {signals['signal']} | VolSurge={signals['volume_surge']:.2f}x"
                    f" | DCF upside={record.get('upside_downside_pct')}%"
                )

        # ── Assemble DataFrames ───────────────────────────────────────────────
        if not all_records:
            logger.warning("No records produced from screen run.")
            return {
                "highs":       pd.DataFrame(),
                "lows":        pd.DataFrame(),
                "all_flagged": pd.DataFrame(),
                "errors":      self.errors,
            }

        all_df   = pd.DataFrame(all_records)
        flagged  = all_df[all_df["signal"].notna()].copy()

        highs_df = flagged[flagged["signal"] == "BREAKOUT_HIGH"].copy()
        lows_df  = flagged[flagged["signal"] == "BREAKDOWN_LOW"].copy()

        # Sort highs: best DCF upside first; lows: biggest discount first
        _sort_col = "upside_downside_pct"
        if not highs_df.empty and _sort_col in highs_df.columns:
            highs_df.sort_values(_sort_col, ascending=False, inplace=True)
        if not lows_df.empty and _sort_col in lows_df.columns:
            lows_df.sort_values(_sort_col, ascending=False, inplace=True)

        logger.info(
            f"Screen complete | "
            f"{len(highs_df)} BREAKOUT_HIGH | "
            f"{len(lows_df)} BREAKDOWN_LOW | "
            f"{len(self.errors)} errors"
        )

        return {
            "highs":       highs_df,
            "lows":        lows_df,
            "all_flagged": flagged,
            "errors":      self.errors,
        }

    # ── Display Formatter ─────────────────────────────────────────────────────
    def format_for_display(self, df: pd.DataFrame) -> list[dict]:
        """
        Prepare a DataFrame for Jinja2 template rendering.

        - Selects and orders display columns
        - Rounds all numeric fields to 2 d.p.
        - Converts NaN → None so Jinja2 can use the `if` guard cleanly
        - Returns a list of plain dicts
        """
        if df is None or df.empty:
            return []

        ordered_cols = [
            "ticker", "company_name", "sector", "industry",
            "current_price", "week_52_high", "week_52_low",
            "pct_from_high", "pct_from_low",
            "volume_surge", "avg_vol_20d", "current_vol",
            "pe_ratio", "forward_pe", "beta",
            "intrinsic_value", "upside_downside_pct", "dcf_stage1_growth",
            "ret_1m", "ret_3m", "ret_6m",
            "news_headlines", "why_text",
            "signal",
        ]
        available = [c for c in ordered_cols if c in df.columns]
        out = df[available].copy()

        numeric_cols = [
            "current_price", "week_52_high", "week_52_low",
            "pct_from_high", "pct_from_low", "volume_surge",
            "pe_ratio", "forward_pe", "beta",
            "intrinsic_value", "upside_downside_pct",
            "ret_1m", "ret_3m", "ret_6m",
        ]
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

        # NaN → None for clean Jinja2 handling
        out = out.where(pd.notnull(out), None)

        return out.to_dict(orient="records")

    # ── 6. News Headlines ─────────────────────────────────────────────────────
    def fetch_news(self, ticker: str, max_items: int = 5) -> list[str]:
        """
        Fetch the most recent news headlines for a ticker via yfinance.
        Returns a list of plain headline strings (publisher appended).
        Never raises — returns empty list on any failure.
        """
        try:
            raw_news = yf.Ticker(ticker).news or []
            headlines = []
            for item in raw_news[:max_items]:
                # yfinance ≥ 0.2.50 nests content inside a 'content' key
                if isinstance(item, dict) and "content" in item:
                    item = item["content"]
                title     = item.get("title", "").strip()
                publisher = item.get("publisher", "").strip()
                if title:
                    headlines.append(f"{title} [{publisher}]" if publisher else title)
            return headlines
        except Exception as exc:
            logger.debug(f"{ticker}: fetch_news failed — {exc}")
            return []

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _log_error(self, ticker: str, step: str, exc: Exception) -> None:
        self.errors.append({
            "ticker": ticker,
            "step":   step,
            "error":  str(exc),
            "time":   datetime.now().isoformat(timespec="seconds"),
        })
