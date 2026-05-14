"""
database.py — SQLite persistence layer for the 52-Week screener
===============================================================
Stores screening results so the app can display data instantly without
waiting for a live API run.  The background screener writes here every
~120 seconds; the Streamlit UI only reads.

Tables
------
screen_runs     — one row per completed screen cycle (universe + timestamp)
screen_results  — one row per flagged stock per run

Conventions
-----------
- news_headlines is stored as a JSON string.
- NaN / None values are stored as SQL NULL.
- Keep only the last MAX_RUNS runs per universe to cap file size.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("database")

_DB_PATH  = Path(__file__).parent / "screener.db"
MAX_RUNS  = 10    # keep last N runs per universe in the DB


# ── Schema ─────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS screen_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    universe     TEXT    NOT NULL,
    run_at       TEXT    NOT NULL,
    highs_count  INTEGER DEFAULT 0,
    lows_count   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS screen_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES screen_runs(id),
    universe             TEXT    NOT NULL,
    signal               TEXT    NOT NULL,   -- BREAKOUT_HIGH | BREAKDOWN_LOW
    ticker               TEXT,
    company_name         TEXT,
    sector               TEXT,
    industry             TEXT,
    current_price        REAL,
    week_52_high         REAL,
    week_52_low          REAL,
    pct_from_high        REAL,
    pct_from_low         REAL,
    volume_surge         REAL,
    avg_vol_20d          INTEGER,
    current_vol          INTEGER,
    pe_ratio             REAL,
    forward_pe           REAL,
    beta                 REAL,
    intrinsic_value      REAL,
    upside_downside_pct  REAL,
    dcf_stage1_growth    REAL,
    ret_1m               REAL,
    ret_3m               REAL,
    ret_6m               REAL,
    news_headlines       TEXT     -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_results_run
    ON screen_results(run_id, signal);
CREATE INDEX IF NOT EXISTS idx_runs_universe
    ON screen_runs(universe, id DESC);
"""

# Columns returned by load_latest_results — matches engine.format_for_display()
_RESULT_COLS = [
    "ticker", "company_name", "sector", "industry",
    "current_price", "week_52_high", "week_52_low",
    "pct_from_high", "pct_from_low",
    "volume_surge", "avg_vol_20d", "current_vol",
    "pe_ratio", "forward_pe", "beta",
    "intrinsic_value", "upside_downside_pct", "dcf_stage1_growth",
    "ret_1m", "ret_3m", "ret_6m",
    "news_headlines", "signal",
]


# ── Connection helper ──────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Open a connection to the screener DB (thread-safe read mode)."""
    return sqlite3.connect(str(_DB_PATH), check_same_thread=False)


# ── Public API ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist.  Safe to call on every startup."""
    with _conn() as con:
        con.executescript(_DDL)
    logger.info(f"DB initialised at {_DB_PATH}")


def save_screen_results(
    universe: str,
    highs_df: pd.DataFrame,
    lows_df:  pd.DataFrame,
) -> int:
    """
    Persist a completed screen run.

    Parameters
    ----------
    universe : 'NIFTY 500' or 'NIFTY 50'
    highs_df : DataFrame of BREAKOUT_HIGH stocks (format_for_display output)
    lows_df  : DataFrame of BREAKDOWN_LOW  stocks (format_for_display output)

    Returns
    -------
    int  — the new run_id
    """
    now = datetime.now().isoformat(timespec="seconds")

    with _conn() as con:
        cur = con.execute(
            "INSERT INTO screen_runs (universe, run_at, highs_count, lows_count) "
            "VALUES (?, ?, ?, ?)",
            (universe, now, len(highs_df), len(lows_df)),
        )
        run_id = cur.lastrowid

        def _insert(df: pd.DataFrame, signal: str) -> None:
            if df is None or df.empty:
                return
            for _, row in df.iterrows():
                headlines = row.get("news_headlines") or []
                con.execute(
                    """
                    INSERT INTO screen_results (
                        run_id, universe, signal,
                        ticker, company_name, sector, industry,
                        current_price, week_52_high, week_52_low,
                        pct_from_high, pct_from_low,
                        volume_surge, avg_vol_20d, current_vol,
                        pe_ratio, forward_pe, beta,
                        intrinsic_value, upside_downside_pct, dcf_stage1_growth,
                        ret_1m, ret_3m, ret_6m,
                        news_headlines
                    ) VALUES (
                        ?,?,?,  ?,?,?,?,  ?,?,?,  ?,?,  ?,?,?,
                        ?,?,?,  ?,?,?,  ?,?,?,  ?
                    )
                    """,
                    (
                        run_id, universe, signal,
                        row.get("ticker"),       row.get("company_name"),
                        row.get("sector"),       row.get("industry"),
                        _f(row.get("current_price")),
                        _f(row.get("week_52_high")),
                        _f(row.get("week_52_low")),
                        _f(row.get("pct_from_high")),
                        _f(row.get("pct_from_low")),
                        _f(row.get("volume_surge")),
                        _i(row.get("avg_vol_20d")),
                        _i(row.get("current_vol")),
                        _f(row.get("pe_ratio")),
                        _f(row.get("forward_pe")),
                        _f(row.get("beta")),
                        _f(row.get("intrinsic_value")),
                        _f(row.get("upside_downside_pct")),
                        _f(row.get("dcf_stage1_growth")),
                        _f(row.get("ret_1m")),
                        _f(row.get("ret_3m")),
                        _f(row.get("ret_6m")),
                        json.dumps(list(headlines)) if headlines else "[]",
                    ),
                )

        _insert(highs_df, "BREAKOUT_HIGH")
        _insert(lows_df,  "BREAKDOWN_LOW")
        con.commit()

    logger.info(f"Saved run #{run_id} | {universe} | {len(highs_df)}H {len(lows_df)}L")
    _cleanup_old_runs(universe)
    return run_id


def load_latest_results(
    universe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Load the most recent screen run for *universe*.

    Returns
    -------
    (highs_df, lows_df, run_at_str)
    All three are None / empty DataFrame if no data exists yet.
    """
    with _conn() as con:
        row = con.execute(
            "SELECT id, run_at FROM screen_runs "
            "WHERE universe = ? ORDER BY id DESC LIMIT 1",
            (universe,),
        ).fetchone()

        if not row:
            return pd.DataFrame(), pd.DataFrame(), None

        run_id, run_at = row

        df = pd.read_sql_query(
            "SELECT " + ", ".join(_RESULT_COLS) + " "
            "FROM screen_results WHERE run_id = ?",
            con,
            params=(run_id,),
        )

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), run_at

    # Deserialise news_headlines from JSON string → list
    df["news_headlines"] = df["news_headlines"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    highs = df[df["signal"] == "BREAKOUT_HIGH"].copy()
    lows  = df[df["signal"] == "BREAKDOWN_LOW"].copy()

    # Sort: highs by best DCF upside first; lows by biggest discount first
    if not highs.empty and "upside_downside_pct" in highs.columns:
        highs = highs.sort_values("upside_downside_pct", ascending=False)
    if not lows.empty and "upside_downside_pct" in lows.columns:
        lows = lows.sort_values("upside_downside_pct", ascending=False)

    return highs, lows, run_at


def get_last_run_info(universe: str) -> Optional[dict]:
    """
    Returns metadata about the most recent completed run, or None.

    Keys: run_at (str), highs_count (int), lows_count (int), seconds_ago (int)
    """
    with _conn() as con:
        row = con.execute(
            "SELECT run_at, highs_count, lows_count FROM screen_runs "
            "WHERE universe = ? ORDER BY id DESC LIMIT 1",
            (universe,),
        ).fetchone()

    if not row:
        return None

    run_at_str, highs, lows = row
    try:
        run_dt    = datetime.fromisoformat(run_at_str)
        secs_ago  = int((datetime.now() - run_dt).total_seconds())
    except Exception:
        secs_ago  = -1

    return {
        "run_at":      run_at_str,
        "highs_count": highs or 0,
        "lows_count":  lows  or 0,
        "seconds_ago": secs_ago,
    }


def db_exists() -> bool:
    """True if the DB file exists and has at least one completed run."""
    if not _DB_PATH.exists():
        return False
    try:
        with _conn() as con:
            n = con.execute("SELECT COUNT(*) FROM screen_runs").fetchone()[0]
        return n > 0
    except Exception:
        return False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _f(v) -> Optional[float]:
    """Safe float coerce — returns None for NaN / None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN check
    except (TypeError, ValueError):
        return None


def _i(v) -> Optional[int]:
    """Safe int coerce."""
    f = _f(v)
    return None if f is None else int(f)


def _cleanup_old_runs(universe: str) -> None:
    """Delete runs older than the most recent MAX_RUNS for this universe."""
    with _conn() as con:
        ids = [
            r[0] for r in con.execute(
                "SELECT id FROM screen_runs WHERE universe = ? ORDER BY id DESC",
                (universe,),
            ).fetchall()
        ]
        to_delete = ids[MAX_RUNS:]
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            con.execute(
                f"DELETE FROM screen_results WHERE run_id IN ({placeholders})",
                to_delete,
            )
            con.execute(
                f"DELETE FROM screen_runs WHERE id IN ({placeholders})",
                to_delete,
            )
            con.commit()
            logger.debug(f"Pruned {len(to_delete)} old runs for {universe}")
