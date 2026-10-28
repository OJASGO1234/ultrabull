# api/intraday_monitor.py
"""Intraday paper-trading monitoring PROTOTYPE -- a separate experiment
from the daily momentum monitor (api/momentum_monitor.py). Nothing here
modifies, extends, or is called by the daily system; the two share only
already-existing, unmodified building blocks (api/paper.py's ledger,
api/live.py's Groww access), never each other's tables or logic.

WHY THIS EXISTS
----------------
The daily monitor runs once a day, after the close, against a validated
momentum strategy (core/strategies/momentum_strategy.py, unchanged --
core/momentum.py, unchanged). This module is a DIFFERENT, DELIBERATELY
UNVALIDATED prototype: it watches 1-minute bars DURING market hours and
proves out the full intraday paper-trading lifecycle (market data -> bar
-> signal -> immutable snapshot -> paper execution -> position monitoring
-> exit -> forced square-off) end to end. There is no backtested intraday
strategy in this repository -- `_evaluate_signal` below is explicitly a
TEST/DEMO-GRADE placeholder rule (three consecutive higher-close 1-minute
bars), never claimed to be profitable or production-ready. Swap it out
once a real, researched intraday strategy exists; nothing else in this
module's pipeline needs to change to support that.

MARKET DATA -- REUSED, NOT DUPLICATED
--------------------------------------
Groww's SDK exposes a native historical-candle endpoint
(`get_historical_candle_data`), but it is single-symbol-per-call with no
batch form -- polling it once per symbol every INTRADAY_POLL_SECONDS for a
multi-symbol universe would multiply upstream calls badly. Instead this
module reuses the SAME shared, already-multi-symbol infrastructure
api/live.py already built for the browser's live ticker:
  - `api.live.feed_snapshot` (added alongside this module) reads the ONE
    shared GrowwFeed websocket connection `/api/live/ws` already uses --
    no second Groww subscription.
  - `api.live.ltp_map` (unmodified -- the SAME batch REST call the daily
    monitor already uses) is the fallback when the feed has no data yet.
Each monitor cycle samples one price snapshot per symbol and feeds it into
a per-symbol 1-minute bar builder (`_ingest_tick`) that this module DOES
own, because building bars from a shared tick/poll snapshot -- and
correctly refusing to treat a still-forming bar as completed -- is
intraday-monitoring logic, not live-quote infrastructure. Bars are
therefore an approximation from periodic sampling (every
INTRADAY_POLL_SECONDS, default 15s -- a handful of samples per minute),
not genuine tick-by-tick OHLC; this is disclosed, not hidden. Volume is
not populated from this feed (Groww's LTP subscription used here carries
no per-trade volume) -- the `volume` field exists in the schema but is
None in live mode; demo mode may supply it explicitly.

ISOLATION FROM REAL / DAILY ACCOUNTS
--------------------------------------
Because this signal engine is explicitly unvalidated, its LIVE paper
trades are written under a dedicated fixed username (`INTRADAY_USER`,
default "intraday-monitor") -- never the caller's own logged-in identity
and never "local" (the daily monitor's default user) -- so an intraday
test trade can never merge into, or be confused with, a real user's daily
paper-account numbers in `paper_positions`/`paper_trades` (both keyed by
`(username, symbol)`). DEMO mode uses a second, separate fixed username
(`INTRADAY_DEMO_USER`, "intraday-demo"), mirroring api/momentum_monitor.py's
own DEMO_USER isolation pattern exactly. All four namespaces -- a real
user's own account, "local" (daily monitor), "intraday-monitor" (this
module, live), "intraday-demo" (this module, demo) -- are fully disjoint
rows in the SAME sqlite tables; none can ever overwrite another.

PAPER ONLY: this module never calls any Groww order-placement endpoint --
`api.live` exposes only read-only quotes/feed here, and every actual
ledger write goes through the SAME `api.paper._fill` every manual trade
and the daily monitor already use.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time as _time_mod
from datetime import datetime, timedelta
from datetime import time as _dtime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .paper import _account as _paper_account, _fill as _paper_fill, _user as _paper_user, _db as _paper_db
from .live import ltp_map as _live_ltp_map, market_open as _live_market_open, feed_snapshot as _live_feed_snapshot
from .momentum_paper import _MOMENTUM_UNIVERSE

router = APIRouter(prefix="/api/intraday", tags=["intraday-monitor"])

IST = ZoneInfo("Asia/Kolkata")

INTRADAY_DEMO_USER = "intraday-demo"


def _parse_hhmm(s: str, name: str) -> _dtime:
    try:
        hh, mm = s.strip().split(":")
        return _dtime(int(hh), int(mm))
    except Exception:
        raise ValueError(f"{name} must be HH:MM (24h, Asia/Kolkata), got {s!r}") from None


# ---- configuration (env-overridable; kept in ONE place) --------------------
INTRADAY_MONITOR_ENABLED = os.environ.get("INTRADAY_MONITOR_ENABLED", "1").strip().lower() not in ("0", "false", "no")
# Master kill switch for the whole feature -- if 0, the background thread
# never even starts (see start_intraday_scheduler). Independent of the
# runtime /start /stop toggle below, which only pauses NEW entries.

INTRADAY_TIMEFRAME = os.environ.get("INTRADAY_TIMEFRAME", "1m")
if INTRADAY_TIMEFRAME != "1m":
    raise ValueError(
        f"INTRADAY_TIMEFRAME={INTRADAY_TIMEFRAME!r} is not supported yet -- this first intraday prototype "
        "only builds 1-minute bars (see task scope: 'do not build 5-second monitoring yet')."
    )

INTRADAY_START_TIME = os.environ.get("INTRADAY_START_TIME", "09:20")
# 09:20 IST: NSE's continuous session opens 09:15; a 5-minute buffer skips
# the opening-auction volatility before the monitor starts sampling bars.
INTRADAY_ENTRY_CUTOFF = os.environ.get("INTRADAY_ENTRY_CUTOFF", "15:00")
# 15:00 IST: stop opening NEW positions 30 minutes before NSE's 15:30
# close, leaving time to manage/exit whatever is already open.
INTRADAY_SQUARE_OFF_TIME = os.environ.get("INTRADAY_SQUARE_OFF_TIME", "15:15")
# 15:15 IST: force-close every open intraday paper position 15 minutes
# before close -- intraday positions must never carry overnight.
INTRADAY_START_TIME_OBJ = _parse_hhmm(INTRADAY_START_TIME, "INTRADAY_START_TIME")
INTRADAY_ENTRY_CUTOFF_OBJ = _parse_hhmm(INTRADAY_ENTRY_CUTOFF, "INTRADAY_ENTRY_CUTOFF")
INTRADAY_SQUARE_OFF_TIME_OBJ = _parse_hhmm(INTRADAY_SQUARE_OFF_TIME, "INTRADAY_SQUARE_OFF_TIME")

INTRADAY_POLL_SECONDS = float(os.environ.get("INTRADAY_POLL_SECONDS", "15"))
# How often the monitor samples market data / runs one cycle. NOT a 5-second
# tick loop -- 15s gives a handful of samples per 1-minute bar without
# hammering the shared feed/REST endpoint.

INTRADAY_USER = os.environ.get("INTRADAY_MONITOR_USER", "intraday-monitor")
# Fixed identity for LIVE (non-demo) intraday paper trades -- deliberately
# NOT the caller's own resolved user and NOT "local" (see module docstring,
# "Isolation from real / daily accounts").

_symbols_env = os.environ.get("INTRADAY_SYMBOLS")
INTRADAY_UNIVERSE = (
    [s.strip().upper() for s in _symbols_env.split(",") if s.strip()] if _symbols_env else list(_MOMENTUM_UNIVERSE)
)
# Defaults to the SAME shared momentum universe (core/event_study.py::UNIVERSE
# via api/momentum_paper.py) rather than hardcoding a second symbol list.

INTRADAY_MAX_OPEN_POSITIONS = int(os.environ.get("INTRADAY_MAX_OPEN_POSITIONS", "3"))
INTRADAY_MAX_TRADES_PER_DAY = int(os.environ.get("INTRADAY_MAX_TRADES_PER_DAY", "10"))
INTRADAY_MAX_DAILY_LOSS = float(os.environ.get("INTRADAY_MAX_DAILY_LOSS", "5000"))
# Absolute rupee amount (prototype default, not fit on any backtest -- there
# is nothing to fit it against, see module docstring).
INTRADAY_RISK_PCT_PER_TRADE = float(os.environ.get("INTRADAY_RISK_PCT_PER_TRADE", "0.5")) / 100.0
INTRADAY_MAX_HOLD_MINUTES = float(os.environ.get("INTRADAY_MAX_HOLD_MINUTES", "45"))
INTRADAY_STALE_DATA_SECONDS = float(os.environ.get("INTRADAY_STALE_DATA_SECONDS", "90"))
INTRADAY_FUTURE_TOLERANCE_SECONDS = float(os.environ.get("INTRADAY_FUTURE_TOLERANCE_SECONDS", "5"))
# A price stamped more than this far AHEAD of the local clock is treated as
# invalid (not "very fresh"). Small allowance for feed/clock jitter only --
# the local clock must be NTP-synchronized; the audit found one 74.6s slow.
INTRADAY_MAX_POSITION_NOTIONAL = float(os.environ.get("INTRADAY_MAX_POSITION_NOTIONAL", "100000"))
# Absolute rupee cap on entry value (qty x price) of ONE position. Default
# Rs 1,00,000 = 10% of the default Rs 10,00,000 paper capital, so three open
# positions use at most ~30%. Prototype default, not fit on any backtest.

STRATEGY_TAG = "Intraday_Test_v0"  # see _evaluate_signal -- NOT a validated strategy


# ---- persistence -------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = _paper_db()  # same sqlite file api/paper.py already uses
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS intraday_signals (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          symbol TEXT NOT NULL,
          yahoo_symbol TEXT NOT NULL,
          strategy TEXT NOT NULL,
          timeframe TEXT NOT NULL,
          signal_timestamp TEXT NOT NULL,
          bar_timestamp TEXT NOT NULL,
          direction TEXT NOT NULL,
          suggested_entry REAL NOT NULL,
          stop_loss REAL, target REAL,
          reason TEXT,
          market_data_ts TEXT,
          status TEXT NOT NULL DEFAULT 'pending',
          UNIQUE(username, symbol, strategy, timeframe, bar_timestamp)
        );
        CREATE TABLE IF NOT EXISTS intraday_executions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_id INTEGER NOT NULL,
          username TEXT NOT NULL,
          symbol TEXT NOT NULL,
          execution_timestamp TEXT NOT NULL,
          actual_entry_price REAL NOT NULL,
          quantity REAL NOT NULL,
          status TEXT NOT NULL,
          paper_trade_id INTEGER,
          closed_at TEXT, exit_price REAL, exit_reason TEXT,
          realized_pnl REAL, return_pct REAL, holding_minutes REAL
        );
        CREATE TABLE IF NOT EXISTS intraday_monitor_toggle (
          username TEXT PRIMARY KEY,
          monitoring_active INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS intraday_monitor_state (
          username TEXT NOT NULL,
          session_date TEXT NOT NULL,
          trades_today INTEGER NOT NULL DEFAULT 0,
          realized_pnl_today REAL NOT NULL DEFAULT 0,
          square_off_done INTEGER NOT NULL DEFAULT 0,
          kill_switch INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          PRIMARY KEY (username, session_date)
        );
        CREATE TABLE IF NOT EXISTS intraday_monitor_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          run_at TEXT NOT NULL,
          result_json TEXT NOT NULL
        );
        """
    )
    # Additive, idempotent migration: existing rows are never touched (their
    # rejection_reason stays NULL). Guarded for a concurrent scheduler/endpoint
    # connection adding the column first.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(intraday_signals)")}
    if "rejection_reason" not in cols:
        try:
            with conn:
                conn.execute("ALTER TABLE intraday_signals ADD COLUMN rejection_reason TEXT")
        except sqlite3.OperationalError:
            pass
    return conn


def _now_ist() -> datetime:
    return datetime.now(IST)


def _session_date_str(now_ist: datetime) -> str:
    return now_ist.astimezone(IST).date().isoformat()


def _get_toggle(conn: sqlite3.Connection, user: str) -> bool:
    row = conn.execute("SELECT monitoring_active FROM intraday_monitor_toggle WHERE username=?", (user,)).fetchone()
    return bool(row[0]) if row else False


def _set_toggle(conn: sqlite3.Connection, user: str, active: bool) -> None:
    with conn:
        conn.execute(
            "INSERT INTO intraday_monitor_toggle (username, monitoring_active) VALUES (?,?) "
            "ON CONFLICT(username) DO UPDATE SET monitoring_active=excluded.monitoring_active",
            (user, 1 if active else 0),
        )


def _get_or_init_state(conn: sqlite3.Connection, user: str, session_date: str) -> dict:
    row = conn.execute(
        "SELECT trades_today, realized_pnl_today, square_off_done, kill_switch, last_error "
        "FROM intraday_monitor_state WHERE username=? AND session_date=?", (user, session_date),
    ).fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO intraday_monitor_state (username, session_date) VALUES (?,?)",
                (user, session_date),
            )
        return _get_or_init_state(conn, user, session_date)
    keys = ["trades_today", "realized_pnl_today", "square_off_done", "kill_switch", "last_error"]
    return dict(zip(keys, row))


def _bump_trades_today(conn: sqlite3.Connection, user: str, session_date: str) -> None:
    with conn:
        conn.execute(
            "UPDATE intraday_monitor_state SET trades_today = trades_today + 1 WHERE username=? AND session_date=?",
            (user, session_date),
        )


def _bump_realized_pnl(conn: sqlite3.Connection, user: str, session_date: str, pnl: float | None) -> None:
    if pnl is None:
        return
    with conn:
        conn.execute(
            "UPDATE intraday_monitor_state SET realized_pnl_today = realized_pnl_today + ? "
            "WHERE username=? AND session_date=?",
            (pnl, user, session_date),
        )


def _set_square_off_done(conn: sqlite3.Connection, user: str, session_date: str) -> None:
    with conn:
        conn.execute(
            "UPDATE intraday_monitor_state SET square_off_done=1 WHERE username=? AND session_date=?",
            (user, session_date),
        )


def _set_kill_switch(conn: sqlite3.Connection, user: str, session_date: str, tripped: bool) -> None:
    with conn:
        conn.execute(
            "UPDATE intraday_monitor_state SET kill_switch=? WHERE username=? AND session_date=?",
            (1 if tripped else 0, user, session_date),
        )


def _set_last_error(conn: sqlite3.Connection, user: str, session_date: str, error_text: str | None) -> None:
    with conn:
        conn.execute(
            "UPDATE intraday_monitor_state SET last_error=? WHERE username=? AND session_date=?",
            (error_text, user, session_date),
        )


def _last_finalized_bar(conn: sqlite3.Connection, user: str) -> dict | None:
    """Most recent bar finalization ever recorded for `user`, read from the
    persisted run log (so it survives restarts and no-finalization cycles):
    {"bar_ts", "finalized_at", "symbol_count", "symbols"} for the newest
    minute finalized in the newest run that finalized any bar, else None."""
    rows = conn.execute(
        "SELECT run_at, result_json FROM intraday_monitor_runs WHERE username=? "
        "AND result_json NOT LIKE '%\"bars_finalized\": []%' ORDER BY id DESC LIMIT 50", (user,),
    ).fetchall()
    for run_at, rj in rows:
        bars = json.loads(rj).get("bars_finalized") or []
        if not bars:
            continue
        newest = max(b["bar_ts"] for b in bars)
        syms = sorted(b["symbol"] for b in bars if b["bar_ts"] == newest)
        return {"bar_ts": newest, "finalized_at": run_at, "symbol_count": len(syms), "symbols": syms}
    return None


def _save_run(conn: sqlite3.Connection, user: str, now_ist: datetime, result: dict) -> None:
    with conn:
        conn.execute(
            "INSERT INTO intraday_monitor_runs (username, run_at, result_json) VALUES (?,?,?)",
            (user, now_ist.isoformat(timespec="seconds"), json.dumps(result, default=str)),
        )


# ---- 1-minute bar builder (in-memory, process-lifetime) --------------------
# Bars are reconstructible market-data state, not money -- a restart simply
# resumes accumulating today's bars from that point on. What MUST survive a
# restart (signals, executions, positions, daily counters) is persisted in
# sqlite above, never here. Demo and live state are kept in SEPARATE dicts
# so synthetic bars can never leak into (or be mistaken for) real ones.
_forming: dict[str, dict] = {}
_bar_hist: dict[str, list[dict]] = {}
_last_known_price: dict[str, float] = {}

_demo_forming: dict[str, dict] = {}
_demo_bar_hist: dict[str, list[dict]] = {}
_demo_last_known_price: dict[str, float] = {}


_ONE_MINUTE = timedelta(minutes=1)


def _floor_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def _ingest_tick(store: dict, hist: dict, symbol: str, price: float, ts: datetime) -> dict | None:
    """Update the forming 1-minute bar for `symbol` with one price sample.
    If this sample belongs to a NEW minute, the PREVIOUS bar is finalized,
    appended to `hist[symbol]` (capped at the last 20), and returned -- the
    just-sampled price only starts the new forming bar, it is never itself
    reported as completed. Returns None while still mid-minute, which is
    exactly what keeps a still-forming candle from ever reaching the signal
    engine."""
    bar_ts = _floor_minute(ts)
    forming = store.get(symbol)
    finalized = None
    if forming is None or forming["bar_ts"] != bar_ts:
        if forming is not None:
            finalized = dict(forming)
            h = hist.setdefault(symbol, [])
            if h and finalized["bar_ts"] - h[-1]["bar_ts"] != _ONE_MINUTE:
                # Continuity rule: minutes with no fresh sample (outage, stale/
                # future/missing prices, restart, overnight) are NOT filled or
                # interpolated -- the run of consecutive bars simply restarts.
                h.clear()
            h.append(finalized)
            del h[:-20]
        store[symbol] = {"bar_ts": bar_ts, "open": price, "high": price, "low": price, "close": price, "volume": None}
    else:
        forming["high"] = max(forming["high"], price)
        forming["low"] = min(forming["low"], price)
        forming["close"] = price
    return finalized


# ---- signal engine (TEST/DEMO-GRADE ONLY -- see module docstring) ---------

def _evaluate_signal(bar_history: list[dict]) -> dict | None:
    """Fires a BUY signal after three consecutive (adjacent-minute)
    higher-close 1-minute bars; stop = lowest low of those 3 bars, target = entry + 2x(entry -
    stop) -- a simple, undocumented-by-research 1:2 placeholder, exactly
    like the daily monitor's own prototype sizing constants are labeled
    "not tuned or researched". This function exists to prove the
    signal -> snapshot -> execution -> exit pipeline end to end, NOT as a
    claim that this rule is profitable. Replace it once a real, backtested
    intraday strategy exists -- nothing else in this module needs to
    change to support that swap."""
    if len(bar_history) < 3:
        return None
    b0, b1, b2 = bar_history[-3], bar_history[-2], bar_history[-1]
    if b1["bar_ts"] - b0["bar_ts"] != _ONE_MINUTE or b2["bar_ts"] - b1["bar_ts"] != _ONE_MINUTE:
        return None  # the three bars must be adjacent minutes -- never bridge a gap
    if not (b0["close"] < b1["close"] < b2["close"]):
        return None
    entry = b2["close"]
    stop = min(b0["low"], b1["low"], b2["low"])
    if stop >= entry:
        return None
    target = round(entry + 2 * (entry - stop), 4)
    return {
        "direction": "BUY", "suggested_entry": round(entry, 4), "stop_loss": round(stop, 4), "target": target,
        "reason": {
            "rule": "three_consecutive_higher_close_1m_bars", "test_signal": True,
            "bars": [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in b.items()} for b in (b0, b1, b2)],
        },
    }


def _fee_rate_per_side(acct: dict) -> float:
    """Fractional charge applied to EACH side's gross value, exactly as
    api/paper.py::_charges does: (brokerage_pct + tax_pct) / 100 from the
    paper account row (defaults 0.0% + 0.12% = 0.12% per side). This is the
    pre-existing paper model, NOT a verified Groww intraday brokerage/STT/
    exchange/GST/stamp schedule -- it must be checked against the intended
    broker, product (intraday MIS vs delivery) and exchange before any P&L
    from it is relied on. Configure it per account via those two columns."""
    return (acct["brokerage_pct"] + acct["tax_pct"]) / 100.0


def _entry_cost(qty: int, entry: float, fee_rate: float) -> float:
    """Cash needed to BUY qty at entry -- gross plus entry-side charges,
    rounded the same way api/paper.py::_charges rounds them."""
    gross = qty * entry
    return gross + round(gross * fee_rate, 2)


def _size_position_checked(cash: float, entry: float, stop: float | None,
                           fee_rate: float = 0.0) -> tuple[int, str | None]:
    """(qty, reject_reason). qty is the largest whole-share quantity that
    satisfies ALL of:
      1. risk: the loss if the stop fills -- price loss + entry charge + exit
         charge at the stop -- is <= cash x INTRADAY_RISK_PCT_PER_TRADE;
      2. notional: qty x entry <= INTRADAY_MAX_POSITION_NOTIONAL;
      3. cash: entry cost INCLUDING the entry charge <= available cash.
    It never widens the stop or moves the entry. If even 1 share breaks the
    risk or notional limit the trade is rejected (qty 0 + reason) rather than
    forced through."""
    if stop is None or entry is None or entry <= 0 or cash <= 0:
        return 0, "invalid entry/stop/cash for sizing"
    price_risk = entry - stop
    if price_risk <= 0:
        return 0, "stop is not below entry"
    loss_per_share = price_risk + fee_rate * (entry + stop)  # buy charge + sell charge at the stop
    risk_amount = cash * INTRADAY_RISK_PCT_PER_TRADE
    qty_risk = int(risk_amount // loss_per_share)
    qty_notional = int(INTRADAY_MAX_POSITION_NOTIONAL // entry)
    qty_cash = int(cash // (entry * (1.0 + fee_rate)))
    qty = min(qty_risk, qty_notional, qty_cash)
    while qty > 0 and _entry_cost(qty, entry, fee_rate) > cash:  # rounding guard: never overspend
        qty -= 1
    if qty >= 1:
        return qty, None
    if qty_risk < 1:
        return 0, (f"1 share risks Rs {loss_per_share:,.2f} (incl. charges) > per-trade risk budget "
                   f"Rs {risk_amount:,.2f}")
    if qty_notional < 1:
        return 0, f"1 share (Rs {entry:,.2f}) > max position notional Rs {INTRADAY_MAX_POSITION_NOTIONAL:,.0f}"
    return 0, f"insufficient paper cash for 1 share incl. charges (have Rs {cash:,.2f})"


def _size_position(cash: float, entry: float, stop: float | None, fee_rate: float = 0.0) -> int:
    return _size_position_checked(cash, entry, stop, fee_rate)[0]


# ---- market data -------------------------------------------------------

def _fetch_market_snapshot(symbols: list[str], now_ist: datetime) -> tuple[dict[str, float], "datetime | dict[str, datetime]", str]:
    """(price_by_symbol, data_as_of, source). data_as_of is either one
    datetime applying to every symbol (REST poll, tests) or a
    {symbol: datetime} map of each symbol's OWN tick time (websocket) --
    _classify_prices consumes both. Prefers the shared
    GrowwFeed websocket snapshot (api.live.feed_snapshot, reused, not
    duplicated); falls back to the existing batch REST poll (api.live.
    ltp_map, the SAME function the daily monitor already uses) when the
    feed has nothing yet. Never fabricates a price: returns {} with
    source='none' if both are empty (e.g. no Groww credentials configured
    -- the same graceful-degradation already documented for the daily
    monitor)."""
    try:
        snap = _live_feed_snapshot(symbols)
    except Exception:
        snap = {}
    if snap:
        # per-symbol tick time; a symbol with no ts is NOT assumed fresh --
        # it gets epoch 0 so _classify_prices marks it stale
        as_of = {s: datetime.fromtimestamp(v["ts"] if v.get("ts") else 0, IST) for s, v in snap.items()}
        return {s: v["ltp"] for s, v in snap.items()}, as_of, "ws"
    try:
        prices = _live_ltp_map(symbols)
    except Exception:
        prices = {}
    if prices:
        return prices, now_ist, "poll"  # a synchronous REST call is fresh as of "now" by construction
    return {}, now_ist, "none"


def _classify_prices(prices: dict[str, float], as_of, now_ist: datetime, expected: list[str]) -> dict:
    """Judge freshness PER SYMBOL against the (NTP-synchronized) local clock.
    A symbol's price is usable only if its own tick age is within
    [-INTRADAY_FUTURE_TOLERANCE_SECONDS, INTRADAY_STALE_DATA_SECONDS]; one
    fresh symbol never vouches for another. `as_of` is one datetime for all
    symbols or a {symbol: datetime} map; a symbol absent from the map has no
    timestamp and is stale. Returns fresh prices plus the stale / future /
    missing symbol lists (missing = expected but no price at all)."""
    fresh: dict[str, float] = {}
    fresh_ts: dict[str, datetime] = {}
    stale: list[str] = []
    future: list[str] = []
    for sym, px in prices.items():
        ts = as_of.get(sym) if isinstance(as_of, dict) else as_of
        if ts is None or px is None:
            stale.append(sym)
            continue
        age = (now_ist - ts).total_seconds()
        if age < -INTRADAY_FUTURE_TOLERANCE_SECONDS:
            future.append(sym)
        elif age > INTRADAY_STALE_DATA_SECONDS:
            stale.append(sym)
        else:
            fresh[sym] = float(px)
            fresh_ts[sym] = ts
    missing = [s for s in expected if s not in prices]
    return {"fresh": fresh, "fresh_ts": fresh_ts, "stale": sorted(stale), "future": sorted(future),
            "missing": sorted(missing)}


# ---- session-window helpers ---------------------------------------------

def _time_on(day, t: _dtime) -> datetime:
    return datetime.combine(day, t, tzinfo=IST)


def _in_entry_window(now_ist: datetime) -> bool:
    d = now_ist.astimezone(IST).date()
    return (
        _live_market_open(now_ist)  # reused, not reimplemented -- excludes weekends (NSE doesn't trade them)
        and _time_on(d, INTRADAY_START_TIME_OBJ) <= now_ist < _time_on(d, INTRADAY_ENTRY_CUTOFF_OBJ)
    )


def _session_active(now_ist: datetime) -> bool:
    d = now_ist.astimezone(IST).date()
    return (
        _live_market_open(now_ist)
        and _time_on(d, INTRADAY_START_TIME_OBJ) <= now_ist < _time_on(d, INTRADAY_SQUARE_OFF_TIME_OBJ)
    )


def _square_off_due(now_ist: datetime) -> bool:
    d = now_ist.astimezone(IST).date()
    return now_ist >= _time_on(d, INTRADAY_SQUARE_OFF_TIME_OBJ)


# ---- signal / execution / close (immutable snapshot semantics) -----------

def _record_signal(conn: sqlite3.Connection, user: str, symbol: str, bar: dict, sig: dict,
                    market_data_ts: datetime, now_ist: datetime) -> dict:
    """Insert an IMMUTABLE signal row if one doesn't already exist for
    (user, symbol, strategy, timeframe, bar_timestamp) -- the idempotency
    key that makes the same 1-minute bar unable to ever produce a second
    signal. Enforced BOTH by a check-then-insert (matching api/momentum_paper
    .py's _record_recommendation convention) AND a UNIQUE constraint as
    defense-in-depth against a race between the manual endpoint and the
    scheduler thread."""
    bar_ts_s = bar["bar_ts"].isoformat(timespec="seconds")
    existing = conn.execute(
        "SELECT id FROM intraday_signals WHERE username=? AND symbol=? AND strategy=? AND timeframe=? AND bar_timestamp=?",
        (user, symbol, STRATEGY_TAG, INTRADAY_TIMEFRAME, bar_ts_s),
    ).fetchone()
    if existing:
        return {"id": existing[0], "already_existed": True}
    now_s = now_ist.isoformat(timespec="seconds")
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO intraday_signals (username, symbol, yahoo_symbol, strategy, timeframe, "
                "signal_timestamp, bar_timestamp, direction, suggested_entry, stop_loss, target, reason, "
                "market_data_ts, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'pending')",
                (user, symbol, symbol, STRATEGY_TAG, INTRADAY_TIMEFRAME, now_s, bar_ts_s, sig["direction"],
                 sig["suggested_entry"], sig["stop_loss"], sig["target"], json.dumps(sig["reason"]),
                 market_data_ts.isoformat(timespec="seconds")),
            )
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT id FROM intraday_signals WHERE username=? AND symbol=? AND strategy=? AND timeframe=? AND bar_timestamp=?",
            (user, symbol, STRATEGY_TAG, INTRADAY_TIMEFRAME, bar_ts_s),
        ).fetchone()
        return {"id": row[0], "already_existed": True}
    return {"id": cur.lastrowid, "already_existed": False}


def _execute_signal(conn: sqlite3.Connection, user: str, signal_id: int, symbol: str, sig: dict,
                     price: float, qty: float, demo: bool, now_ist: datetime) -> dict:
    """Fill a pending signal as an ACTUAL paper trade through the EXISTING,
    unmodified api/paper.py ledger (`_fill`). `actual_entry_price` lives in
    intraday_executions, a table separate from intraday_signals'
    `suggested_entry` -- the two are never merged, matching the daily
    monitor's own suggested-vs-actual separation. Raises RuntimeError if
    the signal is not 'pending' (already executed/cancelled) -- callers
    treat that as a no-op, making repeated execution attempts safe."""
    row = conn.execute("SELECT status FROM intraday_signals WHERE id=?", (signal_id,)).fetchone()
    if row is None or row[0] != "pending":
        raise RuntimeError(f"signal is already {row[0] if row else 'missing'}")
    note = ("DEMO " if demo else "") + f"intraday signal #{signal_id} ({STRATEGY_TAG}) auto-execute"
    _paper_fill(conn, user, symbol, symbol, "BUY", qty, price, sig["stop_loss"], sig["target"], note)
    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    now_s = now_ist.isoformat(timespec="seconds")
    with conn:
        conn.execute("UPDATE intraday_signals SET status='executed' WHERE id=?", (signal_id,))
        cur = conn.execute(
            "INSERT INTO intraday_executions (signal_id, username, symbol, execution_timestamp, actual_entry_price, "
            "quantity, status, paper_trade_id) VALUES (?,?,?,?,?,?,'filled',?)",
            (signal_id, user, symbol, now_s, price, qty, trade_id),
        )
    slippage = round(100.0 * (price - sig["suggested_entry"]) / sig["suggested_entry"], 2) if sig["suggested_entry"] else None
    return {"execution_id": cur.lastrowid, "paper_trade_id": trade_id, "actual_entry_price": price, "qty": qty,
            "suggested_entry": sig["suggested_entry"], "slippage_pct": slippage}


def _reject_signal(conn: sqlite3.Connection, signal_id: int, reason: str) -> None:
    """Terminal status for a signal that could not be executed. Only a
    still-pending signal is touched, so an executed signal can never be
    downgraded; `pending` therefore always means "awaiting an attempt"."""
    with conn:
        conn.execute(
            "UPDATE intraday_signals SET status='rejected', rejection_reason=? WHERE id=? AND status='pending'",
            (reason[:300], signal_id),
        )


def _close_intraday_position(conn: sqlite3.Connection, user: str, exec_row: dict, price: float, reason: str,
                              now_ist: datetime) -> dict:
    realized = _paper_fill(conn, user, exec_row["symbol"], exec_row["symbol"], "SELL",
                            exec_row["quantity"], price, None, None,
                            f"intraday {reason} (signal #{exec_row['signal_id']})")
    closed_at = now_ist
    entered_at = datetime.fromisoformat(exec_row["execution_timestamp"])
    if entered_at.tzinfo is None:
        entered_at = entered_at.replace(tzinfo=IST)
    holding_minutes = round((closed_at - entered_at).total_seconds() / 60.0, 1)
    entry_px = exec_row["actual_entry_price"]
    ret_pct = round(100.0 * (price - entry_px) / entry_px, 2) if entry_px else None
    with conn:
        conn.execute(
            "UPDATE intraday_executions SET closed_at=?, exit_price=?, exit_reason=?, realized_pnl=?, "
            "return_pct=?, holding_minutes=? WHERE id=?",
            (closed_at.isoformat(timespec="seconds"), price, reason, realized, ret_pct, holding_minutes,
             exec_row["execution_id"]),
        )
    return {"execution_id": exec_row["execution_id"], "symbol": exec_row["symbol"], "reason": reason,
            "exit_price": price, "realized_pnl": realized, "return_pct": ret_pct, "holding_minutes": holding_minutes}


# ---- the one monitoring cycle -- called by BOTH the manual endpoint and
# the scheduler thread; there is no second implementation of any of this. --

def run_intraday_cycle(user: str, demo: bool = False, demo_now: datetime | None = None,
                        demo_prices: dict[str, float] | None = None) -> dict:
    now_ist = (demo_now or _now_ist())
    if now_ist.tzinfo is None:
        now_ist = now_ist.replace(tzinfo=IST)
    now_ist = now_ist.astimezone(IST)
    session_date = _session_date_str(now_ist)

    conn = _db()
    try:
        toggle_active = True if demo else _get_toggle(conn, user)
        state = _get_or_init_state(conn, user, session_date)
        in_session = _session_active(now_ist)
        in_entry_window = _in_entry_window(now_ist)
        sq_due = _square_off_due(now_ist)

        result = {
            "ok": True, "mode": "demo" if demo else "live", "run_at": now_ist.isoformat(timespec="seconds"),
            "session_date": session_date, "monitoring_active": toggle_active,
            "in_session": in_session, "in_entry_window": in_entry_window, "square_off_due": sq_due,
            "data": {"source": None, "as_of": None, "stale": False},
            "bars_finalized": [], "signals": {"created": [], "already_existed": []},
            "executions": {"attempted": 0, "filled": [], "skipped": []},
            "position_monitoring": {"checked": 0, "closed": [], "still_open": []},
            "risk": {}, "skipped_reason": None,
        }

        if not demo and not in_session:
            has_open = conn.execute(
                "SELECT 1 FROM intraday_executions WHERE username=? AND closed_at IS NULL", (user,)
            ).fetchone() is not None
            if not has_open:
                result["skipped_reason"] = "outside the configured intraday session (before start, or after square-off with nothing open)"
                _save_run(conn, user, now_ist, result)
                return result
            # else fall through: something is still open (shouldn't normally
            # happen once square-off has run) -- keep monitoring/square off it safely

        store = _demo_forming if demo else _forming
        hist = _demo_bar_hist if demo else _bar_hist
        last_price_cache = _demo_last_known_price if demo else _last_known_price

        # 1. market data
        # `prices` below is ONLY the symbols whose own timestamp passed the
        # per-symbol freshness check; stale, future-dated and missing prices
        # never reach bars, signals, entries or exits.
        if demo:
            prices, source = {s: float(p) for s, p in (demo_prices or {}).items()}, "demo"
            cls = {"fresh": prices, "fresh_ts": {s: now_ist for s in prices}, "stale": [], "future": [], "missing": []}
        else:
            raw_prices, raw_as_of, source = _fetch_market_snapshot(INTRADAY_UNIVERSE, now_ist)
            cls = _classify_prices(raw_prices, raw_as_of, now_ist, INTRADAY_UNIVERSE)
            prices = cls["fresh"]
        # "stale" = no usable price at all; partial staleness is reported per symbol
        stale = (not prices) and not demo
        ts_vals = list(cls["fresh_ts"].values())
        result["data"] = {
            "source": source, "stale": stale,
            "as_of": (max(ts_vals) if ts_vals else now_ist).isoformat(timespec="seconds"),  # newest usable tick
            "oldest_as_of": min(ts_vals).isoformat(timespec="seconds") if ts_vals else None,
            "fresh_count": len(prices), "stale_symbols": cls["stale"], "future_symbols": cls["future"],
            "missing_symbols": cls["missing"],
        }
        for sym, px in prices.items():
            last_price_cache[sym] = float(px)

        # 2. ingest ticks -> finalize completed 1-minute bars
        newly_finalized: dict[str, dict] = {}
        for sym, price in prices.items():
            bar = _ingest_tick(store, hist, sym, float(price), now_ist)
            if bar is not None:
                newly_finalized[sym] = bar
                result["bars_finalized"].append({
                    "symbol": sym, "bar_ts": bar["bar_ts"].isoformat(timespec="seconds"),
                    "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"],
                })

        # 3. signal generation + immediate paper execution (entry window only)
        can_enter = (
            toggle_active and not state["kill_switch"] and in_entry_window and not stale and not sq_due
        )
        open_count = conn.execute(
            "SELECT COUNT(*) FROM intraday_executions WHERE username=? AND closed_at IS NULL", (user,)
        ).fetchone()[0]
        trades_today = state["trades_today"]
        if can_enter:
            for sym, bar in newly_finalized.items():
                if open_count >= INTRADAY_MAX_OPEN_POSITIONS or trades_today >= INTRADAY_MAX_TRADES_PER_DAY:
                    break
                already_open = conn.execute(
                    "SELECT 1 FROM intraday_executions WHERE username=? AND symbol=? AND closed_at IS NULL",
                    (user, sym),
                ).fetchone()
                if already_open:
                    continue
                sig = _evaluate_signal(hist.get(sym, []))
                if sig is None:
                    continue
                rec = _record_signal(conn, user, sym, bar, sig, cls["fresh_ts"][sym], now_ist)
                bucket = result["signals"]["already_existed"] if rec["already_existed"] else result["signals"]["created"]
                bucket.append({"id": rec["id"], "symbol": sym, "suggested_entry": sig["suggested_entry"]})
                if rec["already_existed"]:
                    continue

                fill_price = prices.get(sym, sig["suggested_entry"])
                acct = _paper_account(conn, user)
                qty, why = _size_position_checked(acct["cash"], fill_price, sig["stop_loss"], _fee_rate_per_side(acct))
                result["executions"]["attempted"] += 1
                if qty <= 0:
                    # terminal: a rejected signal is never re-attempted (its bar key already exists)
                    _reject_signal(conn, rec["id"], f"sizing: {why}")
                    result["executions"]["skipped"].append({"id": rec["id"], "symbol": sym, "reason": "rejected", "detail": why})
                    continue
                try:
                    exe = _execute_signal(conn, user, rec["id"], sym, sig, fill_price, qty, demo, now_ist)
                except ValueError as e:  # paper-ledger guard (e.g. insufficient cash): nothing was written
                    _reject_signal(conn, rec["id"], f"ledger: {e}")
                    result["executions"]["skipped"].append({"id": rec["id"], "symbol": sym, "reason": "rejected", "error": str(e)})
                    continue
                except RuntimeError as e:  # signal no longer pending: leave its status alone
                    result["executions"]["skipped"].append({"id": rec["id"], "symbol": sym, "error": str(e)})
                    continue
                result["executions"]["filled"].append({"id": rec["id"], "symbol": sym, **exe})
                open_count += 1
                trades_today += 1
                _bump_trades_today(conn, user, session_date)

        # 4. monitor every open intraday position (SL / target / time-exit / square-off)
        open_rows = conn.execute(
            "SELECT ix.id, ix.signal_id, ix.symbol, ix.actual_entry_price, ix.quantity, ix.execution_timestamp, "
            "isg.stop_loss, isg.target "
            "FROM intraday_executions ix JOIN intraday_signals isg ON isg.id = ix.signal_id "
            "WHERE ix.username=? AND ix.closed_at IS NULL",
            (user,),
        ).fetchall()
        mon = result["position_monitoring"]
        force_square_off = sq_due and not state["square_off_done"]
        for eid, sid, sym, entry_px, qty, exec_ts, sl, tp in open_rows:
            mon["checked"] += 1
            exec_row = {"execution_id": eid, "signal_id": sid, "symbol": sym, "actual_entry_price": entry_px,
                        "quantity": qty, "execution_timestamp": exec_ts}
            price = prices.get(sym)
            if price is not None:
                last_price_cache[sym] = float(price)

            if force_square_off:
                use_price = price if price is not None else last_price_cache.get(sym, entry_px)
                closed = _close_intraday_position(conn, user, exec_row, use_price, "INTRADAY_SQUARE_OFF", now_ist)
                mon["closed"].append(closed)
                _bump_realized_pnl(conn, user, session_date, closed["realized_pnl"])
                continue

            if price is None:  # `prices` holds fresh symbols only
                mon["still_open"].append({"execution_id": eid, "symbol": sym, "note": "no fresh price this cycle"})
                continue

            entered_at = datetime.fromisoformat(exec_ts)
            if entered_at.tzinfo is None:
                entered_at = entered_at.replace(tzinfo=IST)
            holding_minutes = (now_ist - entered_at).total_seconds() / 60.0

            reason = None
            # Stop wins over a simultaneous target hit -- same conservative
            # tie-break convention documented in core/sim/engine.py and
            # reused (unmodified in spirit) by the daily monitor.
            if sl is not None and price <= sl:
                reason = "STOP"
            elif tp is not None and price >= tp:
                reason = "TARGET"
            elif holding_minutes >= INTRADAY_MAX_HOLD_MINUTES:
                reason = "TIME_EXIT"

            if reason is None:
                mon["still_open"].append({
                    "execution_id": eid, "symbol": sym, "current_price": price, "entry_price": entry_px, "qty": qty,
                    "unrealized_pnl": round((price - entry_px) * qty, 2), "holding_minutes": round(holding_minutes, 1),
                })
                continue
            try:
                closed = _close_intraday_position(conn, user, exec_row, price, reason, now_ist)
            except ValueError as e:
                mon["still_open"].append({"execution_id": eid, "symbol": sym, "error": str(e)})
                continue
            mon["closed"].append(closed)
            _bump_realized_pnl(conn, user, session_date, closed["realized_pnl"])

        if force_square_off:
            _set_square_off_done(conn, user, session_date)

        # 5. daily-loss kill switch (checked AFTER this cycle's closes, so a
        # loss realized just now can trip it before the NEXT cycle's entries)
        state_after = _get_or_init_state(conn, user, session_date)
        if (not state_after["kill_switch"]) and state_after["realized_pnl_today"] <= -INTRADAY_MAX_DAILY_LOSS:
            _set_kill_switch(conn, user, session_date, True)
            state_after["kill_switch"] = True

        open_positions_now = conn.execute(
            "SELECT COUNT(*) FROM intraday_executions WHERE username=? AND closed_at IS NULL", (user,)
        ).fetchone()[0]
        result["risk"] = {
            "open_positions": open_positions_now,
            "max_open_positions": INTRADAY_MAX_OPEN_POSITIONS,
            "trades_today": state_after["trades_today"], "max_trades_per_day": INTRADAY_MAX_TRADES_PER_DAY,
            "realized_pnl_today": round(state_after["realized_pnl_today"], 2), "max_daily_loss": INTRADAY_MAX_DAILY_LOSS,
            "kill_switch": bool(state_after["kill_switch"]),
        }
        _save_run(conn, user, now_ist, result)
        return result
    finally:
        conn.close()


# ---- scheduler: same background-thread pattern as api/paper.py's
# start_executor AND api/momentum_monitor.py's start_monitor_scheduler,
# just gated by the /start /stop toggle on top of the session window. -----

def _intraday_tick(now: datetime | None = None) -> None:
    session_date = _session_date_str(now or _now_ist())
    try:
        run_intraday_cycle(INTRADAY_USER, demo=False, demo_now=now)
    except Exception as e:
        conn = _db()
        try:
            _set_last_error(conn, INTRADAY_USER, session_date, str(e))
        finally:
            conn.close()
    else:
        conn = _db()
        try:
            _set_last_error(conn, INTRADAY_USER, session_date, None)  # clear a PRIOR tick's error once a cycle succeeds
        finally:
            conn.close()


_intraday_started = False


def start_intraday_scheduler() -> None:
    """Start the intraday background thread once. No-ops if
    INTRADAY_MONITOR_ENABLED=0. The thread itself always runs (so an open
    position is never left unmonitored) -- run_intraday_cycle internally
    gates NEW entries on the /start /stop toggle, the entry-cutoff window,
    risk limits, and data freshness; only forced square-off and existing-
    position monitoring bypass that toggle, for safety."""
    global _intraday_started
    if _intraday_started or not INTRADAY_MONITOR_ENABLED:
        return
    _intraday_started = True

    def _loop():
        while True:
            try:
                _intraday_tick()
            except Exception:
                pass
            _time_mod.sleep(INTRADAY_POLL_SECONDS)

    threading.Thread(target=_loop, daemon=True, name="intraday-monitor-scheduler").start()


# ---- API -----------------------------------------------------------------

class IntradayRunOnceReq(BaseModel):
    demo: bool = False
    demo_now: str | None = None            # ISO Asia/Kolkata datetime override -- DEMO/TEST ONLY
    demo_prices: dict[str, float] | None = None  # yahoo_symbol -> price tick -- DEMO/TEST ONLY


@router.post("/start")
def intraday_start(request: Request):
    """Resume NEW intraday signal generation / paper execution. Existing
    open positions are monitored (and force-squared-off) regardless of
    this toggle -- see start_intraday_scheduler."""
    conn = _db()
    try:
        _set_toggle(conn, INTRADAY_USER, True)
        return {"ok": True, "monitoring_active": True}
    finally:
        conn.close()


@router.post("/stop")
def intraday_stop(request: Request):
    """Pause NEW intraday signal generation / paper execution (a manual
    kill switch). Existing open positions keep being monitored/squared off."""
    conn = _db()
    try:
        _set_toggle(conn, INTRADAY_USER, False)
        return {"ok": True, "monitoring_active": False}
    finally:
        conn.close()


@router.post("/run-once")
def intraday_run_once(req: IntradayRunOnceReq, request: Request):
    """MANUAL trigger: run exactly ONE intraday cycle now and return its
    result -- calls the SAME run_intraday_cycle the background scheduler
    calls. Live (demo=false) runs always target INTRADAY_USER, the same
    fixed identity the scheduler uses, so manual and scheduled runs are
    identical in effect, not just in code path. demo=true runs target
    INTRADAY_DEMO_USER and accept caller-supplied synthetic prices/clock
    (see module docstring) to demonstrate the full lifecycle on demand."""
    if not req.demo and (req.demo_now or req.demo_prices):
        raise HTTPException(422, "demo_now/demo_prices require demo=true")
    user = INTRADAY_DEMO_USER if req.demo else INTRADAY_USER
    demo_now = None
    if req.demo_now:
        demo_now = datetime.fromisoformat(req.demo_now)
        if demo_now.tzinfo is None:
            demo_now = demo_now.replace(tzinfo=IST)
    return run_intraday_cycle(user, req.demo, demo_now, req.demo_prices)


@router.get("/status")
def intraday_status(request: Request, demo: bool = False):
    """Read-only snapshot for the UI. Never triggers a cycle itself."""
    user = INTRADAY_DEMO_USER if demo else INTRADAY_USER
    conn = _db()
    try:
        last = conn.execute(
            "SELECT result_json FROM intraday_monitor_runs WHERE username=? ORDER BY id DESC LIMIT 1", (user,),
        ).fetchone()
        last_run = json.loads(last[0]) if last else None
        data_info = (last_run or {}).get("data") or {}
        last_bar_info = _last_finalized_bar(conn, user)
        last_bar = last_bar_info["bar_ts"] if last_bar_info else None

        # DEMO mode lets a caller drive an arbitrary injected clock (demo_now)
        # so a walkthrough isn't tied to the real date/session window; reading
        # this status back must use THAT same clock, not real wall-clock time,
        # or a demo run's counters/holding-time would land in the wrong
        # session-date bucket / show a nonsense elapsed time. LIVE mode never
        # injects a clock, so effective_now is always real "now" there.
        if demo and last_run:
            now_ist = datetime.fromisoformat(last_run["run_at"])
            if now_ist.tzinfo is None:
                now_ist = now_ist.replace(tzinfo=IST)
        else:
            now_ist = _now_ist()
        session_date = (last_run or {}).get("session_date") or _session_date_str(now_ist)
        toggle = _get_toggle(conn, user)
        state = _get_or_init_state(conn, user, session_date)

        last_price_cache = _demo_last_known_price if demo else _last_known_price
        open_rows = conn.execute(
            "SELECT ix.id, ix.symbol, ix.actual_entry_price, ix.quantity, ix.execution_timestamp, "
            "isg.stop_loss, isg.target, isg.id "
            "FROM intraday_executions ix JOIN intraday_signals isg ON isg.id = ix.signal_id "
            "WHERE ix.username=? AND ix.closed_at IS NULL ORDER BY ix.id DESC",
            (user,),
        ).fetchall()
        open_positions = []
        for eid, sym, entry_px, qty, exec_ts, sl, tp, sid in open_rows:
            cur = last_price_cache.get(sym)
            entered_at = datetime.fromisoformat(exec_ts)
            if entered_at.tzinfo is None:
                entered_at = entered_at.replace(tzinfo=IST)
            open_positions.append({
                "execution_id": eid, "signal_id": sid, "symbol": sym, "direction": "BUY",
                "actual_entry": entry_px, "current_price": cur, "stop_loss": sl, "target": tp,
                "unrealized_pnl": round((cur - entry_px) * qty, 2) if cur is not None else None,
                "qty": qty, "executed_at": exec_ts,
                "holding_minutes": round((now_ist - entered_at).total_seconds() / 60.0, 1),
            })

        recent = conn.execute(
            "SELECT id, symbol, direction, status, signal_timestamp, suggested_entry, rejection_reason "
            "FROM intraday_signals WHERE username=? ORDER BY id DESC LIMIT 10", (user,),
        ).fetchall()
        recent_signals = [
            {"id": r[0], "symbol": r[1], "direction": r[2], "status": r[3], "signal_timestamp": r[4],
             "suggested_entry": r[5], "rejection_reason": r[6]}
            for r in recent
        ]
        signals_today = conn.execute(
            "SELECT COUNT(*) FROM intraday_signals WHERE username=? AND signal_timestamp LIKE ?",
            (user, f"{session_date}%"),
        ).fetchone()[0]

        return {
            "mode": "demo" if demo else "live",
            "market_open": _live_market_open(),
            "timeframe": INTRADAY_TIMEFRAME,
            "monitoring_active": bool(toggle),
            "in_session": _session_active(now_ist), "in_entry_window": _in_entry_window(now_ist),
            "start_time": INTRADAY_START_TIME, "entry_cutoff": INTRADAY_ENTRY_CUTOFF, "square_off_time": INTRADAY_SQUARE_OFF_TIME,
            "symbols_monitored": len(INTRADAY_UNIVERSE),
            "data_source": data_info.get("source"), "last_data_ts": data_info.get("as_of"), "data_stale": data_info.get("stale", False),
            "data_stale_symbols": data_info.get("stale_symbols", []),
            "data_future_symbols": data_info.get("future_symbols", []),
            "data_missing_symbols": data_info.get("missing_symbols", []),
            "last_bar": last_bar,
            "last_bar_info": last_bar_info,  # None = no bar has ever been finalized for this account
            "signals_today": signals_today,
            "trades_today": state["trades_today"],
            "max_trades_per_day": INTRADAY_MAX_TRADES_PER_DAY,
            "open_positions": open_positions,
            "max_open_positions": INTRADAY_MAX_OPEN_POSITIONS,
            "daily_pnl": round(state["realized_pnl_today"], 2),
            "max_daily_loss": INTRADAY_MAX_DAILY_LOSS,
            "kill_switch": bool(state["kill_switch"]),
            "square_off_done": bool(state["square_off_done"]),
            "recent_signals": recent_signals,
            "last_error": state["last_error"],
        }
    finally:
        conn.close()
