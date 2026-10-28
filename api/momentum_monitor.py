# api/momentum_monitor.py
"""Continuous-monitoring prototype for the momentum strategy: the single
missing piece core/MOMENTUM_STRATEGY.md section 9 called out ("nothing
calls snapshot/execute on a schedule ... a human ... has to trigger
end-of-day snapshotting and next-session execution").

This module adds ONE thing on top of the existing, unmodified pieces: a
monitoring CYCLE that runs snapshot -> execute -> position-check -> close
in a single call, reusing every existing building block rather than
duplicating any of them:

  - core.strategies.momentum_strategy.screen_universe_now (unmodified) for
    the live cross-sectional screen -- the SAME function GET
    /api/momentum/screen and POST /api/momentum/paper/snapshot already use.
  - api.momentum_paper._record_recommendation / _execute_recommendation_row
    (extracted from that module's own endpoints during this change, so the
    single-symbol manual endpoints and this cycle share one code path --
    they cannot silently diverge in what counts as a recommendation or a
    fill).
  - api.paper._fill / _account (unmodified) for every actual paper
    ledger write -- entries AND closes both go through the SAME function
    every manual trade in the app already uses.
  - api.live.ltp_map / market_open (unmodified) for live Groww prices.

WHAT'S NEW HERE (the actual gap this closes):
  1. One endpoint that runs a full cycle instead of requiring a human to
     call /snapshot then /execute by hand for every symbol.
  2. Automatic CLOSING of a momentum-originated position on target/stop/
     expiry -- api/paper.py already computed stop_hit/target_hit as
     read-only flags (see PaperPage's "⚠stop"/"✓target" markers) but never
     acted on them; this module is what actually closes the position.
  3. A "finalized daily bar" guard (see `_finalized_bar_ok`) that refuses
     to generate a NEW recommendation while the market is open, because
     core/stock_explorer.py::load_data deliberately keeps today's
     in-progress session in the daily series during market hours (its own
     docstring: "keeps today's (partial, ~15-min-delayed) candle during
     market hours") -- screen_universe_now anchors on the last date with
     non-NaN data, which does NOT by itself exclude a partial-but-non-NaN
     in-progress candle. This module adds that exclusion.
  4. A DEMO mode, fully isolated from real paper-trading state (see below).

WHAT IS REAL vs WHAT IS DEMO/SIMULATED
-----------------------------------------
- **Live mode** (`demo=False`, the default): real momentum data, real Groww
  LTPs, real fills through the SAME api/paper.py ledger every manual trade
  uses, under the caller's own paper-account username. This is genuine
  forward, point-in-time paper trading -- no real money, no real broker
  orders (this module never calls anything in api/live.py's order-placement
  surface; there isn't one -- Groww is read-only quotes here).
- **Demo mode** (`demo=True`): recommendations still come from REAL momentum
  data (there is nothing to fake there), but execution and position-
  monitoring prices are supplied EXPLICITLY by the caller (`demo_prices`),
  not fetched from Groww -- so a full open -> target/stop -> close lifecycle
  can be demonstrated on demand without waiting for the real market to move.
  Demo runs are written under a SEPARATE, fixed username (`DEMO_USER`,
  "demo-monitor") -- a completely different row-scope in every table this
  module and api/paper.py touch (paper_account/paper_positions/paper_trades
  are all keyed by username) -- so demo activity can NEVER merge with, or
  corrupt the numbers of, a real user's paper account. `POST /api/paper/reset`
  (existing, unmodified) already works per-username, so wiping demo state is
  `POST /api/paper/reset` while authenticated as the demo user, or simply
  ignoring it (it never affects a real account's equity/stats).
- **Automatic scheduling** (added on top of the manually-triggered prototype
  above): a daily in-process scheduler, following the SAME background-thread
  pattern api/paper.py's `start_executor` already uses for the paper-order
  auto-executor (a daemon thread woken by `time.sleep`, not a new
  framework), just at a once-a-day cadence instead of an intraday one. It
  wakes every MOMENTUM_MONITOR_POLL_SECONDS (default 300s -- a coarse clock
  check, NOT the monitoring cadence itself) and, once past
  MOMENTUM_MONITOR_TIME in Asia/Kolkata (default 15:45, chosen to run after
  NSE's 15:30 close with a buffer for the day's daily bar to finalize -- see
  `_finalized_bar_ok`), fires ONE call to `run_monitor_cycle`, the SAME
  function POST /api/momentum/monitor/run calls -- there is exactly one
  monitoring-cycle entry point, never two. Whether it already fired for
  today's Asia/Kolkata calendar date is tracked in a `momentum_monitor_schedule`
  row in this same sqlite file (not in memory), so a process restart can
  never cause a same-day double-fire, and a failed cycle is simply retried
  on the next tick (safe: `run_cycle`'s own idempotency makes retrying an
  already-completed cycle a no-op -- see Idempotency below). Started from
  api/main.py's startup event; set MOMENTUM_MONITOR_ENABLED=0 to disable it.
  An external systemd timer / cron calling `python -m api.momentum_monitor`
  (this module's own `__main__` block) remains available too, matching this
  repo's existing convention for api/ledger.py / core/audit.py -- pick ONE
  of the two mechanisms in production; running both is harmless (idempotent)
  but redundant.

Idempotency: entirely inherited from the existing status machine, not a new
mechanism. A recommendation's natural key is (username, symbol, strategy,
as_of_date) -- already enforced by `_record_recommendation`. A
recommendation can only be executed once (`status` must be 'pending') and
only closed once (`status` must be 'executed') -- so calling this cycle
twice in a row is always safe: the second call creates nothing new, fills
nothing twice, and closes nothing twice.
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

from .paper import _account as _paper_account, _fill as _paper_fill, _user as _paper_user
from .live import ltp_map as _live_ltp_map, market_open as _live_market_open
from .momentum_paper import (
    _db as _mp_db,
    _record_recommendation,
    _execute_recommendation_row,
    _row_to_dict,
    _REC_COLS,
    MomentumConfig,
    screen_universe_now as _run_screen,
    _MOMENTUM_CACHE_DIR,
    _MOMENTUM_LOOKBACK_BARS,
    _MOMENTUM_UNIVERSE,
)

router = APIRouter(prefix="/api/momentum/monitor", tags=["momentum-monitor"])

DEMO_USER = "demo-monitor"

# Prototype MONITORING-LAYER policy constants -- NOT strategy parameters.
# Neither of these lives in core/strategies/momentum_strategy.py::MomentumConfig
# and neither is read from or fitted on any backtest; they are simple,
# documented defaults for this application-layer prototype only.
MAX_HOLD_DAYS = 30       # calendar days after which an open position is closed as EXPIRY
RISK_PCT_PER_TRADE = 0.005   # fraction of current paper cash risked per new position

# ---- automatic scheduling (application-layer policy, env-overridable; kept
# in ONE place rather than scattered through the module) --------------------
IST = ZoneInfo("Asia/Kolkata")


def _parse_hhmm(s: str) -> _dtime:
    try:
        hh, mm = s.strip().split(":")
        return _dtime(int(hh), int(mm))
    except Exception:
        raise ValueError(f"MOMENTUM_MONITOR_TIME must be HH:MM (24h, Asia/Kolkata), got {s!r}") from None


MONITOR_SCHEDULE_ENABLED = os.environ.get("MOMENTUM_MONITOR_ENABLED", "1").strip().lower() not in ("0", "false", "no")
MONITOR_SCHEDULE_TIME = os.environ.get("MOMENTUM_MONITOR_TIME", "15:45")
# 15:45 IST default: NSE cash-market close is 15:30 IST; the 15-minute buffer
# gives the day's daily bar (core/stock_explorer.py::load_data) time to
# finalize before _finalized_bar_ok treats it as tradeable.
MONITOR_SCHEDULE_TIME_OBJ = _parse_hhmm(MONITOR_SCHEDULE_TIME)
MONITOR_SCHEDULE_USER = os.environ.get("MOMENTUM_MONITOR_USER", "local")
# The paper account the unattended (no-HTTP-request) scheduled cycle runs
# as. "local" matches api/paper.py::_user's own fallback when ULTRABULL_AUTH
# is off (the default for local/dev use) -- so the scheduled cycle lands in
# the SAME account the UI shows without needing to log in as anyone.
MONITOR_POLL_SECONDS = float(os.environ.get("MOMENTUM_MONITOR_POLL_SECONDS", "300"))
# How often the scheduler thread wakes to check the clock -- NOT the
# monitoring cadence, which stays once/day. 300s keeps this a coarse clock
# check, not an intraday polling loop.


def _db() -> sqlite3.Connection:
    conn = _mp_db()  # same sqlite file; momentum_recommendations is ensured there
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS momentum_monitor_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT NOT NULL,
          mode TEXT NOT NULL,
          run_at TEXT NOT NULL,
          result_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS momentum_monitor_schedule (
          username TEXT PRIMARY KEY,
          last_fired_date TEXT,
          last_fired_at TEXT,
          last_error TEXT
        );
        """
    )
    return conn


def _now_ist() -> datetime:
    return datetime.now(IST)


def _get_schedule_state(conn: sqlite3.Connection, user: str) -> dict | None:
    row = conn.execute(
        "SELECT last_fired_date, last_fired_at, last_error FROM momentum_monitor_schedule WHERE username=?",
        (user,),
    ).fetchone()
    if row is None:
        return None
    return {"last_fired_date": row[0], "last_fired_at": row[1], "last_error": row[2]}


def _set_schedule_fired(conn: sqlite3.Connection, user: str, fired_date: str, fired_at: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO momentum_monitor_schedule (username, last_fired_date, last_fired_at, last_error) "
            "VALUES (?,?,?,NULL) ON CONFLICT(username) DO UPDATE SET "
            "last_fired_date=excluded.last_fired_date, last_fired_at=excluded.last_fired_at, last_error=NULL",
            (user, fired_date, fired_at),
        )


def _set_schedule_error(conn: sqlite3.Connection, user: str, error_text: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO momentum_monitor_schedule (username, last_fired_date, last_fired_at, last_error) "
            "VALUES (?,NULL,NULL,?) ON CONFLICT(username) DO UPDATE SET last_error=excluded.last_error",
            (user, error_text),
        )


def _should_fire_now(now_ist: datetime, last_fired_date: str | None) -> bool:
    """Is the scheduler due to fire, given the Asia/Kolkata calendar date it
    last fired for? Compares aware datetimes (instant-based), so it is
    correct regardless of what timezone `now_ist` was originally constructed
    in -- there is no naive-local-time ambiguity. Once true for a given
    calendar date it stays true until a fire is recorded, so a process that
    was down through the scheduled time fires on its first tick after
    restart (catch-up), rather than waiting for the next day."""
    today_str = now_ist.astimezone(IST).date().isoformat()
    if last_fired_date == today_str:
        return False
    today_slot = datetime.combine(now_ist.astimezone(IST).date(), MONITOR_SCHEDULE_TIME_OBJ, tzinfo=IST)
    return now_ist >= today_slot


def _next_run_at(now_ist: datetime, last_fired_date: str | None) -> datetime:
    """Next scheduled Asia/Kolkata slot, for display only (see
    `_should_fire_now` for the actual firing decision)."""
    today = now_ist.astimezone(IST).date()
    today_slot = datetime.combine(today, MONITOR_SCHEDULE_TIME_OBJ, tzinfo=IST)
    if last_fired_date != today.isoformat() and now_ist < today_slot:
        return today_slot
    return datetime.combine(today + timedelta(days=1), MONITOR_SCHEDULE_TIME_OBJ, tzinfo=IST)


def _scheduler_status(conn: sqlite3.Connection) -> dict:
    """Read-only scheduler snapshot for the UI (folded into GET .../status)."""
    now_ist = _now_ist()
    state = _get_schedule_state(conn, MONITOR_SCHEDULE_USER)
    last_fired_date = state["last_fired_date"] if state else None
    return {
        "enabled": MONITOR_SCHEDULE_ENABLED,
        "timezone": "Asia/Kolkata",
        "schedule_time": MONITOR_SCHEDULE_TIME,
        "scheduler_user": MONITOR_SCHEDULE_USER,
        "last_scheduled_run_at": state["last_fired_at"] if state else None,
        "next_run_at": (
            _next_run_at(now_ist, last_fired_date).isoformat(timespec="minutes")
            if MONITOR_SCHEDULE_ENABLED else None
        ),
        "last_error": state["last_error"] if state else None,
    }


def _monitor_user(request: Request, demo: bool) -> str:
    return DEMO_USER if demo else _paper_user(request)


def _size_position(cash: float, entry: float, stop: float | None) -> int:
    """Whole shares affordable while risking at most RISK_PCT_PER_TRADE of
    current paper cash against the recommendation's own suggested stop
    distance -- a simple, documented prototype sizing rule (not the
    backtest's ATR-based sizing in core/sim/engine.py, and not claimed to
    be). Returns 0 (never guesses) when the stop is missing or entry<=0."""
    if stop is None or entry is None or entry <= 0:
        return 0
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    risk_amount = cash * RISK_PCT_PER_TRADE
    qty = int(risk_amount // risk_per_share)
    max_affordable = int(cash // entry)
    return max(0, min(qty, max_affordable))


def _finalized_bar_ok(as_of_date: str, demo: bool) -> tuple[bool, str | None]:
    """Refuse to treat today's session as a finalized daily bar while the
    market is open (see module docstring, point 3). Demo mode is exempt --
    it never claims to be a real EOD signal."""
    if demo:
        return True, None
    if _live_market_open():
        today = datetime.now().date().isoformat()
        if as_of_date >= today:
            return False, (
                f"Market is currently open and the latest available daily bar ({as_of_date}) may be a "
                "partially-formed, in-progress session -- refusing to generate new recommendations from it "
                "(daily mode only trades finalized bars). Run again after market close, or use demo mode "
                "to see the full cycle without this guard."
            )
    return True, None


class MonitorRunReq(BaseModel):
    lookback: str = "6M"
    years: float = 2.0
    top_n: int = 5
    use_relative: bool = True
    use_technical: bool = True
    use_market_regime: bool = True
    symbols: str | None = None  # comma-separated universe override; default the shared UNIVERSE
    demo: bool = False
    demo_prices: dict[str, float] | None = None  # yahoo_symbol -> price; DEMO MODE ONLY, never used in live mode
    demo_max_hold_days: int | None = None         # DEMO MODE ONLY override of MAX_HOLD_DAYS, for a fast expiry demo


def run_cycle(
    user: str, demo: bool, lookback: str, years: float, top_n: int,
    use_relative: bool, use_technical: bool, use_market_regime: bool,
    symbols: list[str] | None, demo_prices: dict[str, float] | None,
    max_hold_days: int,
) -> dict:
    """One monitoring cycle: screen -> record recommendations -> execute
    pending ones -> check/close open positions. See module docstring for
    what's real vs demo."""
    conn = _db()
    try:
        universe = symbols or list(_MOMENTUM_UNIVERSE)
        cfg = MomentumConfig(
            lookback_label=lookback, top_n=top_n, use_relative_momentum=use_relative,
            use_technical_confirmation=use_technical, use_market_regime=use_market_regime,
        )

        # ---- steps 1-8: screen the universe, record new recommendations ----
        gen = {"skipped": False, "reason": None, "as_of_date": None,
               "screened": 0, "candidates": 0, "created": [], "already_existed": []}
        screen = _run_screen(universe, years, cfg, cache_dir=_MOMENTUM_CACHE_DIR)
        if screen["as_of"] is None:
            gen["skipped"] = True
            gen["reason"] = "No valid momentum data for this universe/lookback right now."
        else:
            as_of_date = str(screen["as_of"].date())
            gen["as_of_date"] = as_of_date
            gen["screened"] = screen["n_symbols"]
            ok, block_reason = _finalized_bar_ok(as_of_date, demo)
            if not ok:
                gen["skipped"] = True
                gen["reason"] = block_reason
            else:
                config_dict = {"top_n": top_n, "use_relative": use_relative,
                                "use_technical": use_technical, "use_market_regime": use_market_regime}
                for row in screen["rows"]:
                    if row["signal"] != "long":
                        continue
                    gen["candidates"] += 1
                    res = _record_recommendation(conn, user, row["symbol"], cfg, lookback, row, as_of_date, config_dict)
                    bucket = gen["already_existed"] if res["already_existed"] else gen["created"]
                    bucket.append({"id": res["id"], "symbol": row["symbol"], "suggested_entry": res["suggested_entry"]})

        # ---- step 8b: execute pending recommendations for this user --------
        exe = {"attempted": 0, "filled": [], "skipped_market_closed": [], "skipped_insufficient_info": []}
        pending = conn.execute(
            f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE username=? AND status='pending'",
            (user,),
        ).fetchall()
        for rec in pending:
            rec_d = _row_to_dict(rec, _REC_COLS)
            exe["attempted"] += 1

            if demo:
                price = (demo_prices or {}).get(rec_d["yahoo_symbol"], rec_d["suggested_entry"])
            else:
                if not _live_market_open():
                    exe["skipped_market_closed"].append({"id": rec_d["id"], "symbol": rec_d["symbol"]})
                    continue
                price = _live_ltp_map([rec_d["yahoo_symbol"]]).get(rec_d["yahoo_symbol"])
                if price is None:
                    exe["skipped_market_closed"].append(
                        {"id": rec_d["id"], "symbol": rec_d["symbol"], "note": "no live price available"})
                    continue

            acct = _paper_account(conn, user)
            qty = _size_position(acct["cash"], price, rec_d["sl_est"])
            if qty <= 0:
                exe["skipped_insufficient_info"].append({"id": rec_d["id"], "symbol": rec_d["symbol"]})
                continue

            note = ("DEMO monitor cycle auto-execute" if demo else "momentum monitor cycle auto-execute")
            try:
                fill = _execute_recommendation_row(conn, user, rec_d["id"], rec_d, price, qty, note)
            except (ValueError, RuntimeError) as e:
                exe["skipped_insufficient_info"].append({"id": rec_d["id"], "symbol": rec_d["symbol"], "error": str(e)})
                continue
            exe["filled"].append({"id": rec_d["id"], "symbol": rec_d["symbol"], **fill})

        # ---- steps 9-13: monitor + close open (executed) positions ---------
        mon = {"checked": 0, "closed": [], "still_open": []}
        open_recs = conn.execute(
            f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE username=? AND status='executed'",
            (user,),
        ).fetchall()
        for rec in open_recs:
            rec_d = _row_to_dict(rec, _REC_COLS)
            mon["checked"] += 1

            pos = conn.execute(
                "SELECT qty FROM paper_positions WHERE username=? AND symbol=?", (user, rec_d["symbol"]),
            ).fetchone()
            if pos is None:
                # Closed outside this cycle (e.g. a manual Sell) -- reconcile
                # instead of re-checking it forever or crashing.
                with conn:
                    conn.execute(
                        "UPDATE momentum_recommendations SET status='closed', closed_at=?, exit_reason='closed_externally' "
                        "WHERE id=?", (datetime.now().isoformat(timespec="seconds"), rec_d["id"]),
                    )
                mon["closed"].append({"id": rec_d["id"], "symbol": rec_d["symbol"], "reason": "closed_externally"})
                continue
            qty = pos[0]

            if demo:
                price = (demo_prices or {}).get(rec_d["yahoo_symbol"])
                if price is None:
                    mon["still_open"].append({"id": rec_d["id"], "symbol": rec_d["symbol"],
                                               "note": "no demo price supplied this cycle"})
                    continue
            else:
                price = _live_ltp_map([rec_d["yahoo_symbol"]]).get(rec_d["yahoo_symbol"])
                if price is None:
                    mon["still_open"].append({"id": rec_d["id"], "symbol": rec_d["symbol"],
                                               "note": "no live price available"})
                    continue

            days_held = (
                (datetime.now() - datetime.fromisoformat(rec_d["executed_at"])).days
                if rec_d["executed_at"] else 0
            )
            reason = None
            # Same tie-break convention as core/sim/engine.py: a stop hit
            # wins over a simultaneous target hit (documented there as
            # "conservative"); checked first here for the same reason.
            if rec_d["sl_est"] is not None and price <= rec_d["sl_est"]:
                reason = "STOP"
            elif rec_d["tp_est"] is not None and price >= rec_d["tp_est"]:
                reason = "TARGET"
            elif days_held >= max_hold_days:
                reason = "EXPIRY"

            if reason is None:
                mon["still_open"].append({
                    "id": rec_d["id"], "symbol": rec_d["symbol"], "current_price": price,
                    "entry_price": rec_d["actual_entry"], "qty": qty,
                    "unrealized_pnl": round((price - rec_d["actual_entry"]) * qty, 2) if rec_d["actual_entry"] else None,
                    "days_held": days_held,
                })
                continue

            note = ("DEMO " if demo else "") + f"momentum monitor close: {reason} (rec #{rec_d['id']})"
            try:
                realized = _paper_fill(conn, user, rec_d["symbol"], rec_d["yahoo_symbol"], "SELL", qty, price, None, None, note)
            except ValueError as e:
                mon["still_open"].append({"id": rec_d["id"], "symbol": rec_d["symbol"], "error": str(e)})
                continue
            closed_at = datetime.now().isoformat(timespec="seconds")
            with conn:
                conn.execute(
                    "UPDATE momentum_recommendations SET status='closed', closed_at=?, exit_price=?, "
                    "exit_reason=?, close_realized_pnl=? WHERE id=?",
                    (closed_at, price, reason, realized, rec_d["id"]),
                )
            entry = rec_d["actual_entry"]
            ret_pct = round(100.0 * (price - entry) / entry, 2) if entry else None
            mon["closed"].append({
                "id": rec_d["id"], "symbol": rec_d["symbol"], "reason": reason, "exit_price": price,
                "realized_pnl": realized, "return_pct": ret_pct, "days_held": days_held, "closed_at": closed_at,
            })

        run_at = datetime.now().isoformat(timespec="seconds")
        result = {
            "ok": True, "run_at": run_at, "mode": "demo" if demo else "live",
            "market_open": _live_market_open(), "lookback": lookback,
            "recommendation_generation": gen, "executions": exe, "position_monitoring": mon,
            "summary": (
                f"{'[DEMO] ' if demo else ''}{len(gen['created'])} new recommendation(s), "
                f"{len(exe['filled'])} executed, {len(mon['closed'])} position(s) closed, "
                f"{len(mon['still_open'])} still open."
            ),
        }
        with conn:
            conn.execute(
                "INSERT INTO momentum_monitor_runs (username, mode, run_at, result_json) VALUES (?,?,?,?)",
                (user, result["mode"], run_at, json.dumps(result)),
            )
        return result
    finally:
        conn.close()


def run_monitor_cycle(req: MonitorRunReq, user: str) -> dict:
    """THE single entry point for a monitoring cycle -- both POST
    /api/momentum/monitor/run (manual) and the automatic scheduler
    (`_scheduler_tick`) call this and only this. It validates + resolves a
    MonitorRunReq exactly the same way regardless of trigger source, then
    delegates to `run_cycle` (the existing, unmodified screen -> record ->
    execute -> monitor -> close implementation) -- so the two trigger paths
    can never quietly diverge in what a 'run' does."""
    if req.lookback not in _MOMENTUM_LOOKBACK_BARS:
        raise HTTPException(400, f"lookback must be one of {list(_MOMENTUM_LOOKBACK_BARS)}")
    if not req.demo and (req.demo_prices or req.demo_max_hold_days is not None):
        raise HTTPException(422, "demo_prices/demo_max_hold_days require demo=true")
    symbols = [s.strip().upper() for s in req.symbols.split(",") if s.strip()] if req.symbols else None
    max_hold = req.demo_max_hold_days if (req.demo and req.demo_max_hold_days is not None) else MAX_HOLD_DAYS
    return run_cycle(
        user, req.demo, req.lookback, req.years, req.top_n, req.use_relative,
        req.use_technical, req.use_market_regime, symbols, req.demo_prices, max_hold,
    )


@router.post("/run")
def monitor_run(req: MonitorRunReq, request: Request):
    """MANUAL trigger: execute ONE monitoring cycle (screen -> record ->
    execute -> monitor -> close) and return its result. Safe to call
    repeatedly (see module docstring on idempotency) and safe to call
    whether the market is open or closed -- what it actually does adapts,
    and is reported back. Useful for development/demo/testing even once the
    scheduler below is running -- both paths call `run_monitor_cycle`."""
    user = _monitor_user(request, req.demo)
    return run_monitor_cycle(req, user)


# ---------------------------------------------------------------------------
# AUTOMATIC trigger: a daily in-process scheduler. See module docstring for
# how this fits the rest of the app and why (background-thread pattern
# already used by api/paper.py's start_executor; restart-safe via the
# momentum_monitor_schedule table, not in-memory state).
# ---------------------------------------------------------------------------

def _scheduler_tick(now: datetime | None = None) -> None:
    """One scheduler check: fire the daily automatic cycle if due. Any
    exception from the cycle itself is caught and recorded as `last_error`,
    never propagated -- a failed cycle (a transient data-source error, say)
    must not kill the background thread, and must not be recorded as
    'fired' so the next tick retries it. `run_cycle`'s own idempotency
    (natural key = username, symbol, strategy, as_of_date; status-gated
    execute/close) makes any such retry -- or a same-day double-tick, or a
    process restart mid-day -- safe: nothing is created, filled, or closed
    twice. `now` is injectable for tests; production calls always pass the
    real Asia/Kolkata clock."""
    now_ist = (now or _now_ist()).astimezone(IST)
    conn = _db()
    try:
        state = _get_schedule_state(conn, MONITOR_SCHEDULE_USER)
        last_fired_date = state["last_fired_date"] if state else None
        if not _should_fire_now(now_ist, last_fired_date):
            return
        try:
            run_monitor_cycle(MonitorRunReq(), MONITOR_SCHEDULE_USER)
            _set_schedule_fired(
                conn, MONITOR_SCHEDULE_USER,
                now_ist.date().isoformat(), now_ist.isoformat(timespec="seconds"),
            )
        except Exception as e:
            _set_schedule_error(conn, MONITOR_SCHEDULE_USER, str(e))
    finally:
        conn.close()


_scheduler_started = False


def start_monitor_scheduler() -> None:
    """Start the daily automatic-monitoring background thread once. Mirrors
    api/paper.py's start_executor -- same in-process daemon-thread pattern
    already used in this app, just woken every MONITOR_POLL_SECONDS (a
    coarse clock check) instead of every ~12s, because this is a once-a-day
    cycle, not an intraday one. No-ops if MOMENTUM_MONITOR_ENABLED=0."""
    global _scheduler_started
    if _scheduler_started or not MONITOR_SCHEDULE_ENABLED:
        return
    _scheduler_started = True

    def _loop():
        while True:
            try:
                _scheduler_tick()
            except Exception:
                pass
            _time_mod.sleep(MONITOR_POLL_SECONDS)

    threading.Thread(target=_loop, daemon=True, name="momentum-monitor-scheduler").start()


@router.get("/status")
def monitor_status(request: Request, demo: bool = False):
    """Read-only snapshot for the UI: scheduler state (enabled/disabled,
    last + next run), last run, latest recommendations, currently open
    positions (enriched with live price/unrealized P&L where available),
    and recent outcomes. Never triggers a cycle itself."""
    user = _monitor_user(request, demo)
    conn = _db()
    try:
        last = conn.execute(
            "SELECT result_json FROM momentum_monitor_runs WHERE username=? ORDER BY id DESC LIMIT 1", (user,),
        ).fetchone()
        last_run = json.loads(last[0]) if last else None

        recs = conn.execute(
            f"SELECT {','.join(_REC_COLS)} FROM momentum_recommendations WHERE username=? ORDER BY id DESC LIMIT 20",
            (user,),
        ).fetchall()
        all_recs = [_row_to_dict(r, _REC_COLS) for r in recs]
        today_ist = _now_ist().date().isoformat()
        recommendations_today = sum(1 for r in all_recs if r["as_of_date"] == today_ist)

        open_positions = []
        for r in all_recs:
            if r["status"] != "executed":
                continue
            pos = conn.execute(
                "SELECT qty, stop, target FROM paper_positions WHERE username=? AND symbol=?",
                (user, r["symbol"]),
            ).fetchone()
            if pos is None:
                continue
            qty, stop, target = pos
            ltp = None if demo else _live_ltp_map([r["yahoo_symbol"]]).get(r["yahoo_symbol"])
            open_positions.append({
                "recommendation_id": r["id"], "symbol": r["symbol"], "qty": qty,
                "suggested_entry": r["suggested_entry"], "actual_entry": r["actual_entry"],
                "current_price": ltp,
                "unrealized_pnl": round((ltp - r["actual_entry"]) * qty, 2) if (ltp is not None and r["actual_entry"]) else None,
                "stop": stop, "target": target, "executed_at": r["executed_at"],
            })

        return {
            "mode": "demo" if demo else "live",
            "market_open": _live_market_open(),
            "scheduler": _scheduler_status(conn),
            "last_run": last_run,
            "latest_recommendations": all_recs[:5],
            "open_positions": open_positions,
            "recent_outcomes": [r for r in all_recs if r["status"] == "closed"][:5],
            "recommendations_today": recommendations_today,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    # Alternative to the in-process scheduler above: an external systemd
    # timer / cron can call this directly instead --
    #   python -m api.momentum_monitor
    # -- matching this repo's existing convention for api/ledger.py
    # ("python -m api.ledger poll") and core/audit.py. Runs ONE cycle for
    # MONITOR_SCHEDULE_USER and exits; pick ONE of the two mechanisms in
    # production (running both is harmless -- idempotent -- but redundant).
    import sys as _sys
    print(f"running momentum monitor cycle (user={MONITOR_SCHEDULE_USER!r}) ...")
    _result = run_monitor_cycle(MonitorRunReq(), MONITOR_SCHEDULE_USER)
    print(_result["summary"])
    _sys.exit(0 if _result.get("ok") else 1)
