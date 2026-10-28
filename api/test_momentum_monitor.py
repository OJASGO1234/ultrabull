# api/test_momentum_monitor.py
"""Tests for api/momentum_monitor.py (the continuous-monitoring prototype)
and the api/momentum_paper.py helpers it reuses. Fully offline: the momentum
screen, Groww live prices, and market-hours check are all monkeypatched to
canned values -- no network calls, no real broker interaction, and the
sqlite paper-trading db is redirected to a temp file so these tests can
NEVER touch the real data_cache/paper.db a real user's account lives in.

Run with pytest: pytest api/test_momentum_monitor.py
Run standalone:  python api/test_momentum_monitor.py
"""
from __future__ import annotations
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import pandas as pd

import api.paper as paper_mod
import api.momentum_paper as mp_mod
import api.momentum_monitor as mm


# ---------------------------------------------------------------------------
# Isolation: every test gets a fresh temp sqlite file, never the real one.
# ---------------------------------------------------------------------------
def _isolated_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let sqlite create it fresh
    paper_mod.PAPER_DB = path
    return path


def _candidate_row(symbol="TESTCO.NS", entry=100.0, sl=90.0, tp=130.0, rank=1, signal="long"):
    return {
        "symbol": symbol, "rank": rank, "absolute_momentum": 0.12, "relative_momentum": 0.08,
        "technical_confirmed": True, "selected": True, "close": entry,
        "signal": signal, "entry_est": entry, "sl_est": sl, "tp_est": tp,
        "sl_atr": 2.0, "tp_atr": 4.0,
    }


def _fake_screen(as_of: str, rows: list[dict]):
    return {"as_of": pd.Timestamp(as_of), "n_symbols": len(rows), "rows": rows}


def _patch_screen(rows_by_call):
    """rows_by_call: a list of (as_of, rows) tuples; each call to run_cycle
    consumes the next one (so a test can control successive cycles), or a
    single (as_of, rows) tuple reused for every call."""
    calls = {"i": 0}
    if isinstance(rows_by_call, tuple):
        rows_by_call = [rows_by_call] * 100

    def _screen(universe, years, cfg, cache_dir=None):
        as_of, rows = rows_by_call[calls["i"]]
        calls["i"] = min(calls["i"] + 1, len(rows_by_call) - 1)
        return _fake_screen(as_of, rows)

    mm._run_screen = _screen


def _patch_market(open_: bool):
    mm._live_market_open = lambda: open_


def _patch_prices(prices: dict[str, float]):
    mm._live_ltp_map = lambda syms: {s: prices[s] for s in syms if s in prices}


def _run(user="tester", demo=False, **kw):
    kw.setdefault("lookback", "6M")
    kw.setdefault("years", 2.0)
    kw.setdefault("top_n", 5)
    kw.setdefault("use_relative", True)
    kw.setdefault("use_technical", True)
    kw.setdefault("use_market_regime", True)
    kw.setdefault("symbols", ["TESTCO.NS"])
    kw.setdefault("demo_prices", None)
    kw.setdefault("max_hold_days", mm.MAX_HOLD_DAYS)
    return mm.run_cycle(user, demo, kw["lookback"], kw["years"], kw["top_n"], kw["use_relative"],
                         kw["use_technical"], kw["use_market_regime"], kw["symbols"], kw["demo_prices"],
                         kw["max_hold_days"])


# ---------------------------------------------------------------------------
# 1. Monitoring cycle creates a recommendation
# ---------------------------------------------------------------------------
def test_cycle_creates_recommendation():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))
    _patch_market(False)  # market closed -> execution step is skipped, isolates recommendation creation

    res = _run()
    assert res["recommendation_generation"]["skipped"] is False
    created = res["recommendation_generation"]["created"]
    assert len(created) == 1
    assert created[0]["symbol"] == "TESTCO.NS"
    assert created[0]["suggested_entry"] == 100.0

    conn = mm._db()
    row = conn.execute("SELECT suggested_entry, status FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()
    conn.close()
    assert row == (100.0, "pending")


# ---------------------------------------------------------------------------
# 2. Recommendation snapshot is immutable
# ---------------------------------------------------------------------------
def test_recommendation_suggested_entry_is_immutable_after_execution():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=130.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 104.0})  # actual fill differs from suggested_entry

    res = _run()
    filled = res["executions"]["filled"]
    assert len(filled) == 1
    assert filled[0]["suggested_entry"] == 100.0
    assert filled[0]["actual_entry"] == 104.0

    conn = mm._db()
    row = conn.execute("SELECT suggested_entry, actual_entry FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()
    conn.close()
    assert row[0] == 100.0, "suggested_entry must never be overwritten by the actual fill price"
    assert row[1] == 104.0


# ---------------------------------------------------------------------------
# 3 & 8. Duplicate/repeated monitoring cycles do not duplicate anything
# ---------------------------------------------------------------------------
def test_repeated_cycle_same_day_does_not_duplicate_recommendation():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))  # SAME as_of_date every call
    _patch_market(False)

    r1 = _run()
    r2 = _run()
    assert len(r1["recommendation_generation"]["created"]) == 1
    assert len(r2["recommendation_generation"]["created"]) == 0
    assert len(r2["recommendation_generation"]["already_existed"]) == 1

    conn = mm._db()
    n = conn.execute("SELECT COUNT(*) FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()[0]
    conn.close()
    assert n == 1


def test_repeated_cycle_does_not_double_execute():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 101.0})

    r1 = _run()
    assert len(r1["executions"]["filled"]) == 1
    r2 = _run()
    # the same recommendation is no longer 'pending' -> nothing new to execute
    assert len(r2["executions"]["filled"]) == 0
    assert r2["executions"]["attempted"] == 0

    conn = mm._db()
    n_trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE symbol='TESTCO.NS' AND side='BUY'").fetchone()[0]
    conn.close()
    assert n_trades == 1


# ---------------------------------------------------------------------------
# 4. Actual execution price is separate from suggested price (sizing too)
# ---------------------------------------------------------------------------
def test_execution_uses_live_price_not_suggested_price_for_the_fill():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=95.0, tp=120.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 110.0})  # live price has moved well above the suggested entry

    res = _run()
    filled = res["executions"]["filled"][0]
    assert filled["actual_entry"] == 110.0
    assert filled["suggested_entry"] == 100.0
    assert filled["slippage_pct"] == 10.0


# ---------------------------------------------------------------------------
# 5. An open position can be updated (unrealized P&L reflects current price)
# ---------------------------------------------------------------------------
def test_open_position_is_updated_with_current_price_and_unrealized_pnl():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=130.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 100.0})
    r1 = _run()
    qty = r1["executions"]["filled"][0]["qty"]
    assert qty > 0

    _patch_prices({"TESTCO.NS": 105.0})  # price moved up, still inside [sl, tp]
    r2 = _run()
    still_open = r2["position_monitoring"]["still_open"]
    assert len(still_open) == 1
    assert still_open[0]["current_price"] == 105.0
    assert still_open[0]["unrealized_pnl"] == round((105.0 - 100.0) * qty, 2)
    assert len(r2["position_monitoring"]["closed"]) == 0


# ---------------------------------------------------------------------------
# 6. Target/stop closes a position
# ---------------------------------------------------------------------------
def test_target_hit_closes_the_position():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=120.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 100.0})
    r1 = _run()
    assert len(r1["executions"]["filled"]) == 1

    _patch_prices({"TESTCO.NS": 125.0})  # above target
    r2 = _run()
    closed = r2["position_monitoring"]["closed"]
    assert len(closed) == 1
    assert closed[0]["reason"] == "TARGET"
    assert closed[0]["exit_price"] == 125.0

    conn = mm._db()
    pos = conn.execute("SELECT COUNT(*) FROM paper_positions WHERE symbol='TESTCO.NS'").fetchone()[0]
    status = conn.execute("SELECT status, exit_reason FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()
    conn.close()
    assert pos == 0, "the paper position must actually be closed (no shares left)"
    assert status == ("closed", "TARGET")


def test_stop_hit_closes_the_position():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=120.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 100.0})
    _run()
    _patch_prices({"TESTCO.NS": 85.0})  # below stop
    r2 = _run()
    closed = r2["position_monitoring"]["closed"]
    assert len(closed) == 1
    assert closed[0]["reason"] == "STOP"


def test_expiry_closes_a_stale_position():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=1.0, tp=1000.0)]))  # far stop/target -> only expiry can fire
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 100.0})
    r1 = _run()
    assert len(r1["executions"]["filled"]) == 1

    _patch_prices({"TESTCO.NS": 101.0})
    r2 = _run(max_hold_days=0)  # anything held at all is immediately "expired"
    closed = r2["position_monitoring"]["closed"]
    assert len(closed) == 1
    assert closed[0]["reason"] == "EXPIRY"


# ---------------------------------------------------------------------------
# 7. Outcome links back to the recommendation ID
# ---------------------------------------------------------------------------
def test_closed_position_outcome_links_back_to_recommendation_id():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=120.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 100.0})
    r1 = _run()
    rec_id = r1["executions"]["filled"][0]["id"]

    _patch_prices({"TESTCO.NS": 125.0})
    r2 = _run()
    assert r2["position_monitoring"]["closed"][0]["id"] == rec_id

    conn = mm._db()
    row = conn.execute(
        "SELECT paper_trade_id, close_realized_pnl FROM momentum_recommendations WHERE id=?", (rec_id,),
    ).fetchone()
    trade = conn.execute("SELECT symbol, side FROM paper_trades WHERE id=?", (row[0],)).fetchone()
    conn.close()
    assert trade == ("TESTCO.NS", "BUY"), "the recommendation's paper_trade_id must point at its own entry fill"
    assert row[1] is not None  # realized P&L recorded on the recommendation itself


# ---------------------------------------------------------------------------
# 9. No real broker order is sent -- the monitor never touches Groww's order
# surface, only read-only quotes (ltp_map, monkeypatched here) and the local
# paper ledger.
# ---------------------------------------------------------------------------
def test_no_broker_order_function_is_ever_referenced():
    import inspect
    src = inspect.getsource(mm)
    forbidden = ["place_order(", "growwapi", "GrowwAPI(", "get_client().place", "_get_client().place"]
    for token in forbidden:
        assert token not in src, f"momentum_monitor.py must never call a real order-placement API ({token})"
    # and structurally: the only price source is the read-only ltp_map, never a broker client
    assert "_live_ltp_map" in src and "GrowwFeed" not in src


# ---------------------------------------------------------------------------
# Finalized-bar guard: market open blocks NEW recommendations, but existing
# open positions are still monitored.
# ---------------------------------------------------------------------------
def test_market_open_blocks_new_recommendation_generation_in_live_mode():
    _isolated_db()
    today = pd.Timestamp.now().date().isoformat()
    _patch_screen((today, [_candidate_row()]))
    _patch_market(True)

    res = _run()
    assert res["recommendation_generation"]["skipped"] is True
    assert "partially-formed" in res["recommendation_generation"]["reason"]
    conn = mm._db()
    n = conn.execute("SELECT COUNT(*) FROM momentum_recommendations").fetchone()[0]
    conn.close()
    assert n == 0


# ---------------------------------------------------------------------------
# Demo mode is fully isolated from a real (live) user's paper account.
# ---------------------------------------------------------------------------
def test_demo_mode_never_touches_the_real_users_paper_account():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=120.0)]))
    _patch_market(False)  # even with the market closed, demo mode still executes (uses demo_prices)

    res = mm.run_cycle(
        mm.DEMO_USER, True, "6M", 2.0, 5, True, True, True, ["TESTCO.NS"],
        {"TESTCO.NS": 100.0}, mm.MAX_HOLD_DAYS,
    )
    assert len(res["executions"]["filled"]) == 1
    assert res["mode"] == "demo"

    conn = mm._db()
    real_user_trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE username='tester'").fetchone()[0]
    demo_trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE username=?", (mm.DEMO_USER,)).fetchone()[0]
    conn.close()
    assert real_user_trades == 0
    assert demo_trades == 1


# ---------------------------------------------------------------------------
# Automatic scheduling (api/momentum_monitor.py's daily in-process scheduler)
# ---------------------------------------------------------------------------

def _ist(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=mm.IST)


# 1. Scheduler calls the existing monitoring cycle (same recommendation path
#    a manual run would produce).
def test_scheduler_tick_calls_the_existing_monitoring_cycle():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))
    _patch_market(False)

    mm._scheduler_tick(now=_ist(2026, 1, 5, 16, 0))

    conn = mm._db()
    row = conn.execute(
        "SELECT suggested_entry, status FROM momentum_recommendations WHERE symbol='TESTCO.NS' AND username=?",
        (mm.MONITOR_SCHEDULE_USER,),
    ).fetchone()
    conn.close()
    assert row == (100.0, "pending")


# 2. Scheduler does not duplicate a cycle for the same market date, even if
#    ticked again the same day (e.g. two poll wakeups after the slot passed).
def test_scheduler_does_not_duplicate_same_day():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))
    _patch_market(False)

    mm._scheduler_tick(now=_ist(2026, 1, 5, 16, 0))
    mm._scheduler_tick(now=_ist(2026, 1, 5, 16, 5))
    mm._scheduler_tick(now=_ist(2026, 1, 5, 23, 55))

    conn = mm._db()
    n = conn.execute("SELECT COUNT(*) FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()[0]
    state = conn.execute(
        "SELECT last_fired_date FROM momentum_monitor_schedule WHERE username=?", (mm.MONITOR_SCHEDULE_USER,)
    ).fetchone()
    conn.close()
    assert n == 1
    assert state[0] == "2026-01-05"


# 3. Manual endpoint and scheduled execution use the SAME function -- no
#    second implementation of screen/record/execute/monitor anywhere.
def test_manual_and_scheduled_paths_use_the_same_function():
    import inspect
    manual_src = inspect.getsource(mm.monitor_run)
    tick_src = inspect.getsource(mm._scheduler_tick)
    assert "run_monitor_cycle(" in manual_src
    assert "run_monitor_cycle(" in tick_src
    # and run_monitor_cycle itself has exactly one call into the cycle implementation
    rmc_src = inspect.getsource(mm.run_monitor_cycle)
    assert rmc_src.count("run_cycle(") == 1


# 4. A process restart does not cause a duplicate recommendation: scheduler
#    state lives in sqlite (read fresh every tick), never in a Python-level
#    variable, so "restart" is simply another tick against the same db.
def test_scheduler_restart_does_not_duplicate_recommendation():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))
    _patch_market(False)

    mm._scheduler_tick(now=_ist(2026, 1, 5, 16, 0))  # "first boot", fires
    # simulate the process restarting: close/reopen a fresh connection and
    # tick again later the same IST calendar date
    mm._scheduler_tick(now=_ist(2026, 1, 5, 18, 0))

    conn = mm._db()
    n = conn.execute("SELECT COUNT(*) FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()[0]
    conn.close()
    assert n == 1, "restarting mid-day and ticking again must not duplicate today's recommendation"

    # a genuinely NEW trading day is a distinct as_of_date -- that's a new,
    # legitimate recommendation, not a duplicate of the first one
    _patch_screen(("2026-01-06", [_candidate_row()]))
    mm._scheduler_tick(now=_ist(2026, 1, 6, 16, 0))
    conn = mm._db()
    n2 = conn.execute("SELECT COUNT(*) FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchone()[0]
    dates = {r[0] for r in conn.execute("SELECT as_of_date FROM momentum_recommendations WHERE symbol='TESTCO.NS'").fetchall()}
    conn.close()
    assert n2 == 2, "the scheduler must not be stuck forever -- a new trading day fires a new cycle"
    assert dates == {"2026-01-05", "2026-01-06"}


# 5. Monitoring remains paper-only: the scheduler's own code never touches a
#    real broker order-placement surface (mirrors test_no_broker_order_function_is_ever_referenced).
def test_scheduler_never_places_a_real_broker_order():
    import inspect
    src = inspect.getsource(mm._scheduler_tick) + inspect.getsource(mm.start_monitor_scheduler)
    forbidden = ["place_order(", "growwapi", "GrowwAPI(", "get_client().place", "_get_client().place"]
    for token in forbidden:
        assert token not in src, f"scheduler must never call a real order-placement API ({token})"


# 6. Timezone handling is correct for Asia/Kolkata: instant-based comparison,
#    not a naive wall-clock string match against the server's own local time.
def test_should_fire_now_respects_ist_schedule_time():
    slot = mm.MONITOR_SCHEDULE_TIME_OBJ
    before = _ist(2026, 1, 5, slot.hour, max(0, slot.minute - 1))
    at = _ist(2026, 1, 5, slot.hour, slot.minute)
    assert mm._should_fire_now(before, None) is False
    assert mm._should_fire_now(at, None) is True
    assert mm._should_fire_now(at, "2026-01-05") is False  # already fired today


def test_should_fire_now_uses_ist_not_naive_local_time():
    slot = mm.MONITOR_SCHEDULE_TIME_OBJ
    ist_moment = _ist(2026, 1, 5, slot.hour, slot.minute)
    utc_equivalent = ist_moment.astimezone(ZoneInfo("UTC"))
    assert utc_equivalent.hour != ist_moment.hour or utc_equivalent.utcoffset() != ist_moment.utcoffset()
    assert mm._should_fire_now(utc_equivalent, None) is True, (
        "the SAME instant expressed in a different tzinfo must fire identically -- "
        "proves the comparison is instant-based (Asia/Kolkata aware), not a naive local string match"
    )


def test_next_run_at_reflects_schedule_and_fired_state():
    slot = mm.MONITOR_SCHEDULE_TIME_OBJ
    morning = _ist(2026, 1, 5, 9, 0)
    next_run = mm._next_run_at(morning, None)
    assert next_run.date().isoformat() == "2026-01-05"
    assert (next_run.hour, next_run.minute) == (slot.hour, slot.minute)

    already_fired = mm._next_run_at(morning, "2026-01-05")
    assert already_fired.date().isoformat() == "2026-01-06"


# 7. Scheduler failure (e.g. a data-source error) does not corrupt existing
#    paper positions, and is recorded rather than silently swallowed or
#    crashing the loop.
def test_scheduler_failure_does_not_corrupt_existing_positions():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row(entry=100.0, sl=90.0, tp=120.0)]))
    _patch_market(True)
    _patch_prices({"TESTCO.NS": 100.0})

    mm._scheduler_tick(now=_ist(2026, 1, 5, 16, 0))  # opens a position successfully
    conn = mm._db()
    pos_before = conn.execute(
        "SELECT qty FROM paper_positions WHERE username=? AND symbol='TESTCO.NS'", (mm.MONITOR_SCHEDULE_USER,)
    ).fetchone()
    conn.close()
    assert pos_before is not None

    def _boom(*a, **kw):
        raise RuntimeError("data source unavailable")
    mm._run_screen = _boom

    mm._scheduler_tick(now=_ist(2026, 1, 6, 16, 0))  # must not raise, must not touch the existing position

    conn = mm._db()
    pos_after = conn.execute(
        "SELECT qty FROM paper_positions WHERE username=? AND symbol='TESTCO.NS'", (mm.MONITOR_SCHEDULE_USER,)
    ).fetchone()
    state = conn.execute(
        "SELECT last_fired_date, last_error FROM momentum_monitor_schedule WHERE username=?", (mm.MONITOR_SCHEDULE_USER,)
    ).fetchone()
    conn.close()
    assert pos_after == pos_before, "an existing paper position must survive a failed scheduled cycle untouched"
    assert state[0] == "2026-01-05", "a failed attempt must not be recorded as 'fired' -- the next tick should retry"
    assert state[1] is not None and "data source unavailable" in state[1]


# 8. Monitoring status exposes last-run/scheduler information correctly.
def test_scheduler_status_exposes_last_run_info():
    _isolated_db()
    _patch_screen(("2026-01-05", [_candidate_row()]))
    _patch_market(False)

    mm._scheduler_tick(now=_ist(2026, 1, 5, 16, 0))

    conn = mm._db()
    st = mm._scheduler_status(conn)
    conn.close()
    assert st["enabled"] is True
    assert st["timezone"] == "Asia/Kolkata"
    assert st["scheduler_user"] == mm.MONITOR_SCHEDULE_USER
    assert st["last_scheduled_run_at"] is not None
    assert st["last_error"] is None
    assert st["next_run_at"] is not None


def test_disabled_scheduler_does_not_start_a_thread():
    prev = mm.MONITOR_SCHEDULE_ENABLED
    prev_started = mm._scheduler_started
    try:
        mm.MONITOR_SCHEDULE_ENABLED = False
        mm._scheduler_started = False
        mm.start_monitor_scheduler()
        assert mm._scheduler_started is False, "start_monitor_scheduler must no-op when disabled"
    finally:
        mm.MONITOR_SCHEDULE_ENABLED = prev
        mm._scheduler_started = prev_started


if __name__ == "__main__":
    import sys as _sys
    try:
        import pytest
        raise SystemExit(pytest.main([__file__, "-v"]))
    except ImportError:
        g = dict(globals())
        tests = [(k, v) for k, v in g.items() if k.startswith("test_") and callable(v)]
        failed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
        print(f"\n{len(tests) - failed}/{len(tests)} passed")
        _sys.exit(1 if failed else 0)
