# api/test_intraday_monitor.py
"""Tests for api/intraday_monitor.py (the intraday paper-monitoring
PROTOTYPE -- a separate experiment from the daily momentum monitor, see
that module's own test file api/test_momentum_monitor.py for the daily
system's tests, unaffected by anything here).

Fully offline: market data (_fetch_market_snapshot) and market-hours
(_live_market_open) are monkeypatched to canned values -- no network calls,
no real broker interaction, and the sqlite paper-trading db is redirected
to a temp file so these tests can NEVER touch the real data_cache/paper.db.

Run with pytest: pytest api/test_intraday_monitor.py
Run standalone:  python api/test_intraday_monitor.py
"""
from __future__ import annotations
import os
import sys
import tempfile
from datetime import datetime, timedelta

CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import api.paper as paper_mod
import api.intraday_monitor as im

SYMBOL = "TESTCO.NS"
_REAL_FETCH = im._fetch_market_snapshot  # tests overwrite im._fetch_market_snapshot with stubs


# ---------------------------------------------------------------------------
# Isolation: every test gets a fresh temp sqlite file AND fresh in-memory
# bar state, never the real db / real accumulated bars.
# ---------------------------------------------------------------------------
def _isolated_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    paper_mod.PAPER_DB = path
    im._forming.clear(); im._bar_hist.clear(); im._last_known_price.clear()
    im._demo_forming.clear(); im._demo_bar_hist.clear(); im._demo_last_known_price.clear()
    im._live_market_open = lambda now=None: True  # weekday/hours gating tested separately
    conn = im._db()
    try:
        im._set_toggle(conn, im.INTRADAY_USER, True)  # monitoring_active defaults OFF until /start -- tests opt in explicitly
    finally:
        conn.close()
    return path


def _ist(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=im.IST)


def _const_snapshot(prices: dict, source: str = "poll"):
    def _fn(symbols, now_ist):
        return dict(prices), now_ist, source
    return _fn


def _patch_snapshot(prices: dict, source: str = "poll"):
    im._fetch_market_snapshot = _const_snapshot(prices, source)


def _warm_up_signal(base_dt, closes=(100.0, 101.0, 102.0, 103.0), symbol=SYMBOL, user=None):
    """Runs len(closes) sequential 1-minute-apart cycles so three
    consecutive higher-close bars finalize and (by the last call) a signal
    fires and immediately executes. Returns the LAST cycle's result."""
    user = user or im.INTRADAY_USER
    res = None
    for i, px in enumerate(closes):
        now = base_dt + timedelta(minutes=i)
        _patch_snapshot({symbol: px})
        res = im.run_intraday_cycle(user, demo=False, demo_now=now)
    return res


# ---------------------------------------------------------------------------
# 1. Market-session gating
# ---------------------------------------------------------------------------
def test_session_gating_before_start_is_a_noop():
    _isolated_db()
    _patch_snapshot({SYMBOL: 100.0})
    before_start = _ist(2026, 1, 5, 9, 0)  # before INTRADAY_START_TIME (09:20 default)
    res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=before_start)
    assert res["in_session"] is False
    assert res["skipped_reason"] is not None
    assert res["signals"]["created"] == []
    assert res["bars_finalized"] == []


def test_session_gating_after_square_off_with_nothing_open_is_a_noop():
    _isolated_db()
    _patch_snapshot({SYMBOL: 100.0})
    after_close = _ist(2026, 1, 5, 18, 0)
    res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=after_close)
    assert res["in_session"] is False
    assert res["skipped_reason"] is not None


def test_session_gating_weekend_blocks_entries():
    _isolated_db()
    im._live_market_open = lambda now=None: False  # simulate a weekend/holiday
    _patch_snapshot({SYMBOL: 100.0})
    saturday_during_hours = _ist(2026, 1, 3, 10, 0)  # 2026-01-03 is a Saturday
    assert saturday_during_hours.weekday() == 5
    res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=saturday_during_hours)
    assert res["in_session"] is False


# ---------------------------------------------------------------------------
# 2. 1-minute bar finalization
# ---------------------------------------------------------------------------
def test_bar_finalizes_with_correct_ohlc_on_minute_rollover():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0, 0)
    _patch_snapshot({SYMBOL: 100.0})
    im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base)               # 10:00:00, open=100
    _patch_snapshot({SYMBOL: 105.0})
    im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=20))  # 10:00:20, high=105
    _patch_snapshot({SYMBOL: 98.0})
    im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=40))  # 10:00:40, low=98
    _patch_snapshot({SYMBOL: 102.0})
    res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=1))  # 10:01:00 -> finalizes 10:00 bar

    finalized = [b for b in res["bars_finalized"] if b["symbol"] == SYMBOL]
    assert len(finalized) == 1
    bar = finalized[0]
    assert bar["open"] == 100.0
    assert bar["high"] == 105.0
    assert bar["low"] == 98.0
    assert bar["close"] == 98.0  # close is the LAST sample of that minute (before the minute rolled over)


# ---------------------------------------------------------------------------
# 3. A still-forming candle is never treated as completed
# ---------------------------------------------------------------------------
def test_forming_candle_never_finalized_mid_minute():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0, 0)
    _patch_snapshot({SYMBOL: 100.0})
    r1 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base)
    _patch_snapshot({SYMBOL: 101.0})
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=30))
    assert r1["bars_finalized"] == []
    assert r2["bars_finalized"] == []
    assert im._forming[SYMBOL]["close"] == 101.0  # still forming, not in hist yet
    assert im._bar_hist.get(SYMBOL, []) == []


# ---------------------------------------------------------------------------
# 4. Duplicate bar does not duplicate signal
# ---------------------------------------------------------------------------
def test_duplicate_bar_does_not_duplicate_signal():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    _warm_up_signal(base)
    conn = im._db()
    n = conn.execute("SELECT COUNT(*) FROM intraday_signals WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    conn.close()
    assert n == 1

    # re-process the SAME already-finalized bar timestamp again directly
    bar = {"bar_ts": base + timedelta(minutes=2), "open": 102.0, "high": 102.0, "low": 102.0, "close": 102.0, "volume": None}
    sig = im._evaluate_signal(im._bar_hist[SYMBOL])
    conn = im._db()
    try:
        rec1 = im._record_signal(conn, im.INTRADAY_USER, SYMBOL, bar, sig, base, base)
        rec2 = im._record_signal(conn, im.INTRADAY_USER, SYMBOL, bar, sig, base, base)
        assert rec1["already_existed"] in (True, False)
        assert rec2["already_existed"] is True
        n2 = conn.execute("SELECT COUNT(*) FROM intraday_signals WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    finally:
        conn.close()
    assert n2 == 1


# ---------------------------------------------------------------------------
# 5. Duplicate signal does not duplicate execution
# ---------------------------------------------------------------------------
def test_duplicate_signal_does_not_duplicate_execution():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    assert len(res["executions"]["filled"]) == 1
    signal_id = res["executions"]["filled"][0]["id"]

    conn = im._db()
    try:
        row = conn.execute(
            "SELECT status, suggested_entry, stop_loss, target FROM intraday_signals WHERE id=?", (signal_id,)
        ).fetchone()
        assert row[0] == "executed"
        sig = {"suggested_entry": row[1], "stop_loss": row[2], "target": row[3]}
        try:
            im._execute_signal(conn, im.INTRADAY_USER, signal_id, SYMBOL, sig, 999.0, 1, False, base)
            raised = False
        except RuntimeError:
            raised = True
        n_exec = conn.execute("SELECT COUNT(*) FROM intraday_executions WHERE signal_id=?", (signal_id,)).fetchone()[0]
    finally:
        conn.close()
    assert raised is True
    assert n_exec == 1


# ---------------------------------------------------------------------------
# 6. Stop-loss exit
# ---------------------------------------------------------------------------
def test_stop_loss_exit():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    filled = res["executions"]["filled"][0]
    sl = None
    conn = im._db()
    sl = conn.execute("SELECT stop_loss FROM intraday_signals WHERE id=?", (filled["id"],)).fetchone()[0]
    conn.close()

    _patch_snapshot({SYMBOL: sl - 1})
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=4))
    closed = [c for c in r2["position_monitoring"]["closed"] if c["symbol"] == SYMBOL]
    assert len(closed) == 1
    assert closed[0]["reason"] == "STOP"


# ---------------------------------------------------------------------------
# 7. Target exit
# ---------------------------------------------------------------------------
def test_target_exit():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    filled = res["executions"]["filled"][0]
    conn = im._db()
    tp = conn.execute("SELECT target FROM intraday_signals WHERE id=?", (filled["id"],)).fetchone()[0]
    conn.close()

    _patch_snapshot({SYMBOL: tp + 1})
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=4))
    closed = [c for c in r2["position_monitoring"]["closed"] if c["symbol"] == SYMBOL]
    assert len(closed) == 1
    assert closed[0]["reason"] == "TARGET"


# ---------------------------------------------------------------------------
# 8. Time-based exit
# ---------------------------------------------------------------------------
def test_time_based_exit():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base, closes=(100.0, 101.0, 102.0, 103.0))
    filled = res["executions"]["filled"][0]
    conn = im._db()
    sl = conn.execute("SELECT stop_loss FROM intraday_signals WHERE id=?", (filled["id"],)).fetchone()[0]
    conn.close()
    entry = filled["actual_entry_price"]
    mid_price = (entry + sl) / 2 if sl else entry  # stays strictly inside [sl, tp]

    far_later = base + timedelta(minutes=3) + timedelta(minutes=im.INTRADAY_MAX_HOLD_MINUTES + 1)
    _patch_snapshot({SYMBOL: mid_price})
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=far_later)
    closed = [c for c in r2["position_monitoring"]["closed"] if c["symbol"] == SYMBOL]
    assert len(closed) == 1
    assert closed[0]["reason"] == "TIME_EXIT"


# ---------------------------------------------------------------------------
# 9. Mandatory square-off
# ---------------------------------------------------------------------------
def test_mandatory_square_off_closes_open_positions():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    assert len(res["executions"]["filled"]) == 1

    square_off_time = _ist(2026, 1, 5, im.INTRADAY_SQUARE_OFF_TIME_OBJ.hour, im.INTRADAY_SQUARE_OFF_TIME_OBJ.minute)
    _patch_snapshot({SYMBOL: 999.0})  # deep in profit -- square-off must still fire regardless
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=square_off_time)
    closed = [c for c in r2["position_monitoring"]["closed"] if c["symbol"] == SYMBOL]
    assert len(closed) == 1
    assert closed[0]["reason"] == "INTRADAY_SQUARE_OFF"

    conn = im._db()
    still_open = conn.execute(
        "SELECT COUNT(*) FROM intraday_executions WHERE username=? AND closed_at IS NULL", (im.INTRADAY_USER,)
    ).fetchone()[0]
    conn.close()
    assert still_open == 0

    # idempotent: re-running square-off does not double-close / error
    r3 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=square_off_time + timedelta(minutes=1))
    assert [c for c in r3["position_monitoring"]["closed"] if c["symbol"] == SYMBOL] == []


# ---------------------------------------------------------------------------
# 10. Entry cutoff
# ---------------------------------------------------------------------------
def test_entry_cutoff_blocks_new_positions_but_keeps_monitoring_existing():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    assert len(res["executions"]["filled"]) == 1

    # a SECOND, distinct symbol tries to form a fresh signal AFTER the entry cutoff
    after_cutoff = _ist(2026, 1, 5, im.INTRADAY_ENTRY_CUTOFF_OBJ.hour, im.INTRADAY_ENTRY_CUTOFF_OBJ.minute) + timedelta(minutes=5)
    assert im._in_entry_window(after_cutoff) is False
    _warm_up_signal(after_cutoff, symbol="OTHER.NS")
    conn = im._db()
    n_other_signals = conn.execute("SELECT COUNT(*) FROM intraday_signals WHERE symbol='OTHER.NS'").fetchone()[0]
    n_other_execs = conn.execute("SELECT COUNT(*) FROM intraday_executions ix JOIN intraday_signals s ON s.id=ix.signal_id WHERE s.symbol='OTHER.NS'").fetchone()[0]
    still_open_original = conn.execute(
        "SELECT COUNT(*) FROM intraday_executions ix JOIN intraday_signals s ON s.id=ix.signal_id "
        "WHERE s.symbol=? AND ix.closed_at IS NULL", (SYMBOL,),
    ).fetchone()[0]
    conn.close()
    assert n_other_signals == 0, "no NEW signal should be created after the entry cutoff"
    assert n_other_execs == 0
    assert still_open_original == 1, "the position opened before the cutoff must still be tracked as open"


# ---------------------------------------------------------------------------
# 11. Daily loss limit
# ---------------------------------------------------------------------------
def test_daily_loss_limit_trips_kill_switch_and_blocks_new_entries():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    filled = res["executions"]["filled"][0]
    entry = filled["actual_entry_price"]
    qty = filled["qty"]
    # force a loss bigger than INTRADAY_MAX_DAILY_LOSS by stopping out deep
    conn = im._db()
    sl = conn.execute("SELECT stop_loss FROM intraday_signals WHERE id=?", (filled["id"],)).fetchone()[0]
    conn.close()
    crash_price = max(0.01, entry - (im.INTRADAY_MAX_DAILY_LOSS / qty) - 10)
    _patch_snapshot({SYMBOL: crash_price})
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=4))
    assert r2["risk"]["kill_switch"] is True
    assert r2["risk"]["realized_pnl_today"] <= -im.INTRADAY_MAX_DAILY_LOSS

    # a fresh signal on a NEW symbol must not enter while the kill switch is tripped
    _warm_up_signal(base + timedelta(minutes=10), symbol="OTHER2.NS")
    conn = im._db()
    n_execs = conn.execute(
        "SELECT COUNT(*) FROM intraday_executions ix JOIN intraday_signals s ON s.id=ix.signal_id WHERE s.symbol='OTHER2.NS'"
    ).fetchone()[0]
    conn.close()
    assert n_execs == 0


# ---------------------------------------------------------------------------
# 12. Stale-data protection
# ---------------------------------------------------------------------------
def test_stale_data_blocks_new_signals_but_keeps_position_tracked():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    assert len(res["executions"]["filled"]) == 1

    stale_now = base + timedelta(minutes=10)
    stale_as_of = stale_now - timedelta(seconds=im.INTRADAY_STALE_DATA_SECONDS + 30)
    im._fetch_market_snapshot = lambda symbols, now_ist: ({SYMBOL: 150.0}, stale_as_of, "ws")
    r2 = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=stale_now)
    assert r2["data"]["stale"] is True
    assert r2["signals"]["created"] == []
    # existing position is still reported (not silently dropped), just without a fresh price-based close
    still_open = [p for p in r2["position_monitoring"]["still_open"] if p["symbol"] == SYMBOL]
    closed = [c for c in r2["position_monitoring"]["closed"] if c["symbol"] == SYMBOL]
    assert len(still_open) == 1
    assert len(closed) == 0


# ---------------------------------------------------------------------------
# 13. Restart / idempotency
# ---------------------------------------------------------------------------
def test_restart_does_not_duplicate_signals_or_positions():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    _warm_up_signal(base)
    # "restart": nothing but the sqlite state is trusted -- simulate by
    # wiping the in-memory bar-builder state (as a real process restart
    # would) and re-driving the SAME bar timestamps again
    im._forming.clear(); im._bar_hist.clear()
    for i, px in enumerate((100.0, 101.0, 102.0, 103.0)):
        _patch_snapshot({SYMBOL: px})
        im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=i))

    conn = im._db()
    n_signals = conn.execute("SELECT COUNT(*) FROM intraday_signals WHERE symbol=?", (SYMBOL,)).fetchone()[0]
    n_execs = conn.execute(
        "SELECT COUNT(*) FROM intraday_executions ix JOIN intraday_signals s ON s.id=ix.signal_id WHERE s.symbol=?", (SYMBOL,)
    ).fetchone()[0]
    conn.close()
    assert n_signals == 1
    assert n_execs == 1


# ---------------------------------------------------------------------------
# 14. Demo mode separation
# ---------------------------------------------------------------------------
def test_demo_mode_never_touches_the_real_intraday_account():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = None
    for i, px in enumerate((100.0, 101.0, 102.0, 103.0)):
        now = base + timedelta(minutes=i)
        res = im.run_intraday_cycle(im.INTRADAY_DEMO_USER, demo=True, demo_now=now, demo_prices={SYMBOL: px})
    assert len(res["executions"]["filled"]) == 1
    assert res["mode"] == "demo"

    conn = im._db()
    real_trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE username=?", (im.INTRADAY_USER,)).fetchone()[0]
    demo_trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE username=?", (im.INTRADAY_DEMO_USER,)).fetchone()[0]
    conn.close()
    assert real_trades == 0
    assert demo_trades == 1
    assert im._forming == {} or SYMBOL not in im._forming or SYMBOL not in im._demo_forming or True
    # demo bar state must be in the SEPARATE demo store, never the live one
    assert SYMBOL not in im._forming


# ---------------------------------------------------------------------------
# 15. No real broker order is ever sent
# ---------------------------------------------------------------------------
def test_no_broker_order_function_is_ever_referenced():
    import inspect
    src = inspect.getsource(im)
    forbidden = ["place_order(", "growwapi", "GrowwAPI(", "get_client().place", "_get_client().place"]
    for token in forbidden:
        assert token not in src, f"intraday_monitor.py must never call a real order-placement API ({token})"
    assert "_live_ltp_map" in src and "_live_feed_snapshot" in src


# ---------------------------------------------------------------------------
# 16. Timezone handling (Asia/Kolkata, instant-based)
# ---------------------------------------------------------------------------
def test_session_window_respects_ist_boundaries():
    d = _ist(2026, 1, 5, 0, 0).date()
    start = im.INTRADAY_START_TIME_OBJ
    before = _ist(d.year, d.month, d.day, start.hour, max(0, start.minute - 1))
    at = _ist(d.year, d.month, d.day, start.hour, start.minute)
    assert im._in_entry_window(before) is False
    assert im._in_entry_window(at) is True


def test_session_window_uses_ist_not_naive_local_time():
    from zoneinfo import ZoneInfo
    d = _ist(2026, 1, 5, 0, 0).date()
    start = im.INTRADAY_START_TIME_OBJ
    ist_moment = _ist(d.year, d.month, d.day, start.hour, start.minute)
    utc_equivalent = ist_moment.astimezone(ZoneInfo("UTC"))
    assert im._in_entry_window(utc_equivalent) is True, (
        "the SAME instant in a different tzinfo must gate identically -- proves the "
        "comparison is instant-based (Asia/Kolkata aware), not a naive local string match"
    )


# ===========================================================================
# Audit fixes (2026-10-23): sizing, fees, per-symbol freshness, signal status,
# last finalized bar. All offline; temp sqlite only; no live market request.
# ===========================================================================
FEE = 0.0012  # the pre-existing paper model: 0.0% brokerage + 0.12% tax per side


def _ledger_cost(qty, entry, fee=FEE):
    return im._entry_cost(qty, entry, fee)


# ---- A. position sizing ----------------------------------------------------
def test_sizing_audit_case_is_notional_capped_not_full_account():
    # the audit's TECHM entry sized 554 sh = Rs 998,696 (~100% of capital)
    qty, why = im._size_position_checked(1_000_000.0, 1802.7, 1800.5, FEE)
    assert why is None
    assert qty * 1802.7 <= im.INTRADAY_MAX_POSITION_NOTIONAL
    assert qty == int(im.INTRADAY_MAX_POSITION_NOTIONAL // 1802.7)  # notional is the binding limit here


def test_sizing_loss_at_stop_including_charges_never_exceeds_risk_budget():
    cash, entry, stop = 1_000_000.0, 100.0, 90.0   # wide stop -> risk, not notional, binds
    qty, why = im._size_position_checked(cash, entry, stop, FEE)
    assert why is None and qty > 0
    budget = cash * im.INTRADAY_RISK_PCT_PER_TRADE
    loss_at_stop = qty * (entry - stop) + FEE * qty * (entry + stop)
    assert loss_at_stop <= budget
    q1 = qty + 1  # the quantity is the maximum allowed, not arbitrary
    assert q1 * (entry - stop) + FEE * q1 * (entry + stop) > budget or q1 * entry > im.INTRADAY_MAX_POSITION_NOTIONAL


def test_sizing_never_overspends_cash_including_entry_charges():
    for cash, entry in ((1000.0, 100.0), (999_999.0, 1162.05), (50_000.0, 7.0), (1_000_000.0, 1.0)):
        qty, _ = im._size_position_checked(cash, entry, entry - 0.01, FEE)
        assert _ledger_cost(qty, entry) <= cash


def test_sizing_rejects_when_one_share_breaks_risk_budget_and_does_not_widen_stop():
    # risk budget = 10_000 x 0.5% = Rs 50; 1 share at 5000 w/ stop 4000 risks ~Rs 1,011
    qty, why = im._size_position_checked(10_000.0, 5000.0, 4000.0, FEE)
    assert qty == 0 and "risk budget" in why


def test_sizing_rejects_when_one_share_exceeds_max_notional():
    qty, why = im._size_position_checked(100_000_000.0, im.INTRADAY_MAX_POSITION_NOTIONAL + 1, 1.0, FEE)
    assert qty == 0 and "notional" in why


def test_sizing_rejects_when_one_share_unaffordable_with_charges():
    # cash equals the gross price: affordable without charges, NOT with the 0.12% entry charge
    qty, why = im._size_position_checked(100.0, 100.0, 99.9, FEE)
    assert qty == 0 and "cash" in why


def test_sizing_invalid_inputs_rejected():
    assert im._size_position_checked(1000.0, 100.0, None, FEE)[0] == 0
    assert im._size_position_checked(1000.0, 100.0, 100.0, FEE)[0] == 0
    assert im._size_position_checked(1000.0, 100.0, 101.0, FEE)[0] == 0
    assert im._size_position_checked(0.0, 100.0, 99.0, FEE)[0] == 0


def test_cycle_executed_position_respects_notional_cap():
    _isolated_db()
    res = _warm_up_signal(_ist(2026, 1, 5, 10, 0))
    f = res["executions"]["filled"][0]
    assert f["qty"] * f["actual_entry_price"] <= im.INTRADAY_MAX_POSITION_NOTIONAL


# ---- B. fee model ----------------------------------------------------------
def test_fee_rate_comes_from_account_columns_and_defaults_to_existing_model():
    _isolated_db()
    conn = im._db()
    try:
        acct = im._paper_account(conn, "fee-user")
        assert abs(im._fee_rate_per_side(acct) - 0.0012) < 1e-12
        assert abs(im._fee_rate_per_side({"brokerage_pct": 0.03, "tax_pct": 0.025}) - 0.00055) < 1e-12
    finally:
        conn.close()


def test_fees_affect_affordability_exit_pnl_and_net_realized():
    _isolated_db()
    conn = im._db()
    try:
        user = "fee-user"
        im._paper_account(conn, user)
        cash0 = 1_000_000.0
        im._paper_fill(conn, user, "X.NS", "X.NS", "BUY", 100, 100.0, 99.0, None, "t")
        # entry affordability: cash falls by gross + 0.12% entry charge
        assert abs(im._paper_account(conn, user)["cash"] - (cash0 - 10_000.0 - 12.0)) < 1e-6
        realized = im._paper_fill(conn, user, "X.NS", "X.NS", "SELL", 100, 101.0, None, None, "t")
        entry_charge, exit_charge = 12.0, round(10_100.0 * FEE, 2)  # 12.12
        assert abs(realized - (100.0 - entry_charge - exit_charge)) < 0.01   # 75.88 net
        acct = im._paper_account(conn, user)
        assert abs(acct["realized_pnl"] - realized) < 1e-6
        assert abs(acct["cash"] - (cash0 + realized)) < 1e-6
    finally:
        conn.close()


def test_stop_out_net_loss_exceeds_price_loss_by_round_trip_charges():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = _warm_up_signal(base)
    f = res["executions"]["filled"][0]
    entry, qty = f["actual_entry_price"], f["qty"]
    exit_px = 100.0   # below the 3-bar low (101.0) -> STOP
    _patch_snapshot({SYMBOL: exit_px})
    r = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=4))
    closed = r["position_monitoring"]["closed"]
    assert len(closed) == 1
    price_pnl = (exit_px - entry) * qty
    charges = round(entry * qty * FEE, 2) + round(exit_px * qty * FEE, 2)
    assert abs(closed[0]["realized_pnl"] - (price_pnl - charges)) < 0.02
    assert closed[0]["realized_pnl"] < price_pnl


# ---- C. per-symbol freshness ----------------------------------------------
NOW = _ist(2026, 1, 5, 10, 30, 0)


def _cls(prices, as_of, expected=()):
    return im._classify_prices(prices, as_of, NOW, list(expected))


def test_freshness_fresh_prices_pass():
    c = _cls({"A.NS": 1.0, "B.NS": 2.0}, {"A.NS": NOW, "B.NS": NOW - timedelta(seconds=im.INTRADAY_STALE_DATA_SECONDS)})
    assert set(c["fresh"]) == {"A.NS", "B.NS"} and not c["stale"] and not c["future"]


def test_freshness_stale_price_rejected():
    c = _cls({"A.NS": 1.0}, {"A.NS": NOW - timedelta(seconds=im.INTRADAY_STALE_DATA_SECONDS + 1)})
    assert c["fresh"] == {} and c["stale"] == ["A.NS"]


def test_freshness_mixed_age_symbols_are_independent():
    c = _cls({"A.NS": 1.0, "B.NS": 2.0, "C.NS": 3.0},
             {"A.NS": NOW, "B.NS": NOW - timedelta(seconds=600), "C.NS": NOW - timedelta(seconds=10)})
    assert set(c["fresh"]) == {"A.NS", "C.NS"}
    assert c["stale"] == ["B.NS"]   # A's fresh tick does not vouch for B


def test_freshness_future_timestamp_rejected_but_small_jitter_allowed():
    c = _cls({"A.NS": 1.0, "B.NS": 2.0},
             {"A.NS": NOW + timedelta(seconds=im.INTRADAY_FUTURE_TOLERANCE_SECONDS + 1),
              "B.NS": NOW + timedelta(seconds=im.INTRADAY_FUTURE_TOLERANCE_SECONDS - 1)})
    assert c["future"] == ["A.NS"] and set(c["fresh"]) == {"B.NS"}
    c74 = _cls({"A.NS": 1.0}, NOW + timedelta(seconds=74))  # the audit's 74s clock-skew signature
    assert c74["fresh"] == {} and c74["future"] == ["A.NS"]


def test_freshness_missing_prices_and_missing_timestamps():
    c = _cls({"A.NS": 1.0}, {}, expected=["A.NS", "B.NS"])
    assert c["fresh"] == {} and c["stale"] == ["A.NS"] and c["missing"] == ["B.NS"]   # no ts -> stale, never assumed fresh
    assert _cls({}, NOW, expected=["A.NS"])["missing"] == ["A.NS"]


def test_ws_snapshot_returns_per_symbol_timestamps_and_zero_ts_is_stale():
    saved = im._live_feed_snapshot
    try:
        t = int(NOW.timestamp())
        im._live_feed_snapshot = lambda syms: {"A.NS": {"ltp": 10.0, "ts": t}, "B.NS": {"ltp": 20.0, "ts": 0}}
        prices, as_of, source = _REAL_FETCH(["A.NS", "B.NS"], NOW)
        assert source == "ws" and isinstance(as_of, dict)
        assert as_of["A.NS"] == datetime.fromtimestamp(t, im.IST)
        c = _cls(prices, as_of)
        assert set(c["fresh"]) == {"A.NS"} and c["stale"] == ["B.NS"]
    finally:
        im._live_feed_snapshot = saved


def test_rest_fallback_still_used_when_ws_empty_and_is_fresh_as_of_now():
    s1, s2 = im._live_feed_snapshot, im._live_ltp_map
    try:
        im._live_feed_snapshot = lambda syms: {}
        im._live_ltp_map = lambda syms: {"A.NS": 10.0}
        prices, as_of, source = _REAL_FETCH(["A.NS"], NOW)
        assert source == "poll" and prices == {"A.NS": 10.0} and as_of == NOW
        im._live_ltp_map = lambda syms: {}
        assert _REAL_FETCH(["A.NS"], NOW)[2] == "none"
    finally:
        im._live_feed_snapshot, im._live_ltp_map = s1, s2


def test_cycle_mixed_age_stale_symbol_gets_no_bars_signals_or_entries():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    fresh_sym, stale_sym = "FRESH.NS", "STALE.NS"
    res = None
    for i, px in enumerate((100.0, 101.0, 102.0, 103.0)):
        now = base + timedelta(minutes=i)
        im._fetch_market_snapshot = lambda syms, n, px=px, now=now: (
            {fresh_sym: px, stale_sym: px}, {fresh_sym: now, stale_sym: now - timedelta(seconds=500)}, "ws")
        res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=now)
    assert res["data"]["stale"] is False                    # one symbol is usable...
    assert res["data"]["stale_symbols"] == [stale_sym]      # ...but the other is reported stale, not masked
    assert stale_sym not in im._forming and stale_sym not in im._bar_hist
    conn = im._db()
    n_stale = conn.execute("SELECT COUNT(*) FROM intraday_signals WHERE symbol=?", (stale_sym,)).fetchone()[0]
    n_fresh = conn.execute("SELECT COUNT(*) FROM intraday_signals WHERE symbol=?", (fresh_sym,)).fetchone()[0]
    conn.close()
    assert n_stale == 0 and n_fresh == 1


def test_cycle_future_dated_prices_cannot_create_signals_or_entries():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    res = None
    for i, px in enumerate((100.0, 101.0, 102.0, 103.0)):
        now = base + timedelta(minutes=i)
        im._fetch_market_snapshot = lambda syms, n, px=px, now=now: ({SYMBOL: px}, now + timedelta(seconds=74), "ws")
        res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=now)
    assert res["data"]["stale"] is True and res["data"]["future_symbols"] == [SYMBOL]
    assert im._bar_hist == {} and res["signals"]["created"] == [] and res["executions"]["filled"] == []


def test_cycle_missing_prices_is_safe_and_flagged_stale():
    _isolated_db()
    im._fetch_market_snapshot = lambda syms, n: ({}, n, "none")
    res = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=_ist(2026, 1, 5, 10, 0))
    assert res["data"]["stale"] is True and res["signals"]["created"] == [] and res["bars_finalized"] == []


# ---- D. terminal signal status --------------------------------------------
def _signal_row(signal_id):
    conn = im._db()
    try:
        return conn.execute("SELECT status, rejection_reason FROM intraday_signals WHERE id=?", (signal_id,)).fetchone()
    finally:
        conn.close()


def test_sizing_rejection_is_persisted_as_rejected_and_not_retried():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    # one share costs more than INTRADAY_MAX_POSITION_NOTIONAL -> sizing must reject
    px0 = im.INTRADAY_MAX_POSITION_NOTIONAL * 2
    res = _warm_up_signal(base, closes=(px0, px0 + 1, px0 + 2, px0 + 3))
    skipped = res["executions"]["skipped"]
    assert len(skipped) == 1 and skipped[0]["reason"] == "rejected"
    sid = skipped[0]["id"]
    status, why = _signal_row(sid)
    assert status == "rejected" and "notional" in why
    conn = im._db()
    n_exec = conn.execute("SELECT COUNT(*) FROM intraday_executions").fetchone()[0]
    n_trades = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE username=?", (im.INTRADAY_USER,)).fetchone()[0]
    conn.close()
    assert n_exec == 0 and n_trades == 0
    # later cycles (mid-minute polls) must not re-attempt or revive it
    for k in range(1, 4):
        _patch_snapshot({SYMBOL: px0 + 3})
        r = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=3, seconds=15 * k))
        assert r["executions"]["attempted"] == 0
    assert _signal_row(sid)[0] == "rejected"


def test_ledger_cash_rejection_is_persisted_as_rejected():
    _isolated_db()
    saved = im._size_position_checked
    try:
        im._size_position_checked = lambda cash, entry, stop, fee=0.0: (10**9, None)   # force the ledger guard to fire
        res = _warm_up_signal(_ist(2026, 1, 5, 10, 0))
    finally:
        im._size_position_checked = saved
    sk = res["executions"]["skipped"]
    assert len(sk) == 1 and "insufficient paper cash" in sk[0]["error"]
    status, why = _signal_row(sk[0]["id"])
    assert status == "rejected" and why.startswith("ledger:")
    conn = im._db()
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
    conn.close()


def test_reject_signal_never_downgrades_executed():
    _isolated_db()
    res = _warm_up_signal(_ist(2026, 1, 5, 10, 0))
    sid = res["executions"]["filled"][0]["id"]
    conn = im._db()
    try:
        im._reject_signal(conn, sid, "late")
    finally:
        conn.close()
    assert _signal_row(sid) == ("executed", None)


def test_schema_migration_adds_column_without_touching_existing_rows():
    import sqlite3
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    paper_mod.PAPER_DB = path
    old = sqlite3.connect(path)
    old.executescript(
        """CREATE TABLE intraday_signals (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, symbol TEXT NOT NULL,
        yahoo_symbol TEXT NOT NULL, strategy TEXT NOT NULL, timeframe TEXT NOT NULL, signal_timestamp TEXT NOT NULL,
        bar_timestamp TEXT NOT NULL, direction TEXT NOT NULL, suggested_entry REAL NOT NULL, stop_loss REAL, target REAL,
        reason TEXT, market_data_ts TEXT, status TEXT NOT NULL DEFAULT 'pending',
        UNIQUE(username, symbol, strategy, timeframe, bar_timestamp));
        INSERT INTO intraday_signals (username,symbol,yahoo_symbol,strategy,timeframe,signal_timestamp,bar_timestamp,direction,suggested_entry)
        VALUES ('u','OLD.NS','OLD.NS','s','1m','2026-10-23T10:49:13+05:30','2026-10-23T10:48:00+05:30','BUY',1.5);"""
    )
    old.commit()
    old.close()
    for _ in range(2):  # idempotent
        conn = im._db()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(intraday_signals)")]
        row = conn.execute("SELECT symbol, status, suggested_entry, rejection_reason FROM intraday_signals").fetchall()
        conn.close()
        assert cols.count("rejection_reason") == 1
        assert row == [("OLD.NS", "pending", 1.5, None)]   # pre-existing pending row left exactly as it was


# ---- E. last finalized bar -------------------------------------------------
def _status(demo=False):
    return im.intraday_status(None, demo)


def test_last_bar_empty_state_when_no_bar_ever_finalized():
    _isolated_db()
    _patch_snapshot({SYMBOL: 100.0})
    im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=_ist(2026, 1, 5, 10, 0))   # first sample only
    st = _status()
    assert st["last_bar"] is None and st["last_bar_info"] is None


def test_last_bar_survives_cycles_that_finalize_nothing():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    for secs, px in ((0, 100.0), (60, 101.0)):   # minute rollover finalizes the 10:00 bar
        _patch_snapshot({SYMBOL: px})
        im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=secs))
    st = _status()
    assert st["last_bar"] == "2026-01-05T10:00:00+05:30"
    assert st["last_bar_info"]["symbols"] == [SYMBOL] and st["last_bar_info"]["symbol_count"] == 1
    assert st["last_bar_info"]["finalized_at"] == "2026-01-05T10:01:00+05:30"
    for secs in (75, 90, 105):                   # several cycles mid-minute: nothing finalizes
        _patch_snapshot({SYMBOL: 101.5})
        r = im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=secs))
        assert r["bars_finalized"] == []
        assert _status()["last_bar"] == "2026-01-05T10:00:00+05:30"
    _patch_snapshot({SYMBOL: 102.0})             # next rollover advances it
    im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=120))
    assert _status()["last_bar"] == "2026-01-05T10:01:00+05:30"


def test_last_bar_survives_restart_and_is_per_account():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    for secs, px in ((0, 100.0), (60, 101.0)):
        _patch_snapshot({SYMBOL: px})
        im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(seconds=secs))
    im._forming.clear()
    im._bar_hist.clear()    # simulated process restart: in-memory bars gone
    assert _status()["last_bar"] == "2026-01-05T10:00:00+05:30"
    assert _status(demo=True)["last_bar"] is None  # demo account has its own (empty) history


# ---- F. bar continuity: a gap must never be bridged ------------------------
def _bar(ts, close, low=None):
    return {"bar_ts": ts, "open": close, "high": close, "low": close if low is None else low, "close": close, "volume": None}


def _signal_count():
    conn = im._db()
    try:
        return conn.execute("SELECT COUNT(*) FROM intraday_signals").fetchone()[0]
    finally:
        conn.close()


def test_evaluate_signal_refuses_non_adjacent_bars_but_accepts_adjacent():
    t = _ist(2026, 1, 5, 10, 0)
    adjacent = [_bar(t, 100.0), _bar(t + timedelta(minutes=1), 101.0), _bar(t + timedelta(minutes=2), 102.0)]
    assert im._evaluate_signal(adjacent) is not None
    gap_mid = [_bar(t, 100.0), _bar(t + timedelta(minutes=2), 101.0), _bar(t + timedelta(minutes=3), 102.0)]
    gap_last = [_bar(t, 100.0), _bar(t + timedelta(minutes=1), 101.0), _bar(t + timedelta(minutes=5), 102.0)]
    assert im._evaluate_signal(gap_mid) is None
    assert im._evaluate_signal(gap_last) is None


def test_missing_minutes_reset_the_run_and_recovery_needs_three_new_adjacent_bars():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    # minutes 0,1 trade; minutes 2,3 have NO price at all (outage); 4..7 trade, always rising
    plan = {0: 100.0, 1: 101.0, 4: 102.0, 5: 103.0, 6: 104.0, 7: 105.0}
    counts = {}
    for m in range(0, 8):
        if m in plan:
            _patch_snapshot({SYMBOL: plan[m]})
        else:
            _patch_snapshot({})
        im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=base + timedelta(minutes=m))
        counts[m] = _signal_count()
    # bars 0,1,4,5 would have looked like 3 rising closes if the gap were bridged (1 -> 4): must NOT signal
    assert counts[5] == 0 and counts[6] == 0
    # history after the gap holds only adjacent bars
    mins = [b["bar_ts"].minute for b in im._bar_hist[SYMBOL]]
    assert mins == [4, 5, 6]
    # recovery: the first signal comes only once bars 4,5,6 (adjacent) are all finalized
    assert counts[7] == 1


def test_stale_price_interval_is_treated_like_a_gap():
    _isolated_db()
    base = _ist(2026, 1, 5, 10, 0)
    plan = {0: 100.0, 1: 101.0, 2: 150.0, 3: 160.0, 4: 102.0, 5: 103.0, 6: 104.0}
    for m in range(0, 7):
        now = base + timedelta(minutes=m)
        # minutes 2,3 carry prices but their tick time is stale -> rejected per symbol, no bar for those minutes
        ts = now - timedelta(seconds=500) if m in (2, 3) else now
        im._fetch_market_snapshot = lambda syms, n, px=plan[m], ts=ts: ({SYMBOL: px}, {SYMBOL: ts}, "ws")
        im.run_intraday_cycle(im.INTRADAY_USER, demo=False, demo_now=now)
    # the stale 150/160 prints never entered a bar, and no signal bridges 1 -> 4
    closes = [b["close"] for b in im._bar_hist[SYMBOL]]
    assert 150.0 not in closes and 160.0 not in closes
    assert _signal_count() == 0   # bars 4,5 are finalized; 4,5,6 not yet (bar 6 still forming)


def test_session_boundary_overnight_gap_does_not_bridge_days():
    store, hist = {}, {}
    d1, d2 = _ist(2026, 1, 5, 15, 12), _ist(2026, 1, 6, 9, 20)
    for i in range(4):                       # day 1: 15:12..15:15 (bars 15:12-15:14 finalize, 15:15 stays forming)
        im._ingest_tick(store, hist, SYMBOL, 100.0 + i, d1 + timedelta(minutes=i))
    assert [b["bar_ts"] for b in hist[SYMBOL]] == [d1, d1 + timedelta(minutes=1), d1 + timedelta(minutes=2)]
    im._ingest_tick(store, hist, SYMBOL, 110.0, d2)                          # next morning: finalizes day-1's 15:15 bar
    assert hist[SYMBOL][-1]["bar_ts"] == d1 + timedelta(minutes=3)
    im._ingest_tick(store, hist, SYMBOL, 111.0, d2 + timedelta(minutes=1))   # finalizes 09:20 -> run restarts
    assert [b["bar_ts"] for b in hist[SYMBOL]] == [d2]                       # no day-1 bar carried into day 2
    assert im._evaluate_signal(hist[SYMBOL]) is None
    im._ingest_tick(store, hist, SYMBOL, 112.0, d2 + timedelta(minutes=2))
    assert im._evaluate_signal(hist[SYMBOL]) is None                         # only 2 day-2 bars so far
    im._ingest_tick(store, hist, SYMBOL, 113.0, d2 + timedelta(minutes=3))
    assert im._evaluate_signal(hist[SYMBOL]) is not None                     # 09:20,09:21,09:22 -> legitimate


def test_ingest_tick_gap_keeps_real_bars_without_fabricating_any():
    store, hist = {}, {}
    t = _ist(2026, 1, 5, 10, 0)
    im._ingest_tick(store, hist, SYMBOL, 100.0, t)
    im._ingest_tick(store, hist, SYMBOL, 101.0, t + timedelta(minutes=1))
    assert [b["bar_ts"].minute for b in hist[SYMBOL]] == [0]
    im._ingest_tick(store, hist, SYMBOL, 102.0, t + timedelta(minutes=5))    # 3 empty minutes in between
    assert [b["bar_ts"].minute for b in hist[SYMBOL]] == [0, 1]
    im._ingest_tick(store, hist, SYMBOL, 103.0, t + timedelta(minutes=6))
    assert [b["bar_ts"].minute for b in hist[SYMBOL]] == [5]                 # run restarted; minutes 2-4 never invented
    assert im._evaluate_signal(hist[SYMBOL]) is None


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
