# core/sim/test_engine_regressions.py
"""
Regression tests for the three engine fixes from the audit series:
  1. A delayed-entry position (delay_bars >= 2) must never be exit-checked
     (SL/TP/trail/time-stop) on a bar before its own entry_bar.
  2. Invalid ATR (NaN or negative) at the entry bar must not open a
     position with an undefined or inverted SL/TP.
  3. fixed_cash position sizing must never imply notional greater than
     available equity.

Each test constructs the exact synthetic scenario that previously exposed
the bug (see core/audit_sim_engine.py and core/audit_atr_sizing_edge_cases.py),
and asserts the FIXED behavior. Run with: pytest core/sim/test_engine_regressions.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sim.engine import simulate


def make_ohlcv(n, base=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": np.full(n, base), "High": np.full(n, base),
                          "Low": np.full(n, base), "Close": np.full(n, base),
                          "Volume": np.full(n, 1_000_000.0)}, index=idx)


def base_cfg():
    return {
        "execution": {"delay_bars": 0, "enter_at": "close", "max_open_positions": 1,
                      "cooldown_bars": 0, "max_hold_bars": 0},
        "risk": {"capital": 1_000_000.0, "max_risk_per_trade": 0.005,
                 "position_sizing": "atr", "fixed_cash_per_trade": 50_000.0},
        "filters": {"min_atr": 0.0, "warmup_cutoff_bars": 0},
        "equity": {"mtm": False, "mtm_haircut_bps": 0.0},
    }


def sig_row(bar_idx, side="long", sl_atr=1.0, tp_atr=2.0, tag="TEST", **kw):
    row = {"bar_idx": bar_idx, "side": side, "sl_atr": sl_atr, "tp_atr": tp_atr, "tag": tag}
    row.update(kw)
    return row


# ---------------------------------------------------------------------------
# 1. delay_bars >= 2 must not allow exit-before-entry
# ---------------------------------------------------------------------------
def test_delay_bars_2_no_exit_before_entry():
    n = 8
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 2
    cfg["execution"]["enter_at"] = "open"
    cfg["execution"]["max_open_positions"] = 1

    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[4], "Open"] = 100.0     # entry bar (t+2 = bar 4)
    # bar 3 == t+1, ONE BAR BEFORE the real entry: drive low far below the
    # would-be SL level. Before the fix, this incorrectly exited the position.
    df.loc[df.index[3], "Low"] = 50.0
    df.loc[df.index[3], "Open"] = 100.0
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    sig = pd.DataFrame([sig_row(bar_idx=2, side="long", sl_atr=1.0, tp_atr=50.0, delay_bars=2)])

    trades, _, fills = simulate(df, sig, ind, cfg)

    # The position must not have been exited on bar 3 (before its entry_bar=4).
    if len(trades):
        for _, t in trades.iterrows():
            assert t["exit_bar"] >= t["entry_bar"], (
                f"exit_bar ({t['exit_bar']}) is before entry_bar ({t['entry_bar']}) — "
                "delayed-entry position was exit-checked before it actually entered"
            )
    # The entry fill itself must still have happened at bar 4, unaffected by bar 3's move.
    entry_fills = fills[fills["type"] == "ENTRY"]
    assert len(entry_fills) == 1
    assert int(entry_fills.iloc[0]["bar"]) == 4


def test_delay_bars_1_unaffected_by_fix():
    """The production-default delay_bars=1 case must behave exactly as before:
    entry and a same-bar SL breach on the real entry bar is legitimate."""
    n = 8
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 1
    cfg["execution"]["enter_at"] = "open"

    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[3], "Open"] = 100.0
    df.loc[df.index[3], "Low"] = 50.0   # this IS the entry bar (t+1) under delay=1
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    sig = pd.DataFrame([sig_row(bar_idx=2, side="long", sl_atr=1.0, tp_atr=50.0, delay_bars=1)])

    trades, _, _ = simulate(df, sig, ind, cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert int(t["entry_bar"]) == 3
    assert int(t["exit_bar"]) == 3
    assert t["reason"] == "SL"


# ---------------------------------------------------------------------------
# 2. Invalid ATR (NaN / negative) must not open an unmanaged/inverted position
# ---------------------------------------------------------------------------
def test_nan_atr_signal_rejected():
    n = 8
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[3], "High"] = 110.0
    atr = np.full(n, 2.0)
    atr[0] = np.nan  # ATR is NaN specifically on the entry bar
    ind = pd.DataFrame({"ATR": atr}, index=df.index)
    cfg = base_cfg()
    cfg["execution"]["max_hold_bars"] = 3
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    trades, _, fills = simulate(df, sig, ind, cfg)
    assert len(trades) == 0, "a NaN-ATR signal must be rejected, not opened with undefined sl/tp"
    assert len(fills) == 0, "no ENTRY fill should be recorded for a rejected NaN-ATR signal"


def test_negative_atr_signal_rejected():
    n = 6
    df = make_ohlcv(n, base=100.0)
    ind = pd.DataFrame({"ATR": np.full(n, -2.0)}, index=df.index)
    cfg = base_cfg()
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    trades, _, fills = simulate(df, sig, ind, cfg)
    assert len(trades) == 0, "a negative-ATR signal must be rejected, not opened with inverted sl/tp"
    assert len(fills) == 0


def test_positive_atr_signal_still_opens_normally():
    """Sanity check: the ATR validation must not reject legitimate positive ATR."""
    n = 6
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[1], "High"] = 110.0
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    cfg = base_cfg()
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    trades, _, _ = simulate(df, sig, ind, cfg)
    assert len(trades) == 1
    assert trades.iloc[0]["reason"] == "TP"


# ---------------------------------------------------------------------------
# 3. fixed_cash sizing must be capped at available equity
# ---------------------------------------------------------------------------
def test_fixed_cash_capped_at_equity():
    n = 6
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[1], "High"] = 110.0
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    cfg = base_cfg()
    cfg["risk"]["position_sizing"] = "fixed_cash"
    cfg["risk"]["capital"] = 10_000.0
    cfg["risk"]["fixed_cash_per_trade"] = 50_000.0  # 5x starting capital
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    trades, _, _ = simulate(df, sig, ind, cfg)
    assert len(trades) == 1
    t = trades.iloc[0]
    notional = float(t["size_shares"]) * float(t["entry_price"])
    assert notional <= cfg["risk"]["capital"] + 1e-6, (
        f"fixed_cash notional ({notional}) exceeds starting capital ({cfg['risk']['capital']})"
    )


def test_fixed_cash_normal_case_unaffected():
    """Sanity check: when fixed_cash_per_trade is within equity, sizing is unchanged."""
    n = 6
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[1], "High"] = 110.0
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    cfg = base_cfg()
    cfg["risk"]["position_sizing"] = "fixed_cash"
    cfg["risk"]["capital"] = 1_000_000.0
    cfg["risk"]["fixed_cash_per_trade"] = 50_000.0
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    trades, _, _ = simulate(df, sig, ind, cfg)
    assert len(trades) == 1
    notional = float(trades.iloc[0]["size_shares"]) * float(trades.iloc[0]["entry_price"])
    assert abs(notional - 50_000.0) < 1.0


if __name__ == "__main__":
    import sys
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        # pytest not installed in this environment -- fall back to running every
        # test_* function directly so the regressions are still checked.
        fns = [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]
        passed, failed = 0, 0
        for name in fns:
            try:
                globals()[name]()
                print(f"PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        sys.exit(1 if failed else 0)
