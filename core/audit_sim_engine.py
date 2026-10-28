# core/audit_sim_engine.py
"""
Independent mechanics audit of the backtesting engine (sim/engine.py::simulate).

Does NOT modify sim/engine.py, sim/tune.py, pipeline.py, or any other
production/config file. simulate() is imported and called read-only, exactly
as any other caller (pipeline.py, tune.py) would call it. Every expected
value below is computed independently in this script from the synthetic
input, BEFORE looking at simulate()'s actual output, so the comparison is a
real check and not a restatement of the engine's own arithmetic.

If a check fails or looks suspicious, this script only documents it
(PASS / FAIL / INVESTIGATE + evidence) — it does not attempt a fix.

Run:  python audit_sim_engine.py   (from the core/ directory)
Output: core/audit_sim_engine_results.csv (+ printed summary)
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd

from sim.engine import simulate

RESULTS: list[dict] = []


def record(item: str, check: str, expected, actual, verdict: str, evidence: str):
    RESULTS.append({
        "item": item, "check": check,
        "expected": str(expected), "actual": str(actual),
        "verdict": verdict, "evidence": evidence,
    })
    print(f"[{verdict:10s}] {item} :: {check}")
    if verdict != "PASS":
        print(f"             expected={expected!r}  actual={actual!r}")
        print(f"             evidence: {evidence}")


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------
def make_ohlcv(n: int, base: float = 100.0) -> pd.DataFrame:
    """Flat OHLCV series (open=high=low=close=base) the caller then edits
    bar-by-bar for the specific scenario under test."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": np.full(n, base), "High": np.full(n, base),
        "Low": np.full(n, base), "Close": np.full(n, base),
        "Volume": np.full(n, 1_000_000.0),
    }, index=idx)


def make_ind(n: int, atr: float | np.ndarray, index=None) -> pd.DataFrame:
    idx = index if index is not None else pd.date_range("2024-01-01", periods=n, freq="B")
    atr_arr = np.full(n, float(atr)) if np.isscalar(atr) else np.asarray(atr, dtype=float)
    return pd.DataFrame({"ATR": atr_arr}, index=idx)


def base_cfg() -> dict:
    """Minimal engine cfg mirroring pipeline.py's *base* dict shape but
    WITHOUT the costs block pipeline.py only adds later in __main__ — this
    is the engine's true zero-config default, used as the item-5 baseline."""
    return {
        "execution": {"delay_bars": 1, "enter_at": "Close", "max_open_positions": 1,
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
# 1) ENTRY TIMING
# ---------------------------------------------------------------------------
def audit_entry_timing():
    item = "1_entry_timing"

    # delay_bars=1 (production default), enter_at="open"
    n = 10
    df = make_ohlcv(n)
    df.loc[df.index[4], "Open"] = 105.0   # entry bar should be 3+1=4
    ind = make_ind(n, atr=2.0)
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 1
    cfg["execution"]["enter_at"] = "open"
    sig = pd.DataFrame([sig_row(bar_idx=3)])

    trades, eq, fills = simulate(df, sig, ind, cfg)
    expected_entry_bar = 4
    expected_entry_px = 105.0
    if len(trades) == 1:
        actual_entry_bar = int(trades.iloc[0]["entry_bar"])
        actual_entry_px = float(trades.iloc[0]["entry_price"])
        ok = (actual_entry_bar == expected_entry_bar) and abs(actual_entry_px - expected_entry_px) < 1e-9
        record(item, "delay_bars=1, signal at bar 3 -> entry at bar 4, price=open[4]",
               f"entry_bar={expected_entry_bar}, entry_price={expected_entry_px}",
               f"entry_bar={actual_entry_bar}, entry_price={actual_entry_px}",
               "PASS" if ok else "FAIL",
               f"trades_df row: {trades.iloc[0].to_dict()}")
    else:
        record(item, "delay_bars=1, signal at bar 3 -> entry at bar 4, price=open[4]",
               "1 trade", f"{len(trades)} trades", "FAIL", f"trades_df: {trades.to_dict('records')}")

    # delay_bars=0
    df2 = make_ohlcv(n)
    df2.loc[df2.index[3], "Close"] = 110.0
    cfg2 = base_cfg()
    cfg2["execution"]["delay_bars"] = 0
    cfg2["execution"]["enter_at"] = "close"
    sig2 = pd.DataFrame([sig_row(bar_idx=3)])
    trades2, _, _ = simulate(df2, sig2, ind, cfg2)
    if len(trades2) == 1:
        ok = int(trades2.iloc[0]["entry_bar"]) == 3 and abs(float(trades2.iloc[0]["entry_price"]) - 110.0) < 1e-9
        record(item, "delay_bars=0, enter_at=close -> same-bar entry at bar 3, price=close[3]",
               "entry_bar=3, entry_price=110.0",
               f"entry_bar={int(trades2.iloc[0]['entry_bar'])}, entry_price={float(trades2.iloc[0]['entry_price'])}",
               "PASS" if ok else "FAIL", f"trades_df row: {trades2.iloc[0].to_dict()}")
    else:
        record(item, "delay_bars=0, enter_at=close -> same-bar entry",
               "1 trade", f"{len(trades2)} trades", "FAIL", str(trades2.to_dict("records")))

    # signal near end of data where entry_bar would fall off the array -> must be skipped
    df3 = make_ohlcv(5)
    ind3 = make_ind(5, atr=2.0)
    cfg3 = base_cfg()
    cfg3["execution"]["delay_bars"] = 1
    sig3 = pd.DataFrame([sig_row(bar_idx=4)])  # entry_bar = 5, len(close)=5 -> out of range
    trades3, _, _ = simulate(df3, sig3, ind3, cfg3)
    ok = len(trades3) == 0
    record(item, "signal at last bar with delay=1 -> entry_bar out of range, must be skipped",
           "0 trades", f"{len(trades3)} trades", "PASS" if ok else "FAIL",
           f"trades_df: {trades3.to_dict('records')}")


# ---------------------------------------------------------------------------
# 2) EXIT TIMING (SL/TP, intrabar vs close, gap handling)
# ---------------------------------------------------------------------------
def audit_exit_timing():
    item = "2_exit_timing"
    n = 8
    df = make_ohlcv(n, base=100.0)
    # entry at bar1 open=100 (delay=1, signal at bar0), atr=2 -> sl=100-1*2=98, tp=100+2*2=104
    df.loc[df.index[3], "Low"] = 97.0   # bar3: intrabar low pierces SL (98) without gap-through-open
    df.loc[df.index[3], "Open"] = 99.5  # open still above SL level -> should fill AT sl level, not open
    ind = make_ind(n, atr=2.0)
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 1
    cfg["execution"]["enter_at"] = "open"
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    trades, eq, fills = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        ok = t["reason"] == "SL" and abs(float(t["exit_price"]) - 98.0) < 1e-9 and int(t["exit_bar"]) == 3
        record(item, "intrabar SL touch (low<=sl, open>sl) -> exit AT sl level, not at open/close",
               "reason=SL, exit_price=98.0, exit_bar=3",
               f"reason={t['reason']}, exit_price={t['exit_price']}, exit_bar={t['exit_bar']}",
               "PASS" if ok else "FAIL", f"trade row: {t.to_dict()}")
    else:
        record(item, "intrabar SL touch", "1 trade", f"{len(trades)} trades", "FAIL", str(trades.to_dict("records")))

    # gap-through-open case: open itself already below SL level
    df2 = make_ohlcv(n, base=100.0)
    df2.loc[df2.index[3], "Open"] = 95.0   # gapped below sl=98
    df2.loc[df2.index[3], "Low"] = 94.0
    ind2 = make_ind(n, atr=2.0)
    sig2 = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades2, _, _ = simulate(df2, sig2, ind2, cfg)
    if len(trades2) == 1:
        t = trades2.iloc[0]
        ok = t["reason"] == "SL_GAP" and abs(float(t["exit_price"]) - 95.0) < 1e-9
        record(item, "gap-through-open (open < sl level) -> fill AT open, reason=SL_GAP",
               "reason=SL_GAP, exit_price=95.0",
               f"reason={t['reason']}, exit_price={t['exit_price']}",
               "PASS" if ok else "FAIL", f"trade row: {t.to_dict()}")
    else:
        record(item, "gap-through-open", "1 trade", f"{len(trades2)} trades", "FAIL", str(trades2.to_dict("records")))

    # TP intrabar touch
    df3 = make_ohlcv(n, base=100.0)
    df3.loc[df3.index[3], "High"] = 105.0  # tp=104, high pierces it
    df3.loc[df3.index[3], "Open"] = 100.5
    ind3 = make_ind(n, atr=2.0)
    sig3 = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades3, _, _ = simulate(df3, sig3, ind3, cfg)
    if len(trades3) == 1:
        t = trades3.iloc[0]
        ok = t["reason"] == "TP" and abs(float(t["exit_price"]) - 104.0) < 1e-9
        record(item, "intrabar TP touch (high>=tp, open<tp) -> exit AT tp level",
               "reason=TP, exit_price=104.0",
               f"reason={t['reason']}, exit_price={t['exit_price']}",
               "PASS" if ok else "FAIL", f"trade row: {t.to_dict()}")
    else:
        record(item, "intrabar TP touch", "1 trade", f"{len(trades3)} trades", "FAIL", str(trades3.to_dict("records")))

    # short side sanity: SL above entry, TP below
    df4 = make_ohlcv(n, base=100.0)
    df4.loc[df4.index[3], "High"] = 103.0  # short sl = 100+1*2=102, high pierces
    df4.loc[df4.index[3], "Open"] = 100.5
    ind4 = make_ind(n, atr=2.0)
    sig4 = pd.DataFrame([sig_row(bar_idx=0, side="short", sl_atr=1.0, tp_atr=2.0)])
    trades4, _, _ = simulate(df4, sig4, ind4, cfg)
    if len(trades4) == 1:
        t = trades4.iloc[0]
        ok = t["reason"] == "SL" and abs(float(t["exit_price"]) - 102.0) < 1e-9
        record(item, "short side: intrabar SL touch (high>=sl) -> exit AT sl level",
               "reason=SL, exit_price=102.0",
               f"reason={t['reason']}, exit_price={t['exit_price']}",
               "PASS" if ok else "FAIL", f"trade row: {t.to_dict()}")
    else:
        record(item, "short side intrabar SL", "1 trade", f"{len(trades4)} trades", "FAIL", str(trades4.to_dict("records")))


# ---------------------------------------------------------------------------
# 3) SAME-BAR SL + TP
# ---------------------------------------------------------------------------
def audit_same_bar_sl_tp():
    item = "3_same_bar_sl_tp"
    n = 8
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 1
    cfg["execution"]["enter_at"] = "open"

    # long: entry@100, sl=98, tp=104; bar 3 both low<=98 AND high>=104
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[3], "Low"] = 90.0
    df.loc[df.index[3], "High"] = 110.0
    df.loc[df.index[3], "Open"] = 100.0   # open between sl and tp -> no gap override
    ind = make_ind(n, atr=2.0)
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        ok = t["reason"] == "SL" and abs(float(t["exit_price"]) - 98.0) < 1e-9
        record(item, "long: same bar hits both SL(98) and TP(104) -> engine must pick SL (pessimistic)",
               "reason=SL, exit_price=98.0",
               f"reason={t['reason']}, exit_price={t['exit_price']}",
               "PASS" if ok else "FAIL",
               f"code: `if hit_sl and hit_tp: exit_px, reason = sl_eff, \"SL\"  # conservative`; trade row: {t.to_dict()}")
    else:
        record(item, "long same-bar SL+TP", "1 trade", f"{len(trades)} trades", "FAIL", str(trades.to_dict("records")))

    # short: entry@100, sl=102, tp=96; bar 3 both high>=102 AND low<=96
    df2 = make_ohlcv(n, base=100.0)
    df2.loc[df2.index[3], "High"] = 110.0
    df2.loc[df2.index[3], "Low"] = 90.0
    df2.loc[df2.index[3], "Open"] = 100.0
    ind2 = make_ind(n, atr=2.0)
    sig2 = pd.DataFrame([sig_row(bar_idx=0, side="short", sl_atr=1.0, tp_atr=2.0)])
    trades2, _, _ = simulate(df2, sig2, ind2, cfg)
    if len(trades2) == 1:
        t = trades2.iloc[0]
        ok = t["reason"] == "SL" and abs(float(t["exit_price"]) - 102.0) < 1e-9
        record(item, "short: same bar hits both SL(102) and TP(96) -> engine must pick SL (pessimistic)",
               "reason=SL, exit_price=102.0",
               f"reason={t['reason']}, exit_price={t['exit_price']}",
               "PASS" if ok else "FAIL", f"trade row: {t.to_dict()}")
    else:
        record(item, "short same-bar SL+TP", "1 trade", f"{len(trades2)} trades", "FAIL", str(trades2.to_dict("records")))


# ---------------------------------------------------------------------------
# 4) ATR-BASED POSITION SIZING
# ---------------------------------------------------------------------------
def audit_position_sizing():
    item = "4_position_sizing"
    n = 5
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 0
    cfg["execution"]["enter_at"] = "close"
    cfg["risk"]["capital"] = 1_000_000.0
    cfg["risk"]["max_risk_per_trade"] = 0.005
    cfg["risk"]["position_sizing"] = "atr"

    # base case: atr=2, sl_atr=1 -> stop_dist=2, risk_cap=1e6*0.005=5000, shares=5000/2=2500
    # NOTE: trades_df only contains CLOSED trades, so a bar must be added that
    # actually triggers TP/SL — otherwise a correctly-sized, still-open
    # position produces 0 rows here and would look like a sizing failure
    # when it is really a test-harness gap, not an engine issue.
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[1], "High"] = 110.0   # closes any tp_atr<=2.5 long entered at 100 with atr<=4
    ind = make_ind(n, atr=2.0)
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    expected_shares = (1_000_000.0 * 0.005) / (2.0 * 1.0)
    if len(trades) == 1:
        actual_shares = float(trades.iloc[0]["size_shares"])
        ok = abs(actual_shares - expected_shares) < 1e-6
        record(item, "shares = (equity*max_risk_per_trade)/(atr*sl_atr)  [atr=2, sl_atr=1]",
               expected_shares, actual_shares, "PASS" if ok else "FAIL",
               f"trade row: {trades.iloc[0].to_dict()}")
    else:
        record(item, "base sizing case", "1 trade", f"{len(trades)} trades", "FAIL", str(trades.to_dict("records")))

    # doubling ATR should halve shares
    ind_2x = make_ind(n, atr=4.0)
    trades_2x, _, _ = simulate(df, sig, ind_2x, cfg)
    if len(trades) == 1 and len(trades_2x) == 1:
        s1 = float(trades.iloc[0]["size_shares"])
        s2 = float(trades_2x.iloc[0]["size_shares"])
        ok = abs(s2 - s1 / 2.0) < 1e-6
        record(item, "doubling ATR (2 -> 4) halves position size, all else equal",
               f"{s1/2.0}", f"{s2}", "PASS" if ok else "FAIL",
               f"atr=2 shares={s1}, atr=4 shares={s2}")

    # ATR = 0: stop_dist floored at max(1e-8, atr*sl_atr_k)=1e-8 -> huge implied shares,
    # but capped by `shares = min(shares, equity/entry_px)` -> should equal equity/entry_px
    ind0 = make_ind(n, atr=0.0)
    trades0, _, _ = simulate(df, sig, ind0, cfg)
    expected_capped = cfg["risk"]["capital"] / 100.0   # entry price 100
    if len(trades0) == 1:
        actual = float(trades0.iloc[0]["size_shares"])
        ok = abs(actual - expected_capped) < 1e-6
        verdict = "PASS" if ok else "INVESTIGATE"
        record(item, "ATR=0: risk-based sizing formula degenerates (division by ~1e-8) but is "
                     "capped by `min(shares, equity/entry_px)` -> full-equity position, not a crash/NaN",
               f"shares == equity/entry_px == {expected_capped}", actual, verdict,
               "ATR=0 does not raise or produce NaN, but it silently converts a risk-sized trade "
               "into a full-notional (unlimited-risk) trade with no SL distance protection — the "
               "risk cap is defeated, not enforced, when ATR data is missing/zero. "
               f"trade row: {trades0.iloc[0].to_dict()}")
    else:
        record(item, "ATR=0 sizing", "1 trade", f"{len(trades0)} trades", "INVESTIGATE", str(trades0.to_dict("records")))

    # sanity: sizing never implies notional beyond available equity (fixed_cash method too)
    cfg_fc = copy.deepcopy(cfg)
    cfg_fc["risk"]["position_sizing"] = "fixed_cash"
    cfg_fc["risk"]["fixed_cash_per_trade"] = 50_000.0
    trades_fc, _, _ = simulate(df, sig, ind, cfg_fc)
    if len(trades_fc) == 1:
        notional = float(trades_fc.iloc[0]["size_shares"]) * float(trades_fc.iloc[0]["entry_price"])
        ok = abs(notional - 50_000.0) < 1.0
        record(item, "fixed_cash sizing: notional == fixed_cash_per_trade (50,000)",
               50_000.0, notional, "PASS" if ok else "FAIL", f"trade row: {trades_fc.iloc[0].to_dict()}")


# ---------------------------------------------------------------------------
# 5) TRANSACTION COSTS
# ---------------------------------------------------------------------------
def audit_transaction_costs():
    item = "5_transaction_costs"
    n = 6
    cfg_default = base_cfg()  # no "costs" key at all
    cfg_default["execution"]["delay_bars"] = 0
    cfg_default["execution"]["enter_at"] = "close"

    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[3], "High"] = 110.0   # hits tp
    ind = make_ind(n, atr=2.0)
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])

    # A) costs omitted entirely
    trades_a, _, _ = simulate(df, sig, ind, cfg_default)
    if len(trades_a) == 1:
        t = trades_a.iloc[0]
        ok = (abs(float(t["entry_price"]) - 100.0) < 1e-9 and
              abs(float(t["exit_price"]) - 104.0) < 1e-9 and
              abs(float(t["fees"])) < 1e-9)
        expected_shares = (1_000_000.0 * 0.005) / (2.0 * 1.0)
        expected_pnl = (104.0 - 100.0) * expected_shares
        pnl_ok = abs(float(t["pnl"]) - expected_pnl) < 1e-6
        record(item, "(A) costs key absent -> _cost_cfg defaults ALL to 0.0: no slippage, no fees, "
                     "entry/exit prices unadjusted, fees=0.0, pnl == raw (exit-entry)*shares",
               f"entry=100.0, exit=104.0, fees=0.0, pnl={expected_pnl}",
               f"entry={t['entry_price']}, exit={t['exit_price']}, fees={t['fees']}, pnl={t['pnl']}",
               "PASS" if (ok and pnl_ok) else "FAIL",
               "sim/engine.py::_cost_cfg reads cfg.get('costs', {}) — pipeline.py's BASE cfg dict "
               "(module top, before __main__ mutates it) has no 'costs' key, so any caller that "
               "reuses that base dict without pipeline.py's later `cfg.setdefault('costs', ...)` "
               f"block runs with zero costs by default. trade row: {t.to_dict()}")
    else:
        record(item, "(A) zero-cost baseline", "1 trade", f"{len(trades_a)} trades", "FAIL", str(trades_a.to_dict("records")))

    # B) production pipeline.py costs: per_share=0, per_trade=0, bps_fee=0, slippage_bps=2.0
    cfg_prod = copy.deepcopy(cfg_default)
    cfg_prod["costs"] = {"per_share": 0.0, "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 2.0}
    trades_b, _, _ = simulate(df, sig, ind, cfg_prod)
    if len(trades_b) == 1:
        t = trades_b.iloc[0]
        # long entry: raw_px * (1 + slip); long exit: raw exit level * (1 - slip)
        expected_entry = 100.0 * (1 + 2.0 / 10_000.0)
        expected_exit = 104.0 * (1 - 2.0 / 10_000.0)
        expected_shares = (1_000_000.0 * 0.005) / (2.0 * 1.0)
        expected_pnl = (expected_exit - expected_entry) * expected_shares  # fees=0 (per_share/per_trade/bps_fee all 0)
        ok = (abs(float(t["entry_price"]) - expected_entry) < 1e-6 and
              abs(float(t["exit_price"]) - expected_exit) < 1e-6 and
              abs(float(t["fees"])) < 1e-9 and
              abs(float(t["pnl"]) - expected_pnl) < 1e-6)
        record(item, "(B) production costs (slippage_bps=2.0, all fee fields 0): entry pays +2bps, "
                     "exit receives -2bps (adverse both ways), fees still 0.0, pnl reflects slippage only",
               f"entry={expected_entry:.4f}, exit={expected_exit:.4f}, fees=0.0, pnl={expected_pnl:.4f}",
               f"entry={t['entry_price']}, exit={t['exit_price']}, fees={t['fees']}, pnl={t['pnl']}",
               "PASS" if ok else "FAIL",
               "pipeline.py __main__ sets cfg['costs'] = (per_share=0, per_trade=0, bps_fee=0, "
               "slippage_bps=2.0) AFTER the module-level cfg used for signal/detection work is "
               "defined — i.e. non-zero cost usage is opt-in and caller-specific, not the engine's "
               f"own default. trade row: {t.to_dict()}")
    else:
        record(item, "(B) production cost config", "1 trade", f"{len(trades_b)} trades", "FAIL", str(trades_b.to_dict("records")))

    # C) non-zero per_share / per_trade / bps_fee fee accounting sanity
    cfg_fees = copy.deepcopy(cfg_default)
    cfg_fees["costs"] = {"per_share": 0.01, "per_trade": 20.0, "bps_fee": 5.0, "slippage_bps": 0.0}
    trades_c, _, _ = simulate(df, sig, ind, cfg_fees)
    if len(trades_c) == 1:
        t = trades_c.iloc[0]
        shares = float(t["size_shares"])
        entry_px = float(t["entry_price"])
        exit_px = float(t["exit_price"])
        fee_entry_expected = 20.0 + 0.01 * shares + (5.0 / 10_000.0) * abs(entry_px * shares)
        fee_exit_expected = 20.0 + 0.01 * shares + (5.0 / 10_000.0) * abs(exit_px * shares)
        expected_fees = fee_entry_expected + fee_exit_expected
        ok = abs(float(t["fees"]) - expected_fees) < 1e-4
        record(item, "(C) fee formula: per_trade + per_share*shares + bps_fee*notional/1e4, charged on both fills",
               expected_fees, float(t["fees"]), "PASS" if ok else "FAIL", f"trade row: {t.to_dict()}")
    else:
        record(item, "(C) fee accounting", "1 trade", f"{len(trades_c)} trades", "FAIL", str(trades_c.to_dict("records")))


# ---------------------------------------------------------------------------
# 6) MAX_OPEN_POSITIONS
# ---------------------------------------------------------------------------
def audit_max_open_positions():
    item = "6_max_open_positions"
    n = 10
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 0
    cfg["execution"]["enter_at"] = "close"
    cfg["execution"]["max_open_positions"] = 1

    # NOTE: entries are checked via the `fills` frame (ENTRY events are recorded
    # immediately on fill) rather than trades_df, which only contains CLOSED
    # trades — an sl_atr/tp_atr wide enough to never trigger would otherwise
    # show 0 rows in trades_df regardless of whether entries behaved correctly.
    df = make_ohlcv(n, base=100.0)
    ind = make_ind(n, atr=2.0)
    # two signals same bar, both would otherwise fill; max_open=1
    sig = pd.DataFrame([
        sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=50.0, tag="A"),
        sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=50.0, tag="B"),
    ])
    trades, _, fills = simulate(df, sig, ind, cfg)
    entry_tags = list(fills.loc[fills["type"] == "ENTRY", "tag"]) if len(fills) else []
    ok = entry_tags == ["A"]
    record(item, "max_open_positions=1, 2 simultaneous signals -> only the first (by sig order) fills, "
                 "second is dropped (not queued for later)",
           "ENTRY fills == ['A']", f"ENTRY fills == {entry_tags}",
           "PASS" if ok else "FAIL", f"fills: {fills.to_dict('records')}")

    # second signal on a LATER bar while position A still open -> also must be skipped
    sig2 = pd.DataFrame([
        sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=50.0, tag="A"),
        sig_row(bar_idx=2, side="long", sl_atr=1.0, tp_atr=50.0, tag="B"),
    ])
    _, _, fills2 = simulate(df, sig2, ind, cfg)
    entry_tags2 = list(fills2.loc[fills2["type"] == "ENTRY", "tag"]) if len(fills2) else []
    ok2 = entry_tags2 == ["A"]
    record(item, "max_open_positions=1, position A still open when signal B fires on a later bar -> B skipped entirely",
           "ENTRY fills == ['A']", f"ENTRY fills == {entry_tags2}",
           "PASS" if ok2 else "FAIL", f"fills: {fills2.to_dict('records')}")

    # max_open_positions=2 -> both should fill (checked via ENTRY fills, positions never closed)
    cfg2 = copy.deepcopy(cfg)
    cfg2["execution"]["max_open_positions"] = 2
    _, _, fills3 = simulate(df, sig, ind, cfg2)
    entry_tags3 = sorted(fills3.loc[fills3["type"] == "ENTRY", "tag"]) if len(fills3) else []
    ok3 = entry_tags3 == ["A", "B"]
    record(item, "max_open_positions=2, 2 simultaneous signals -> both fill",
           "ENTRY fills == ['A', 'B']", f"ENTRY fills == {entry_tags3}",
           "PASS" if ok3 else "FAIL", f"fills: {fills3.to_dict('records')}")


# ---------------------------------------------------------------------------
# 7) CAPITAL / EQUITY / P&L ACCOUNTING
# ---------------------------------------------------------------------------
def audit_capital_accounting():
    item = "7_capital_accounting"
    n = 10
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 0
    cfg["execution"]["enter_at"] = "close"
    cfg["risk"]["capital"] = 1_000_000.0
    cfg["risk"]["max_risk_per_trade"] = 0.005

    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[2], "High"] = 106.0   # trade 1 exits TP on bar 2
    df.loc[df.index[6], "Low"] = 92.0     # trade 2 exits SL on bar 6
    ind = make_ind(n, atr=2.0)
    sig = pd.DataFrame([
        sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0, tag="T1"),
        sig_row(bar_idx=3, side="long", sl_atr=1.0, tp_atr=2.0, tag="T2"),
    ])
    trades, eq, _ = simulate(df, sig, ind, cfg)

    # Position sizing re-reads `equity` at EACH entry (see _size_shares), so
    # trade 2's share count must be computed off the equity AFTER trade 1's
    # realized pnl, not off the static starting capital — sizing compounds
    # between sequential trades by design.
    shares1 = (1_000_000.0 * 0.005) / (2.0 * 1.0)
    pnl1_expected = (104.0 - 100.0) * shares1   # tp
    equity_after_t1 = 1_000_000.0 + pnl1_expected
    shares2 = (equity_after_t1 * 0.005) / (2.0 * 1.0)
    pnl2_expected = (98.0 - 100.0) * shares2    # sl
    expected_final_equity = equity_after_t1 + pnl2_expected

    ok_trades = len(trades) == 2
    if ok_trades:
        pnl1_actual = float(trades.iloc[0]["pnl"])
        pnl2_actual = float(trades.iloc[1]["pnl"])
        ok_pnl = abs(pnl1_actual - pnl1_expected) < 1e-6 and abs(pnl2_actual - pnl2_expected) < 1e-6
        record(item, "per-trade PnL matches independently computed (exit-entry)*shares for each of 2 "
                     "sequential trades, with trade 2's size correctly compounding off post-trade-1 equity",
               f"pnl1={pnl1_expected}, pnl2={pnl2_expected} (shares1={shares1}, shares2={shares2})",
               f"pnl1={pnl1_actual}, pnl2={pnl2_actual}",
               "PASS" if ok_pnl else "FAIL", f"trades: {trades.to_dict('records')}")

    final_eq_actual = float(eq.iloc[-1])
    ok_final = abs(final_eq_actual - expected_final_equity) < 1e-6
    record(item, "final equity == start_capital + sum(realized pnl); equity only updates on exits (realized-only, mtm off)",
           expected_final_equity, final_eq_actual, "PASS" if ok_final else "FAIL",
           f"equity_curve tail: {eq.tail(5).tolist()}")

    # sizing never implies notional beyond available equity (checked structurally via the sizing cap)
    # cross-check: entry notional <= starting equity for a case with a very tight/near-zero ATR
    ind_tiny = make_ind(n, atr=0.0001)
    sig_tiny = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=50.0)])
    trades_tiny, _, _ = simulate(df, sig_tiny, ind_tiny, cfg)
    if len(trades_tiny) == 1:
        notional = float(trades_tiny.iloc[0]["size_shares"]) * float(trades_tiny.iloc[0]["entry_price"])
        ok = notional <= cfg["risk"]["capital"] + 1e-6
        record(item, "near-zero ATR: notional must never exceed starting capital (equity/entry_px cap enforced)",
               f"<= {cfg['risk']['capital']}", notional, "PASS" if ok else "FAIL",
               f"trade row: {trades_tiny.iloc[0].to_dict()}")


# ---------------------------------------------------------------------------
# 8) LOOK-AHEAD / DELAYED-ENTRY BEHAVIOR  (delay_bars=2 explicit test)
# ---------------------------------------------------------------------------
def audit_lookahead_delay():
    item = "8_lookahead_delay"
    n = 8
    cfg = base_cfg()
    cfg["execution"]["delay_bars"] = 2
    cfg["execution"]["enter_at"] = "open"
    cfg["execution"]["max_open_positions"] = 1

    # signal at bar 2 -> entry should be bar 4 (t+2). atr=2, sl_atr=1 -> sl = open[4]-2.
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[4], "Open"] = 100.0     # entry price -> sl=98, tp (sl_atr wide) irrelevant here
    # bar 3 == t+1 (ONE BAR BEFORE actual entry_bar=4): drive low far below the
    # would-be SL level to see if the engine exits a position that hasn't
    # actually entered yet.
    df.loc[df.index[3], "Low"] = 50.0
    df.loc[df.index[3], "Open"] = 100.0
    ind = make_ind(n, atr=2.0)
    sig = pd.DataFrame([sig_row(bar_idx=2, side="long", sl_atr=1.0, tp_atr=50.0, delay_bars=2)])

    trades, eq, fills = simulate(df, sig, ind, cfg)

    if len(trades) == 0:
        record(item, "delay_bars=2: signal@2, entry should be bar4; bar3's low=50 must NOT be able to "
                     "exit a position that hasn't entered yet",
               "0 or 1 trades, and IF any trade exists entry_bar<=exit_bar",
               "0 trades", "INVESTIGATE",
               "No trade was produced at all — could not evaluate the entry_bar vs exit_bar ordering "
               f"claim. fills: {fills.to_dict('records') if len(fills) else '[]'}")
    else:
        t = trades.iloc[0]
        entry_bar = int(t["entry_bar"])
        exit_bar = int(t["exit_bar"])
        ordering_ok = exit_bar >= entry_bar
        verdict = "PASS" if (ordering_ok and entry_bar == 4) else "FAIL"
        record(item,
               "delay_bars=2: signal@bar2 -> entry_bar must be 4 (t+2); the position must not be "
               "exit-checked against bar 3 (t+1, one bar BEFORE the real entry) — exit_bar must be >= entry_bar",
               "entry_bar=4, exit_bar>=4 (position untouched at bar 3 despite low=50 there)",
               f"entry_bar={entry_bar}, exit_bar={exit_bar}, reason={t['reason']}",
               verdict,
               "sim/engine.py appends the position dict to `open_positions` in the SAME loop "
               "iteration `t` the signal fires (inside the `for t in range(len(close))` entries "
               "block), tagged with pos['entry_bar']=t+delay for LATER reference, but the dict "
               "itself is already live in `open_positions` starting the very next iteration "
               "(t+1) — one full bar (or more, for delay_bars>1) BEFORE the position has actually "
               "been filled at its stated entry_bar. The exit-check block at the top of the loop "
               "does not filter open_positions by `t >= pos['entry_bar']`, so it evaluates this "
               "not-yet-entered position's SL/TP against every intervening bar's high/low. "
               f"trade row: {t.to_dict()}; fills: {fills.to_dict('records')}")

    # Now the production-default case (delay_bars=1) for comparison: the "one bar before
    # entry" window doesn't exist (added at t, first checked at t+1 == entry_bar itself),
    # so this should NOT reproduce the issue.
    cfg1 = copy.deepcopy(cfg)
    cfg1["execution"]["delay_bars"] = 1
    df1 = make_ohlcv(n, base=100.0)
    df1.loc[df1.index[3], "Open"] = 100.0
    df1.loc[df1.index[3], "Low"] = 50.0   # this IS the entry bar (t+1) under delay=1
    ind1 = make_ind(n, atr=2.0)
    sig1 = pd.DataFrame([sig_row(bar_idx=2, side="long", sl_atr=1.0, tp_atr=50.0, delay_bars=1)])
    trades1, _, _ = simulate(df1, sig1, ind1, cfg1)
    if len(trades1) == 1:
        t = trades1.iloc[0]
        entry_bar = int(t["entry_bar"])
        exit_bar = int(t["exit_bar"])
        ok = entry_bar == 3 and exit_bar == 3   # same-day SL after opening at 100, sl=98, low=50 -> hits SL same bar
        record(item, "delay_bars=1 (production default): entry_bar==exit_bar==3 is CORRECT here "
                     "(entry and the SL breach are the same, real entry bar) — included for contrast with the delay=2 case above",
               "entry_bar=3, exit_bar=3, reason=SL",
               f"entry_bar={entry_bar}, exit_bar={exit_bar}, reason={t['reason']}",
               "PASS" if ok else "INVESTIGATE", f"trade row: {t.to_dict()}")
    else:
        record(item, "delay_bars=1 contrast case", "1 trade", f"{len(trades1)} trades", "INVESTIGATE", str(trades1.to_dict("records")))


# ---------------------------------------------------------------------------
# 9) BOUNDARY CASES
# ---------------------------------------------------------------------------
def audit_boundary_cases():
    item = "9_boundary_cases"

    # empty signals
    df = make_ohlcv(5)
    ind = make_ind(5, atr=2.0)
    cfg = base_cfg()
    trades, eq, fills = simulate(df, pd.DataFrame(columns=["bar_idx", "side", "sl_atr", "tp_atr", "tag"]), ind, cfg)
    ok = len(trades) == 0 and len(eq) == 5 and (eq == cfg["risk"]["capital"]).all()
    record(item, "empty signals df -> 0 trades, equity curve flat at starting capital for all bars",
           "0 trades, flat equity", f"{len(trades)} trades, equity={eq.tolist()}",
           "PASS" if ok else "FAIL", f"eq: {eq.tolist()}")

    # missing ATR column entirely -> zeros, ties into position-sizing item 4's ATR=0 finding
    df2 = make_ohlcv(6, base=100.0)
    ind_no_atr = pd.DataFrame(index=df2.index)  # no "ATR" column at all
    cfg2 = base_cfg()
    cfg2["execution"]["delay_bars"] = 0
    cfg2["execution"]["enter_at"] = "close"
    sig2 = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades2, _, _ = simulate(df2, sig2, ind_no_atr, cfg2)
    if len(trades2) == 1:
        t = trades2.iloc[0]
        # atr=0 -> sl=tp=entry_price exactly; next bar's flat OHLC (=entry price) touches both -> SL wins (item 3 rule)
        ok = abs(float(t["sl_price"]) - 100.0) < 1e-9 and abs(float(t["tp_price"]) - 100.0) < 1e-9
        record(item, "ind_df has no ATR column -> engine falls back to zeros -> sl_price==tp_price==entry_price "
                     "(zero-width stop/target); next bar's flat OHLC satisfies both simultaneously -> exits SL (pessimistic rule from item 3)",
               "sl_price=tp_price=100.0, immediate SL exit next bar",
               f"sl_price={t['sl_price']}, tp_price={t['tp_price']}, reason={t['reason']}, exit_bar={t['exit_bar']}",
               "PASS" if ok else "INVESTIGATE",
               f"trade row: {t.to_dict()}. Combined with item 4's ATR=0 finding: missing ATR data "
               "silently produces a full-notional position with a zero-distance stop that fires on "
               "the very next bar regardless of direction, rather than being rejected or held back "
               "by a data-quality check.")
    else:
        record(item, "missing ATR column entirely", "1 trade", f"{len(trades2)} trades", "INVESTIGATE", str(trades2.to_dict("records")))

    # simultaneous signals, same bar, different tags, max_open=2 -> fill order is sig row order
    df3 = make_ohlcv(6, base=100.0)
    ind3 = make_ind(6, atr=2.0)
    cfg3 = base_cfg()
    cfg3["execution"]["delay_bars"] = 0
    cfg3["execution"]["enter_at"] = "close"
    cfg3["execution"]["max_open_positions"] = 5
    sig3 = pd.DataFrame([
        sig_row(bar_idx=1, side="long", sl_atr=1.0, tp_atr=2.0, tag="Z_last_alpha"),
        sig_row(bar_idx=1, side="long", sl_atr=1.0, tp_atr=2.0, tag="A_first_alpha"),
    ])
    # tp_atr=2.0 is never reached on a flat series, so trades_df (closed-only)
    # would stay empty regardless of fill order — use `fills` (ENTRY events
    # are recorded immediately) to check ordering instead.
    _, _, fills3 = simulate(df3, sig3, ind3, cfg3)
    entry_order = list(fills3.loc[fills3["type"] == "ENTRY", "tag"]) if len(fills3) else []
    ok = entry_order == ["Z_last_alpha", "A_first_alpha"]
    record(item, "simultaneous same-bar signals: fill order follows original signals-DataFrame row "
                 "order (stable sort by bar_idx), NOT alphabetical/score order",
           "ENTRY fill order = ['Z_last_alpha', 'A_first_alpha'] (input row order preserved)",
           f"ENTRY fill order = {entry_order}",
           "PASS" if ok else "INVESTIGATE",
           "Callers that rely on simultaneous signals being prioritized by score/quality must "
           "pre-sort the signals DataFrame themselves — simulate() applies no such ordering; it "
           "only stable-sorts by bar_idx. " + f"fills: {fills3.to_dict('records')}")

    # position still open at the very last bar of the dataset: mtm_on=False vs True
    df4 = make_ohlcv(6, base=100.0)
    ind4 = make_ind(6, atr=2.0)
    cfg4a = base_cfg()
    cfg4a["execution"]["delay_bars"] = 0
    cfg4a["execution"]["enter_at"] = "close"
    cfg4a["execution"]["max_hold_bars"] = 0  # never time-stops
    cfg4a["equity"]["mtm"] = False
    sig4 = pd.DataFrame([sig_row(bar_idx=1, side="long", sl_atr=100.0, tp_atr=100.0)])  # SL/TP unreachable
    trades4a, eq4a, _ = simulate(df4, sig4, ind4, cfg4a)
    ok_a = len(trades4a) == 0 and (eq4a == cfg4a["risk"]["capital"]).all()
    record(item, "mtm_on=False: position open at dataset end -> never appears in trades_df (no exit "
                 "row = no realized pnl), equity curve stays at last REALIZED value the whole time "
                 "(open position's unrealized pnl is invisible to the equity curve)",
           "0 trades, equity flat at starting capital throughout",
           f"{len(trades4a)} trades, equity={eq4a.tolist()}",
           "PASS" if ok_a else "FAIL", f"eq: {eq4a.tolist()}")

    cfg4b = copy.deepcopy(cfg4a)
    cfg4b["equity"]["mtm"] = True
    cfg4b["equity"]["mtm_haircut_bps"] = 0.0
    df4b = make_ohlcv(6, base=100.0)
    df4b.loc[df4b.index[5], "Close"] = 130.0  # mark the open long up on the final bar
    trades4b, eq4b, _ = simulate(df4b, sig4, ind4, cfg4b)
    shares = (cfg4b["risk"]["capital"] * cfg4b["risk"]["max_risk_per_trade"]) / (2.0 * 100.0)
    expected_final_eq = cfg4b["risk"]["capital"] + (130.0 - 100.0) * shares
    ok_b = len(trades4b) == 0 and abs(float(eq4b.iloc[-1]) - expected_final_eq) < 1e-6
    record(item, "mtm_on=True: still-open position at dataset end -> still 0 realized trades in "
                 "trades_df, but final equity_curve value reflects UNREALIZED mark-to-close pnl "
                 "(no haircut here) instead of flat starting capital",
           f"0 trades, final equity={expected_final_eq}",
           f"{len(trades4b)} trades, final equity={float(eq4b.iloc[-1])}",
           "PASS" if ok_b else "FAIL", f"eq tail: {eq4b.tail(3).tolist()}")


def main():
    audit_entry_timing()
    audit_exit_timing()
    audit_same_bar_sl_tp()
    audit_position_sizing()
    audit_transaction_costs()
    audit_max_open_positions()
    audit_capital_accounting()
    audit_lookahead_delay()
    audit_boundary_cases()

    report = pd.DataFrame(RESULTS)
    report.to_csv("audit_sim_engine_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(report.groupby(["item", "verdict"]).size().unstack(fill_value=0).to_string())
    print(f"\nOverall: {(report['verdict']=='PASS').sum()} PASS, "
          f"{(report['verdict']=='FAIL').sum()} FAIL, "
          f"{(report['verdict']=='INVESTIGATE').sum()} INVESTIGATE, "
          f"out of {len(report)} checks.")
    print("\nSaved: audit_sim_engine_results.csv")


if __name__ == "__main__":
    main()
