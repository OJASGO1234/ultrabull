# core/audit_atr_sizing_edge_cases.py
"""
Extended ATR/risk-sizing edge-case audit — follow-up to audit_sim_engine.py's
position-sizing checks (which covered ATR=0, ATR doubling, and near-zero ATR
capped-notional). This script covers the edge cases NOT yet exercised:
NaN ATR mid-series, negative ATR (should not occur upstream but tested
defensively since sim/engine.py does not itself validate the sign),
extremely large ATR, max_risk_per_trade >100%, and fixed_cash exceeding
starting equity.

Does NOT modify sim/engine.py or any production file — simulate() is
imported and called exactly as every prior audit calls it. All expected
values are computed independently before comparison.

Run:  python audit_atr_sizing_edge_cases.py   (from core/ directory)
Output: core/audit_atr_sizing_edge_cases_results.csv (+ printed report)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sim.engine import simulate

RESULTS = []


def record(check, expected, actual, verdict, evidence):
    RESULTS.append({"check": check, "expected": str(expected), "actual": str(actual),
                     "verdict": verdict, "evidence": evidence})
    print(f"[{verdict:11s}] {check}")
    if verdict != "PASS":
        print(f"              expected={expected!r}  actual={actual!r}")
        print(f"              evidence: {evidence}")


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


def audit_nan_atr_midseries():
    n = 8
    df = make_ohlcv(n, base=100.0)
    df.loc[df.index[3], "High"] = 110.0  # forces a TP if sl/tp computed sanely
    atr = np.full(n, 2.0)
    atr[0] = np.nan  # ATR is NaN specifically on the entry bar
    ind = pd.DataFrame({"ATR": atr}, index=df.index)
    cfg = base_cfg()
    # NaN sl/tp means the position can never hit either level (NaN comparisons are
    # always False) — force a TIME exit via max_hold_bars so trades_df (closed-only)
    # actually shows a row to inspect, rather than 0 rows looking like "no bug".
    cfg["execution"]["max_hold_bars"] = 3
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        # atr_e = float(nan) if len(atr) else 0.0 -- engine does NOT check np.isnan, so
        # atr_e stays NaN, and sl/tp/shares all become NaN too.
        sl_is_nan = pd.isna(t["sl_price"])
        tp_is_nan = pd.isna(t["tp_price"])
        shares_is_nan = pd.isna(t["size_shares"])
        verdict = "INVESTIGATE" if (sl_is_nan or tp_is_nan or shares_is_nan) else "PASS"
        record("NaN ATR on entry bar: engine's `atr_e = float(atr[entry_bar])` does not "
               "check np.isnan, so a NaN ATR silently propagates into sl_price/tp_price/size_shares",
               "engine should either reject the signal or produce a well-defined sl/tp/shares",
               f"sl_price={t['sl_price']}, tp_price={t['tp_price']}, size_shares={t['size_shares']}, reason={t['reason']}",
               verdict,
               f"trade row: {t.to_dict()}. A NaN sl_price/tp_price means `low[t]<=sl_eff`/"
               "`high[t]>=tp_eff` compare against NaN, which numpy evaluates as False for both — "
               "so a NaN-ATR position can never hit its (undefined) SL/TP and will ride until "
               "max_hold_bars or dataset end, with an undefined/NaN share count.")
    else:
        record("NaN ATR on entry bar", "1 trade", f"{len(trades)} trades", "INVESTIGATE", str(trades.to_dict("records")))


def audit_negative_atr():
    """ATR should never be negative in practice (it's built from |high-low| etc.),
    but sim/engine.py itself performs no validation on the sign — this checks what
    happens if a negative value reaches it anyway (e.g. from a future indicator bug
    or bad data feed), defensively."""
    n = 6
    df = make_ohlcv(n, base=100.0)
    ind = pd.DataFrame({"ATR": np.full(n, -2.0)}, index=df.index)
    cfg = base_cfg()
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        # sl = raw_px - sl_atr*atr_e = 100 - 1*(-2) = 102 (ABOVE entry for a LONG stop-loss)
        # tp = raw_px + tp_atr*atr_e = 100 + 2*(-2) = 96  (BELOW entry for a long take-profit)
        sl_above_entry = float(t["sl_price"]) > float(t["entry_price"])
        tp_below_entry = float(t["tp_price"]) < float(t["entry_price"])
        verdict = "FAIL" if (sl_above_entry and tp_below_entry) else "INVESTIGATE"
        record("Negative ATR (defensive, should never occur upstream): sl/tp formulas are "
               "`raw_px -/+ atr_e*k` with no sign check on atr_e",
               "engine should reject or clamp a negative ATR rather than invert SL/TP sides",
               f"entry={t['entry_price']}, sl_price={t['sl_price']} (above entry: {sl_above_entry}), "
               f"tp_price={t['tp_price']} (below entry: {tp_below_entry}), reason={t['reason']}",
               verdict,
               f"trade row: {t.to_dict()}. With ATR=-2, a LONG position's stop-loss (102) sits ABOVE "
               "its entry (100) and its take-profit (96) sits BELOW entry — i.e. the SL/TP sides are "
               "inverted from what 'stop-loss'/'take-profit' are supposed to mean for a long. This is "
               "a defensive/edge-case finding: negative ATR should never reach simulate() from the "
               "production indicator pipeline (ATR is derived from |high-low|/|high-close|/|low-close|, "
               "all non-negative), but engine.py itself has no guard against it if it ever did.")
    else:
        record("Negative ATR", "1 trade", f"{len(trades)} trades", "INVESTIGATE", str(trades.to_dict("records")))


def audit_extremely_large_atr():
    n = 6
    df = make_ohlcv(n, base=100.0)
    ind = pd.DataFrame({"ATR": np.full(n, 1_000_000.0)}, index=df.index)  # absurdly large ATR
    cfg = base_cfg()
    cfg["execution"]["max_hold_bars"] = 2  # sl/tp are unreachable at this ATR; force a TIME exit
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        expected_shares = (1_000_000.0 * 0.005) / (1_000_000.0 * 1.0)  # = 0.005 shares
        ok = abs(float(t["size_shares"]) - expected_shares) < 1e-9
        record("Extremely large ATR (1,000,000): stop distance dominates, shares should shrink to "
               "a tiny (near-zero, non-integer) fraction rather than error",
               expected_shares, float(t["size_shares"]), "PASS" if ok else "INVESTIGATE",
               f"trade row: {t.to_dict()}. Note: size_shares=0.005 is not a whole number of shares — "
               "sim/engine.py's sizing model treats shares as continuous, never rounds to a tradable "
               "lot size. This is a modeling simplification, not unique to the large-ATR case, but "
               "it's most visible here.")
    else:
        record("Extremely large ATR", "1 trade", f"{len(trades)} trades", "INVESTIGATE", str(trades.to_dict("records")))


def audit_risk_over_100pct():
    n = 6
    df = make_ohlcv(n, base=100.0)
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    df.loc[df.index[1], "High"] = 110.0  # force a TP close so trades_df shows a row
    cfg = base_cfg()
    cfg["risk"]["max_risk_per_trade"] = 1.5   # 150% of equity risked per trade -- clearly invalid input
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        notional = float(t["size_shares"]) * float(t["entry_price"])
        capped_at_equity = abs(notional - cfg["risk"]["capital"]) < 1.0
        record("max_risk_per_trade=1.5 (150%, clearly invalid config): does the "
               "`min(shares, equity/entry_px)` notional cap still protect the account?",
               f"notional capped at equity ({cfg['risk']['capital']})",
               f"notional={notional}", "PASS" if capped_at_equity else "FAIL",
               f"trade row: {t.to_dict()}. sim/engine.py performs NO validation that "
               "max_risk_per_trade is in a sane range (e.g. <=1.0) — a config typo (1.5 instead of "
               "0.015, say) is caught only incidentally by the equity/entry_px notional cap, not by "
               "any explicit bounds check on the risk parameter itself.")
    else:
        record("max_risk_per_trade > 100%", "1 trade", f"{len(trades)} trades", "INVESTIGATE", str(trades.to_dict("records")))


def audit_fixed_cash_exceeds_equity():
    n = 6
    df = make_ohlcv(n, base=100.0)
    ind = pd.DataFrame({"ATR": np.full(n, 2.0)}, index=df.index)
    df.loc[df.index[1], "High"] = 110.0  # force a TP close so trades_df shows a row
    cfg = base_cfg()
    cfg["risk"]["position_sizing"] = "fixed_cash"
    cfg["risk"]["capital"] = 10_000.0
    cfg["risk"]["fixed_cash_per_trade"] = 50_000.0  # 5x the starting capital
    sig = pd.DataFrame([sig_row(bar_idx=0, side="long", sl_atr=1.0, tp_atr=2.0)])
    trades, _, _ = simulate(df, sig, ind, cfg)
    if len(trades) == 1:
        t = trades.iloc[0]
        notional = float(t["size_shares"]) * float(t["entry_price"])
        # _size_shares' fixed_cash branch returns cash/entry_px directly, with NO equity cap
        # applied in that branch (the `min(shares, equity/entry_px)` line only exists in the
        # "atr" branch of _size_shares) — so this is expected to FAIL the "notional <= equity" check.
        within_equity = notional <= cfg["risk"]["capital"] + 1e-6
        record("fixed_cash sizing: fixed_cash_per_trade (50,000) > starting capital (10,000) — "
               "is notional still capped at available equity?",
               f"notional <= {cfg['risk']['capital']} (equity cap enforced)",
               f"notional={notional}",
               "PASS" if within_equity else "FAIL",
               f"trade row: {t.to_dict()}. sim/engine.py::_size_shares' `fixed_cash` branch is:\n"
               "  `return max(0.0, cash) / max(1e-9, entry_px)`\n"
               "with NO `min(shares, equity/entry_px)` cap — that cap exists only in the `atr` "
               "sizing branch. So `position_sizing='fixed_cash'` can size a position at MORE than "
               "the account's entire starting capital if `fixed_cash_per_trade` is misconfigured "
               "relative to `capital`, with no protection at all.")
    else:
        record("fixed_cash exceeds equity", "1 trade", f"{len(trades)} trades", "INVESTIGATE", str(trades.to_dict("records")))


def main():
    audit_nan_atr_midseries()
    audit_negative_atr()
    audit_extremely_large_atr()
    audit_risk_over_100pct()
    audit_fixed_cash_exceeds_equity()

    df = pd.DataFrame(RESULTS)
    df.to_csv("audit_atr_sizing_edge_cases_results.csv", index=False)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(df["verdict"].value_counts().to_string())
    print("\nSaved: audit_atr_sizing_edge_cases_results.csv")
    print("\nNOTE: findings here are documented only, per instructions — sim/engine.py was NOT modified.")


if __name__ == "__main__":
    main()
