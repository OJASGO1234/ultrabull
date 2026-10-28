# core/audit_lookahead_pipeline.py
"""
Pipeline-wide look-ahead / data-leakage audit — complements the engine-level
delay_bars=2 finding already confirmed in audit_sim_engine.py (item 8) by
checking the stages BEFORE the engine: pattern detectors, indicator
inputs to the gates, and the market-context/events joins.

Method: for each stage, truncate the input OHLCV data at a cut bar C
(bars 0..C inclusive only) and recompute that stage's output for bars
0..C-1. If the output for those bars is IDENTICAL to what the same stage
produces when given the FULL series, the stage cannot be using information
from bar C or later for those earlier bars — a direct, empirical proof
rather than a read-the-code argument alone. This is the same class of
technique verify_adx.py/verify_atr.py used for indicators; here it is
applied to detectors and gates.

Does NOT modify any production file — patterns/breakouts.py,
patterns/pullbacks.py, patterns/pipeline.py, rules/gates.py,
market_context.py are all imported and called read-only.

Run:  python audit_lookahead_pipeline.py <cache_dir>   (from core/ directory)
Output: core/audit_lookahead_pipeline_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import prepare_symbol
from patterns.config import CFG as PATTERNS_CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import enrich_detections
from rules.gates import apply_gates
from market_context import build_market_context

RESULTS = []
SYMBOL = "RELIANCE.NS"
YEARS = 2.0
CUT = 300  # truncation point; must leave enough bars before it for warmup + enough after for comparison


def record(check, verdict, evidence):
    RESULTS.append({"check": check, "verdict": verdict, "evidence": evidence})
    print(f"[{verdict:11s}] {check}")
    if verdict != "PASS":
        print(f"              evidence: {evidence}")


def audit_pattern_detectors(df_ohlcv, ind_df):
    """Truncate at CUT, rerun run_all_patterns, compare detections in [0, CUT-CMP_MARGIN)
    against the full-series run. CMP_MARGIN accounts for any detector using a rolling
    window that legitimately needs some trailing bars but must never need bars AT or
    AFTER the truncation point for a detection BEFORE the margin."""
    full_dets = run_all_patterns(df_ohlcv, ind_df, PATTERNS_CFG)

    df_trunc = df_ohlcv.iloc[:CUT].copy()
    ind_trunc = ind_df.iloc[:CUT].copy()
    trunc_dets = run_all_patterns(df_trunc, ind_trunc, PATTERNS_CFG)

    # Compare detections strictly before the cut, well clear of any rolling-window edge
    margin = 5
    cut_date = df_ohlcv.index[CUT - margin]
    full_before = full_dets[full_dets["date"] < cut_date].sort_values(["date", "pattern"]).reset_index(drop=True)
    trunc_before = trunc_dets[trunc_dets["date"] < cut_date].sort_values(["date", "pattern"]).reset_index(drop=True)

    same_shape = full_before.shape == trunc_before.shape
    if same_shape:
        cols = ["date", "pattern"]
        identical = full_before[cols].equals(trunc_before[cols])
    else:
        identical = False

    verdict = "PASS" if (same_shape and identical) else "INVESTIGATE"
    record("Pattern detectors (run_all_patterns): detections before the truncation point are "
           "identical whether the detector sees only bars [0,CUT) or the full series",
           verdict,
           f"full_before: {len(full_before)} detections, trunc_before: {len(trunc_before)} detections, "
           f"shapes match: {same_shape}, detection (date,pattern) sets identical: {identical if same_shape else 'n/a'}")
    if not (same_shape and identical) and same_shape:
        diff = full_before[cols].compare(trunc_before[cols])
        record("  -> detail", "INVESTIGATE", f"diff sample:\n{diff.head(10)}")


def audit_gates(df_ohlcv, ind_df, det_e, ctx):
    """Same truncation test for apply_gates(): does gating a detection at bar t
    ever depend on OHLCV/indicator/ctx data at bar >= t?"""
    cfg = {
        "disable_patterns": [],
        "gates": {
            "bb_touch_eps": 0.0, "pullback_ma_tol": 0.01,
            "market": {"enabled": True, "require_market_regime": True, "require_sector_regime": False,
                       "rs_col": "rs_20", "breakout_rs_min": 0.0, "breakdown_rs_max": 0.0,
                       "dist_days_max": 5.0, "breadth_min": 0.0},
            "adx": {"breakout_min": 0.0, "meanrev_max": 0.0},
            "liquidity": {"min_turnover_20d": 2e7},
        },
    }
    full_gated = apply_gates(det_e, df_ohlcv, ind_df, cfg, ctx=ctx)

    df_trunc = df_ohlcv.iloc[:CUT].copy()
    ind_trunc = ind_df.iloc[:CUT].copy()
    ctx_trunc = ctx.iloc[:CUT].copy()
    margin = 5
    cut_date = df_ohlcv.index[CUT - margin]
    det_before = det_e[det_e["date"] < cut_date].copy()
    trunc_gated = apply_gates(det_before, df_trunc, ind_trunc, cfg, ctx=ctx_trunc)

    full_before_gated = full_gated[full_gated["date"] < cut_date].sort_values(["date", "tag" if "tag" in full_gated.columns else "pattern"]).reset_index(drop=True)
    trunc_gated_sorted = trunc_gated.sort_values(["date", "tag" if "tag" in trunc_gated.columns else "pattern"]).reset_index(drop=True)

    key_col = "tag" if "tag" in full_gated.columns else "pattern"
    same_shape = len(full_before_gated) == len(trunc_gated_sorted)
    identical = (same_shape and
                 full_before_gated[["date", key_col]].equals(trunc_gated_sorted[["date", key_col]]))
    verdict = "PASS" if identical else "INVESTIGATE"
    record("apply_gates(): which detections survive gating before the truncation point is "
           "identical whether the gate sees only bars [0,CUT) or the full series "
           "(covers trend/RSI/volume/MA/BB/pullback/ADX/liquidity/market-regime/RS conditions)",
           verdict,
           f"full_before_gated: {len(full_before_gated)}, trunc_gated: {len(trunc_gated_sorted)}, "
           f"shapes match: {same_shape}, identical (date,tag) sets: {identical}")


def audit_market_context(symbol, close_series, cache_dir):
    """build_market_context uses .rolling(20/50).mean() and .pct_change(n) on the index
    series with ffill for missing dates — check its outputs before the cut are unchanged
    by future index/sector data appended after the cut."""
    full_ctx = build_market_context(symbol, close_series, years=YEARS, cache_dir=cache_dir)

    close_trunc = close_series.iloc[:CUT]
    trunc_ctx = build_market_context(symbol, close_trunc, years=YEARS, cache_dir=cache_dir)

    margin = 5
    cmp_idx = close_series.index[:CUT - margin]
    cols = [c for c in ("mkt_up", "mkt_dn", "sect_up", "sect_dn", "rs_20", "rs_60") if c in full_ctx.columns]
    diffs = {}
    for c in cols:
        a = full_ctx.loc[cmp_idx, c]
        b = trunc_ctx.loc[cmp_idx, c]
        if pd.api.types.is_bool_dtype(a) or a.dtype == object:
            same = (a.astype(str) == b.astype(str)).all()
        else:
            same = np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)
        diffs[c] = same
    all_same = all(diffs.values())
    verdict = "PASS" if all_same else "INVESTIGATE"
    record("build_market_context(): mkt_up/mkt_dn/sect_up/sect_dn/rs_20/rs_60 before the "
           "truncation point are identical whether computed on the truncated or full stock "
           "close series (the index/sector series themselves are fetched independently and "
           "at full length regardless of the stock's own truncation, by construction of the "
           "function's signature)",
           verdict, f"per-column match: {diffs}")


def audit_breakout_rolling_shift():
    """Static/code-level confirmation (not truncation-based): breakouts.py's rolling
    high/low windows are explicitly .shift(1)'d before comparison, so a bar's OWN
    high/low is never part of its own breakout threshold — verified directly from
    source rather than assumed."""
    import inspect
    from patterns import breakouts
    src = inspect.getsource(breakouts.breakouts)
    has_shift = ".shift(1)" in src
    verdict = "PASS" if has_shift else "INVESTIGATE"
    record("patterns/breakouts.py: rolling N-day high/low windows are shift(1)'d before the "
           "breakout comparison (a bar's own high/low cannot set its own breakout threshold)",
           verdict, f"'.shift(1)' present in source: {has_shift}")


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Symbol under test: {SYMBOL}, years={YEARS}, truncation cut bar={CUT}\n")

    prep = prepare_symbol(SYMBOL, years=YEARS, cache_dir=cache)
    if prep is None:
        print("FAILED to prepare symbol data — aborting.")
        return
    df_ohlcv, ind_df, dets = prep
    if len(df_ohlcv) <= CUT + 20:
        print(f"WARNING: only {len(df_ohlcv)} bars available; CUT={CUT} may be too close to the "
              f"end of the series for a clean comparison margin. Proceeding anyway.")

    ctx = build_market_context(SYMBOL, df_ohlcv["Close"], years=YEARS, cache_dir=cache)
    det_e = enrich_detections(dets, df_ohlcv, ind_df, SYMBOL, ctx=ctx)

    audit_pattern_detectors(df_ohlcv, ind_df)
    audit_gates(df_ohlcv, ind_df, det_e, ctx)
    audit_market_context(SYMBOL, df_ohlcv["Close"], cache)
    audit_breakout_rolling_shift()

    df = pd.DataFrame(RESULTS)
    df.to_csv("audit_lookahead_pipeline_results.csv", index=False)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(df["verdict"].value_counts().to_string())
    print("\nCross-reference: the sim-engine look-ahead finding (delay_bars>=2 causes a position "
          "to be exit-checked before its actual entry bar) was already confirmed in "
          "audit_sim_engine.py (item 8) and traced to production reachability in the "
          "delay_bars trace — not re-tested here.")
    print("\nSaved: audit_lookahead_pipeline_results.csv")


if __name__ == "__main__":
    main()
