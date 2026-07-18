# core/evaluate_gates.py
"""
Measure whether the gate stack improves the detection pool.

Compares mean forward-return excess (event_study methodology) of:
  raw      — all detections
  legacy   — apply_gates without market context (trend/vol/RSI/band gates)
  market   — apply_gates WITH market/sector regime + relative-strength gates

Run:  python evaluate_gates.py <cache_dir>   (from the core/ directory)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import os

from event_study import UNIVERSE, prepare_symbol, compute_event_rows
from patterns.postprocess import enrich_detections
from rules.gates import apply_gates
from market_context import build_market_context
from stock_explorer import load_events


def _load_events_cached(symbol: str, df_ohlcv, cache_dir: str | None):
    if cache_dir:
        path = os.path.join(cache_dir, f"events_{symbol.replace('&','_')}.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    ev = load_events(symbol, df_ohlcv.rename(columns=str.lower))
    if cache_dir:
        ev.to_pickle(path)
    return ev

GATE_CFG = {
    # NOTE: deliberately does NOT include the short-side prunes (BB_SellFade,
    # Breakdown20/55) so their stats stay visible for future re-evaluation.
    "disable_patterns": ["RSI_60_Down", "RSI_40_Up", "Doji", "MACD_Cross_Zeroish_Up"],
    "gates": {
        "bb_touch_eps": 0.0,
        "pullback_ma_tol": 0.01,
        "market": {
            "enabled": True,
            "require_market_regime": True,
            "require_sector_regime": False,
            "rs_col": "rs_20",
            "breakout_rs_min": 0.0,
            "breakdown_rs_max": 0.0,
        },
        "events": {
            "enabled": True,
            "earnings_blackout_before": 3,
            "earnings_blackout_after": 1,
        },
        # ADX gates evaluated and REJECTED: they select late, extended entries
        # (breakout excess @10 fell from +29.7 to +6.5 bps with breakout_min=20)
        "adx": {"breakout_min": 0.0, "meanrev_max": 0.0},
        "liquidity": {"min_turnover_20d": 2e7},
    },
}


def collect(years: float = 4, cache_dir: str | None = None):
    import copy
    cfg_breadth = copy.deepcopy(GATE_CFG)
    cfg_breadth["gates"]["market"]["breadth_min"] = 0.45
    cfg_dd = copy.deepcopy(GATE_CFG)
    cfg_dd["gates"]["market"]["dist_days_max"] = 5.0

    ev_raw, ev_legacy, ev_market, ev_events = [], [], [], []
    ev_breadth, ev_dd = [], []
    for sym in UNIVERSE:
        try:
            prep = prepare_symbol(sym, years=years, cache_dir=cache_dir)
            if prep is None:
                continue
            df_ohlcv, ind_df, dets = prep
            det_e = enrich_detections(dets, df_ohlcv, ind_df, sym)

            g_legacy = apply_gates(det_e, df_ohlcv, ind_df, GATE_CFG)
            ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
            g_market = apply_gates(det_e, df_ohlcv, ind_df, GATE_CFG, ctx=ctx)
            try:
                sym_events = _load_events_cached(sym, df_ohlcv, cache_dir)
            except Exception:
                sym_events = None
            g_events = apply_gates(det_e, df_ohlcv, ind_df, GATE_CFG, ctx=ctx, events=sym_events)
            g_breadth = apply_gates(det_e, df_ohlcv, ind_df, cfg_breadth, ctx=ctx, events=sym_events)
            g_dd = apply_gates(det_e, df_ohlcv, ind_df, cfg_dd, ctx=ctx, events=sym_events)

            for pool, dst in ((det_e, ev_raw), (g_legacy, ev_legacy),
                              (g_market, ev_market), (g_events, ev_events),
                              (g_breadth, ev_breadth), (g_dd, ev_dd)):
                rows = compute_event_rows(pool, df_ohlcv, sym)
                if rows is not None:
                    dst.append(rows)
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    return (pd.concat(x, ignore_index=True) if x else pd.DataFrame()
            for x in (ev_raw, ev_legacy, ev_market, ev_events, ev_breadth, ev_dd))


def summarize_pool(events: pd.DataFrame, label: str, horizon: int = 5) -> pd.DataFrame:
    e = events[events["horizon"] == horizon]
    g = e.groupby("pattern")["excess"]
    out = pd.DataFrame({
        f"n_{label}": g.size(),
        f"bps_{label}": g.mean() * 1e4,
    })
    total = pd.DataFrame({f"n_{label}": [len(e)], f"bps_{label}": [e["excess"].mean() * 1e4]},
                         index=["== ALL =="])
    return pd.concat([out, total])


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    raw, legacy, market, ev, breadth, dd = collect(cache_dir=cache)
    pd.set_option("display.width", 240)
    for h in (5, 10):
        tbl = summarize_pool(raw, "raw", h).join(
            summarize_pool(legacy, "legacy", h), how="outer").join(
            summarize_pool(market, "market", h), how="outer").join(
            summarize_pool(ev, "mkt+earn", h), how="outer").join(
            summarize_pool(breadth, "+breadth45", h), how="outer").join(
            summarize_pool(dd, "+distday5", h), how="outer")
        print(f"\n=== Mean excess (bps) @ {h}-bar horizon: raw | legacy | +market/RS | +earnings | +breadth>=45% | +distdays<5 ===")
        print(tbl.round(1).to_string())
