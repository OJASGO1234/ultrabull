# core/audit_baseline_backtest.py
"""
Baseline backtest audit — runs the EXISTING production strategy exactly as
configured, across the standard 37-symbol/4y universe used by every prior
audit (evaluate_gates.py, walkforward.py), and documents its behavior.

Does NOT modify pipeline.py, rules/signals.py, sim/engine.py, sim/metrics.py,
sim/report.py, rules/gates.py, or any config file. Every production
component (enrich_detections, apply_gates, compile_signals, simulate,
summarize, by_tag, audit_exits) is imported and called exactly as
pipeline.py's own __main__ block calls them — only the cfg dict construction
is reproduced HERE (as literal, unmodified copies of pipeline.py's cfg
dict and its __main__ mutations, lines 20-55 and 120-181), because
pipeline.py's __main__ block itself only ever runs one hardcoded symbol and
this script needs to loop over a universe.

CONFIG PROVENANCE NOTE: this cfg is a byte-for-byte reproduction of
pipeline.py's cfg (module-level dict + its __main__ mutations). It was
cross-checked against api/main.py::build_cfg(), which is functionally
identical, with ONE documented difference: build_cfg() additionally merges
"steward" runtime overrides (core/steward.py::load_overrides()) into
disable_patterns — human-approved, session-mutable state, not a fixed
config value. This script intentionally excludes that runtime merge (it
is not safe/meaningful to import api/main.py — that module wires up
FastAPI, auth, and DB/env state at import time) and reports its exact
disable_patterns list below so the discrepancy is visible, not hidden.

Does NOT tune, optimize, or change ANY strategy parameter. All values
below are copy-pasted from pipeline.py, not invented.

Run:  python audit_baseline_backtest.py <cache_dir>   (from core/ directory)
Output: core/audit_baseline_backtest_results.csv (+ printed report)
"""
from __future__ import annotations
import copy
import json
import os
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from stock_explorer import load_data, compute_indicators, load_events
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.config import CFG as PATTERNS_CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import enrich_detections, warmup_cutoff
from rules.gates import apply_gates
from rules.signals import compile_signals
from sim.engine import simulate
from sim.metrics import summarize, by_tag, audit_exits
from market_context import build_market_context

YEARS = 4.0  # matches evaluate_gates.py / walkforward.py's universe-run convention;
             # pipeline.py's own __main__ hardcodes a single symbol, not a universe.


# ---------------------------------------------------------------------------
# EXACT reproduction of pipeline.py's cfg (lines 20-55) + its __main__
# mutations (lines 120-181). Nothing here is invented or tuned — every value
# is copied verbatim from pipeline.py.
# ---------------------------------------------------------------------------
def build_production_cfg() -> dict:
    cfg = {
        "execution": {
            "delay_bars": 1,
            "enter_at": "Close",
            "max_open_positions": 1,
            "cooldown_bars": int(PATTERNS_CFG.get("cooldowns", {}).get("bars", 0)),
            "max_hold_bars": 0,
        },
        "risk": {
            "capital": 1_000_000.0,
            "max_risk_per_trade": 0.005,
            "position_sizing": "atr",
            "fixed_cash_per_trade": 50_000.0,
        },
        "filters": {
            "min_atr": 0.0,
            "warmup_cutoff_bars": 60,
        },
        "patterns": {
            "Breakout20": {"side": "long",  "sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
            "Breakout55": {"side": "long",  "sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.5},
            "Breakdown20": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
            "Breakdown55": {"side": "short","sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.5},
            "PullbackToMA20": {"side": "long","sl_atr": 1.2, "tp_atr": 2.4, "trail_atr": 1.0},
            "BB_BuyDip": {"side": "long",  "sl_atr": 1.2, "tp_atr": 2.0, "trail_atr": 0.8},
            "BB_SellFade": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 1.0},
            "RSI_40_Up": {"side": "long",  "sl_atr": 1.5, "tp_atr": 3.0},
            "RSI_60_Down": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0},
            "Hammer": {"side": "long", "sl_atr": 1.5, "tp_atr": 3.0},
            "BullishEngulfing": {"side": "long","sl_atr": 1.5, "tp_atr": 3.0},
            "BearishEngulfing": {"side": "short","sl_atr": 1.5, "tp_atr": 3.0},
            "Doji": {"side": "long","sl_atr": 1.2, "tp_atr": 2.0},
            "MACD_Cross_Zeroish_Down": {"side": "short", "sl_atr": 1.5, "tp_atr": 3.0},
        },
        "defaults": {"sl_atr": 1.5, "tp_atr": 3.0, "trail_atr": 0.0},
    }

    # ---- pipeline.py __main__ mutations (lines 120-181), verbatim ----
    cfg.setdefault("costs", {})
    cfg["costs"].update({
        "per_share": 0.0,
        "per_trade": 0.0,
        "bps_fee": 0.0,
        "slippage_bps": 2.0,
    })
    cfg["execution"]["max_hold_bars"] = 10
    cfg["patterns"]["BB_BuyDip"].update({"tp_atr": 1.6, "trail_atr": 1.0})
    cfg["patterns"]["BB_SellFade"].update({"tp_atr": 1.6, "trail_atr": 1.2})
    cfg.setdefault("gates", {})
    cfg["gates"]["bb_touch_eps"] = 0.0
    cfg["gates"]["pullback_ma_tol"] = 0.01
    cfg["execution"].update({"enter_at": "Open", "delay_bars": 1})
    cfg["disable_patterns"] = [
        "RSI_60_Down", "RSI_40_Up", "Doji", "MACD_Cross_Zeroish_Up",
        "BB_SellFade", "Breakdown20", "Breakdown55", "VCP_Setup",
    ]
    cfg.setdefault("equity", {})
    cfg["equity"]["mtm"] = True
    cfg["equity"]["mtm_haircut_bps"] = 2.0
    cfg["gates"]["market"] = {
        "enabled": True,
        "require_market_regime": True,
        "require_sector_regime": False,
        "rs_col": "rs_20",
        "breakout_rs_min": 0.0,
        "breakdown_rs_max": 0.0,
        "dist_days_max": 5.0,
        "breadth_min": 0.0,
    }
    cfg["gates"]["events"] = {
        "enabled": True,
        "earnings_blackout_before": 3,
        "earnings_blackout_after": 1,
    }
    cfg["gates"]["adx"] = {"breakout_min": 0.0, "meanrev_max": 0.0}
    cfg["gates"]["liquidity"] = {"min_turnover_20d": 2e7}
    return cfg


def _load_cached(symbol: str, years: float, cache_dir: str | None) -> pd.DataFrame:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{symbol.replace('&','_')}_{years}y.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    df = load_data(symbol, years)
    if cache_dir:
        df.to_pickle(path)
    return df


def _load_events_cached(symbol: str, df_ohlcv, cache_dir: str | None):
    if cache_dir:
        path = os.path.join(cache_dir, f"events_{symbol.replace('&','_')}.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    ev = load_events(symbol, df_ohlcv.rename(columns=str.lower))
    if cache_dir:
        ev.to_pickle(path)
    return ev


def run_symbol(symbol: str, cfg: dict, cache_dir: str | None) -> dict:
    """Exact production call chain for one symbol. Returns a result dict,
    or a dict with 'error' set if data was missing/invalid — never raises,
    so one bad symbol doesn't kill the universe run (mirrors how
    evaluate_gates.py/walkforward.py already handle per-symbol failures)."""
    result = {"symbol": symbol, "error": None}
    try:
        df_raw = _load_cached(symbol, YEARS, cache_dir)
        df_ohlcv = extract_symbol_df(df_raw, symbol=symbol, case="title")
        if len(df_ohlcv) < 120:
            result["error"] = f"insufficient bars ({len(df_ohlcv)} < 120)"
            return result

        ind = compute_indicators(extract_symbol_df(df_raw, symbol=symbol, case="lower"))
        ind_df = build_ind_df_from_vbt(df_ohlcv, ind)

        detections = run_all_patterns(df_ohlcv, ind_df, PATTERNS_CFG)

        ctx = build_market_context(symbol, df_ohlcv["Close"], years=YEARS, cache_dir=cache_dir)

        det_e = enrich_detections(detections, df_ohlcv, ind_df, symbol, ctx=ctx)
        cut = warmup_cutoff(ind_df)
        det_e = det_e[det_e["date"] >= cut]

        try:
            events = _load_events_cached(symbol, df_ohlcv, cache_dir)
        except Exception as e:
            events = None
            result["events_error"] = str(e)

        det_gated = apply_gates(det_e, df_ohlcv, ind_df, cfg, ctx=ctx, events=events)
        signals = compile_signals(det_gated, df_ohlcv, ind_df, cfg)
        trades, equity, fills = simulate(df_ohlcv, signals, ind_df, cfg)

        result.update({
            "df_ohlcv": df_ohlcv, "ind_df": ind_df, "detections": detections,
            "det_gated": det_gated, "signals": signals,
            "trades": trades, "equity": equity, "fills": fills,
            "date_min": df_ohlcv.index.min(), "date_max": df_ohlcv.index.max(),
            "events_available": events is not None and len(events) > 0,
        })
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = build_production_cfg()

    print("=" * 100)
    print("EXACT CONFIGURATION VALUES USED (byte-for-byte from pipeline.py, unmodified)")
    print("=" * 100)
    print(json.dumps(cfg, indent=2, default=str))

    results = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)
        results.append(r)
        if r["error"]:
            print(f"  {sym}: DATA ISSUE -> {r['error']}")
        else:
            print(f"  {sym}: {len(r['detections'])} raw detections, "
                  f"{len(r['det_gated'])} gated, {len(r['signals'])} signals, "
                  f"{len(r['trades'])} closed trades")

    ok = [r for r in results if not r["error"]]
    bad = [r for r in results if r["error"]]

    # ---- delay_bars actually present in generated signals (not just cfg) ----
    all_signals = pd.concat([r["signals"] for r in ok if len(r["signals"])], ignore_index=True) if ok else pd.DataFrame()
    delay_bars_seen = sorted(all_signals["delay_bars"].unique().tolist()) if len(all_signals) else []
    delay_bars_by_tag = (all_signals.groupby("tag")["delay_bars"].unique().apply(lambda a: sorted(a.tolist()))
                          if len(all_signals) else pd.Series(dtype=object))

    # ---- pooled trades (NOT a portfolio — each symbol has its own separate capital) ----
    all_trades = pd.concat([r["trades"] for r in ok if len(r["trades"])], ignore_index=True) if ok else pd.DataFrame()
    for r in ok:
        if len(r["trades"]):
            pass  # trades already carry 'tag'; symbol attached below for pooled view
    trades_with_symbol = []
    for r in ok:
        if len(r["trades"]):
            t = r["trades"].copy()
            t["symbol"] = r["symbol"]
            trades_with_symbol.append(t)
    all_trades_sym = pd.concat(trades_with_symbol, ignore_index=True) if trades_with_symbol else pd.DataFrame()

    # open/unclosed positions per symbol: signals that filled (ENTRY in fills) but
    # never produced a matching row in trades_df (still open at dataset end)
    open_positions_rows = []
    for r in ok:
        fills = r["fills"]
        trades = r["trades"]
        if len(fills) == 0:
            continue
        n_entries = int((fills["type"] == "ENTRY").sum())
        n_closed = len(trades)
        n_open = n_entries - n_closed
        open_positions_rows.append({"symbol": r["symbol"], "entries": n_entries,
                                     "closed_trades": n_closed, "open_at_end": n_open})
    open_positions_df = pd.DataFrame(open_positions_rows)
    total_open = int(open_positions_df["open_at_end"].sum()) if len(open_positions_df) else 0

    # ---- pooled trade-level statistics (via existing sim.metrics, unmodified) ----
    pooled_summary = summarize(all_trades_sym, pd.Series(dtype=float))  # equity fields intentionally NOT populated (see below)
    # summarize() computes return_total/max_drawdown/sharpe from `equity` — passing an
    # empty series here deliberately suppresses those fields, because pooling trades
    # across symbols with SEPARATE capital pools does not correspond to a real equity
    # curve; per-symbol equity metrics are reported separately below instead.
    pooled_by_tag = by_tag(all_trades_sym)
    pooled_exit_reasons = all_trades_sym["reason"].value_counts().to_dict() if len(all_trades_sym) else {}
    pooled_side_counts = all_trades_sym["side"].value_counts().to_dict() if len(all_trades_sym) else {}
    pooled_side_pnl = all_trades_sym.groupby("side")["pnl"].sum().to_dict() if len(all_trades_sym) else {}

    # audit_exits: production's own SL/TP-vs-slippage consistency check, called unmodified
    exit_mismatches = audit_exits(all_trades_sym, cfg) if len(all_trades_sym) else pd.DataFrame()

    # ---- per-symbol metrics (each symbol's OWN separate capital/equity curve) ----
    per_symbol_rows = []
    for r in ok:
        m = summarize(r["trades"], r["equity"])
        per_symbol_rows.append({
            "symbol": r["symbol"],
            "date_min": r["date_min"], "date_max": r["date_max"],
            "raw_detections": len(r["detections"]), "gated_detections": len(r["det_gated"]),
            "signals": len(r["signals"]), "closed_trades": m.get("trades", 0),
            "win_rate": m.get("win_rate", np.nan),
            "avg_win": m.get("avg_win", np.nan), "avg_loss": m.get("avg_loss", np.nan),
            "profit_factor": m.get("profit_factor", np.nan),
            "expectancy": m.get("expectancy", np.nan),
            "median_hold_bars": m.get("median_hold_bars", np.nan),
            "return_total_pct": m.get("return_total", np.nan) * 100 if "return_total" in m else np.nan,
            "max_drawdown_pct": m.get("max_drawdown", np.nan) * 100 if "max_drawdown" in m else np.nan,
            "sharpe_daily": m.get("sharpe_daily", np.nan),
            "events_available": r["events_available"],
        })
    per_symbol_df = pd.DataFrame(per_symbol_rows)

    print("\n" + "=" * 100)
    print("SYMBOLS WITH DATA ISSUES")
    print("=" * 100)
    if bad:
        for r in bad:
            print(f"  {r['symbol']}: {r['error']}")
    else:
        print("  none")

    print("\n" + "=" * 100)
    print("DELAY_BARS ACTUALLY PRESENT IN GENERATED SIGNALS (not just cfg default)")
    print("=" * 100)
    print(f"distinct values observed: {delay_bars_seen}")
    print("by pattern tag:")
    print(delay_bars_by_tag.to_string() if len(delay_bars_by_tag) else "  (no signals generated)")

    print("\n" + "=" * 100)
    print("COST CONFIGURATION ACTUALLY PASSED TO simulate()")
    print("=" * 100)
    print(json.dumps(cfg["costs"], indent=2))

    print("\n" + "=" * 100)
    print(f"max_open_positions = {cfg['execution']['max_open_positions']}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("SL/TP PARAMETERS (per pattern, sl_atr / tp_atr / trail_atr)")
    print("=" * 100)
    print(pd.DataFrame(cfg["patterns"]).T.to_string())

    print("\n" + "=" * 100)
    print("ATR SIZING PARAMETERS")
    print("=" * 100)
    print(json.dumps(cfg["risk"], indent=2))

    print("\n" + "=" * 100)
    print("PER-SYMBOL METRICS (each symbol has its OWN separate capital/equity curve —")
    print("these are NOT comparable to a shared-portfolio backtest)")
    print("=" * 100)
    pd.set_option("display.width", 220)
    print(per_symbol_df.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("OPEN/UNCLOSED POSITIONS AT DATASET END (per symbol)")
    print("=" * 100)
    print(open_positions_df.to_string(index=False) if len(open_positions_df) else "  none")
    print(f"\nTotal open/unclosed positions across universe: {total_open}")

    print("\n" + "=" * 100)
    print("POOLED TRADE-LEVEL STATISTICS (all symbols' CLOSED trades concatenated — this")
    print("is a trade-count/PnL pool, NOT a portfolio return; return_total/max_drawdown/")
    print("sharpe are intentionally omitted here because there is no shared capital base)")
    print("=" * 100)
    print(json.dumps({k: v for k, v in pooled_summary.items()
                       if k not in ("return_total", "max_drawdown", "sharpe_daily")}, indent=2, default=str))
    print(f"\ntotal signals (all symbols): {len(all_signals)}")
    print(f"total closed trades (all symbols): {len(all_trades_sym)}")
    print(f"total open/unclosed positions (all symbols): {total_open}")

    print("\nExits by reason (pooled):")
    print(json.dumps(pooled_exit_reasons, indent=2))

    print("\nLong vs short: trade counts (pooled):")
    print(json.dumps(pooled_side_counts, indent=2))
    print("Long vs short: total PnL (pooled):")
    print(json.dumps(pooled_side_pnl, indent=2, default=str))

    print("\nTrades by pattern, PnL by pattern (pooled, via sim.metrics.by_tag, unmodified):")
    print(pooled_by_tag.round(2).to_string())

    print("\nSL/TP exit-price audit (sim.metrics.audit_exits, unmodified) — mismatches found:")
    print(f"  {len(exit_mismatches)} mismatch(es)" if len(exit_mismatches) else "  0 mismatches")
    if len(exit_mismatches):
        print(exit_mismatches[["symbol", "entry_bar", "exit_bar", "reason", "exit_price",
                                "sl_price", "tp_price"]].to_string(index=False)
              if "symbol" in exit_mismatches.columns else exit_mismatches.to_string(index=False))

    # ---- save results ----
    out_rows = []
    for _, row in per_symbol_df.iterrows():
        d = row.to_dict()
        d["row_type"] = "per_symbol"
        out_rows.append(d)
    pooled_row = {"row_type": "pooled_summary"}
    pooled_row.update({k: v for k, v in pooled_summary.items()})
    pooled_row["total_signals"] = len(all_signals)
    pooled_row["total_open_unclosed"] = total_open
    out_rows.append(pooled_row)
    for reason, cnt in pooled_exit_reasons.items():
        out_rows.append({"row_type": "exit_reason", "reason": reason, "count": cnt})
    for side, cnt in pooled_side_counts.items():
        out_rows.append({"row_type": "side_count", "side": side, "count": cnt,
                          "total_pnl": pooled_side_pnl.get(side)})
    for tag, row in pooled_by_tag.iterrows():
        d = row.to_dict()
        d["row_type"] = "by_pattern"
        d["tag"] = tag
        out_rows.append(d)
    for _, row in open_positions_df.iterrows():
        d = row.to_dict()
        d["row_type"] = "open_positions"
        out_rows.append(d)
    for r in bad:
        out_rows.append({"row_type": "data_issue", "symbol": r["symbol"], "error": r["error"]})

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("audit_baseline_backtest_results.csv", index=False)
    print("\nSaved: audit_baseline_backtest_results.csv")


if __name__ == "__main__":
    main()
