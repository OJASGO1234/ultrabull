# core/audit_walkforward_pattern_diagnosis.py
"""
Diagnose WHY the strategy fails chronological out-of-sample validation
(confirmed in audit_walkforward_validation.py: baseline PF 1.40 tune ->
0.75 valid, expectancy +814 -> -660) by breaking the same train/valid split
down PER PATTERN, to see which patterns drive the decline.

Reuses build_production_cfg() and run_symbol() DIRECTLY from
audit_baseline_backtest.py (imported, not retyped) — the exact same config
object already used by audit_baseline_backtest.py and
audit_market_regime_ablation.py, not walkforward.py's separate hand-written
BASE_CFG (confirmed different in several keys). Each symbol is simulated
ONCE, full history, via the unmodified production chain
(enrich_detections -> apply_gates -> compile_signals -> simulate); the
resulting trades are then split post-hoc by entry_bar against a 70% split
bar, exactly mirroring walkforward.py's own split convention. simulate()
never sees the split boundary, so there is no re-fit, re-gate, or
config difference between the two periods — only which historical trades
fall before/after the boundary.

Does NOT modify any production file. Does NOT tune, optimize, or change any
threshold/gate/pattern/parameter.

Run:  python audit_walkforward_pattern_diagnosis.py <cache_dir>   (from core/)
Output: core/audit_walkforward_pattern_diagnosis_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from sim.metrics import summarize
from audit_baseline_backtest import build_production_cfg, run_symbol

SPLIT = 0.7


def collect_trades(cache: str | None) -> tuple[pd.DataFrame, list, list]:
    cfg = build_production_cfg()
    all_trades = []
    bad = []
    ok_symbols = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)
        if r["error"]:
            bad.append(r)
            print(f"  {sym}: DATA ISSUE -> {r['error']}")
            continue
        ok_symbols.append(sym)
        trades = r["trades"]
        n = len(r["df_ohlcv"])
        split_bar = int(n * SPLIT)
        print(f"  {sym}: {n} bars, split_bar={split_bar}, {len(trades)} closed trades")
        if len(trades) == 0:
            continue
        t = trades.copy()
        t["symbol"] = sym
        t["period"] = np.where(t["entry_bar"] < split_bar, "train", "valid")
        all_trades.append(t)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return trades_df, ok_symbols, bad


def _metrics(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {
            "completed_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
            "expectancy": np.nan, "avg_trade_pnl": np.nan, "median_trade_pnl": np.nan,
            "total_pnl": np.nan,
        }
    m = summarize(g, pd.Series(dtype=float))
    return {
        "completed_trades": m.get("trades", len(g)),
        "win_rate": m.get("win_rate", np.nan),
        "profit_factor": m.get("profit_factor", np.nan),
        "expectancy": m.get("expectancy", np.nan),
        "avg_trade_pnl": float(g["pnl"].mean()),
        "median_trade_pnl": float(g["pnl"].median()),
        "total_pnl": float(g["pnl"].sum()),
    }


def by_pattern_period(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tag, period), g in trades.groupby(["tag", "period"]):
        row = {"tag": tag, "period": period}
        row.update(_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows)


def exit_reason_dist(trades: pd.DataFrame, tag: str | None = None) -> dict:
    g = trades if tag is None else trades[trades["tag"] == tag]
    return g["reason"].value_counts().to_dict() if len(g) else {}


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting trades: exact production config (build_production_cfg from "
          "audit_baseline_backtest.py), full-history single simulation per symbol, "
          "post-hoc 70/30 chronological split by entry_bar.\n")

    trades, ok_symbols, bad = collect_trades(cache)
    print(f"\n{len(ok_symbols)}/{len(UNIVERSE)} symbols OK, {len(bad)} data issues")
    print(f"Total closed trades collected: {len(trades)}")

    if len(trades) == 0:
        print("No trades collected -- aborting.")
        return

    pd.set_option("display.width", 240)

    # ---- 1 & 2: overall pooled train/valid summary ----
    pooled_train = _metrics(trades[trades["period"] == "train"])
    pooled_valid = _metrics(trades[trades["period"] == "valid"])
    pooled_delta = {k: (pooled_valid[k] - pooled_train[k]
                         if isinstance(pooled_train.get(k), (int, float)) and isinstance(pooled_valid.get(k), (int, float))
                         else np.nan)
                     for k in pooled_train}

    print("\n" + "=" * 100)
    print("1. OVERALL POOLED TRAINING SUMMARY (trade-pool across 37 symbols, NOT a portfolio return)")
    print("=" * 100)
    print(pooled_train)
    print("\n" + "=" * 100)
    print("2. OVERALL POOLED VALIDATION SUMMARY")
    print("=" * 100)
    print(pooled_valid)
    print("\nPooled delta (valid - train):")
    print(pooled_delta)

    # ---- per-pattern x period table ----
    pp = by_pattern_period(trades)
    pivot = pp.pivot(index="tag", columns="period",
                      values=["completed_trades", "win_rate", "profit_factor", "expectancy",
                              "avg_trade_pnl", "median_trade_pnl", "total_pnl"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    for metric in ["completed_trades", "win_rate", "profit_factor", "expectancy",
                   "avg_trade_pnl", "median_trade_pnl", "total_pnl"]:
        tcol, vcol = f"{metric}_train", f"{metric}_valid"
        if tcol in pivot.columns and vcol in pivot.columns:
            pivot[f"delta_{metric}"] = pivot[vcol] - pivot[tcol]

    pivot = pivot.sort_values("delta_total_pnl", ascending=True).reset_index(drop=True)

    print("\n" + "=" * 100)
    print("PER-PATTERN TRAIN vs VALID (ranked by delta_total_pnl: largest deterioration first)")
    print("=" * 100)
    show_cols = ["tag", "completed_trades_train", "completed_trades_valid",
                 "win_rate_train", "win_rate_valid", "profit_factor_train", "profit_factor_valid",
                 "expectancy_train", "expectancy_valid", "total_pnl_train", "total_pnl_valid",
                 "delta_total_pnl", "delta_expectancy", "delta_win_rate", "delta_profit_factor"]
    show_cols = [c for c in show_cols if c in pivot.columns]
    print(pivot[show_cols].round(4).to_string(index=False))

    # ---- exit reason distribution per pattern per period ----
    exit_rows = []
    for tag in trades["tag"].unique():
        for period in ("train", "valid"):
            g = trades[(trades["tag"] == tag) & (trades["period"] == period)]
            dist = g["reason"].value_counts().to_dict()
            for reason, cnt in dist.items():
                exit_rows.append({"tag": tag, "period": period, "reason": reason, "count": cnt})
    exit_df = pd.DataFrame(exit_rows)

    print("\n" + "=" * 100)
    print("EXIT REASON DISTRIBUTION BY PATTERN AND PERIOD")
    print("=" * 100)
    exit_pivot = exit_df.pivot_table(index=["tag", "reason"], columns="period", values="count", fill_value=0)
    print(exit_pivot.to_string())

    # ---- pattern contribution to overall decline ----
    total_delta_pnl = pooled_delta["total_pnl"]
    contrib = pivot[["tag", "delta_total_pnl"]].copy()
    contrib["pct_of_total_decline"] = (contrib["delta_total_pnl"] / total_delta_pnl * 100.0
                                        if total_delta_pnl != 0 else np.nan)
    contrib = contrib.sort_values("delta_total_pnl", ascending=True).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"3. PATTERN CONTRIBUTION TO OVERALL DECLINE (total pooled delta_total_pnl = {total_delta_pnl:.2f})")
    print("=" * 100)
    print(contrib.round(3).to_string(index=False))

    # ---- long/short breakdown ----
    side_rows = []
    for (side, period), g in trades.groupby(["side", "period"]):
        row = {"side": side, "period": period}
        row.update(_metrics(g))
        side_rows.append(row)
    side_df = pd.DataFrame(side_rows)
    side_pivot = side_df.pivot(index="side", columns="period",
                                values=["completed_trades", "win_rate", "profit_factor",
                                        "expectancy", "total_pnl"])
    side_pivot.columns = [f"{a}_{b}" for a, b in side_pivot.columns]
    side_pivot = side_pivot.reset_index()
    for metric in ["completed_trades", "win_rate", "profit_factor", "expectancy", "total_pnl"]:
        tcol, vcol = f"{metric}_train", f"{metric}_valid"
        if tcol in side_pivot.columns and vcol in side_pivot.columns:
            side_pivot[f"delta_{metric}"] = side_pivot[vcol] - side_pivot[tcol]

    print("\n" + "=" * 100)
    print("4. LONG vs SHORT BREAKDOWN (train vs valid)")
    print("=" * 100)
    print(side_pivot.round(4).to_string(index=False))

    # ---- save CSV ----
    out_rows = []
    for _, row in pivot.iterrows():
        d = row.to_dict(); d["section"] = "PER_PATTERN"; out_rows.append(d)
    tr = dict(pooled_train); tr["section"] = "POOLED_TRAIN"; out_rows.append(tr)
    va = dict(pooled_valid); va["section"] = "POOLED_VALID"; out_rows.append(va)
    dl = dict(pooled_delta); dl["section"] = "POOLED_DELTA"; out_rows.append(dl)
    for _, row in exit_df.iterrows():
        d = row.to_dict(); d["section"] = "EXIT_REASONS"; out_rows.append(d)
    for _, row in contrib.iterrows():
        d = row.to_dict(); d["section"] = "PATTERN_CONTRIBUTION"; out_rows.append(d)
    for _, row in side_pivot.iterrows():
        d = row.to_dict(); d["section"] = "LONG_SHORT"; out_rows.append(d)
    for r in bad:
        out_rows.append({"section": "DATA_ISSUE", "symbol": r["symbol"], "error": r["error"]})

    out_df = pd.DataFrame(out_rows)
    cols = ["section"] + [c for c in out_df.columns if c != "section"]
    out_df = out_df[cols]
    out_df.to_csv("audit_walkforward_pattern_diagnosis_results.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    worst = contrib.iloc[0]
    print(f"Largest single-pattern deterioration: {worst['tag']} "
          f"(delta_total_pnl={worst['delta_total_pnl']:.2f}, "
          f"{worst['pct_of_total_decline']:.1f}% of total pooled decline)")
    print(f"\nSaved: audit_walkforward_pattern_diagnosis_results.csv ({len(out_df)} rows)")
    print("\nNOTE: pooled/per-pattern figures are trade-pool aggregates across 37 independently-")
    print("capitalized symbols, NOT a shared-capital portfolio return. No parameter was changed,")
    print("no gate was modified, no pattern was disabled based on these results.")


if __name__ == "__main__":
    main()
