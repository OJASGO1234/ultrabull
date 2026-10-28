# core/audit_remaining_ablations.py
"""
Four more controlled single-variable ablations, using the exact same
methodology as audit_market_regime_ablation.py (which covered
gates.market.require_market_regime): earnings blackout, breadth floor,
distribution-day ceiling, and the breakout relative-strength requirement.

Reuses build_production_cfg()/YEARS/run_symbol() from
audit_baseline_backtest.py and dict_diff()/run_config()/build_report() from
audit_market_regime_ablation.py — imported, not retyped. Does not modify
any production file or either prior audit script.

Each ablation is Config A (baseline, unchanged) vs Config B (deepcopy of A
with EXACTLY one leaf changed). The recursive dict diff is asserted to
contain exactly that one path before the experiment runs; if not, this
script aborts for that ablation without running it.

Ablation definitions (why each single-key change is the right isolation):
  1. earnings_blackout: gates.events.enabled True -> False. This is the only
     flag gates.py checks for the whole earnings-blackout block
     (`if events is not None and len(events) and bool(egate.get("enabled"))`),
     so flipping it is a clean on/off switch.
  2. breadth: gates.market.breadth_min 0.0 -> 0.45. Baseline ships this OFF
     (0.0 is a no-op per gates.py's `if breadth_min > 0`), matching
     evaluate_gates.py's original cfg_breadth variant. This ablation asks
     "what if we turned it ON", not "what if we turned it off" (it's already
     off in the baseline).
  3. distribution_days: gates.market.dist_days_max 5.0 -> 0.0. Baseline ships
     this ON (5.0); 0.0 is gates.py's own no-op sentinel
     (`if dd_max > 0`), so this ablation asks "what if we turned it off".
  4. relative_strength: gates.market.breakout_rs_min 0.0 -> -1e9. Only
     Breakout20/Breakout55 signals are ever RS-gated in this universe
     (Breakdown20/55 are in disable_patterns, so breakdown_rs_max never
     matters in practice — confirmed in the gate audit) — a permissive
     -1e9 floor makes the `rs_i >= rs_min` condition always true, i.e.
     disables the requirement, via a single changed key.

Run:  python audit_remaining_ablations.py <cache_dir>   (from core/ directory)
Output: core/audit_remaining_ablations_results.csv (+ printed report)
"""
from __future__ import annotations
import copy
import json
import sys
import numpy as np
import pandas as pd

from audit_baseline_backtest import build_production_cfg
from audit_market_regime_ablation import dict_diff, run_config, build_report

ABLATIONS = [
    {
        "name": "earnings_blackout",
        "path": "gates.events.enabled",
        "mutate": lambda cfg: cfg["gates"]["events"].update({"enabled": False}),
        "expected_a": True, "expected_b": False,
        "description": "Earnings blackout ON (baseline) vs OFF",
    },
    {
        "name": "breadth_floor",
        "path": "gates.market.breadth_min",
        "mutate": lambda cfg: cfg["gates"]["market"].update({"breadth_min": 0.45}),
        "expected_a": 0.0, "expected_b": 0.45,
        "description": "Breadth floor OFF (baseline, 0.0=no-op) vs ON (0.45)",
    },
    {
        "name": "distribution_days",
        "path": "gates.market.dist_days_max",
        "mutate": lambda cfg: cfg["gates"]["market"].update({"dist_days_max": 0.0}),
        "expected_a": 5.0, "expected_b": 0.0,
        "description": "Distribution-day ceiling ON (baseline, 5.0) vs OFF (0.0=no-op)",
    },
    {
        "name": "relative_strength_breakout",
        "path": "gates.market.breakout_rs_min",
        "mutate": lambda cfg: cfg["gates"]["market"].update({"breakout_rs_min": -1e9}),
        "expected_a": 0.0, "expected_b": -1e9,
        "description": "Breakout relative-strength requirement ON (baseline, rs>=0) vs effectively OFF (rs>=-1e9)",
    },
]


def run_ablation(ablation: dict, cache: str | None) -> dict | None:
    name = ablation["name"]
    cfg_a = build_production_cfg()
    cfg_b = copy.deepcopy(cfg_a)
    ablation["mutate"](cfg_b)

    diffs = dict_diff(cfg_a, cfg_b)
    print(f"\n{'='*100}\nABLATION: {name} — {ablation['description']}\n{'='*100}")
    print(f"Config diff (A -> B): {diffs}")

    expected = ablation["path"]
    ok = (len(diffs) == 1 and diffs[0][0] == expected
          and diffs[0][1] == ablation["expected_a"] and diffs[0][2] == ablation["expected_b"])
    if not ok:
        print(f"ABORTING '{name}': diff does not match the expected single-variable change "
              f"({expected}: {ablation['expected_a']!r} -> {ablation['expected_b']!r}). Skipping.")
        return None
    print(f"ASSERTION PASSED for '{name}': exactly one diff, at the expected path.")

    run_a = run_config(f"{name}_A", cfg_a, cache)
    run_b = run_config(f"{name}_B", cfg_b, cache)
    rep_a = build_report(f"{name}_A", run_a)
    rep_b = build_report(f"{name}_B", run_b)

    ok_a = rep_a["n_ok"] == rep_a["n_total"] or len(rep_a["bad"]) > 0
    ok_b = rep_b["n_ok"] == rep_b["n_total"] or len(rep_b["bad"]) > 0
    print(f"  A: {rep_a['n_ok']}/{rep_a['n_total']} symbols OK, {len(rep_a['bad'])} hard failures")
    print(f"  B: {rep_b['n_ok']}/{rep_b['n_total']} symbols OK, {len(rep_b['bad'])} hard failures")

    merged = rep_a["per_symbol_df"].merge(rep_b["per_symbol_df"], on="symbol", suffixes=("_A", "_B"), how="outer")
    delta_rows = []
    for _, row in merged.iterrows():
        delta_rows.append({
            "symbol": row["symbol"],
            "delta_completed_trades": row.get("completed_trades_B", np.nan) - row.get("completed_trades_A", np.nan),
            "delta_win_rate": row.get("win_rate_B", np.nan) - row.get("win_rate_A", np.nan),
            "delta_total_pnl": row.get("total_pnl_B", np.nan) - row.get("total_pnl_A", np.nan),
            "delta_return_total_pct": row.get("return_total_pct_B", np.nan) - row.get("return_total_pct_A", np.nan),
            "delta_max_drawdown_pct": row.get("max_drawdown_pct_B", np.nan) - row.get("max_drawdown_pct_A", np.nan),
            "delta_sharpe_daily": row.get("sharpe_daily_B", np.nan) - row.get("sharpe_daily_A", np.nan),
        })
    delta_per_symbol_df = pd.DataFrame(delta_rows)

    delta_pooled = {
        "delta_completed_trades": rep_b["pooled"]["completed_trades"] - rep_a["pooled"]["completed_trades"],
        "delta_win_rate": rep_b["pooled"]["win_rate"] - rep_a["pooled"]["win_rate"],
        "delta_profit_factor": rep_b["pooled"]["profit_factor"] - rep_a["pooled"]["profit_factor"],
        "delta_expectancy": rep_b["pooled"]["expectancy"] - rep_a["pooled"]["expectancy"],
        "delta_avg_trade_pnl": rep_b["pooled"]["avg_trade_pnl"] - rep_a["pooled"]["avg_trade_pnl"],
        "delta_median_trade_pnl": rep_b["pooled"]["median_trade_pnl"] - rep_a["pooled"]["median_trade_pnl"],
        "delta_total_pnl_trade_pool": (rep_b["pooled"]["total_pnl_trade_pool_NOT_portfolio_return"]
                                        - rep_a["pooled"]["total_pnl_trade_pool_NOT_portfolio_return"]),
    }

    print(f"\n  Pooled A (trade-pool, NOT portfolio): {rep_a['pooled']}")
    print(f"  Pooled B (trade-pool, NOT portfolio): {rep_b['pooled']}")
    print(f"  Pooled deltas (B-A): {delta_pooled}")

    return {
        "name": name, "description": ablation["description"], "diff": diffs,
        "rep_a": rep_a, "rep_b": rep_b,
        "delta_per_symbol_df": delta_per_symbol_df, "delta_pooled": delta_pooled,
    }


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    out_rows = []
    summaries = []

    for ablation in ABLATIONS:
        res = run_ablation(ablation, cache)
        if res is None:
            out_rows.append({"section": "ABORTED", "ablation": ablation["name"],
                              "reason": "config diff did not match expected single-variable change"})
            continue
        name = res["name"]
        for p, va, vb in res["diff"]:
            out_rows.append({"section": "CONFIG_DIFF", "ablation": name, "path": p, "value_A": va, "value_B": vb})
        for _, row in res["rep_a"]["per_symbol_df"].iterrows():
            d = row.to_dict(); d["section"] = "PER_SYMBOL_A"; d["ablation"] = name; out_rows.append(d)
        for _, row in res["rep_b"]["per_symbol_df"].iterrows():
            d = row.to_dict(); d["section"] = "PER_SYMBOL_B"; d["ablation"] = name; out_rows.append(d)
        pa = dict(res["rep_a"]["pooled"]); pa["section"] = "POOLED_A"; pa["ablation"] = name; out_rows.append(pa)
        pb = dict(res["rep_b"]["pooled"]); pb["section"] = "POOLED_B"; pb["ablation"] = name; out_rows.append(pb)
        for reason, cnt in res["rep_a"]["exit_reasons"].items():
            out_rows.append({"section": "EXIT_REASONS_A", "ablation": name, "reason": reason, "count": cnt})
        for reason, cnt in res["rep_b"]["exit_reasons"].items():
            out_rows.append({"section": "EXIT_REASONS_B", "ablation": name, "reason": reason, "count": cnt})
        for tag, row in res["rep_a"]["pooled_by_tag"].iterrows():
            d = row.to_dict(); d["section"] = "PATTERNS_A"; d["ablation"] = name; d["tag"] = tag; out_rows.append(d)
        for tag, row in res["rep_b"]["pooled_by_tag"].iterrows():
            d = row.to_dict(); d["section"] = "PATTERNS_B"; d["ablation"] = name; d["tag"] = tag; out_rows.append(d)
        for _, row in res["delta_per_symbol_df"].iterrows():
            d = row.to_dict(); d["section"] = "DELTA_PER_SYMBOL"; d["ablation"] = name; out_rows.append(d)
        dp = dict(res["delta_pooled"]); dp["section"] = "DELTA_POOLED"; dp["ablation"] = name; out_rows.append(dp)
        for r in res["rep_a"]["bad"]:
            out_rows.append({"section": "HARD_FAILURE_A", "ablation": name, "symbol": r["symbol"], "error": r["error"]})
        for r in res["rep_b"]["bad"]:
            out_rows.append({"section": "HARD_FAILURE_B", "ablation": name, "symbol": r["symbol"], "error": r["error"]})

        summaries.append({
            "ablation": name, "description": ablation["description"],
            "pooled_A": res["rep_a"]["pooled"], "pooled_B": res["rep_b"]["pooled"],
            "delta_pooled": res["delta_pooled"],
        })

    out_df = pd.DataFrame(out_rows)
    cols = ["section", "ablation"] + [c for c in out_df.columns if c not in ("section", "ablation")]
    out_df = out_df[cols]
    out_df.to_csv("audit_remaining_ablations_results.csv", index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY — ALL 4 ABLATIONS")
    print("=" * 100)
    for s in summaries:
        print(f"\n[{s['ablation']}] {s['description']}")
        print(f"  A: {s['pooled_A']}")
        print(f"  B: {s['pooled_B']}")
        print(f"  Delta (B-A): {s['delta_pooled']}")
    print(f"\nSaved: audit_remaining_ablations_results.csv ({len(out_df)} rows)")
    print("\nNote: pooled figures are trade-pool aggregates across 37 independently-capitalized "
          "symbol runs, NOT a shared-capital portfolio return.")


if __name__ == "__main__":
    main()
