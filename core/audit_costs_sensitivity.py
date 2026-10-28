# core/audit_costs_sensitivity.py
"""
Costs/slippage sensitivity sweep — measures how sensitive the baseline
strategy's pooled trade-pool P&L is to transaction-cost assumptions.

This is NOT optimization: no threshold is chosen or recommended, no config
is changed in production, and the production cfg's actual slippage_bps=2.0
is only ONE of several levels swept for comparison. The purpose is purely
to answer "how much of the edge survives realistic costs, and at what cost
level does it disappear" — a sensitivity measurement, not a search for a
better setting.

Reuses build_production_cfg()/YEARS/run_symbol() from
audit_baseline_backtest.py unmodified. Only the `costs` sub-dict is varied
per sweep point; every other cfg key (patterns, gates, execution, risk,
disable_patterns) is held at the exact baseline value.

Run:  python audit_costs_sensitivity.py <cache_dir>   (from core/ directory)
Output: core/audit_costs_sensitivity_results.csv (+ printed report)
"""
from __future__ import annotations
import copy
import json
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from sim.metrics import summarize, by_tag
from audit_baseline_backtest import build_production_cfg, run_symbol

# Sweep points. slippage_bps=2.0/fees=0 is the ACTUAL production value
# (labeled below); the rest are comparison points only, not proposals.
SWEEP = [
    {"name": "zero_cost",          "per_share": 0.0,  "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 0.0},
    {"name": "production_2bps",    "per_share": 0.0,  "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 2.0},
    {"name": "slippage_5bps",      "per_share": 0.0,  "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 5.0},
    {"name": "slippage_10bps",     "per_share": 0.0,  "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 10.0},
    {"name": "slippage_20bps",     "per_share": 0.0,  "per_trade": 0.0, "bps_fee": 0.0, "slippage_bps": 20.0},
    {"name": "plus_brokerage_5bps_fee", "per_share": 0.0, "per_trade": 20.0, "bps_fee": 5.0, "slippage_bps": 2.0},
    {"name": "plus_brokerage_10bps_fee", "per_share": 0.0, "per_trade": 20.0, "bps_fee": 10.0, "slippage_bps": 5.0},
]
PRODUCTION_LABEL = "production_2bps"


def run_point(point: dict, cache: str | None) -> dict:
    cfg = build_production_cfg()
    cfg["costs"] = {"per_share": point["per_share"], "per_trade": point["per_trade"],
                     "bps_fee": point["bps_fee"], "slippage_bps": point["slippage_bps"]}

    results = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)
        results.append(r)
    ok = [r for r in results if not r["error"]]
    bad = [r for r in results if r["error"]]

    trades_with_symbol = []
    for r in ok:
        if len(r["trades"]):
            t = r["trades"].copy()
            t["symbol"] = r["symbol"]
            trades_with_symbol.append(t)
    all_trades = pd.concat(trades_with_symbol, ignore_index=True) if trades_with_symbol else pd.DataFrame()

    pooled_summary = summarize(all_trades, pd.Series(dtype=float))
    pooled_by_tag = by_tag(all_trades)
    total_fees = float(all_trades["fees"].sum()) if len(all_trades) else 0.0
    median_pnl = float(all_trades["pnl"].median()) if len(all_trades) else np.nan
    total_pnl = float(all_trades["pnl"].sum()) if len(all_trades) else 0.0

    return {
        "name": point["name"], "costs": cfg["costs"],
        "n_ok": len(ok), "n_total": len(UNIVERSE), "bad": bad,
        "completed_trades": pooled_summary.get("trades", 0),
        "win_rate": pooled_summary.get("win_rate", np.nan),
        "avg_trade_pnl": pooled_summary.get("expectancy", np.nan),
        "median_trade_pnl": median_pnl,
        "profit_factor": pooled_summary.get("profit_factor", np.nan),
        "expectancy": pooled_summary.get("expectancy", np.nan),
        "total_pnl_trade_pool": total_pnl,
        "total_fees_paid": total_fees,
        "by_tag": pooled_by_tag,
    }


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    by_tag_rows = []

    for point in SWEEP:
        print(f"\n{'='*100}\nRunning cost point: {point['name']} -> {point}\n{'='*100}")
        res = run_point(point, cache)
        print(f"  {res['n_ok']}/{res['n_total']} symbols OK, {len(res['bad'])} hard failures")
        print(f"  trades={res['completed_trades']}  win_rate={res['win_rate']:.4f}  "
              f"profit_factor={res['profit_factor']:.4f}  expectancy={res['expectancy']:.2f}  "
              f"total_pnl_pool={res['total_pnl_trade_pool']:.2f}  total_fees={res['total_fees_paid']:.2f}")
        rows.append({k: v for k, v in res.items() if k not in ("costs", "by_tag", "bad")} |
                    {"costs_json": json.dumps(res["costs"])})
        for tag, r in res["by_tag"].iterrows():
            d = r.to_dict(); d["sweep_point"] = point["name"]; d["tag"] = tag
            by_tag_rows.append(d)
        if res["bad"]:
            for b in res["bad"]:
                print(f"    HARD FAILURE: {b['symbol']}: {b['error']}")

    df = pd.DataFrame(rows)
    df["is_production_value"] = df["name"] == PRODUCTION_LABEL

    prod_row = df[df["name"] == PRODUCTION_LABEL].iloc[0]
    df["delta_expectancy_vs_production"] = df["expectancy"] - prod_row["expectancy"]
    df["delta_profit_factor_vs_production"] = df["profit_factor"] - prod_row["profit_factor"]
    df["delta_total_pnl_vs_production"] = df["total_pnl_trade_pool"] - prod_row["total_pnl_trade_pool"]

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print("SENSITIVITY TABLE (production_2bps row = cfg actually used in production)")
    print("=" * 100)
    show_cols = ["name", "is_production_value", "completed_trades", "win_rate", "profit_factor",
                 "expectancy", "median_trade_pnl", "total_pnl_trade_pool", "total_fees_paid",
                 "delta_expectancy_vs_production", "delta_profit_factor_vs_production"]
    print(df[show_cols].round(4).to_string(index=False))

    by_tag_df = pd.DataFrame(by_tag_rows)
    df.to_csv("audit_costs_sensitivity_results.csv", index=False)
    by_tag_df.to_csv("audit_costs_sensitivity_by_pattern.csv", index=False)

    print("\n" + "=" * 100)
    print("INTERPRETATION NOTE")
    print("=" * 100)
    print("This is a SENSITIVITY sweep, not an optimization: no cost level here is recommended,\n"
          "chosen, or fed back into production config. 'production_2bps' is the row matching the\n"
          "cfg actually used in pipeline.py/api/main.py today; all other rows are comparison points\n"
          "only, to show how much of the trade-pool P&L survives higher assumed costs. Figures are\n"
          "trade-pool aggregates across 37 independently-capitalized symbols, NOT a portfolio return.")
    print("\nSaved: audit_costs_sensitivity_results.csv, audit_costs_sensitivity_by_pattern.csv")


if __name__ == "__main__":
    main()
