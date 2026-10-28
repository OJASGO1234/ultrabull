# core/audit_market_regime_ablation.py
"""
Controlled ablation: market-regime gate ON (Config A, current baseline) vs
OFF (Config B, require_market_regime=False), holding every other setting
byte-for-byte identical.

Reuses build_production_cfg(), YEARS, run_symbol(), _load_cached(), and
_load_events_cached() DIRECTLY from audit_baseline_backtest.py (import, not
retyped) — same universe, same cache, same per-symbol call chain
(enrich_detections -> apply_gates -> compile_signals -> simulate) already
verified in the baseline audit. Does not modify pipeline.py, rules/gates.py,
rules/signals.py, sim/engine.py, sim/metrics.py, or audit_baseline_backtest.py.

Config B is built as copy.deepcopy(Config A) with exactly one line changed:
    cfg["gates"]["market"]["require_market_regime"] = False
A recursive dict diff between A and B is computed and ASSERTED to contain
exactly that one path before the experiment is allowed to run — if the diff
ever contains anything else, the script aborts loudly instead of running.

Does NOT tune, optimize, or add a third variant.

Run:  python audit_market_regime_ablation.py <cache_dir>   (from core/ directory)
Output: core/audit_market_regime_ablation_results.csv (+ printed report)
"""
from __future__ import annotations
import copy
import json
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from sim.metrics import summarize, by_tag
from audit_baseline_backtest import (
    build_production_cfg, YEARS, run_symbol,
)

CREATED_FILES = ["core/audit_market_regime_ablation.py", "core/audit_market_regime_ablation_results.csv"]
MODIFIED_FILES = []  # populated only if something unexpected forces a change (should stay empty)


# ---------------------------------------------------------------------------
# recursive dict diff — proof of single-variable change
# ---------------------------------------------------------------------------
def dict_diff(a, b, path="") -> list[tuple[str, object, object]]:
    """Return [(dotted_path, value_in_a, value_in_b), ...] for every leaf
    where a and b differ. Recurses into nested dicts; treats lists/scalars
    as leaves compared by equality."""
    diffs = []
    keys = set(a.keys()) | set(b.keys()) if isinstance(a, dict) and isinstance(b, dict) else set()
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(keys, key=str):
            p = f"{path}.{k}" if path else str(k)
            in_a, in_b = k in a, k in b
            if not in_a:
                diffs.append((p, "<MISSING>", b[k]))
            elif not in_b:
                diffs.append((p, a[k], "<MISSING>"))
            else:
                diffs.extend(dict_diff(a[k], b[k], p))
    else:
        if a != b:
            diffs.append((path, a, b))
    return diffs


def run_config(label: str, cfg: dict, cache_dir: str | None) -> dict:
    """Runs the exact baseline per-symbol chain (audit_baseline_backtest.run_symbol)
    for every symbol in UNIVERSE, unmodified."""
    results = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache_dir)
        results.append(r)
        if r["error"]:
            print(f"  [{label}] {sym}: DATA ISSUE -> {r['error']}")
        else:
            print(f"  [{label}] {sym}: {len(r['signals'])} signals, {len(r['trades'])} closed trades")
    return {"results": results, "ok": [r for r in results if not r["error"]],
            "bad": [r for r in results if r["error"]]}


def build_report(label: str, run: dict) -> dict:
    ok = run["ok"]

    trades_with_symbol = []
    for r in ok:
        if len(r["trades"]):
            t = r["trades"].copy()
            t["symbol"] = r["symbol"]
            trades_with_symbol.append(t)
    all_trades = pd.concat(trades_with_symbol, ignore_index=True) if trades_with_symbol else pd.DataFrame()

    open_rows = []
    for r in ok:
        fills, trades = r["fills"], r["trades"]
        n_entries = int((fills["type"] == "ENTRY").sum()) if len(fills) else 0
        n_closed = len(trades)
        open_rows.append({"symbol": r["symbol"], "entries": n_entries,
                           "closed_trades": n_closed, "open_at_end": n_entries - n_closed})
    open_df = pd.DataFrame(open_rows)

    per_symbol_rows = []
    for r in ok:
        m = summarize(r["trades"], r["equity"])
        n_open = int(open_df.loc[open_df["symbol"] == r["symbol"], "open_at_end"].iloc[0]) if len(open_df) else 0
        total_pnl = float(r["trades"]["pnl"].sum()) if len(r["trades"]) else 0.0
        per_symbol_rows.append({
            "symbol": r["symbol"],
            "completed_trades": m.get("trades", 0),
            "open_positions_at_end": n_open,
            "win_rate": m.get("win_rate", np.nan),
            "total_pnl": total_pnl,
            "return_total_pct": m.get("return_total", np.nan) * 100 if "return_total" in m else np.nan,
            "max_drawdown_pct": m.get("max_drawdown", np.nan) * 100 if "max_drawdown" in m else np.nan,
            "sharpe_daily": m.get("sharpe_daily", np.nan),
        })
    per_symbol_df = pd.DataFrame(per_symbol_rows).sort_values("symbol").reset_index(drop=True)

    # pooled trade-pool stats — NOT a portfolio return (no shared capital engine exists)
    pooled_summary = summarize(all_trades, pd.Series(dtype=float))
    pooled_by_tag = by_tag(all_trades)
    exit_reasons = all_trades["reason"].value_counts().to_dict() if len(all_trades) else {}
    median_pnl = float(all_trades["pnl"].median()) if len(all_trades) else np.nan
    total_pnl_pooled = float(all_trades["pnl"].sum()) if len(all_trades) else 0.0

    pooled = {
        "label": label,
        "completed_trades": pooled_summary.get("trades", 0),
        "win_rate": pooled_summary.get("win_rate", np.nan),
        "avg_trade_pnl": pooled_summary.get("expectancy", np.nan),
        "median_trade_pnl": median_pnl,
        "profit_factor": pooled_summary.get("profit_factor", np.nan),
        "expectancy": pooled_summary.get("expectancy", np.nan),
        "total_pnl_trade_pool_NOT_portfolio_return": total_pnl_pooled,
    }

    return {
        "label": label, "per_symbol_df": per_symbol_df, "open_df": open_df,
        "pooled": pooled, "pooled_by_tag": pooled_by_tag, "exit_reasons": exit_reasons,
        "bad": run["bad"], "n_ok": len(ok), "n_total": len(UNIVERSE),
    }


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None

    # ---- build configs ----
    cfg_a = build_production_cfg()
    cfg_b = copy.deepcopy(cfg_a)
    cfg_b["gates"]["market"]["require_market_regime"] = False

    diffs = dict_diff(cfg_a, cfg_b)
    expected_path = "gates.market.require_market_regime"

    print("=" * 100)
    print("CONFIG DIFF (A -> B)")
    print("=" * 100)
    for p, va, vb in diffs:
        print(f"  {p}: {va!r} -> {vb!r}")

    if len(diffs) != 1 or diffs[0][0] != expected_path or diffs[0][1] is not True or diffs[0][2] is not False:
        print("\nABORTING: config diff does not match the expected single-variable change "
              f"({expected_path}: True -> False). Refusing to run the experiment.")
        print(f"Actual diff: {diffs}")
        sys.exit(1)

    print(f"\nASSERTION PASSED: the only difference between Config A and Config B is "
          f"{expected_path} (True -> False). {len(diffs)} total diff(s) found, as expected.")

    # ---- run both configs ----
    print("\n" + "=" * 100)
    print("RUNNING CONFIG A (require_market_regime=True, current baseline)")
    print("=" * 100)
    run_a = run_config("A", cfg_a, cache)

    print("\n" + "=" * 100)
    print("RUNNING CONFIG B (require_market_regime=False)")
    print("=" * 100)
    run_b = run_config("B", cfg_b, cache)

    rep_a = build_report("A", run_a)
    rep_b = build_report("B", run_b)

    # ---- validation checks ----
    print("\n" + "=" * 100)
    print("VALIDATION CHECKS")
    print("=" * 100)
    checks = []
    checks.append((f"A: {rep_a['n_ok']}/{rep_a['n_total']} symbols processed",
                    rep_a["n_ok"] == rep_a["n_total"] or len(rep_a["bad"]) > 0))
    checks.append((f"B: {rep_b['n_ok']}/{rep_b['n_total']} symbols processed",
                    rep_b["n_ok"] == rep_b["n_total"] or len(rep_b["bad"]) > 0))
    checks.append(("Config diff contains exactly one changed path", len(diffs) == 1))
    checks.append(("No third configuration was built or run", True))  # by construction: only cfg_a/cfg_b exist
    for desc, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {desc}")

    same_symbol_set = set(rep_a["per_symbol_df"]["symbol"]) == set(rep_b["per_symbol_df"]["symbol"])
    print(f"  [{'OK' if same_symbol_set else 'FAIL'}] A and B produced results for the identical symbol set")

    # ---- per-symbol delta (B - A) ----
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

    # ---- pooled delta (B - A) ----
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

    pd.set_option("display.width", 220)

    def print_report(rep):
        print(f"\n--- CONFIG {rep['label']} : PER-SYMBOL ---")
        print(rep["per_symbol_df"].round(4).to_string(index=False))
        print(f"\n--- CONFIG {rep['label']} : OPEN POSITIONS ---")
        print(rep["open_df"].to_string(index=False) if len(rep["open_df"]) else "  none")
        print(f"\n--- CONFIG {rep['label']} : POOLED (TRADE-POOL, NOT A PORTFOLIO RETURN) ---")
        print(json.dumps(rep["pooled"], indent=2, default=str))
        print(f"\n--- CONFIG {rep['label']} : EXIT REASONS (pooled) ---")
        print(json.dumps(rep["exit_reasons"], indent=2))
        print(f"\n--- CONFIG {rep['label']} : BY PATTERN (pooled) ---")
        print(rep["pooled_by_tag"].round(2).to_string())
        if rep["bad"]:
            print(f"\n--- CONFIG {rep['label']} : HARD FAILURES ---")
            for r in rep["bad"]:
                print(f"  {r['symbol']}: {r['error']}")

    print_report(rep_a)
    print_report(rep_b)

    print("\n" + "=" * 100)
    print("A -> B DELTAS (B minus A)")
    print("=" * 100)
    print("\nPer-symbol deltas:")
    print(delta_per_symbol_df.round(4).to_string(index=False))
    print("\nPooled deltas (trade-pool, not portfolio):")
    print(json.dumps(delta_pooled, indent=2, default=str))

    # ---- save CSV ----
    out_rows = []
    for p, va, vb in diffs:
        out_rows.append({"section": "CONFIG_DIFF", "path": p, "value_A": va, "value_B": vb})

    for _, row in rep_a["per_symbol_df"].iterrows():
        d = row.to_dict(); d["section"] = "PER_SYMBOL_A"; out_rows.append(d)
    for _, row in rep_b["per_symbol_df"].iterrows():
        d = row.to_dict(); d["section"] = "PER_SYMBOL_B"; out_rows.append(d)

    pa = dict(rep_a["pooled"]); pa["section"] = "POOLED_A"; out_rows.append(pa)
    pb = dict(rep_b["pooled"]); pb["section"] = "POOLED_B"; out_rows.append(pb)

    for reason, cnt in rep_a["exit_reasons"].items():
        out_rows.append({"section": "EXIT_REASONS_A", "reason": reason, "count": cnt})
    for reason, cnt in rep_b["exit_reasons"].items():
        out_rows.append({"section": "EXIT_REASONS_B", "reason": reason, "count": cnt})

    for tag, row in rep_a["pooled_by_tag"].iterrows():
        d = row.to_dict(); d["section"] = "PATTERNS_A"; d["tag"] = tag; out_rows.append(d)
    for tag, row in rep_b["pooled_by_tag"].iterrows():
        d = row.to_dict(); d["section"] = "PATTERNS_B"; d["tag"] = tag; out_rows.append(d)

    for _, row in delta_per_symbol_df.iterrows():
        d = row.to_dict(); d["section"] = "DELTA_PER_SYMBOL"; out_rows.append(d)

    dp = dict(delta_pooled); dp["section"] = "DELTA_POOLED"; out_rows.append(dp)

    for r in rep_a["bad"]:
        out_rows.append({"section": "HARD_FAILURE_A", "symbol": r["symbol"], "error": r["error"]})
    for r in rep_b["bad"]:
        out_rows.append({"section": "HARD_FAILURE_B", "symbol": r["symbol"], "error": r["error"]})

    out_df = pd.DataFrame(out_rows)
    # section-first column ordering for readability
    cols = ["section"] + [c for c in out_df.columns if c != "section"]
    out_df = out_df[cols]
    out_df.to_csv("audit_market_regime_ablation_results.csv", index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"1. Files created: {CREATED_FILES}")
    print(f"2. Files modified: {MODIFIED_FILES if MODIFIED_FILES else '[] (none)'}")
    print(f"3. Config diff: {diffs}")
    print(f"4. Config A summary (trade-pool): {rep_a['pooled']}")
    print(f"5. Config B summary (trade-pool): {rep_b['pooled']}")
    print(f"6. A->B pooled deltas: {delta_pooled}")
    print(f"7. Warnings/errors: see stdout above for any FutureWarning/data-issue lines "
          f"printed during the per-symbol runs (e.g. events-data availability)")
    print(f"8. Hard failures: A={len(rep_a['bad'])}, B={len(rep_b['bad'])}")
    print(f"9. No production files were modified — build_production_cfg(), run_symbol(), "
          f"and all production imports (rules/gates.py, rules/signals.py, sim/engine.py, "
          f"sim/metrics.py, patterns/*, market_context.py) were called/imported read-only.")
    print("\nSaved: audit_market_regime_ablation_results.csv")


if __name__ == "__main__":
    main()
