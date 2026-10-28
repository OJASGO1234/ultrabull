# core/audit_pattern_component_ablation.py
"""
Production-equivalent component ablation: for each currently-traded pattern,
run the exact baseline config vs a counterfactual with ONLY that pattern
added to disable_patterns (everything else byte-identical), and compare
WHOLE-STRATEGY (all remaining patterns, pooled) performance -- training and
validation -- to see which pattern's removal moves overall validation
performance, and in which direction.

Reuses build_production_cfg()/run_symbol() from audit_baseline_backtest.py
and dict_diff() from audit_market_regime_ablation.py (imported, not
retyped) -- same proof-of-single-variable-change discipline as every prior
ablation in this series, adapted for a list-valued key (disable_patterns):
the counterfactual config is asserted to differ from baseline in EXACTLY
one way -- disable_patterns gains exactly the one pattern under test and
nothing else changes anywhere in the config.

The 70/30 chronological per-symbol train/valid split is the same one used
by every walk-forward script in this series (split_bar = 0.7 * symbol's
own bar count, by entry_bar).

Does NOT modify any production file. Does NOT optimize or search any
threshold/combination -- one pattern disabled at a time, nothing combined.

Run:  python audit_pattern_component_ablation.py <cache_dir>
Output: core/audit_pattern_component_ablation_results.csv (+ printed report)
"""
from __future__ import annotations
import copy
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from audit_baseline_backtest import build_production_cfg, run_symbol
from audit_market_regime_ablation import dict_diff

SPLIT = 0.7
OUT_ROWS = []


def log(section, pattern, rows):
    for r in rows:
        d = dict(r); d["section"] = section; d["pattern"] = pattern
        OUT_ROWS.append(d)


def summary_stats(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {"n": 0, "win_rate": np.nan, "profit_factor": np.nan, "expectancy": np.nan, "total_pnl": 0.0}
    wins, losses = g[g.pnl > 0], g[g.pnl <= 0]
    gp, gl = float(wins["pnl"].sum()), -float(losses["pnl"].sum())
    return {
        "n": len(g), "win_rate": float((g.pnl > 0).mean()),
        "profit_factor": (gp / gl) if gl > 0 else np.inf,
        "expectancy": float(g["pnl"].mean()), "total_pnl": float(g["pnl"].sum()),
    }


def run_cfg(cfg, cache) -> pd.DataFrame:
    all_trades = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)
        if r["error"]:
            continue
        trades = r["trades"]
        if len(trades) == 0:
            continue
        n = len(r["df_ohlcv"])
        split_bar = int(n * SPLIT)
        t = trades.copy()
        t["symbol"] = sym
        t["period"] = np.where(t["entry_bar"] < split_bar, "train", "valid")
        all_trades.append(t)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


def assert_single_pattern_diff(cfg_a, cfg_b, pattern):
    """Every key except disable_patterns must be identical; disable_patterns
    must differ by EXACTLY adding `pattern`."""
    a_wo = {k: v for k, v in cfg_a.items() if k != "disable_patterns"}
    b_wo = {k: v for k, v in cfg_b.items() if k != "disable_patterns"}
    diffs = dict_diff(a_wo, b_wo)
    if diffs:
        raise AssertionError(f"Unexpected config diff outside disable_patterns for {pattern}: {diffs}")
    added = set(cfg_b["disable_patterns"]) - set(cfg_a["disable_patterns"])
    removed = set(cfg_a["disable_patterns"]) - set(cfg_b["disable_patterns"])
    if added != {pattern} or removed:
        raise AssertionError(f"disable_patterns diff for {pattern} is not exactly +{{{pattern}}}: "
                              f"added={added}, removed={removed}")


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    cfg_baseline = build_production_cfg()

    print("Running BASELINE (exact production config, all currently-traded patterns active)...")
    baseline_trades = run_cfg(cfg_baseline, cache)
    print(f"Baseline: {len(baseline_trades)} total trades "
          f"(train={len(baseline_trades[baseline_trades.period=='train'])}, "
          f"valid={len(baseline_trades[baseline_trades.period=='valid'])})")

    traded_patterns = sorted(baseline_trades["tag"].unique())
    print(f"Currently-traded patterns found in baseline: {traded_patterns}\n")

    base_train = baseline_trades[baseline_trades.period == "train"]
    base_valid = baseline_trades[baseline_trades.period == "valid"]
    s_base_train, s_base_valid = summary_stats(base_train), summary_stats(base_valid)
    log("BASELINE", "ALL", [{"period": "train", **s_base_train}, {"period": "valid", **s_base_valid}])

    pd.set_option("display.width", 240)
    comparison_rows = []

    for pat in traded_patterns:
        print("=" * 100)
        print(f"COUNTERFACTUAL: disable '{pat}' (all else identical to baseline)")
        print("=" * 100)
        cfg_cf = copy.deepcopy(cfg_baseline)
        cfg_cf["disable_patterns"] = sorted(set(cfg_cf["disable_patterns"]) | {pat})
        assert_single_pattern_diff(cfg_baseline, cfg_cf, pat)
        print(f"Config diff verified: disable_patterns gains exactly '{pat}', nothing else changed.")

        cf_trades = run_cfg(cfg_cf, cache)
        cf_train = cf_trades[cf_trades.period == "train"]
        cf_valid = cf_trades[cf_trades.period == "valid"]
        s_cf_train, s_cf_valid = summary_stats(cf_train), summary_stats(cf_valid)

        print(f"\nTRAINING   baseline: {s_base_train}")
        print(f"TRAINING   counterfactual (no {pat}): {s_cf_train}")
        print(f"VALIDATION baseline: {s_base_valid}")
        print(f"VALIDATION counterfactual (no {pat}): {s_cf_valid}")

        def delta(a, b, k):
            va, vb = a.get(k), b.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and np.isfinite(va) and np.isfinite(vb):
                return vb - va
            return np.nan

        row = {
            "pattern": pat,
            "train_n_baseline": s_base_train["n"], "train_n_cf": s_cf_train["n"],
            "train_win_rate_baseline": s_base_train["win_rate"], "train_win_rate_cf": s_cf_train["win_rate"],
            "train_pf_baseline": s_base_train["profit_factor"], "train_pf_cf": s_cf_train["profit_factor"],
            "train_expectancy_baseline": s_base_train["expectancy"], "train_expectancy_cf": s_cf_train["expectancy"],
            "train_total_pnl_baseline": s_base_train["total_pnl"], "train_total_pnl_cf": s_cf_train["total_pnl"],
            "train_delta_total_pnl": delta(s_base_train, s_cf_train, "total_pnl"),
            "train_delta_expectancy": delta(s_base_train, s_cf_train, "expectancy"),
            "train_delta_pf": delta(s_base_train, s_cf_train, "profit_factor"),
            "valid_n_baseline": s_base_valid["n"], "valid_n_cf": s_cf_valid["n"],
            "valid_win_rate_baseline": s_base_valid["win_rate"], "valid_win_rate_cf": s_cf_valid["win_rate"],
            "valid_pf_baseline": s_base_valid["profit_factor"], "valid_pf_cf": s_cf_valid["profit_factor"],
            "valid_expectancy_baseline": s_base_valid["expectancy"], "valid_expectancy_cf": s_cf_valid["expectancy"],
            "valid_total_pnl_baseline": s_base_valid["total_pnl"], "valid_total_pnl_cf": s_cf_valid["total_pnl"],
            "valid_delta_total_pnl": delta(s_base_valid, s_cf_valid, "total_pnl"),
            "valid_delta_expectancy": delta(s_base_valid, s_cf_valid, "expectancy"),
            "valid_delta_pf": delta(s_base_valid, s_cf_valid, "profit_factor"),
        }
        comparison_rows.append(row)
        log("COUNTERFACTUAL", pat, [{"period": "train", **s_cf_train}, {"period": "valid", **s_cf_valid}])
        print()

    comp_df = pd.DataFrame(comparison_rows).sort_values("valid_delta_total_pnl", ascending=False).reset_index(drop=True)

    print("=" * 100)
    print("SUMMARY: effect of disabling EACH pattern (one at a time) on whole-strategy performance")
    print("Positive valid_delta_total_pnl means REMOVING that pattern would have IMPROVED validation")
    print("(i.e. that pattern is currently hurting validation performance); negative means removing")
    print("it would have made validation WORSE (i.e. that pattern is currently helping).")
    print("=" * 100)
    show_cols = ["pattern", "valid_n_baseline", "valid_n_cf", "valid_pf_baseline", "valid_pf_cf",
                 "valid_delta_pf", "valid_expectancy_baseline", "valid_expectancy_cf", "valid_delta_expectancy",
                 "valid_delta_total_pnl", "train_delta_total_pnl"]
    print(comp_df[show_cols].round(3).to_string(index=False))

    log("COMPARISON", "ALL", [{**r} for r in comp_df.to_dict("records")])
    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_pattern_component_ablation_results.csv", index=False)

    print(f"\nSaved: audit_pattern_component_ablation_results.csv ({len(out_df)} rows)")
    print("\nNOTE: one pattern disabled at a time, no combinations tested, no threshold searched or")
    print("optimized. Diagnosis only -- no production file/parameter changed.")


if __name__ == "__main__":
    main()
