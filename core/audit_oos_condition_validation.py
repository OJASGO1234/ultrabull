# core/audit_oos_condition_validation.py
"""
Out-of-sample test of the two training-identified conditions:
  - Hammer x higher breadth
  - PullbackToMA20 x higher ADX + higher ATR%

The threshold for "higher" is FROZEN from the training period only: it is
the exact 67th-percentile (upper-tertile) cut point computed by
pd.qcut(..., q=3, retbins=True) on each pattern's TRAINING trades in
audit_training_period_condition_analysis.py / audit_breadth_adx_atr_robustness.py.
That numeric boundary is recomputed identically here (same function, same
training-only data) and then applied AS-IS to validation-period trades via
a plain >= comparison -- nothing is re-derived, re-quantiled, or re-fit
using validation data. BearishEngulfing is excluded (no clean relationship
was found and no filter was proposed for it).

Reuses collect_trade_features()/FOCUS_PATTERNS from
audit_walkforward_market_condition_diagnosis.py and group_metrics() from
audit_training_period_condition_analysis.py (imported, not retyped). Does
NOT modify any production file, does NOT search for or tune any threshold.
This is a diagnostic experiment only.

Run:  python audit_oos_condition_validation.py <cache_dir>
Output: core/audit_oos_condition_validation_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from audit_walkforward_market_condition_diagnosis import collect_trade_features
from audit_training_period_condition_analysis import group_metrics

OUT_ROWS = []


def log(section, tag, rows):
    for r in rows:
        d = dict(r); d["section"] = section; d["tag"] = tag
        OUT_ROWS.append(d)


def frozen_upper_tertile_cut(train_series: pd.Series) -> float:
    """The exact 67th-percentile boundary pd.qcut(q=3) would have used on
    this TRAINING series -- i.e. the boundary between 'mid' and 'high'.
    Returned as a single frozen number, to be applied with >= on any data
    (training or validation) from here on."""
    s = pd.to_numeric(train_series, errors="coerce").dropna()
    _, edges = pd.qcut(s, q=3, retbins=True, duplicates="drop")
    # edges = [min, q33, q67, max] (or fewer if duplicates dropped) -- the
    # boundary that starts the "high" group is the second-to-last edge.
    return float(edges[-2])


def per_symbol_distribution(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, sg in g.groupby("symbol"):
        m = group_metrics(sg)
        rows.append({"symbol": sym, **m})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def analyze_pattern(trades: pd.DataFrame, pat: str, condition_fn, frozen_thresholds: dict):
    train = trades[(trades.tag == pat) & (trades.period == "train")]
    valid = trades[(trades.tag == pat) & (trades.period == "valid")]

    print("\n" + "=" * 100)
    print(f"PATTERN: {pat}   frozen thresholds (from TRAINING only): {frozen_thresholds}")
    print("=" * 100)

    train_baseline = group_metrics(train)
    valid_baseline = group_metrics(valid)
    print(f"\nTraining baseline (unfiltered, n={len(train)}): {train_baseline}")
    print(f"Validation baseline (unfiltered, n={len(valid)}): {valid_baseline}")
    log("BASELINE", pat, [{"dataset": "train_unfiltered", **train_baseline},
                            {"dataset": "valid_unfiltered", **valid_baseline}])

    retain_mask = condition_fn(valid)
    retained = valid[retain_mask]
    rejected = valid[~retain_mask]

    print(f"\nValidation trades: {len(valid)} total -> retained={len(retained)}, rejected={len(rejected)}")
    ret_m = group_metrics(retained)
    rej_m = group_metrics(rejected)
    print(f"Retained (condition TRUE):   {ret_m}")
    print(f"Rejected (condition FALSE):  {rej_m}")
    log("FILTERED_VALID", pat, [{"group": "retained", "n_valid_total": len(valid), **ret_m},
                                  {"group": "rejected", "n_valid_total": len(valid), **rej_m}])

    print("\nComparison vs unfiltered validation baseline:")
    cmp_rows = []
    for k in ["win_rate", "profit_factor", "expectancy", "total_pnl"]:
        cmp_rows.append({"metric": k, "unfiltered_valid": valid_baseline.get(k),
                          "retained_valid": ret_m.get(k),
                          "delta": (ret_m.get(k, np.nan) - valid_baseline.get(k, np.nan))
                                   if isinstance(ret_m.get(k), (int, float)) and isinstance(valid_baseline.get(k), (int, float)) else np.nan})
    cmp_df = pd.DataFrame(cmp_rows)
    print(cmp_df.round(4).to_string(index=False))
    log("COMPARISON", pat, [{**r} for r in cmp_df.to_dict("records")])

    # % of the train->valid deterioration that filtering closes
    train_pnl, valid_pnl, retained_pnl = train_baseline["total_pnl"], valid_baseline["total_pnl"], ret_m["total_pnl"]
    deterioration = train_pnl - valid_pnl
    recovered = retained_pnl - valid_pnl
    pct_recovered = (recovered / deterioration * 100.0) if deterioration != 0 else np.nan
    print(f"\nOriginal train->valid P&L deterioration: {deterioration:.2f} "
          f"(train={train_pnl:.2f}, unfiltered_valid={valid_pnl:.2f})")
    print(f"After applying frozen condition, validation total P&L: {retained_pnl:.2f}")
    print(f"Amount of deterioration 'recovered' by filtering: {recovered:.2f} "
          f"({pct_recovered:.1f}% of the original deterioration)")
    log("DETERIORATION_RECOVERY", pat, [{"train_pnl": train_pnl, "unfiltered_valid_pnl": valid_pnl,
                                           "retained_valid_pnl": retained_pnl, "deterioration": deterioration,
                                           "recovered": recovered, "pct_recovered": pct_recovered}])

    print(f"\nPer-symbol distribution of RETAINED validation trades "
          f"(n={len(retained)}, {retained['symbol'].nunique()} distinct symbols):")
    psb = per_symbol_distribution(retained)
    print(psb.round(2).to_string(index=False) if len(psb) else "  (no retained trades)")
    log("PER_SYMBOL_RETAINED", pat, [{**r} for r in psb.to_dict("records")])

    improves = (ret_m.get("profit_factor", np.nan) > valid_baseline.get("profit_factor", np.nan) and
                ret_m.get("expectancy", np.nan) > valid_baseline.get("expectancy", np.nan))
    print(f"\n-> Does the frozen condition improve validation performance vs unfiltered baseline? {improves}")
    log("VERDICT", pat, [{"improves_validation": improves, "pct_deterioration_recovered": pct_recovered}])
    return improves, pct_recovered


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting per-trade entry-time features (train+valid), freezing the 'higher' "
          "threshold for each condition from TRAINING data only, applying as-is to "
          "validation-period trades.\n")

    trades, bad = collect_trade_features(cache)
    print(f"Total trades: {len(trades)} ({len(bad)} data issues)")

    # ---- freeze thresholds from TRAINING data only ----
    hammer_train = trades[(trades.tag == "Hammer") & (trades.period == "train")]
    breadth_thr = frozen_upper_tertile_cut(hammer_train["breadth"])

    pb_train = trades[(trades.tag == "PullbackToMA20") & (trades.period == "train")]
    adx_thr = frozen_upper_tertile_cut(pb_train["adx"])
    atr_thr = frozen_upper_tertile_cut(pb_train["atr_pct"])

    print(f"\nFROZEN thresholds (training-derived, upper-tertile boundaries):")
    print(f"  Hammer: breadth >= {breadth_thr:.4f}")
    print(f"  PullbackToMA20: adx >= {adx_thr:.4f}  AND  atr_pct >= {atr_thr:.4f}")
    log("FROZEN_THRESHOLDS", "Hammer", [{"feature": "breadth", "threshold": breadth_thr}])
    log("FROZEN_THRESHOLDS", "PullbackToMA20", [{"feature": "adx", "threshold": adx_thr},
                                                   {"feature": "atr_pct", "threshold": atr_thr}])

    pd.set_option("display.width", 240)

    hammer_improves, hammer_pct = analyze_pattern(
        trades, "Hammer",
        condition_fn=lambda v: pd.to_numeric(v["breadth"], errors="coerce") >= breadth_thr,
        frozen_thresholds={"breadth": round(breadth_thr, 4)},
    )

    pb_improves, pb_pct = analyze_pattern(
        trades, "PullbackToMA20",
        condition_fn=lambda v: (pd.to_numeric(v["adx"], errors="coerce") >= adx_thr) &
                                 (pd.to_numeric(v["atr_pct"], errors="coerce") >= atr_thr),
        frozen_thresholds={"adx": round(adx_thr, 4), "atr_pct": round(atr_thr, 4)},
    )

    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_oos_condition_validation_results.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Hammer x breadth>=frozen:        improves validation? {hammer_improves}  "
          f"({hammer_pct:.1f}% of deterioration recovered)")
    print(f"PullbackToMA20 x ADX+ATR%>=frozen: improves validation? {pb_improves}  "
          f"({pb_pct:.1f}% of deterioration recovered)")
    print(f"\nSaved: audit_oos_condition_validation_results.csv ({len(out_df)} rows)")
    print("\nNOTE: thresholds frozen from training only, never re-derived or adjusted using "
          "validation data. Diagnostic experiment only -- no production file/parameter changed.")


if __name__ == "__main__":
    main()
