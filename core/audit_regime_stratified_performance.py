# core/audit_regime_stratified_performance.py
"""
Regime-stratified performance audit of the FULL baseline strategy (all
patterns, all 37 symbols) -- not just the 3 patterns investigated
previously. Splits every closed trade by the market-context variables
already computed by production code at entry time (market regime, breadth,
distribution days, ADX, ATR%), compares training vs validation within each
group, and reports which conditions are associated with the out-of-sample
deterioration already confirmed in audit_walkforward_validation.py.

Bin boundaries are FIXED, PRE-SPECIFIED, round numbers chosen independently
of this data -- NOT searched, NOT data-derived quantiles:
  - ADX: conventional Wilder bands (<20 weak, 20-25 moderate, 25-40
    trending, >40 strong) -- textbook interpretation, not fit to this data.
  - ATR%: round percentage bands (<1.5%, 1.5-2.5%, 2.5-4%, >4%).
  - breadth: round fractional bands (<0.3, 0.3-0.5, 0.5-0.7, >0.7).
  - distribution days: 0-2 / 3-4 / 5+ -- the 5+ boundary matches
    production's OWN existing gate threshold (gates.market.dist_days_max
    = 5.0 in build_production_cfg), a real reference point already in the
    codebase, not something searched for here.
  - market regime: the existing categorical uptrend/downtrend/neutral.

Reuses collect_trade_features() from
audit_walkforward_market_condition_diagnosis.py (imported, not retyped) --
same config, same universe, same train/valid split, same feature
definitions. Does NOT modify any production file, does NOT search for or
propose any new threshold/gate. Diagnosis only.

Run:  python audit_regime_stratified_performance.py <cache_dir>
Output: core/audit_regime_stratified_performance_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from audit_walkforward_market_condition_diagnosis import collect_trade_features
from audit_training_period_condition_analysis import group_metrics

OUT_ROWS = []


def log(section, rows):
    for r in rows:
        d = dict(r); d["section"] = section
        OUT_ROWS.append(d)


def bin_adx(x):
    if pd.isna(x): return "unknown"
    if x < 20: return "weak(<20)"
    if x < 25: return "moderate(20-25)"
    if x < 40: return "trending(25-40)"
    return "strong(>=40)"


def bin_atr_pct(x):
    if pd.isna(x): return "unknown"
    if x < 1.5: return "low(<1.5%)"
    if x < 2.5: return "medium(1.5-2.5%)"
    if x < 4.0: return "high(2.5-4%)"
    return "very_high(>=4%)"


def bin_breadth(x):
    if pd.isna(x): return "unknown"
    if x < 0.3: return "weak(<0.3)"
    if x < 0.5: return "below_avg(0.3-0.5)"
    if x < 0.7: return "above_avg(0.5-0.7)"
    return "strong(>=0.7)"


def bin_dist_days(x):
    if pd.isna(x): return "unknown"
    if x <= 2: return "low(0-2)"
    if x <= 4: return "elevated(3-4)"
    return "high(>=5)"  # matches production's own gates.market.dist_days_max=5.0


BIN_FUNCS = {
    "adx_band": ("adx", bin_adx),
    "atr_pct_band": ("atr_pct", bin_atr_pct),
    "breadth_band": ("breadth", bin_breadth),
    "dist_days_band": ("dist_days", bin_dist_days),
    "mkt_regime": ("mkt_regime", lambda x: x),
}

ORDER = {
    "adx_band": ["weak(<20)", "moderate(20-25)", "trending(25-40)", "strong(>=40)", "unknown"],
    "atr_pct_band": ["low(<1.5%)", "medium(1.5-2.5%)", "high(2.5-4%)", "very_high(>=4%)", "unknown"],
    "breadth_band": ["weak(<0.3)", "below_avg(0.3-0.5)", "above_avg(0.5-0.7)", "strong(>=0.7)", "unknown"],
    "dist_days_band": ["low(0-2)", "elevated(3-4)", "high(>=5)", "unknown"],
    "mkt_regime": ["uptrend", "downtrend", "neutral"],
}


def stratify(trades: pd.DataFrame, dim: str) -> pd.DataFrame:
    src_col, fn = BIN_FUNCS[dim]
    trades = trades.copy()
    trades["_grp"] = trades[src_col].apply(fn)

    rows = []
    for period in ("train", "valid"):
        p = trades[trades["period"] == period]
        n_total = len(p)
        for grp in ORDER[dim]:
            g = p[p["_grp"] == grp]
            if len(g) == 0 and grp not in p["_grp"].unique():
                continue
            m = group_metrics(g)
            rows.append({"dimension": dim, "group": grp, "period": period,
                         "pct_of_trades": (len(g) / n_total * 100.0) if n_total else np.nan, **m})
    return pd.DataFrame(rows)


def print_and_compare(dim: str, df: pd.DataFrame):
    print("\n" + "=" * 100)
    print(f"DIMENSION: {dim}")
    print("=" * 100)
    pivot = df.pivot(index="group", columns="period",
                      values=["n", "pct_of_trades", "win_rate", "profit_factor", "expectancy", "total_pnl"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    order = [g for g in ORDER[dim] if g in pivot.index]
    pivot = pivot.reindex(order)
    for metric in ["n", "pct_of_trades", "win_rate", "profit_factor", "expectancy", "total_pnl"]:
        tcol, vcol = f"{metric}_train", f"{metric}_valid"
        if tcol in pivot.columns and vcol in pivot.columns:
            pivot[f"delta_{metric}"] = pivot[vcol] - pivot[tcol]
    print(pivot.round(3).to_string())
    return pivot


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting per-trade entry-time features for the FULL baseline strategy "
          "(all patterns, all 37 symbols), same config/split as every prior walk-forward script.\n")

    trades, bad = collect_trade_features(cache)
    print(f"Total trades: {len(trades)} ({len(bad)} data issues)")
    train, valid = trades[trades.period == "train"], trades[trades.period == "valid"]
    print(f"Training: {len(train)} trades, Validation: {len(valid)} trades")

    overall_train, overall_valid = group_metrics(train), group_metrics(valid)
    print(f"\nOverall baseline (for reference): train={overall_train}")
    print(f"                                   valid={overall_valid}")
    log("OVERALL", [{"period": "train", **overall_train}, {"period": "valid", **overall_valid}])

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)

    pivots = {}
    for dim in BIN_FUNCS:
        df = stratify(trades, dim)
        log("STRATIFIED", [{**r} for r in df.to_dict("records")])
        pivot = print_and_compare(dim, df)
        pivots[dim] = pivot

    # ---- identify which groups are most associated with the OOS deterioration ----
    print("\n" + "=" * 100)
    print("WHICH GROUPS ARE MOST ASSOCIATED WITH THE OOS DETERIORATION?")
    print("(ranked by delta_total_pnl within each dimension; more negative = more deterioration; "
          "also shown: whether that group's share of trades GREW in validation)")
    print("=" * 100)
    assoc_rows = []
    for dim, pivot in pivots.items():
        if "delta_total_pnl" not in pivot.columns:
            continue
        for grp, row in pivot.iterrows():
            if pd.isna(row.get("delta_total_pnl")):
                continue
            assoc_rows.append({
                "dimension": dim, "group": grp,
                "delta_total_pnl": row.get("delta_total_pnl"),
                "pct_of_trades_train": row.get("pct_of_trades_train"),
                "pct_of_trades_valid": row.get("pct_of_trades_valid"),
                "delta_pct_of_trades": (row.get("pct_of_trades_valid", np.nan) - row.get("pct_of_trades_train", np.nan)),
                "win_rate_train": row.get("win_rate_train"), "win_rate_valid": row.get("win_rate_valid"),
            })
    assoc_df = pd.DataFrame(assoc_rows).sort_values("delta_total_pnl").reset_index(drop=True)
    print(assoc_df.round(3).to_string(index=False))
    log("ASSOCIATION_RANKING", [{**r} for r in assoc_df.to_dict("records")])

    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_regime_stratified_performance_results.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY (observations only, no threshold proposed)")
    print("=" * 100)
    worst = assoc_df.iloc[0]
    print(f"Largest single-group deterioration: {worst['dimension']} = '{worst['group']}' "
          f"(delta_total_pnl={worst['delta_total_pnl']:.2f})")
    grown = assoc_df[assoc_df["delta_pct_of_trades"] > 0].sort_values("delta_pct_of_trades", ascending=False)
    print("\nGroups whose SHARE of trades grew from train to valid (composition shift):")
    print(grown[["dimension", "group", "pct_of_trades_train", "pct_of_trades_valid", "delta_pct_of_trades"]].round(2).to_string(index=False))

    print(f"\nSaved: audit_regime_stratified_performance_results.csv ({len(out_df)} rows)")
    print("\nNOTE: diagnosis only. Bins are fixed/conventional, not searched or optimized. "
          "No new gate proposed. No production file/parameter changed.")


if __name__ == "__main__":
    main()
