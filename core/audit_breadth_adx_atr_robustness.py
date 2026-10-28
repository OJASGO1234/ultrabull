# core/audit_breadth_adx_atr_robustness.py
"""
Robustness follow-up to audit_training_period_condition_analysis.py, which
found (training data only):
  - Hammer x breadth: clean monotonic win_rate/expectancy improvement
    across pre-specified training-tertiles (low/mid/high).
  - PullbackToMA20 x ADX and x ATR% individually: clean monotonic
    improvement each.
  - BearishEngulfing: no feature showed a clean monotonic relationship.

This script checks whether those THREE findings are robust (not an
artifact of a handful of stocks or a narrow time window) using TRAINING
DATA ONLY -- no validation-period data is read or used to pick anything.
Group boundaries are the SAME pre-specified training tertiles as the prior
script (recomputed identically here, not re-searched or adjusted). For
PullbackToMA20, ADX and ATR% are combined into 3 pre-specified joint
groups (both_high / both_low / mixed) using those same individual tertile
cuts -- a fixed combination rule decided in advance, not optimized.

Robustness checks, for Hammer x breadth and PullbackToMA20 x (ADX, ATR%):
  1. Feature distributions (describe()) for winners vs losers.
  2. Sample sizes per group.
  3. win_rate / profit_factor / expectancy / total_pnl per group (as before).
  4. Per-stock consistency: how many distinct symbols populate each group,
     and a leave-one-symbol-out check on the best-performing group (does
     removing any single symbol flip the group from net-positive to
     net-negative, or change the rank order of the groups?).
  5. Temporal stability: training period split into an early half and a
     late half (by entry_bar, within training only) -- does the group
     ranking hold in both halves?

For BearishEngulfing: reports the "no clean relationship" finding as-is
and investigates ONE specific structural question raised by the prior
script's output (the ma_spread_pct "low" tertile's outsized profit factor)
-- whether it is concentrated in a small number of symbols/dates -- without
inventing or proposing any new filter.

Reuses collect_trade_features()/FOCUS_PATTERNS from
audit_walkforward_market_condition_diagnosis.py and
tertile_bins()/group_metrics() from audit_training_period_condition_analysis.py
(imported, not retyped). Does NOT modify any production file, does NOT
search for or tune any threshold. Diagnosis only.

Run:  python audit_breadth_adx_atr_robustness.py <cache_dir>
Output: core/audit_breadth_adx_atr_robustness_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from audit_walkforward_market_condition_diagnosis import collect_trade_features
from audit_training_period_condition_analysis import tertile_bins, group_metrics

OUT_ROWS = []


def log_rows(section, tag, rows):
    for r in rows:
        d = dict(r); d["section"] = section; d["tag"] = tag
        OUT_ROWS.append(d)


def per_symbol_breakdown(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, sg in g.groupby("symbol"):
        m = group_metrics(sg)
        rows.append({"symbol": sym, **m})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def leave_one_symbol_out(g: pd.DataFrame) -> pd.DataFrame:
    """For the group g, recompute metrics with each symbol removed in turn."""
    rows = []
    for sym in g["symbol"].unique():
        rest = g[g["symbol"] != sym]
        m = group_metrics(rest)
        rows.append({"excluded_symbol": sym, "n_removed": len(g[g["symbol"] == sym]), **m})
    return pd.DataFrame(rows)


def temporal_halves(g: pd.DataFrame) -> pd.DataFrame:
    """Split by entry_bar median WITHIN this group's own trades (still training-only
    data; this just asks whether the effect holds in the earlier vs later part of
    the training window, not a second split of the whole dataset)."""
    if len(g) < 10:
        return pd.DataFrame()
    med = g["entry_bar"].median()
    early = g[g["entry_bar"] <= med]
    late = g[g["entry_bar"] > med]
    rows = [{"half": "early_training", **group_metrics(early)},
            {"half": "late_training", **group_metrics(late)}]
    return pd.DataFrame(rows)


def analyze_hammer_breadth(train: pd.DataFrame):
    pat = "Hammer"
    pt = train[train["tag"] == pat].copy()
    print("\n" + "=" * 100)
    print(f"HAMMER x BREADTH  (training n={len(pt)})")
    print("=" * 100)

    win, lose = pt[pt.pnl > 0], pt[pt.pnl <= 0]
    print(f"\nSample sizes: n_win={len(win)}, n_lose={len(lose)}")
    print("\nBreadth distribution, winners vs losers:")
    desc = pd.DataFrame({"win": win["breadth"].describe(), "lose": lose["breadth"].describe()})
    print(desc.round(4).to_string())
    log_rows("FEATURE_DIST", pat, [{"feature": "breadth", "group": "win", **desc["win"].to_dict()},
                                     {"feature": "breadth", "group": "lose", **desc["lose"].to_dict()}])

    bins = tertile_bins(pt, "breadth")
    pt["_bin"] = bins
    print("\nPre-specified training-tertile groups (breadth):")
    group_rows = []
    for grp in ["low", "mid", "high"]:
        g = pt[pt["_bin"] == grp]
        m = group_metrics(g)
        n_symbols = g["symbol"].nunique()
        row = {"group": grp, **m, "n_distinct_symbols": n_symbols}
        group_rows.append(row)
    gdf = pd.DataFrame(group_rows)
    print(gdf.round(4).to_string(index=False))
    log_rows("GROUP_METRICS", pat, [{**r, "feature": "breadth"} for r in group_rows])

    # robustness: per-stock breakdown + leave-one-out on the best group (high)
    best = pt[pt["_bin"] == "high"]
    print(f"\nPer-symbol breakdown of the 'high' breadth group (n={len(best)}, "
          f"{best['symbol'].nunique()} distinct symbols):")
    psb = per_symbol_breakdown(best)
    print(psb.round(2).to_string(index=False))
    log_rows("PER_SYMBOL", pat, [{**r, "feature": "breadth", "group": "high"} for r in psb.to_dict("records")])

    print("\nLeave-one-symbol-out on the 'high' breadth group (does any single symbol "
          "flip the result?):")
    loo = leave_one_symbol_out(best)
    print(loo.round(2).to_string(index=False))
    log_rows("LEAVE_ONE_OUT", pat, [{**r, "feature": "breadth", "group": "high"} for r in loo.to_dict("records")])
    base_total_pnl = group_metrics(best)["total_pnl"]
    flips = loo[(loo["total_pnl"] <= 0) != (base_total_pnl <= 0)]
    print(f"-> Removing any single symbol flips total_pnl sign: {len(flips) > 0} "
          f"({len(flips)}/{len(loo)} symbols would flip it)")

    print("\nTemporal stability within training period (early vs late half, per breadth group):")
    for grp in ["low", "mid", "high"]:
        g = pt[pt["_bin"] == grp]
        th = temporal_halves(g)
        if len(th):
            print(f"  [{grp}]")
            print("  " + th.round(2).to_string(index=False).replace("\n", "\n  "))
            log_rows("TEMPORAL_STABILITY", pat, [{**r, "feature": "breadth", "group": grp} for r in th.to_dict("records")])


def analyze_pullback_adx_atr(train: pd.DataFrame):
    pat = "PullbackToMA20"
    pt = train[train["tag"] == pat].copy()
    print("\n" + "=" * 100)
    print(f"PULLBACKTOMA20 x (ADX, ATR%) COMBINED  (training n={len(pt)})")
    print("=" * 100)

    win, lose = pt[pt.pnl > 0], pt[pt.pnl <= 0]
    print(f"\nSample sizes: n_win={len(win)}, n_lose={len(lose)}")
    for feat in ["adx", "atr_pct"]:
        print(f"\n{feat} distribution, winners vs losers:")
        desc = pd.DataFrame({"win": win[feat].describe(), "lose": lose[feat].describe()})
        print(desc.round(4).to_string())
        log_rows("FEATURE_DIST", pat, [{"feature": feat, "group": "win", **desc["win"].to_dict()},
                                         {"feature": feat, "group": "lose", **desc["lose"].to_dict()}])

    # pre-specified combination: use the SAME individual tertile cuts as the prior
    # script, then combine into 3 fixed joint groups. This rule is decided in
    # advance (not searched): both_high if ADX=high AND ATR%=high; both_low if
    # ADX=low AND ATR%=low; everything else is "mixed".
    adx_bins = tertile_bins(pt, "adx")
    atr_bins = tertile_bins(pt, "atr_pct")
    pt["_adx_bin"] = adx_bins
    pt["_atr_bin"] = atr_bins

    def joint_group(row):
        if row["_adx_bin"] == "high" and row["_atr_bin"] == "high":
            return "both_high"
        if row["_adx_bin"] == "low" and row["_atr_bin"] == "low":
            return "both_low"
        return "mixed"

    pt["_joint"] = pt.apply(joint_group, axis=1)
    print("\nPre-specified joint groups (ADX tertile x ATR% tertile, fixed combination rule):")
    group_rows = []
    for grp in ["both_low", "mixed", "both_high"]:
        g = pt[pt["_joint"] == grp]
        m = group_metrics(g)
        n_symbols = g["symbol"].nunique()
        row = {"group": grp, **m, "n_distinct_symbols": n_symbols}
        group_rows.append(row)
    gdf = pd.DataFrame(group_rows)
    print(gdf.round(4).to_string(index=False))
    log_rows("GROUP_METRICS", pat, [{**r, "feature": "adx_atr_joint"} for r in group_rows])

    best = pt[pt["_joint"] == "both_high"]
    print(f"\nPer-symbol breakdown of the 'both_high' group (n={len(best)}, "
          f"{best['symbol'].nunique()} distinct symbols):")
    psb = per_symbol_breakdown(best)
    print(psb.round(2).to_string(index=False))
    log_rows("PER_SYMBOL", pat, [{**r, "feature": "adx_atr_joint", "group": "both_high"} for r in psb.to_dict("records")])

    print("\nLeave-one-symbol-out on the 'both_high' group:")
    loo = leave_one_symbol_out(best)
    print(loo.round(2).to_string(index=False))
    log_rows("LEAVE_ONE_OUT", pat, [{**r, "feature": "adx_atr_joint", "group": "both_high"} for r in loo.to_dict("records")])
    base_total_pnl = group_metrics(best)["total_pnl"]
    flips = loo[(loo["total_pnl"] <= 0) != (base_total_pnl <= 0)]
    print(f"-> Removing any single symbol flips total_pnl sign: {len(flips) > 0} "
          f"({len(flips)}/{len(loo)} symbols would flip it)")

    print("\nTemporal stability within training period (early vs late half, per joint group):")
    for grp in ["both_low", "mixed", "both_high"]:
        g = pt[pt["_joint"] == grp]
        th = temporal_halves(g)
        if len(th):
            print(f"  [{grp}]")
            print("  " + th.round(2).to_string(index=False).replace("\n", "\n  "))
            log_rows("TEMPORAL_STABILITY", pat, [{**r, "feature": "adx_atr_joint", "group": grp} for r in th.to_dict("records")])


def analyze_bearish_engulfing_structural(train: pd.DataFrame):
    pat = "BearishEngulfing"
    pt = train[train["tag"] == pat].copy()
    print("\n" + "=" * 100)
    print(f"BEARISHENGULFING  (training n={len(pt)}) -- NO clean relationship found "
          f"(per audit_training_period_condition_analysis.py: 0/5 features monotonic)")
    print("=" * 100)
    print("\nNo new filter is proposed. Investigating the one structural anomaly flagged "
          "previously: the ma_spread_pct 'low' tertile showed profit_factor=6.16 vs "
          "~1.3-1.4 for mid/high -- is that one bin concentrated in a few symbols/dates?")

    bins = tertile_bins(pt, "ma_spread_pct")
    pt["_bin"] = bins
    low = pt[pt["_bin"] == "low"]
    print(f"\n'low' ma_spread_pct group: n={len(low)}, {low['symbol'].nunique()} distinct symbols")
    psb = per_symbol_breakdown(low)
    print(psb.round(2).to_string(index=False))
    log_rows("STRUCTURAL_CHECK", pat, [{**r, "feature": "ma_spread_pct", "group": "low"} for r in psb.to_dict("records")])

    top_share_n = psb.iloc[0]["n"] / len(low) if len(psb) else np.nan
    top_share_pnl = psb.iloc[0]["total_pnl"] / low["pnl"].sum() if len(low) and low["pnl"].sum() != 0 else np.nan
    print(f"\nTop single symbol ({psb.iloc[0]['symbol'] if len(psb) else 'n/a'}) share of this "
          f"group's trade count: {top_share_n:.1%}, share of this group's total P&L: {top_share_pnl:.1%}")

    date_range = pd.Series(low["entry_bar"]).describe()
    print(f"\nentry_bar spread within this group (to check for date-clustering): {date_range.to_dict()}")
    log_rows("STRUCTURAL_CHECK", pat, [{"feature": "ma_spread_pct", "group": "low",
                                          "top_symbol_trade_share": top_share_n,
                                          "top_symbol_pnl_share": top_share_pnl,
                                          "entry_bar_min": date_range.get("min"), "entry_bar_max": date_range.get("max"),
                                          "entry_bar_std": date_range.get("std")}])

    concentrated = top_share_pnl > 0.5 if pd.notna(top_share_pnl) else False
    verdict = ("concentrated in one symbol" if concentrated else
               "spread across multiple symbols, not a single-stock artifact")
    print(f"\n-> Structural finding: the outsized profit factor in this bin is {verdict}.")
    log_rows("STRUCTURAL_CHECK", pat, [{"feature": "ma_spread_pct", "group": "low", "verdict": verdict}])


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting per-trade entry-time features (training period only), reusing the "
          "exact feature/bin definitions from the prior two diagnostic scripts.\n")
    trades, bad = collect_trade_features(cache)
    train = trades[trades["period"] == "train"].copy()
    print(f"Training-period trades: {len(train)} (of {len(trades)} total, {len(bad)} data issues)")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 200)

    analyze_hammer_breadth(train)
    analyze_pullback_adx_atr(train)
    analyze_bearish_engulfing_structural(train)

    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_breadth_adx_atr_robustness_results.csv", index=False)
    print(f"\nSaved: audit_breadth_adx_atr_robustness_results.csv ({len(out_df)} rows)")
    print("\nNOTE: training period only, no validation-period data read or used to choose anything.")
    print("Diagnosis/research only -- no threshold searched for or tuned, no production file changed,")
    print("no new filter proposed for BearishEngulfing.")


if __name__ == "__main__":
    main()
