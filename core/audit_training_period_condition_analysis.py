# core/audit_training_period_condition_analysis.py
"""
Follow-up to audit_walkforward_market_condition_diagnosis.py: that script
found that winners and losers of Hammer/BearishEngulfing/PullbackToMA20
differ on ADX/RSI/ATR%/MA-spread/breadth, and that the win/lose gap on
those features (especially for Hammer) widened in the validation period.

This script tests the hypothesis "these patterns need stronger trend/
momentum conditions to work" using ONLY the training period (first 70%,
same split as every prior walk-forward script) — no validation-period data
is read, let alone used to pick anything. Group boundaries (tertiles) are
computed strictly from each pattern's OWN training-period feature values,
not searched for and not chosen to maximize a metric.

Reuses collect_trade_features() and FOCUS_PATTERNS directly from
audit_walkforward_market_condition_diagnosis.py (imported, not retyped) —
identical entry-time feature definitions (market regime, ATR%, ADX, RSI,
MA20/MA50 spread, RS_20, distribution-days, breadth), identical config,
identical universe, identical split. Does NOT modify any production file,
does NOT search for a threshold, does NOT change any parameter/gate/
pattern/SL/TP. Diagnosis only.

Method: for each pattern and each continuous feature, TRAINING-only trades
are split into three PRE-SPECIFIED, fixed groups using training-data
tertiles (33rd/67th percentile cut points, pandas qcut) labeled
"low"/"mid"/"high" by raw feature value — no directionality is assumed in
the binning itself. Market regime (categorical) is grouped as-is
(uptrend/downtrend/neutral). For each group: trade count, win rate, profit
factor, expectancy, total P&L. Whether "high" beats "low" (or vice versa,
depending on the feature's expected direction for that pattern's own
directional bias, discussed per-pattern) is reported as an observation,
not tuned to.

Run:  python audit_training_period_condition_analysis.py <cache_dir>
Output: core/audit_training_period_condition_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from audit_walkforward_market_condition_diagnosis import collect_trade_features, FOCUS_PATTERNS

CONTINUOUS_FEATURES = ["adx", "rsi", "atr_pct", "ma_spread_pct", "breadth"]

# Each pattern trades a single, fixed side by construction (patterns.side in
# pipeline.py's cfg) -- there is no long/short split WITHIN a pattern to do;
# noted explicitly per pattern below rather than fabricating one.
PATTERN_SIDE = {"Hammer": "long", "BearishEngulfing": "short", "PullbackToMA20": "long"}


def group_metrics(g: pd.DataFrame) -> dict:
    n = len(g)
    if n == 0:
        return {"n": 0, "win_rate": np.nan, "profit_factor": np.nan, "expectancy": np.nan, "total_pnl": np.nan}
    pnl = g["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gp, gl = float(wins.sum()), -float(losses.sum())
    return {
        "n": n,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": (gp / gl) if gl > 0 else np.inf,
        "expectancy": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
    }


def tertile_bins(train: pd.DataFrame, feature: str) -> pd.Series | None:
    """Fixed, pre-specified tertile cut points computed ONLY from this
    pattern's training-period values of `feature`. Returns None if there
    aren't enough distinct values to form 3 groups."""
    s = pd.to_numeric(train[feature], errors="coerce")
    valid = s.dropna()
    if len(valid) < 15 or valid.nunique() < 3:
        return None
    try:
        bins = pd.qcut(s, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return None
    if bins.dropna().nunique() < 2:
        return None
    return bins


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print("Collecting per-trade entry-time features (same as "
          "audit_walkforward_market_condition_diagnosis.py) and restricting to the "
          "TRAINING period ONLY (first 70%, same split as every prior walk-forward script).\n")

    trades, bad = collect_trade_features(cache)
    train_all = trades[trades["period"] == "train"].copy()
    print(f"Training-period trades collected: {len(train_all)} (of {len(trades)} total, {len(bad)} data issues)")

    pd.set_option("display.width", 240)
    out_rows = []

    for pat in FOCUS_PATTERNS:
        side = PATTERN_SIDE[pat]
        pt = train_all[train_all["tag"] == pat].copy()
        print("\n" + "=" * 100)
        print(f"PATTERN: {pat}  (side={side}, fixed by construction -- no long/short split within this pattern)")
        print(f"Training-period n={len(pt)}")
        print("=" * 100)

        overall = group_metrics(pt)
        print(f"Overall (all training trades): {overall}")
        out_rows.append({"section": "OVERALL", "tag": pat, "feature": "ALL", "group": "ALL", **overall})

        # (a) winners vs losers, continuous features -- same comparison as the
        # prior script, restricted here to training only, for reference.
        win, lose = pt[pt.pnl > 0], pt[pt.pnl <= 0]
        print("\n-- Winners vs losers (training only) --")
        wl_rows = []
        for feat in CONTINUOUS_FEATURES:
            wv = pd.to_numeric(win[feat], errors="coerce").mean()
            lv = pd.to_numeric(lose[feat], errors="coerce").mean()
            wl_rows.append({"feature": feat, "win_mean": wv, "lose_mean": lv, "delta_win_minus_lose": wv - lv})
        wl_df = pd.DataFrame(wl_rows)
        print(wl_df.round(3).to_string(index=False))
        for _, r in wl_df.iterrows():
            d = r.to_dict(); d["section"] = "WIN_VS_LOSE_TRAIN_ONLY"; d["tag"] = pat; out_rows.append(d)

        # (b) tertile groups per feature -- pre-specified bins from training data only
        print("\n-- Condition-strength groups (tertiles, cut points from TRAINING data only) --")
        for feat in CONTINUOUS_FEATURES:
            bins = tertile_bins(pt, feat)
            if bins is None:
                print(f"  [{feat}] insufficient distinct values for tertile split -- skipped")
                out_rows.append({"section": "TERTILE_GROUPS", "tag": pat, "feature": feat, "group": "SKIPPED_insufficient_data"})
                continue
            pt[f"_bin_{feat}"] = bins
            rows = []
            for grp in ["low", "mid", "high"]:
                g = pt[pt[f"_bin_{feat}"] == grp]
                m = group_metrics(g)
                rows.append({"group": grp, **m})
            gdf = pd.DataFrame(rows)
            print(f"\n  [{feat}]")
            print("  " + gdf.round(4).to_string(index=False).replace("\n", "\n  "))
            # monotonic check: does win_rate/expectancy increase low->mid->high (or decrease)?
            wr = gdf.set_index("group")["win_rate"]
            exp = gdf.set_index("group")["expectancy"]
            wr_increasing = wr["low"] <= wr["mid"] <= wr["high"] if wr.notna().all() else None
            wr_decreasing = wr["low"] >= wr["mid"] >= wr["high"] if wr.notna().all() else None
            exp_increasing = exp["low"] <= exp["mid"] <= exp["high"] if exp.notna().all() else None
            exp_decreasing = exp["low"] >= exp["mid"] >= exp["high"] if exp.notna().all() else None
            monotonic = "increasing" if (wr_increasing and exp_increasing) else \
                        "decreasing" if (wr_decreasing and exp_decreasing) else "not monotonic"
            print(f"  -> win_rate/expectancy pattern low->mid->high: {monotonic}")
            for _, r in gdf.iterrows():
                d = r.to_dict(); d["section"] = "TERTILE_GROUPS"; d["tag"] = pat; d["feature"] = feat
                d["monotonic"] = monotonic
                out_rows.append(d)

        # (c) market regime groups (categorical, as-is)
        print("\n-- Market regime groups --")
        regime_rows = []
        for regime in pt["mkt_regime"].unique():
            g = pt[pt["mkt_regime"] == regime]
            m = group_metrics(g)
            regime_rows.append({"regime": regime, **m})
        rdf = pd.DataFrame(regime_rows)
        print(rdf.round(4).to_string(index=False))
        for _, r in rdf.iterrows():
            d = r.to_dict(); d["section"] = "REGIME_GROUPS"; d["tag"] = pat; out_rows.append(d)

        print(f"\n-- Long/short split -- N/A: '{pat}' is a fixed-side pattern (side='{side}' in "
              f"pipeline.py's patterns config); every training trade for this tag is already {side}-only.")

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv("audit_training_period_condition_results.csv", index=False)

    print("\n" + "=" * 100)
    print("SUMMARY: is 'stronger trend/momentum conditions' hypothesis supported IN TRAINING DATA?")
    print("=" * 100)
    tert = out_df[out_df["section"] == "TERTILE_GROUPS"][["tag", "feature", "monotonic"]].drop_duplicates()
    print(tert.to_string(index=False))

    print(f"\nSaved: audit_training_period_condition_results.csv ({len(out_df)} rows)")
    print("\nNOTE: training period only. No validation-period data was read or used to choose bins.")
    print("Diagnosis only -- no threshold was searched for or tuned, no production file/parameter changed.")


if __name__ == "__main__":
    main()
