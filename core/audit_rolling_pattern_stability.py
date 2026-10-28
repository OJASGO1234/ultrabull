# core/audit_rolling_pattern_stability.py
"""
Chronological rolling / nested component-stability audit: is the
Hammer/BearishEngulfing deterioration found in the single 70/30 split
stable across MULTIPLE unseen chronological windows, or specific to that
one validation period?

Method: each symbol's own bar range is split into 8 equal consecutive
chronological blocks (a fixed, pre-specified number of windows -- not
searched or tuned), giving 7 consecutive (train=block i, test=block i+1)
window pairs per pattern, i=0..6. This is a genuine ROLLING walk-forward:
block 1 is "unseen" relative to block 0's classification, block 2 is
unseen relative to block 1's, etc. -- 7 independent train->immediately-
following-test evaluations, not one split repeated.

For each window pair and each pattern:
  - measure PF/expectancy/n in the TRAINING block only
  - freeze a classification from that training block: "positive" if
    expectancy > 0, else "negative" (a fixed, pre-declared rule -- not a
    threshold search)
  - evaluate the SAME classification question on the immediately
    following TEST block (does expectancy stay the same sign?)
  - no threshold or rule is adjusted using test-block data at any point

Reuses run_cfg()/summary_stats() logic already established in
audit_pattern_component_ablation.py (build_production_cfg()/run_symbol()
imported from audit_baseline_backtest.py) -- ONE full-universe baseline
run, no counterfactuals needed here. Does NOT modify any production file,
does NOT optimize or search any threshold/combination. Diagnosis only.

Run:  python audit_rolling_pattern_stability.py <cache_dir>
Output: core/audit_rolling_pattern_stability_results.csv (+ printed report)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from event_study import UNIVERSE
from audit_baseline_backtest import build_production_cfg, run_symbol

N_BLOCKS = 8  # fixed, pre-specified number of rolling windows (not searched)
FOCUS_PATTERNS = ["Hammer", "BearishEngulfing", "PullbackToMA20"]
OUT_ROWS = []


def log(section, rows):
    for r in rows:
        d = dict(r); d["section"] = section
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


def collect_all_trades(cache):
    cfg = build_production_cfg()
    all_trades = []
    bad = []
    for sym in UNIVERSE:
        r = run_symbol(sym, cfg, cache)
        if r["error"]:
            bad.append(r)
            continue
        trades = r["trades"]
        if len(trades) == 0:
            continue
        n = len(r["df_ohlcv"])
        t = trades.copy()
        t["symbol"] = sym
        t["n_bars"] = n
        t["block"] = np.minimum(N_BLOCKS - 1, (t["entry_bar"] / n * N_BLOCKS).astype(int))
        all_trades.append(t)
    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(), bad


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Collecting full-universe baseline trades (exact production config), splitting each "
          f"symbol's own bar range into {N_BLOCKS} equal chronological blocks "
          f"({N_BLOCKS - 1} consecutive rolling train->test window pairs).\n")

    trades, bad = collect_all_trades(cache)
    print(f"Total trades: {len(trades)} ({len(bad)} data issues)")

    pd.set_option("display.width", 240)

    all_window_rows = []
    for pat in FOCUS_PATTERNS:
        pt = trades[trades.tag == pat]
        print("\n" + "=" * 100)
        print(f"PATTERN: {pat}  (total n={len(pt)} across {N_BLOCKS} blocks)")
        print("=" * 100)

        block_stats = {b: summary_stats(pt[pt.block == b]) for b in range(N_BLOCKS)}
        print("Per-block stats (for reference):")
        bdf = pd.DataFrame(block_stats).T
        bdf.index.name = "block"
        print(bdf.round(3).to_string())

        window_rows = []
        for i in range(N_BLOCKS - 1):
            train_stats, test_stats = block_stats[i], block_stats[i + 1]
            train_class = "positive" if (train_stats["n"] > 0 and train_stats["expectancy"] > 0) else (
                          "negative" if train_stats["n"] > 0 else "no_data")
            test_class = "positive" if (test_stats["n"] > 0 and test_stats["expectancy"] > 0) else (
                         "negative" if test_stats["n"] > 0 else "no_data")
            consistent = (train_class == test_class) if "no_data" not in (train_class, test_class) else None
            window_rows.append({
                "pattern": pat, "window": i, "train_block": i, "test_block": i + 1,
                "train_n": train_stats["n"], "test_n": test_stats["n"],
                "train_pf": train_stats["profit_factor"], "test_pf": test_stats["profit_factor"],
                "train_expectancy": train_stats["expectancy"], "test_expectancy": test_stats["expectancy"],
                "train_classification": train_class, "test_classification": test_class,
                "consistent": consistent,
            })
        wdf = pd.DataFrame(window_rows)
        print(f"\nRolling window pairs (train block i -> test block i+1), frozen classification check:")
        print(wdf[["window", "train_n", "test_n", "train_pf", "test_pf", "train_expectancy",
                    "test_expectancy", "train_classification", "test_classification", "consistent"]]
              .round(3).to_string(index=False))

        valid_windows = wdf[wdf["consistent"].notna()]
        consistency_rate = valid_windows["consistent"].mean() if len(valid_windows) else np.nan
        n_test_positive = (wdf["test_classification"] == "positive").sum()
        n_test_negative = (wdf["test_classification"] == "negative").sum()
        print(f"\nConsistency rate (test matches frozen train classification): "
              f"{consistency_rate:.1%} ({int(valid_windows['consistent'].sum())}/{len(valid_windows)} windows)")
        print(f"Test-block classification breakdown: positive={n_test_positive}, negative={n_test_negative}, "
              f"no_data={(wdf['test_classification']=='no_data').sum()}")

        all_window_rows.extend(window_rows)
        log("PER_BLOCK", [{"pattern": pat, "block": b, **s} for b, s in block_stats.items()])
        log("WINDOW_CONSISTENCY_SUMMARY", [{"pattern": pat, "consistency_rate": consistency_rate,
                                              "n_test_positive": int(n_test_positive), "n_test_negative": int(n_test_negative)}])

    log("ROLLING_WINDOWS", all_window_rows)

    print("\n" + "=" * 100)
    print("SUMMARY: is the deterioration PERSISTENT (non-generalizing component) or PERIOD-SPECIFIC?")
    print("=" * 100)
    summary_rows = []
    for pat in FOCUS_PATTERNS:
        wdf = pd.DataFrame([r for r in all_window_rows if r["pattern"] == pat])
        valid = wdf[wdf["consistent"].notna()]
        rate = valid["consistent"].mean() if len(valid) else np.nan
        neg_test_windows = (wdf["test_classification"] == "negative").sum()
        pos_test_windows = (wdf["test_classification"] == "positive").sum()
        verdict = ("persistently non-generalizing" if pos_test_windows > 0 and neg_test_windows >= pos_test_windows * 2 else
                   "period-specific / mixed" if pos_test_windows > 0 else
                   "persistently negative across all windows" if neg_test_windows == len(wdf) else "insufficient data")
        summary_rows.append({"pattern": pat, "consistency_rate": rate,
                              "test_windows_positive": int(pos_test_windows), "test_windows_negative": int(neg_test_windows),
                              "verdict": verdict})
        print(f"  {pat}: consistency_rate={rate:.1%}, test-block positive={pos_test_windows}, "
              f"negative={neg_test_windows} -> {verdict}")
    log("FINAL_VERDICT", summary_rows)

    out_df = pd.DataFrame(OUT_ROWS)
    out_df.to_csv("audit_rolling_pattern_stability_results.csv", index=False)
    print(f"\nSaved: audit_rolling_pattern_stability_results.csv ({len(out_df)} rows)")
    print("\nNOTE: classification rule (expectancy sign) fixed in advance, not searched. No threshold")
    print("optimized, no combination tested, no production file/parameter changed.")


if __name__ == "__main__":
    main()
