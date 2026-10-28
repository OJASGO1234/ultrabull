# core/verify_rsi_multiple_testing.py
"""
Multiple-testing correction for the RSI zone-entry study.

Input: core/rsi_zone_entry_study.csv (20 rows = 5 zones x 4 horizons), produced
by core/rsi_zone_entry_study.py. That CSV is read-only here; this script does
not modify it, rsi_zone_study.py, event_study.py, or any production file.

WHAT INFORMATION THE CSV CONTAINS AND WHY IT IS SUFFICIENT
------------------------------------------------------------
For each (zone, horizon) the CSV stores: n (event count), n_symbols (distinct
symbols contributing, i.e. the number of clusters), mean_excess_bps, and two
already-computed t-statistics: t_stat_ordinary (i.i.d. assumption, df = n-1)
and t_stat_cluster (symbol-clustered sandwich estimator, df = n_symbols-1,
the conventional cluster-robust df choice -- one degree of freedom lost per
cluster mean, matching the C/(C-1) small-cluster correction already baked
into t_stat_cluster when it was computed).

A p-value is a deterministic function of (t-statistic, degrees of freedom)
under the assumed reference distribution -- it requires no additional
information beyond what a t-stat + its df already encode. Computing
p = 2 * (1 - T_df.cdf(|t|)) from a stored t-stat and stored df is therefore
NOT "inventing" a number from the t-stat alone in the sense the task warns
against (e.g. guessing a p-value from a t-stat computed under a DIFFERENT,
unstated df, or fabricating a df). Here the df is explicit and matches
exactly how each t-stat was originally computed in rsi_zone_entry_study.py:
  - t_stat_ordinary was computed as mean/(sd/sqrt(n)) -> reference df = n-1.
  - t_stat_cluster was computed as a clustered sandwich mean estimator with
    the C/(C-1) small-cluster correction -> reference df = n_symbols-1
    (standard practice: one df per cluster, matching the correction already
    applied to the cluster variance).
No event-level (per-detection) data is required for this step -- the CSV's
aggregated (t, n, n_symbols) triple is sufficient to recover valid p-values
for both the ordinary and cluster-robust tests. If a reviewer wants an exact
(non-t-distribution-approximated) permutation or bootstrap p-value instead,
THAT would require the underlying event-level excess-return rows (i.e. the
per-detection table, not just the aggregate), which this CSV does not carry
and rsi_zone_entry_study.py does not currently save to disk.

WHICH INFERENCE THE BH CORRECTION IS APPLIED TO
------------------------------------------------
BH-FDR is applied to the CLUSTER-ROBUST p-values, not the ordinary ones, and
reported as the primary/conservative result. Rationale: the ordinary t-stat
assumes i.i.d. events, which the RSI zone-entry study already documents as
an unrealistic assumption (events from the same symbol share return-regime
and volatility-regime correlation even after de-duplicating consecutive
zone-dwell bars). Running FDR correction on top of an already-anti-conservative
p-value would understate the true false-discovery risk. The ordinary p-values
and their own BH-adjusted column are still reported alongside for
transparency/comparison, but the cluster-robust BH result is what the
verdicts below are based on.

Run:  python verify_rsi_multiple_testing.py     (from the core/ directory)
Output: core/rsi_multiple_testing.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

IN_CSV = "rsi_zone_entry_study.csv"
OUT_CSV = "rsi_multiple_testing.csv"
Q = 0.05  # FDR level


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment. Returns adjusted p-values
    (q-values) aligned to the original (unsorted) order of `pvals`."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    adj_sorted = ranked * n / ranks
    # enforce monotonicity (step-up): adjusted p-values must be non-decreasing
    # when read from largest rank down to smallest
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    out = np.empty(n)
    out[order] = adj_sorted
    return out


def main():
    df = pd.read_csv(IN_CSV)
    required = {"zone", "horizon", "n", "n_symbols", "mean_excess_bps",
                "t_stat_ordinary", "t_stat_cluster"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"STOP: rsi_zone_entry_study.csv is missing required columns: {missing}. "
            "Cannot reconstruct p-values without them."
        )
    if len(df) != 20:
        print(f"WARNING: expected 20 zone x horizon rows, found {len(df)}.")

    df = df.copy()
    df["df_ordinary"] = df["n"] - 1
    df["df_cluster"] = df["n_symbols"] - 1

    # two-sided p-values from the t-distribution with the matching df
    df["p_ordinary"] = 2 * stats.t.sf(np.abs(df["t_stat_ordinary"]), df["df_ordinary"])
    df["p_cluster"] = 2 * stats.t.sf(np.abs(df["t_stat_cluster"]), df["df_cluster"])

    df["p_ordinary_bh"] = bh_adjust(df["p_ordinary"].to_numpy())
    df["p_cluster_bh"] = bh_adjust(df["p_cluster"].to_numpy())

    df["survives_fdr_ordinary_q05"] = df["p_ordinary_bh"] < Q
    df["survives_fdr_cluster_q05"] = df["p_cluster_bh"] < Q

    cols = ["zone", "horizon", "n", "n_symbols", "mean_excess_bps",
            "t_stat_ordinary", "p_ordinary", "p_ordinary_bh", "survives_fdr_ordinary_q05",
            "t_stat_cluster", "p_cluster", "p_cluster_bh", "survives_fdr_cluster_q05"]
    out = df[cols].sort_values(["zone", "horizon"])
    out.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 220)
    print(f"20 tests (5 zones x 4 horizons) read from {IN_CSV}.")
    print("BH-FDR applied separately to ordinary and cluster-robust p-values (q=0.05).\n")
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")

    # --- primary hypotheses at the 5-bar horizon ---
    h1 = df[(df["zone"] == "RSI_40_65") & (df["horizon"] == 5)].iloc[0]
    h2 = df[(df["zone"] == "RSI>70") & (df["horizon"] == 5)].iloc[0]

    print("\n=== Primary hypotheses @ +5 bars ===")
    print("H1: RSI 40-65 has POSITIVE excess return at +5 bars.")
    print(f"    point estimate = {h1['mean_excess_bps']:+.2f} bps, n={int(h1['n'])}, n_symbols={int(h1['n_symbols'])}")
    print(f"    ordinary : t={h1['t_stat_ordinary']:+.2f}  p={h1['p_ordinary']:.4f}  BH-adj p={h1['p_ordinary_bh']:.4f}  "
          f"survives q=0.05: {bool(h1['p_ordinary_bh'] < Q)}")
    print(f"    cluster  : t={h1['t_stat_cluster']:+.2f}  p={h1['p_cluster']:.4f}  BH-adj p={h1['p_cluster_bh']:.4f}  "
          f"survives q=0.05: {bool(h1['p_cluster_bh'] < Q)}")

    print("\nH2: RSI >70 has NEGATIVE excess return at +5 bars.")
    print(f"    point estimate = {h2['mean_excess_bps']:+.2f} bps, n={int(h2['n'])}, n_symbols={int(h2['n_symbols'])}")
    print(f"    ordinary : t={h2['t_stat_ordinary']:+.2f}  p={h2['p_ordinary']:.4f}  BH-adj p={h2['p_ordinary_bh']:.4f}  "
          f"survives q=0.05: {bool(h2['p_ordinary_bh'] < Q)}")
    print(f"    cluster  : t={h2['t_stat_cluster']:+.2f}  p={h2['p_cluster']:.4f}  BH-adj p={h2['p_cluster_bh']:.4f}  "
          f"survives q=0.05: {bool(h2['p_cluster_bh'] < Q)}")


if __name__ == "__main__":
    main()
