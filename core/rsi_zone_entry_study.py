# core/rsi_zone_entry_study.py
"""
RSI-ZONE-ENTRY forward-return event study (overlap-corrected follow-up to
rsi_zone_study.py).

Problem being addressed: rsi_zone_study.py counted EVERY bar where RSI sat
inside a zone. Because RSI is autocorrelated, a stock sitting at RSI=55 for
10 consecutive days contributed ~10 highly overlapping, non-independent rows
to the RSI_40_65 cohort, which inflates the ordinary t-stat (computed under
an i.i.d. assumption) far beyond what the true information content supports.

Fix: only count the FIRST bar of a zone visit -- i.e. a "zone entry" -- as an
event. A zone entry on date t requires:
    RSI[t]   is inside the zone, AND
    RSI[t-1] is NOT inside the zone (or RSI[t-1] is NaN, i.e. t is the first
             valid RSI bar after warmup).
Consecutive bars where RSI stays inside the same zone are NOT separate
events. If RSI leaves the zone and later re-enters it, that is a new event.
This mirrors the exact walk given in the task spec:
    38 -> 42 -> 48 -> 55 -> 61 -> 67   (zone 40-65): only 42 is an event.
    69 -> 72 -> 75 -> 73 -> 68         (zone >70):    only 72 is an event.

Multi-zone membership on the same day: the five zones (<35, 35-40, 40-65,
65-70, >70) are contiguous and non-overlapping by construction (each is a
half-open interval on the RSI number line, partitioning [0,100] with the
boundaries at 35/40/65/70). A single RSI value can therefore belong to at
most one zone on a given day -- multi-zone qualification is structurally
impossible here, so no special handling/tie-breaking is needed. (Contrast
with core/patterns, where a bar could independently qualify for unrelated
pattern types; that is not the case for a partition of the RSI axis.)

Everything else mirrors rsi_zone_study.py / core/event_study.py exactly:
  - Same 37-symbol UNIVERSE, same 4-year lookback, same stock_explorer data
    loader, same vbt.RSI.run(close, window=14) RSI (unmodified, unverified
    changes made to it).
  - Same warmup_cutoff() applied before evaluating any zone.
  - Same forward-return convention: fwd[h] = close.shift(-h)/close - 1,
    strictly AFTER the event day (h >= 1), so there is no overlap between
    the RSI value that defines the event and the days used to measure the
    outcome.
  - Same baseline convention: symbol's unconditional (unsigned) mean forward
    return at that horizon, computed once per symbol/horizon over the WHOLE
    series (not conditioned on any zone).

ROBUSTNESS: symbol-clustered standard error / t-stat
------------------------------------------------------
Even after moving to zone-entry events, events from the SAME symbol can still
be correlated (e.g. a stock's own drift/volatility regime, or overlapping
+10-bar forward windows from two entries 3 days apart). To get a more
conservative significance estimate we add a cluster-robust t-stat with the
symbol as the clustering unit, computed directly with pandas/numpy (a
one-way cluster-robust variance estimator, the same idea as a clustered OLS
SE with a single regressor equal to 1):

    For cluster c (symbol) with n_c events and excess values e_{c,i}:
        S_c = sum_i e_{c,i}                      (cluster sum)
        n   = total events, C = number of clusters (symbols)
        mean = (sum_c S_c) / n
    Cluster-robust variance of the mean (Liang-Zeger / CR0 sandwich for a
    single-parameter "intercept-only" model, i.e. clustered mean estimator):
        Var_cluster = (1/n^2) * sum_c ( S_c - n_c * mean )^2
        se_cluster  = sqrt(Var_cluster)
        t_cluster   = mean / se_cluster
    This is the standard "clustered SE of a mean" formula: treat each
    cluster's total deviation from the overall mean as one super-residual,
    sum their squares, and scale by 1/n^2 (equivalent to the cluster-robust
    sandwich estimator for a regression of excess on a constant, clustered
    by symbol, without the small-cluster df correction -- see note in code).
    No small-C degrees-of-freedom correction is applied (there are only 37
    clusters/symbols at most, so this is already a conservative estimate;
    a common small-sample correction is C/(C-1), which is applied below).

Run:  python rsi_zone_entry_study.py         (from the core/ directory)
Output: core/rsi_zone_entry_study.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from stock_explorer import load_data, compute_indicators
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.postprocess import warmup_cutoff
from event_study import UNIVERSE, HORIZONS, _load_cached

ZONES = [
    ("RSI<35",      -np.inf, 35.0),
    ("RSI_35_40",    35.0,   40.0),
    ("RSI_40_65",    40.0,   65.0),
    ("RSI_65_70",    65.0,   70.0),
    ("RSI>70",       70.0,   np.inf),
]


def _zone_label(rsi: pd.Series) -> pd.Series:
    """Map each RSI value to its zone name (or NaN if RSI itself is NaN).
    Zones are a strict partition, so this is unambiguous."""
    labels = pd.Series(index=rsi.index, dtype=object)
    for name, lo, hi in ZONES:
        mask = (rsi > lo) & (rsi <= hi) if lo != -np.inf else (rsi <= hi)
        labels[mask & rsi.notna()] = name
    return labels


def prepare_symbol(symbol: str, years: float = 4, cache_dir: str | None = None):
    df = _load_cached(symbol, years, cache_dir)
    df_ohlcv = extract_symbol_df(df, symbol=symbol, case="title")
    if len(df_ohlcv) < 120:
        return None
    ind = compute_indicators(extract_symbol_df(df, symbol=symbol, case="lower"))
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)
    cut = warmup_cutoff(ind_df)
    df_ohlcv = df_ohlcv[df_ohlcv.index >= cut]
    ind_df = ind_df[ind_df.index >= cut]
    if df_ohlcv.empty:
        return None
    return df_ohlcv, ind_df


def compute_entry_rows(df_ohlcv: pd.DataFrame, ind_df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Zone-ENTRY events only: today's RSI in zone, yesterday's RSI not in
    that same zone (or no valid prior RSI). Forward returns strictly after
    the entry day."""
    close = df_ohlcv["Close"]
    rsi = ind_df["RSI"].reindex(close.index)
    zone_today = _zone_label(rsi)
    zone_yesterday = zone_today.shift(1)  # only uses PAST info -> no lookahead
    is_entry = zone_today.notna() & (zone_today != zone_yesterday)

    fwd = {h: (close.shift(-h) / close - 1.0) for h in HORIZONS}

    rows = []
    for d, zname in zone_today[is_entry].items():
        for h in HORIZONS:
            f_h = fwd[h]
            baseline = float(f_h.mean())
            f = f_h.get(d, np.nan)
            if pd.isna(f):
                continue
            rows.append({
                "symbol": symbol, "zone": zname, "date": d,
                "horizon": h, "ret": float(f), "baseline": baseline,
                "excess": float(f) - baseline,
            })
    return pd.DataFrame(rows) if rows else None


def study_symbol(symbol: str, years: float = 4, cache_dir: str | None = None) -> pd.DataFrame | None:
    prep = prepare_symbol(symbol, years=years, cache_dir=cache_dir)
    if prep is None:
        return None
    df_ohlcv, ind_df = prep
    return compute_entry_rows(df_ohlcv, ind_df, symbol)


def _cluster_robust_t(g: pd.DataFrame) -> tuple[float, float]:
    """Clustered SE/t-stat of the mean excess, clustering by symbol.
    Returns (se_cluster, t_cluster). See module docstring for the formula.
    Applies the standard C/(C-1) small-cluster correction."""
    n = len(g)
    mean = g["excess"].mean()
    cluster_sums = g.groupby("symbol")["excess"].apply(lambda s: s.sum() - len(s) * mean)
    C = cluster_sums.shape[0]
    if C < 2 or n == 0:
        return np.nan, np.nan
    var_cluster = (cluster_sums ** 2).sum() / (n ** 2)
    var_cluster *= C / (C - 1)  # small-cluster correction
    se_cluster = np.sqrt(var_cluster)
    t_cluster = mean / se_cluster if se_cluster > 0 else np.nan
    return se_cluster, t_cluster


def aggregate(events: pd.DataFrame) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mu = g["excess"].mean()
        sd = g["excess"].std(ddof=1)
        t_ordinary = mu / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
        _, t_cluster = _cluster_robust_t(g)
        return pd.Series({
            "n": n,
            "n_symbols": g["symbol"].nunique(),
            "mean_ret_bps": g["ret"].mean() * 1e4,
            "median_ret_bps": g["ret"].median() * 1e4,
            "baseline_bps": g["baseline"].mean() * 1e4,
            "mean_excess_bps": mu * 1e4,
            "median_excess_bps": g["excess"].median() * 1e4,
            "hit_rate": float((g["ret"] > 0).mean()),
            "t_stat_ordinary": t_ordinary,
            "t_stat_cluster": t_cluster,
        })
    out = events.groupby(["zone", "horizon"]).apply(_agg, include_groups=False)
    return out.reset_index()


def run_study(universe=UNIVERSE, years: float = 4, cache_dir: str | None = None,
              verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for sym in universe:
        try:
            ev = study_symbol(sym, years=years, cache_dir=cache_dir)
            if ev is not None and len(ev):
                all_rows.append(ev)
                if verbose:
                    n_entries = ev[ev["horizon"] == ev["horizon"].iloc[0]].shape[0]
                    print(f"  {sym}: {ev['date'].nunique()} zone-entry dates")
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    events = pd.concat(all_rows, ignore_index=True)
    table = aggregate(events)
    return events, table


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"RSI zone-ENTRY study over {len(UNIVERSE)} symbols, 4y daily bars…")
    events, table = run_study(cache_dir=cache)
    pd.set_option("display.width", 200)
    zone_order = [z[0] for z in ZONES]
    table["zone"] = pd.Categorical(table["zone"], categories=zone_order, ordered=True)
    table = table.sort_values(["zone", "horizon"])
    print("\n=== RSI zone-ENTRY forward-return table (excess vs unconditional long drift) ===")
    print(table.round(2).to_string(index=False))
    table.to_csv("rsi_zone_entry_study.csv", index=False)
    print("\nSaved: rsi_zone_entry_study.csv")

    print("\n=== Comparison @ 5-bar horizon ===")
    prev = {"RSI_40_65": (9.50, 3.24), "RSI>70": (-15.68, -3.29)}
    t5 = table[table["horizon"] == 5]
    for zone in ("RSI_40_65", "RSI>70"):
        row = t5[t5["zone"] == zone]
        if row.empty:
            continue
        r = row.iloc[0]
        prev_bps, prev_t = prev[zone]
        print(f"{zone}: PREVIOUS per-bar excess={prev_bps:+.2f}bps t={prev_t:+.2f}  |  "
              f"NEW entry-based excess={r['mean_excess_bps']:+.2f}bps "
              f"t_ordinary={r['t_stat_ordinary']:+.2f} t_cluster={r['t_stat_cluster']:+.2f} "
              f"n={int(r['n'])} n_symbols={int(r['n_symbols'])}")
