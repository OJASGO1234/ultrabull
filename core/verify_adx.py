# core/verify_adx.py
"""
Independent verification of UltraBull's custom ADX implementation.

Target under test: core/vbt_to_pd_ind.py::adx(df_ohlcv, n=14)

    up, dn = h.diff(), -l.diff()
    plus_dm  = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = max(h-l, |h-c.shift()|, |l-c.shift()|)
    atr_w = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr_w
    dx  = 100 * |pdi - mdi| / (pdi + mdi)
    ADX = dx.ewm(alpha=1/n, adjust=False).mean()

This is a standard Wilder-family ADX: the +DM/-DM/TR/DX definitions match
the textbook Wilder (1978) formulas. The one place a real convention choice
is made is the SMOOTHING: UltraBull uses pandas' ewm(alpha=1/n, adjust=False),
which recursively computes

    y_0 = x_0
    y_t = (1 - alpha) * y_{t-1} + alpha * x_t          for t >= 1

Wilder's ORIGINAL 1978 smoothing method is subtly different in its seed:

    seed  = simple average of the first n raw values (x_1..x_n)
    y_n   = seed
    y_t   = y_{t-1} - y_{t-1}/n + x_t                   for t > n
          = (1 - 1/n) * y_{t-1} + (1/n) * x_t            (same recursion, alpha=1/n)

So both conventions use IDENTICAL recursive smoothing (alpha = 1/n each
step) -- the only difference is the INITIAL VALUE the recursion is seeded
with: pandas' ewm seeds with the very first data point (bar 0), while
classic Wilder seeds with a simple average of the first n data points (bar
n-1) and only starts recursing after that. Because the recursion multiplies
the seed's influence by (1-1/n) every subsequent bar, this is a WARM-UP /
INITIALIZATION difference that decays geometrically and is NOT expected to
produce material differences far into a >200-bar series -- but it does mean
the two conventions will NOT match to 1e-6 near the start of a series, and
because ADX itself is a further ewm of ewm-of-ewm (TR/DM smoothed, then DI
computed, then DX smoothed again), initialization differences compound
through three nested recursions before ADX stabilizes.

Both conventions are "valid Wilder ADX" in the sense that they use Wilder's
TR/DM/DI/DX definitions and Wilder's 1/n recursive smoothing constant; they
differ only in cold-start behavior. A third convention exists in some
libraries (e.g. certain TA-Lib builds) that seeds the first TR/DM SMA
window differently again (using bar n, not bar n-1) or drops the first n
bars as unusable/NaN rather than emitting a partially-converged value.

This script builds an INDEPENDENT reference implementation from scratch
(classic Wilder seeded-SMA convention, coded directly here, NOT copied from
vbt_to_pd_ind.py) and compares it against UltraBull's implementation using
the SAME data-loading path UltraBull itself uses (stock_explorer.load_data
+ vbt_to_pd_ind.extract_symbol_df), across 4 NSE symbols and >=1 year of
daily bars. TA-Lib and pandas_ta are not installed in this environment
(checked: `import talib` / `import pandas_ta` both fail) -- the classic
Wilder recursive formula (hand-coded here) is used as the independent
reference instead, as instructed as the fallback.

Run:  python verify_adx.py           (from the core/ directory)
Output: core/verify_adx_results.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from stock_explorer import load_data
from vbt_to_pd_ind import extract_symbol_df, adx as ultrabull_adx

SYMBOLS = ["CAMS.NS", "RELIANCE.NS", "HINDCOPPER.NS", "SWSOLAR.NS"]
YEARS = 1.5   # >= 1 year requested; padded so the 14-period recursion has
              # plenty of bars to converge past its warm-up before the
              # comparison window (see WARMUP_SKIP below)
N = 14
TOL = 1e-6
WARMUP_SKIP = 60   # bars to skip before comparing, to isolate "does it
                    # converge to the same recursion" from "does it agree
                    # during the unavoidable cold-start difference"


def reference_wilder_adx(df_ohlcv: pd.DataFrame, n: int = 14) -> pd.Series:
    """Independent, from-scratch classic Wilder (1978) ADX.

    Definitions (identical to UltraBull's for TR/+DM/-DM/DX -- these are
    the textbook Wilder formulas, not implementation choices):
      +DM_t = high_t - high_{t-1}   if that is positive AND exceeds the
              corresponding -DM candidate, else 0
      -DM_t = low_{t-1} - low_t     if that is positive AND exceeds the
              corresponding +DM candidate, else 0
      TR_t  = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
      DX_t  = 100 * |+DI_t - -DI_t| / (+DI_t + -DI_t)
      ADX   = Wilder-smoothed DX

    Smoothing (the convention under test): classic Wilder seeded-SMA, i.e.
      seed (at index n-1, 0-based) = simple mean of the first n raw values
      for t > n-1: y_t = y_{t-1} - y_{t-1}/n + x_t   (equivalently
                    y_t = (1 - 1/n)*y_{t-1} + (1/n)*x_t)
    This is applied independently to TR, +DM, -DM (first smoothing pass)
    and again to DX (second/final smoothing pass), exactly mirroring the
    two-stage smoothing structure of UltraBull's implementation, but with
    the SMA-seeded recursion instead of pandas' ewm(adjust=False) (which
    seeds with the first raw value).
    """
    cols = {str(c).lower(): c for c in df_ohlcv.columns}
    h = df_ohlcv[cols["high"]].astype(float).reset_index(drop=True)
    l = df_ohlcv[cols["low"]].astype(float).reset_index(drop=True)
    c = df_ohlcv[cols["close"]].astype(float).reset_index(drop=True)

    up = h.diff()
    dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    def wilder_smooth(x: pd.Series, n: int) -> pd.Series:
        vals = x.to_numpy(dtype=float)
        out = np.full(len(vals), np.nan)
        if len(vals) < n:
            return pd.Series(out, index=x.index)
        # classic seed: simple average of the first n values (indices 0..n-1).
        # Average-scale Wilder recursion: y_t = y_{t-1} - y_{t-1}/n + x_t/n
        # (equivalently y_t = (1-1/n)*y_{t-1} + (1/n)*x_t). NOTE: this must
        # divide the incoming raw value by n too, to stay average-scale and
        # consistent with the mean-seeded start -- using "+ x_t" instead of
        # "+ x_t/n" here (as an early draft of this script did) mixes the
        # sum-scale recursion with a mean-scale seed and makes the series
        # diverge/inflate; that was caught and fixed before reporting any
        # comparison numbers.
        seed = np.nanmean(vals[:n])
        out[n - 1] = seed
        for t in range(n, len(vals)):
            prev = out[t - 1]
            out[t] = prev - prev / n + vals[t] / n
        return pd.Series(out, index=x.index)

    atr_w = wilder_smooth(tr, n)
    pdi_s = wilder_smooth(plus_dm, n)
    mdi_s = wilder_smooth(minus_dm, n)
    pdi = 100 * pdi_s / atr_w
    mdi = 100 * mdi_s / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_out = wilder_smooth(dx, n)
    adx_out.index = df_ohlcv.index
    return adx_out


def compare_symbol(symbol: str) -> dict:
    df_raw = load_data(symbol, YEARS)
    df_ohlcv = extract_symbol_df(df_raw, symbol=symbol, case="title")

    ub = ultrabull_adx(df_ohlcv, n=N)
    ref = reference_wilder_adx(df_ohlcv, n=N)

    comp = pd.DataFrame({"ub": ub, "ref": ref}).iloc[WARMUP_SKIP:]
    comp = comp.dropna()
    diff = (comp["ub"] - comp["ref"]).abs()

    n_bars = len(comp)
    max_diff = float(diff.max()) if n_bars else np.nan
    mean_diff = float(diff.mean()) if n_bars else np.nan
    mismatches = int((diff > TOL).sum()) if n_bars else 0

    last10 = pd.DataFrame({"ub": ub, "ref": ref}).dropna().tail(10)
    last10_diff = (last10["ub"] - last10["ref"]).abs()

    return {
        "symbol": symbol,
        "bars_compared": n_bars,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "mismatches": mismatches,
        "tolerance": TOL,
        "last10_ub": last10["ub"].round(4).tolist(),
        "last10_ref": last10["ref"].round(4).tolist(),
        "last10_diff": last10_diff.round(6).tolist(),
    }


def diagnose_symbol(symbol: str) -> dict:
    """No-skip diagnostic: compare UB vs reference ADX from the earliest
    comparable bar (no WARMUP_SKIP), to test whether the mismatch is really
    just initialization/seed decay or something else."""
    df_raw = load_data(symbol, YEARS)
    df_ohlcv = extract_symbol_df(df_raw, symbol=symbol, case="title")

    ub = ultrabull_adx(df_ohlcv, n=N)
    ref = reference_wilder_adx(df_ohlcv, n=N)

    comp = pd.DataFrame({"ub": ub, "ref": ref}).dropna()
    diff = (comp["ub"] - comp["ref"]).abs()

    n_bars = len(comp)
    idx = comp.index

    mismatch_mask = diff > TOL
    mismatch_idx = idx[mismatch_mask]
    first_mismatch = mismatch_idx[0] if len(mismatch_idx) else None
    last_mismatch = mismatch_idx[-1] if len(mismatch_idx) else None

    # first point where diff <= TOL AND stays <= TOL for every subsequent bar
    below = (diff <= TOL).to_numpy()
    conv_point = None
    for i in range(len(below)):
        if below[i:].all():
            conv_point = idx[i]
            break

    final_diff = float(diff.iloc[-1]) if n_bars else np.nan

    def sample_rows(center_label, n_before=2, n_after=2):
        if center_label is None or center_label not in idx:
            return []
        pos = idx.get_loc(center_label)
        lo = max(0, pos - n_before)
        hi = min(n_bars, pos + n_after + 1)
        rows = []
        for p in range(lo, hi):
            rows.append({
                "date": idx[p],
                "ub_adx": round(float(comp["ub"].iloc[p]), 6),
                "ref_adx": round(float(comp["ref"].iloc[p]), 6),
                "abs_diff": round(float(diff.iloc[p]), 6),
            })
        return rows

    first_valid = idx[0] if n_bars else None
    bar14 = idx[14] if n_bars > 14 else None
    bar30 = idx[30] if n_bars > 30 else None
    bar60 = idx[60] if n_bars > 60 else None

    samples = {
        "first_valid": sample_rows(first_valid),
        "bar_14": sample_rows(bar14),
        "bar_30": sample_rows(bar30),
        "bar_60": sample_rows(bar60),
        "first_convergence": sample_rows(conv_point),
        "final_10": sample_rows(idx[-1], n_before=9, n_after=0) if n_bars else [],
    }

    result = {
        "symbol": symbol,
        "total_comparable_bars": n_bars,
        "first_mismatch": first_mismatch,
        "last_mismatch": last_mismatch,
        "n_mismatches": int(mismatch_mask.sum()),
        "max_abs_diff": float(diff.max()) if n_bars else np.nan,
        "mean_abs_diff": float(diff.mean()) if n_bars else np.nan,
        "first_convergence_point": conv_point,
        "final_abs_diff": final_diff,
    }

    print(f"\n--- diagnostic: {symbol} ---")
    print(f"  total comparable bars      : {n_bars}")
    print(f"  first mismatch             : {first_mismatch}")
    print(f"  last mismatch              : {last_mismatch}")
    print(f"  number of mismatches       : {result['n_mismatches']}")
    print(f"  max abs diff                : {result['max_abs_diff']:.6f}")
    print(f"  mean abs diff                : {result['mean_abs_diff']:.6f}")
    print(f"  first sustained convergence : {conv_point}")
    print(f"  final abs diff              : {final_diff:.6f}")
    for label, rows in samples.items():
        print(f"  [{label}]")
        for row in rows:
            print(f"    {row['date']}  ub={row['ub_adx']:.6f}  ref={row['ref_adx']:.6f}  diff={row['abs_diff']:.6f}")

    return result, samples


def run_diagnostics():
    all_results = []
    for sym in SYMBOLS:
        try:
            result, samples = diagnose_symbol(sym)
        except Exception as e:
            print(f"  FAILED diagnostic for {sym}: {e}")
            continue
        row = dict(result)
        for label, rows in samples.items():
            row[f"sample_{label}"] = rows
        all_results.append(row)

    pd.DataFrame(all_results).to_csv("verify_adx_diagnostics.csv", index=False)
    print("\nSaved: verify_adx_diagnostics.csv")


def main():
    results = []
    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        try:
            r = compare_symbol(sym)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        results.append(r)
        print(f"  bars compared        : {r['bars_compared']}")
        print(f"  max abs diff          : {r['max_abs_diff']:.6f}")
        print(f"  mean abs diff         : {r['mean_abs_diff']:.6f}")
        print(f"  mismatches (>{TOL})    : {r['mismatches']}")
        print(f"  last 10 UltraBull ADX : {r['last10_ub']}")
        print(f"  last 10 reference ADX : {r['last10_ref']}")
        print(f"  last 10 abs diffs     : {r['last10_diff']}")

    out_rows = []
    for r in results:
        out_rows.append({
            "symbol": r["symbol"],
            "bars_compared": r["bars_compared"],
            "max_abs_diff": r["max_abs_diff"],
            "mean_abs_diff": r["mean_abs_diff"],
            "mismatches": r["mismatches"],
            "tolerance": r["tolerance"],
            "last10_ub": r["last10_ub"],
            "last10_ref": r["last10_ref"],
            "last10_diff": r["last10_diff"],
        })
    pd.DataFrame(out_rows).to_csv("verify_adx_results.csv", index=False)
    print("\nSaved: verify_adx_results.csv")


if __name__ == "__main__":
    main()
    run_diagnostics()
