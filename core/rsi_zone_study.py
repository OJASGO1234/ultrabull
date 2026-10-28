# core/rsi_zone_study.py
"""
Independent RSI-ZONE forward-return event study.

Purpose: test the user-facing claim (core/indicators.py, core/advice.py) that
"RSI 40-65 is the one RSI zone with tested edge", and that RSI>70 fresh longs
are negative. Unlike core/event_study.py (which studies discrete pattern
*events*, e.g. RSI crossing 40 or 60), this script studies RSI *level cohorts*
directly: for every bar where RSI falls in a given zone, what is the forward
close-to-close return at +1/+3/+5/+10 bars?

Methodology mirrors core/event_study.py exactly, so the results are
comparable:
  - Same 37-symbol UNIVERSE, same 4-year lookback, same data loader
    (stock_explorer.load_data -> compute_indicators, i.e. the SAME
    vbt.RSI.run(close, window=14) SMA-based RSI already verified in
    verify_rsi.py -- NOT re-implemented, NOT changed).
  - Same forward-return convention: fwd[h] = close.shift(-h)/close - 1
    (simple close-to-close, no execution delay, no costs -- a signal-quality
    measure, exactly like event_study.py).
  - Same baseline convention: the symbol's unconditional mean forward return
    over the FULL series at that horizon (long-only sign, since RSI zones
    have no independent "direction" the way a bullish/bearish pattern does --
    see note below).
  - Same warmup cutoff (core.patterns.postprocess.warmup_cutoff) is applied
    before evaluating any RSI cohort, so bars where RSI/MA/ATR are still
    warming up are excluded.

RSI condition timing: the RSI value used for cohort membership on date t is
the CLOSE-based RSI(14) computed through bar t (i.e. ind_df["RSI"].loc[t]),
which uses only price data up to and including bar t's own close. No future
RSI values are used. The forward return is then measured strictly AFTER date
t (close[t+1..t+h]), so there is no overlap between the conditioning
information and the outcome.

Note on baseline/sign: event_study.py signs each event by the pattern's
declared direction (bullish/bearish) before comparing to a similarly-signed
baseline. RSI zone membership is not itself a directional call in the same
sense (a trader entering because RSI is in a given zone is implicitly going
LONG), so here every cohort is treated as a long-side (unsigned, sign=+1)
question: "if I bought because RSI was in this zone, what forward return
would I have gotten, vs. the symbol's unconditional (unsigned) mean forward
return over the same horizon?" This matches how indicators.py/advice.py
phrase the claim ("fresh longs tested NEGATIVE" / "momentum without
overheat") -- both are long-entry framings.

Run:  python rsi_zone_study.py            (from the core/ directory)
Output: core/rsi_zone_study.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from stock_explorer import load_data, compute_indicators
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.postprocess import warmup_cutoff
from event_study import UNIVERSE, HORIZONS, _load_cached  # same universe, horizons, loader

ZONES = [
    ("RSI<35",      -np.inf, 35.0),
    ("RSI_35_40",    35.0,   40.0),
    ("RSI_40_65",    40.0,   65.0),
    ("RSI_65_70",    65.0,   70.0),
    ("RSI>70",       70.0,   np.inf),
]


def prepare_symbol(symbol: str, years: float = 4, cache_dir: str | None = None):
    """Load data, compute indicators. Returns (df_ohlcv, ind_df) or None. Mirrors
    event_study.prepare_symbol but skips pattern detection (not needed here)."""
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


def compute_zone_rows(df_ohlcv: pd.DataFrame, ind_df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """For every bar, tag its RSI zone and compute signed (long-only) forward
    returns at each horizon, same close-to-close convention as event_study.py."""
    close = df_ohlcv["Close"]
    rsi = ind_df["RSI"].reindex(close.index)
    fwd = {h: (close.shift(-h) / close - 1.0) for h in HORIZONS}

    rows = []
    for zone_name, lo, hi in ZONES:
        mask = (rsi > lo) & (rsi <= hi) if lo != -np.inf else (rsi <= hi)
        mask = mask & rsi.notna()
        dates = close.index[mask]
        for h in HORIZONS:
            f_h = fwd[h]
            baseline = float(f_h.mean())  # unconditional (long, unsigned) drift for this symbol/horizon
            for d in dates:
                f = f_h.get(d, np.nan)
                if pd.isna(f):
                    continue
                rows.append({
                    "symbol": symbol, "zone": zone_name, "date": d,
                    "horizon": h, "ret": float(f), "baseline": baseline,
                    "excess": float(f) - baseline,
                })
    return pd.DataFrame(rows) if rows else None


def study_symbol(symbol: str, years: float = 4, cache_dir: str | None = None) -> pd.DataFrame | None:
    prep = prepare_symbol(symbol, years=years, cache_dir=cache_dir)
    if prep is None:
        return None
    df_ohlcv, ind_df = prep
    return compute_zone_rows(df_ohlcv, ind_df, symbol)


def aggregate(events: pd.DataFrame) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mu = g["excess"].mean()
        sd = g["excess"].std(ddof=1)
        t = mu / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
        return pd.Series({
            "n": n,
            "mean_ret_bps": g["ret"].mean() * 1e4,
            "median_ret_bps": g["ret"].median() * 1e4,
            "baseline_bps": g["baseline"].mean() * 1e4,
            "mean_excess_bps": mu * 1e4,
            "median_excess_bps": g["excess"].median() * 1e4,
            "hit_rate": float((g["ret"] > 0).mean()),
            "t_stat": t,
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
                    print(f"  {sym}: {ev['date'].nunique()} bars tagged across zones")
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    events = pd.concat(all_rows, ignore_index=True)
    table = aggregate(events)
    return events, table


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"RSI-zone study over {len(UNIVERSE)} symbols, 4y daily bars…")
    events, table = run_study(cache_dir=cache)
    pd.set_option("display.width", 160)
    zone_order = [z[0] for z in ZONES]
    table["zone"] = pd.Categorical(table["zone"], categories=zone_order, ordered=True)
    table = table.sort_values(["zone", "horizon"])
    print("\n=== RSI zone forward-return table (excess vs unconditional long drift) ===")
    print(table.round(2).to_string(index=False))
    table.to_csv("rsi_zone_study.csv", index=False)
    print("\nSaved: rsi_zone_study.csv")
