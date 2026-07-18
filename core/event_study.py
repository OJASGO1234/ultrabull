# core/event_study.py
"""
Per-pattern forward-return event study across a basket of NSE symbols.

For every pattern detection, measure the direction-signed forward return at
+1/+3/+5/+10 bars and compare it against the symbol's unconditional forward
return over the same period ("excess"). A pattern with no positive, reasonably
significant excess has no measurable edge and should not be traded.

Run:  python event_study.py            (from the core/ directory)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from stock_explorer import load_data, compute_indicators
from vbt_to_pd_ind import build_ind_df_from_vbt, extract_symbol_df
from patterns.config import CFG
from patterns.pipeline import run_all_patterns
from patterns.postprocess import direction_of, warmup_cutoff

UNIVERSE = [
    # index heavyweights / diverse sectors
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "ITC.NS", "HINDUNILVR.NS", "TITAN.NS",
    "ASIANPAINT.NS", "BHARTIARTL.NS", "MARUTI.NS", "M&M.NS",
    "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "ADANIPORTS.NS",
    "ULTRACEMCO.NS", "ADANIENT.NS", "INDUSINDBK.NS", "EICHERMOT.NS", "VEDL.NS",
    # midcaps incl. the demo symbol
    "HINDCOPPER.NS", "SWSOLAR.NS",
]

HORIZONS = [1, 3, 5, 10]


def _load_cached(symbol: str, years: float, cache_dir: str | None) -> pd.DataFrame:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{symbol.replace('&','_')}_{years}y.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    df = load_data(symbol, years)
    if cache_dir:
        df.to_pickle(path)
    return df


def prepare_symbol(symbol: str, years: float = 4, cache_dir: str | None = None):
    """Load data, compute indicators, run detectors. Returns (df_ohlcv, ind_df, dets) or None."""
    df = _load_cached(symbol, years, cache_dir)
    df_ohlcv = extract_symbol_df(df, symbol=symbol, case="title")
    if len(df_ohlcv) < 120:
        return None
    ind = compute_indicators(extract_symbol_df(df, symbol=symbol, case="lower"))
    ind_df = build_ind_df_from_vbt(df_ohlcv, ind)

    dets = run_all_patterns(df_ohlcv, ind_df, CFG)
    if dets is None or dets.empty:
        return None
    cut = warmup_cutoff(ind_df)
    dets = dets[dets["date"] >= cut]
    if dets.empty:
        return None
    return df_ohlcv, ind_df, dets


def compute_event_rows(dets: pd.DataFrame, df_ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Event rows for a set of detections: pattern, date, horizon, signed fwd ret, excess."""
    close = df_ohlcv["Close"]
    fwd = {h: (close.shift(-h) / close - 1.0) for h in HORIZONS}

    rows = []
    for r in dets.itertuples(index=False):
        direction = direction_of(r.pattern)
        if direction == "neutral":
            continue
        sign = 1.0 if direction == "bullish" else -1.0
        for h in HORIZONS:
            f = fwd[h].get(r.date, np.nan)
            if pd.isna(f):
                continue
            baseline = sign * float(fwd[h].mean())   # unconditional drift, signed
            rows.append({
                "symbol": symbol, "pattern": r.pattern, "date": r.date,
                "horizon": h, "signed_ret": sign * float(f),
                "excess": sign * float(f) - baseline,
            })
    return pd.DataFrame(rows) if rows else None


def study_symbol(symbol: str, years: float = 4, cache_dir: str | None = None) -> pd.DataFrame | None:
    """Return event rows for one symbol: pattern, date, horizon, signed fwd ret, excess."""
    prep = prepare_symbol(symbol, years=years, cache_dir=cache_dir)
    if prep is None:
        return None
    df_ohlcv, ind_df, dets = prep
    return compute_event_rows(dets, df_ohlcv, symbol)


def aggregate(events: pd.DataFrame) -> pd.DataFrame:
    """Per (pattern, horizon): count, mean/median excess in bps, hit rate, t-stat."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mu = g["excess"].mean()
        sd = g["excess"].std(ddof=1)
        t = mu / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
        return pd.Series({
            "n": n,
            "mean_excess_bps": mu * 1e4,
            "median_excess_bps": g["excess"].median() * 1e4,
            "hit_rate": float((g["signed_ret"] > 0).mean()),
            "t_stat": t,
        })
    out = events.groupby(["pattern", "horizon"]).apply(_agg, include_groups=False)
    return out.reset_index()


def run_study(universe=UNIVERSE, years: float = 4, cache_dir: str | None = None,
              verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_events = []
    for sym in universe:
        try:
            ev = study_symbol(sym, years=years, cache_dir=cache_dir)
            if ev is not None and len(ev):
                all_events.append(ev)
                if verbose:
                    print(f"  {sym}: {ev['date'].nunique()} detection dates")
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    events = pd.concat(all_events, ignore_index=True)
    table = aggregate(events)
    return events, table


def verdicts(table: pd.DataFrame, horizon: int = 5, min_n: int = 30,
             min_bps: float = 0.0, min_t: float = 1.0) -> pd.DataFrame:
    """Simple prune rule at the primary horizon: needs positive mean excess with t >= min_t."""
    t5 = table[table["horizon"] == horizon].copy()
    t5["verdict"] = np.where(
        (t5["n"] >= min_n) & (t5["mean_excess_bps"] > min_bps) & (t5["t_stat"] >= min_t),
        "KEEP", "PRUNE")
    t5.loc[t5["n"] < min_n, "verdict"] = "TOO_FEW"
    return t5.sort_values("mean_excess_bps", ascending=False)


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Event study over {len(UNIVERSE)} symbols, 4y daily bars…")
    events, table = run_study(cache_dir=cache)
    pd.set_option("display.width", 160)
    print("\n=== Per-pattern forward-return table (excess vs unconditional drift) ===")
    print(table.round(2).to_string(index=False))
    print("\n=== Verdicts @ 5-bar horizon ===")
    print(verdicts(table).round(2).to_string(index=False))
    events.to_csv("event_study_events.csv", index=False)
    table.to_csv("event_study_table.csv", index=False)
