# core/build_breadth.py
"""
Market breadth: % of universe symbols closing above their own 50DMA, per date.

Breadth answers the question NIFTY alone can't: is the advance broad or is the
index being carried by five heavyweights? Precomputed here (like the base-rate
table) so the API can read it instantly; refresh with a cron/manual run.

Run:  python build_breadth.py <price_cache_dir>     (from core/)
Writes core/breadth.csv with columns: date, breadth (0..1), n_symbols
"""
from __future__ import annotations
import sys

import pandas as pd

from event_study import UNIVERSE, _load_cached
from vbt_to_pd_ind import extract_symbol_df


def main(cache_dir: str | None, years: float = 4):
    above = []
    for sym in UNIVERSE:
        try:
            df = _load_cached(sym, years, cache_dir)
            px = extract_symbol_df(df, symbol=sym, case="title")["Close"]
            ma50 = px.rolling(50).mean()
            above.append((px > ma50).astype(float).rename(sym))
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    mat = pd.concat(above, axis=1)
    out = pd.DataFrame({
        "breadth": mat.mean(axis=1, skipna=True).round(4),
        "n_symbols": mat.notna().sum(axis=1),
    })
    out = out[out["n_symbols"] >= max(5, len(above) // 2)]
    out.index.name = "date"
    out.to_csv("breadth.csv")
    print(f"breadth.csv: {len(out)} dates, latest {out.index[-1].date()} "
          f"breadth={out['breadth'].iloc[-1]:.0%} over {int(out['n_symbols'].iloc[-1])} symbols")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
