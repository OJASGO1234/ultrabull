# core/build_baserates.py
"""
Precompute state-conditioned forward-return base rates from the universe.

For every bar of every universe symbol, record the state vector
    market regime (NIFTY MA20 vs MA50)  x
    stock trend  (MA20 vs MA50)         x
    relative strength 20d vs sector (+/-) x
    RSI band (<=40 / 40-60 / >60)
and the forward return at 1, 5, 21 and 63 trading days. Pool across symbols,
then store per-state P(up), median and mean move per horizon in
core/state_baserates.json. core/outlook.py looks todays state up in the table —
the app's "up / down / sideways" call is therefore an empirical base rate, not
an opinion.

Run:  python build_baserates.py <price_cache_dir>    (from core/)
"""
from __future__ import annotations
import json
import sys

import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol
from market_context import build_market_context

HORIZONS = {"1d": 1, "1w": 5, "1m": 21, "3m": 63}
WARM = 60


def state_series(df_ohlcv: pd.DataFrame, ind_df: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    trend = np.where(ind_df["MA20"] > ind_df["MA50"], "up", "dn")
    mkt = np.where(ctx["mkt_up"].astype(bool), "up", "dn")
    rs = np.where(ctx["rs_20"].fillna(0) >= 0, "+", "-")
    rsi = ind_df["RSI"]
    band = np.select([rsi <= 40, rsi <= 60], ["low", "mid"], default="high")
    keys = [f"mkt={m}|trend={t}|rs={r}|rsi={b}" for m, t, r, b in zip(mkt, trend, rs, band)]
    return pd.Series(keys, index=df_ohlcv.index)


def main(cache_dir: str | None):
    rows = []
    for sym in UNIVERSE:
        try:
            prep = prepare_symbol(sym, years=4, cache_dir=cache_dir)
            if prep is None:
                continue
            df, ind, _ = prep
            ctx = build_market_context(sym, df["Close"], years=4, cache_dir=cache_dir)
            states = state_series(df, ind, ctx)
            close = df["Close"]
            frame = {"state": states}
            for name, h in HORIZONS.items():
                frame[name] = close.shift(-h) / close - 1.0
            rows.append(pd.DataFrame(frame).iloc[WARM:])
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    pool = pd.concat(rows, ignore_index=True)

    table: dict = {"_meta": {
        "symbols": len(rows), "bars": int(len(pool)),
        "horizons": HORIZONS, "note": "pooled 4y daily, NSE universe",
    }}
    # unconditional fallback
    def _stats(v: pd.Series) -> dict:
        v = v.dropna()
        if not len(v):
            return {"n": 0, "p_up": None, "median": None, "mean": None, "q25": None, "q75": None}
        return {
            "n": int(len(v)),
            "p_up": round(float((v > 0).mean()), 4),
            "median": round(float(v.median()), 5),
            "mean": round(float(v.mean()), 5),
            "q25": round(float(v.quantile(0.25)), 5),
            "q75": round(float(v.quantile(0.75)), 5),
        }

    table["_all"] = {name: _stats(pool[name]) for name in HORIZONS}
    for state, g in pool.groupby("state"):
        table[state] = {name: _stats(g[name]) for name in HORIZONS}

    with open("state_baserates.json", "w") as f:
        json.dump(table, f, indent=1)
    print(f"{len(pool)} pooled bars, {len(table) - 2} states -> state_baserates.json")
    # quick readability check
    for st in sorted(k for k in table if not k.startswith("_")):
        r = table[st]
        print(f"{st:38s} n={r['1w']['n']:5d}  p_up 1w={r['1w']['p_up']:.2f}  3m={r['3m']['p_up']:.2f}  med3m={r['3m']['median']:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
