# core/validate_scores.py
"""
Does the rebuilt detection score actually rank setups?

Buckets all enriched detections (universe-wide) into score bands and reports
mean forward-return excess per band. A useful score shows excess rising
monotonically with the band; a decorative one shows a flat line.

Run:  python validate_scores.py <cache_dir>   (from the core/ directory)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol, compute_event_rows
from patterns.postprocess import enrich_detections
from market_context import build_market_context

BANDS = [(0, 45), (45, 55), (55, 65), (65, 75), (75, 101)]


def collect(years: float = 4, cache_dir: str | None = None) -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        try:
            prep = prepare_symbol(sym, years=years, cache_dir=cache_dir)
            if prep is None:
                continue
            df_ohlcv, ind_df, dets = prep
            ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
            det_e = enrich_detections(dets, df_ohlcv, ind_df, sym, ctx=ctx)
            rows = compute_event_rows(det_e, df_ohlcv, sym)
            if rows is None:
                continue
            scores = det_e[["pattern", "date", "score"]].drop_duplicates(["pattern", "date"])
            rows = rows.merge(scores, on=["pattern", "date"], how="left")
            frames.append(rows)
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    events = collect(cache_dir=cache)
    pd.set_option("display.width", 160)
    for h in (5, 10):
        e = events[(events["horizon"] == h) & events["score"].notna()].copy()
        e["band"] = pd.cut(e["score"], bins=[b[0] for b in BANDS] + [BANDS[-1][1]],
                           labels=[f"{lo}-{hi-1}" for lo, hi in BANDS], right=False)
        g = e.groupby("band", observed=True)
        tbl = pd.DataFrame({
            "n": g.size(),
            "mean_excess_bps": g["excess"].mean() * 1e4,
            "hit_rate": g.apply(lambda x: (x["signed_ret"] > 0).mean(), include_groups=False),
        })
        print(f"\n=== Score bands vs forward excess @ {h}-bar horizon ===")
        print(tbl.round(2).to_string())
