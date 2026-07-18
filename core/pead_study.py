# core/pead_study.py
"""
Post-Earnings Announcement Drift (PEAD) on the NSE universe.

Classic finding (US markets): the initial earnings reaction continues to
drift the same direction for weeks. Here we test it on our universe:
  reaction = close-to-close move on the first trading day at/after each
  earnings date; drift = forward return 5/10/20 bars AFTER the reaction day,
  signed by the reaction direction, vs the symbol's unconditional drift.

Run:  python pead_study.py <price_cache_dir>    (from core/)
"""
from __future__ import annotations
import sys

import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol
from evaluate_gates import _load_events_cached

HORIZONS = (5, 10, 20)
BUCKETS = [(-np.inf, -0.03, "big miss <-3%"), (-0.03, -0.01, "miss -3..-1%"),
           (-0.01, 0.01, "flat ±1%"), (0.01, 0.03, "beat +1..3%"),
           (0.03, np.inf, "big beat >+3%")]


def main(cache_dir: str | None):
    rows = []
    for sym in UNIVERSE:
        try:
            prep = prepare_symbol(sym, years=4, cache_dir=cache_dir)
            if prep is None:
                continue
            df, _, _ = prep
            ev = _load_events_cached(sym, df, cache_dir)
            if ev is None or not len(ev):
                continue
            edates = pd.to_datetime(ev.loc[ev["type"].astype(str).str.upper() == "EARNINGS", "date"])
            if not len(edates):
                continue
            close = df["Close"]
            idx = close.index
            fwd = {h: close.shift(-h) / close - 1.0 for h in HORIZONS}
            base = {h: float(fwd[h].mean()) for h in HORIZONS}
            for d in edates:
                p = idx.searchsorted(d.normalize())
                if p <= 0 or p >= len(idx) - 1:
                    continue
                rd = idx[p]  # reaction day (announcement day or next session)
                reaction = float(close.loc[rd] / close.iloc[p - 1] - 1.0)
                sign = 1.0 if reaction >= 0 else -1.0
                for h in HORIZONS:
                    f = fwd[h].loc[rd]
                    if pd.isna(f):
                        continue
                    rows.append({
                        "symbol": sym, "date": rd, "reaction": reaction, "horizon": h,
                        "signed_drift": sign * float(f),
                        "excess": sign * float(f) - sign * base[h],
                    })
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")

    ev = pd.DataFrame(rows)
    print(f"\n{ev['date'].nunique()} earnings reactions across {ev['symbol'].nunique()} symbols\n")
    print("=== Drift AFTER the reaction day, signed by reaction direction (bps, excess vs drift) ===")
    for lo, hi, label in BUCKETS:
        sub = ev[(ev["reaction"] > lo) & (ev["reaction"] <= hi)]
        if not len(sub):
            continue
        line = f"{label:16s}"
        for h in HORIZONS:
            s = sub[sub["horizon"] == h]["excess"]
            t = s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan
            line += f" | {h:2d}d: {s.mean() * 1e4:+7.1f} bps (n={len(s):3d}, t={t:+.1f})"
        print(line)
    print("\n(positive = the initial reaction direction CONTINUED — drift exists)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
