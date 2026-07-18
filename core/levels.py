# core/levels.py
"""
Support / resistance levels the way discretionary pros mark them:

1. Find confirmed swing points — a bar whose high (low) is the highest (lowest)
   of the k bars on each side. These are the prices where supply/demand
   actually flipped, not arbitrary lines.
2. Cluster nearby swings into a zone (within a fraction of ATR): a level that
   was touched four times across three months is one strong zone, not four
   weak lines.
3. Score each zone by touch count and recency (half-life decay) — old levels
   fade, retested levels strengthen.
4. Add the 52-week high/low, which act as S/R on their own.

Returned levels are tagged support/resistance relative to the latest close.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HALF_LIFE_BARS = 120  # recency decay for touch weight


def _swings(df: pd.DataFrame, k: int) -> tuple[pd.Series, pd.Series]:
    hi, lo = df["High"], df["Low"]
    win = 2 * k + 1
    swing_hi = hi[hi == hi.rolling(win, center=True).max()].dropna()
    swing_lo = lo[lo == lo.rolling(win, center=True).min()].dropna()
    return swing_hi, swing_lo


def find_levels(
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    k: int = 3,
    cluster_atr_frac: float = 0.75,
    max_each_side: int = 3,
    max_distance_pct: float = 0.25,
) -> list[dict]:
    """Return clustered S/R levels near the current price, strongest first."""
    if len(df_ohlcv) < 2 * k + 10:
        return []

    close = float(df_ohlcv["Close"].iloc[-1])
    n = len(df_ohlcv)
    atr_now = float(ind_df["ATR"].iloc[-1]) if "ATR" in ind_df.columns and np.isfinite(ind_df["ATR"].iloc[-1]) else close * 0.02
    tol = max(1e-9, cluster_atr_frac * atr_now)

    pos_lookup = pd.Series(np.arange(n), index=df_ohlcv.index)

    swing_hi, swing_lo = _swings(df_ohlcv, k)
    points = [(float(p), int(pos_lookup[ts]), "high") for ts, p in swing_hi.items()]
    points += [(float(p), int(pos_lookup[ts]), "low") for ts, p in swing_lo.items()]
    if not points:
        return []
    points.sort(key=lambda x: x[0])

    # greedy price clustering
    clusters: list[list[tuple[float, int, str]]] = []
    for pt in points:
        if clusters and abs(pt[0] - np.mean([q[0] for q in clusters[-1]])) <= tol:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])

    levels = []
    for cl in clusters:
        weights = [0.5 ** ((n - 1 - bar) / HALF_LIFE_BARS) for _, bar, _ in cl]
        price = float(np.average([p for p, _, _ in cl], weights=weights))
        strength = float(sum(weights))
        last_bar = max(bar for _, bar, _ in cl)
        levels.append({
            "price": round(price, 2),
            "touches": len(cl),
            "strength": strength,
            "last_touch": df_ohlcv.index[last_bar],
            "kind": "support" if price <= close else "resistance",
            "source": "swings",
        })

    # 52-week extremes (act as S/R even without multiple touches)
    look = min(n, 252)
    hi52 = float(df_ohlcv["High"].iloc[-look:].max())
    lo52 = float(df_ohlcv["Low"].iloc[-look:].min())
    for price, kind in ((hi52, "resistance" if hi52 > close else "support"),
                       (lo52, "support" if lo52 < close else "resistance")):
        if not any(abs(price - lv["price"]) <= tol for lv in levels):
            levels.append({
                "price": round(price, 2), "touches": 1, "strength": 1.0,
                "last_touch": df_ohlcv.index[-1], "kind": kind, "source": "52w",
            })

    # keep levels near the current price, strongest first, top-N each side
    levels = [lv for lv in levels if abs(lv["price"] - close) / close <= max_distance_pct]
    max_s = max((lv["strength"] for lv in levels), default=1.0)
    for lv in levels:
        lv["strength"] = round(100.0 * lv["strength"] / max_s, 0)
        lv["distance_pct"] = round((lv["price"] / close - 1.0) * 100, 2)

    supports = sorted([lv for lv in levels if lv["kind"] == "support"],
                      key=lambda lv: -lv["price"])[:max_each_side]
    resistances = sorted([lv for lv in levels if lv["kind"] == "resistance"],
                         key=lambda lv: lv["price"])[:max_each_side]
    out = supports + resistances
    out.sort(key=lambda lv: lv["price"])
    return out
