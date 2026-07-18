# core/idea_lab.py
"""
Natural-language idea testing, the safe way.

Inspired by finny-ai's prompt->strategy->backtest loop, with one structural
difference: ideas do NOT compile to arbitrary generated code. They compile to a
constrained rule spec (feature / operator / value conjunctions) evaluated by
the same event-study methodology every other claim in this project passed
through. No exec(), no lookahead risk, and every answer ships with sample
size, t-stat, and a recent-12-months stability split (edges here decay — the
walk-forward proved it).

Used by the chat assistant's `test_rule` tool and POST /api/idea.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from event_study import UNIVERSE, _load_cached
from vbt_to_pd_ind import extract_symbol_df, adx as adx_calc
from market_context import (
    SECTOR_INDEX, SYMBOL_SECTOR, _load_index_close,
    load_breadth, distribution_days,
)

HORIZONS = (5, 10, 21)
WARM = 60

FEATURES = [
    "rsi", "adx", "atr_pct", "vol_mult20",
    "close_vs_ma20_pct", "close_vs_ma50_pct", "off_52w_high_pct",
    "ret_5_pct", "ret_20_pct", "ret_60_pct",
    "macd_above_signal", "mkt_up", "rs_20_pct",
    "breadth_pct", "dist_days",
]
OPS = {"<", "<=", ">", ">=", "=="}

_frames: dict[str, pd.DataFrame] = {}
_cache_dir: str | None = None


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _feature_frame(symbol: str, cache_dir: str | None) -> pd.DataFrame | None:
    df = _load_cached(symbol, 4, cache_dir)
    px = extract_symbol_df(df, symbol=symbol, case="title")
    if len(px) < WARM + 30:
        return None
    c, v, h, l = px["Close"], px["Volume"], px["High"], px["Low"]

    f = pd.DataFrame(index=px.index)
    f["rsi"] = _rsi(c)
    f["adx"] = adx_calc(px)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100
    f["vol_mult20"] = v / v.rolling(20).mean()
    f["close_vs_ma20_pct"] = (c / c.rolling(20).mean() - 1) * 100
    f["close_vs_ma50_pct"] = (c / c.rolling(50).mean() - 1) * 100
    f["off_52w_high_pct"] = (c / c.rolling(252, min_periods=60).max() - 1) * 100
    for n in (5, 20, 60):
        f[f"ret_{n}_pct"] = c.pct_change(n) * 100
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_above_signal"] = (macd > macd.ewm(span=9, adjust=False).mean()).astype(float)

    mkt = _load_index_close("^NSEI", 4.5, cache_dir)
    if mkt is not None:
        m = mkt.reindex(px.index).ffill()
        f["mkt_up"] = (m.rolling(20).mean() > m.rolling(50).mean()).astype(float)
    else:
        f["mkt_up"] = np.nan
    sect_key = SYMBOL_SECTOR.get(symbol)
    sect = _load_index_close(SECTOR_INDEX[sect_key], 4.5, cache_dir) if sect_key else mkt
    if sect is not None:
        s = sect.reindex(px.index).ffill()
        f["rs_20_pct"] = (c.pct_change(20) - s.pct_change(20)) * 100
    else:
        f["rs_20_pct"] = np.nan
    b = load_breadth()
    f["breadth_pct"] = b.reindex(px.index).ffill() * 100 if b is not None else np.nan
    dd = distribution_days(4.3)
    f["dist_days"] = dd.reindex(px.index).ffill() if dd is not None else np.nan

    for hz in HORIZONS:
        f[f"_fwd_{hz}"] = c.shift(-hz) / c - 1.0
    return f.iloc[WARM:]


def _frame(symbol: str, cache_dir: str | None) -> pd.DataFrame | None:
    global _cache_dir
    if cache_dir != _cache_dir:
        _frames.clear()
        _cache_dir = cache_dir
    if symbol not in _frames:
        try:
            _frames[symbol] = _feature_frame(symbol, cache_dir)
        except Exception:
            _frames[symbol] = None
    return _frames[symbol]


def evaluate_rule(conditions: list[dict], direction: str = "long",
                  name: str = "idea", cache_dir: str | None = None) -> dict:
    """Pooled forward-return stats for bars where ALL conditions hold."""
    if not conditions or len(conditions) > 6:
        raise ValueError("Provide 1-6 conditions")
    for c in conditions:
        if c.get("feature") not in FEATURES:
            raise ValueError(f"Unknown feature {c.get('feature')!r}. Allowed: {FEATURES}")
        if c.get("op") not in OPS:
            raise ValueError(f"Unknown op {c.get('op')!r}. Allowed: {sorted(OPS)}")
        float(c["value"])  # must be numeric
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    sign = 1.0 if direction == "long" else -1.0

    rows, base_rows = [], []
    used = 0
    for sym in UNIVERSE:
        f = _frame(sym, cache_dir)
        if f is None:
            continue
        used += 1
        mask = pd.Series(True, index=f.index)
        for c in conditions:
            col = f[c["feature"]]
            v = float(c["value"])
            op = c["op"]
            m = (col < v) if op == "<" else (col <= v) if op == "<=" else \
                (col > v) if op == ">" else (col >= v) if op == ">=" else (col == v)
            mask &= m.fillna(False)
        for hz in HORIZONS:
            fwd = f[f"_fwd_{hz}"]
            base_mu = float(fwd.mean())
            sel = fwd[mask].dropna()
            base_rows.append({"h": hz, "n": int(fwd.notna().sum())})
            for ts, val in sel.items():
                rows.append({"h": hz, "date": ts, "signed": sign * float(val),
                             "excess": sign * (float(val) - base_mu)})

    ev = pd.DataFrame(rows)
    total_bars = sum(b["n"] for b in base_rows if b["h"] == HORIZONS[0])
    out = {"name": name, "direction": direction, "conditions": conditions,
           "symbols_used": used, "coverage_pct": None, "horizons": {}, "recent_12m": {},
           "methodology": "signed forward return minus each symbol's own drift, pooled "
                          "across the 4y x 35-symbol NSE universe; recent_12m = same "
                          "stat on the last 12 months only (stability check)."}

    if ev.empty:
        out["horizons"] = {str(hz): {"n": 0} for hz in HORIZONS}
        return out

    cutoff = ev["date"].max() - pd.Timedelta(days=365)
    for hz in HORIZONS:
        e = ev[ev["h"] == hz]
        if not len(e):
            continue
        mu, sd, n = e["excess"].mean(), e["excess"].std(ddof=1), len(e)
        out["horizons"][str(hz)] = {
            "n": int(n),
            "mean_excess_bps": round(float(mu) * 1e4, 1),
            "t_stat": round(float(mu / (sd / np.sqrt(n))), 2) if n > 2 and sd > 0 else None,
            "hit_rate": round(float((e["signed"] > 0).mean()), 3),
        }
        r = e[e["date"] >= cutoff]
        if len(r) >= 10:
            rmu, rsd, rn = r["excess"].mean(), r["excess"].std(ddof=1), len(r)
            out["recent_12m"][str(hz)] = {
                "n": int(rn),
                "mean_excess_bps": round(float(rmu) * 1e4, 1),
                "t_stat": round(float(rmu / (rsd / np.sqrt(rn))), 2) if rn > 2 and rsd > 0 else None,
            }
    if total_bars:
        n5 = out["horizons"].get(str(HORIZONS[0]), {}).get("n", 0)
        out["coverage_pct"] = round(n5 / total_bars * 100, 2)
    return out
