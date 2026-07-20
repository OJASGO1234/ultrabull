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
    # raw OHLC + ATR carried for trade-level backtests (stop/target hit checks)
    f["_open"], f["_high"], f["_low"], f["_close"] = px["Open"], h, l, c
    f["_atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
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


def backtest_strategy(
    conditions: list[dict],
    direction: str = "long",
    sl_atr: float = 1.5,
    tp_atr: float = 3.0,
    max_hold: int = 10,
    name: str = "strategy",
    cache_dir: str | None = None,
) -> dict:
    """Trade-level backtest across the 4y x 35-symbol NSE universe.

    On each bar where ALL entry conditions hold and no trade is open for that
    symbol, enter at the NEXT bar's open. Exit at whichever comes first: stop
    (sl_atr x ATR against entry), target (tp_atr x ATR in favour), or max_hold
    bars (exit at close). Intrabar, a bar that touches both stop and target is
    scored as the STOP (pessimistic, matching the project simulator). Returns
    win rate, avg win/loss, profit factor, expectancy, and a recent-12m split.

    Deliberately simpler than the production simulator (fixed 1-unit sizing, no
    slippage/fees, no gap modelling) — it answers "does this entry+exit rule
    have an edge?", not "what P&L would I have made". Costs are noted so the
    chat can hedge accordingly.
    """
    if not conditions or len(conditions) > 6:
        raise ValueError("Provide 1-6 entry conditions")
    for c in conditions:
        if c.get("feature") not in FEATURES:
            raise ValueError(f"Unknown feature {c.get('feature')!r}. Allowed: {FEATURES}")
        if c.get("op") not in OPS:
            raise ValueError(f"Unknown op {c.get('op')!r}. Allowed: {sorted(OPS)}")
        float(c["value"])
    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")
    sl_atr, tp_atr = float(sl_atr), float(tp_atr)
    max_hold = int(max_hold)
    if sl_atr <= 0 or tp_atr <= 0:
        raise ValueError("sl_atr and tp_atr must be positive")
    if not (1 <= max_hold <= 60):
        raise ValueError("max_hold must be 1-60 bars")
    long = direction == "long"

    trades = []
    used = 0
    for sym in UNIVERSE:
        f = _frame(sym, cache_dir)
        if f is None:
            continue
        used += 1
        mask = pd.Series(True, index=f.index)
        for c in conditions:
            col, v, op = f[c["feature"]], float(c["value"]), c["op"]
            m = (col < v) if op == "<" else (col <= v) if op == "<=" else \
                (col > v) if op == ">" else (col >= v) if op == ">=" else (col == v)
            mask &= m.fillna(False)

        o = f["_open"].to_numpy(); hi = f["_high"].to_numpy()
        lo = f["_low"].to_numpy(); cl = f["_close"].to_numpy()
        atr = f["_atr"].to_numpy(); sig = mask.to_numpy()
        dates = f.index
        n = len(f)
        i = 0
        while i < n - 1:
            if not sig[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
                i += 1
                continue
            entry = o[i + 1]
            if not np.isfinite(entry):
                i += 1
                continue
            a = atr[i]
            stop = entry - sl_atr * a if long else entry + sl_atr * a
            targ = entry + tp_atr * a if long else entry - tp_atr * a
            exit_px, reason, held = None, None, 0
            for j in range(i + 1, min(i + 1 + max_hold, n)):
                held = j - i
                if long:
                    if lo[j] <= stop:  # stop checked first (pessimistic)
                        exit_px, reason = stop, "SL"; break
                    if hi[j] >= targ:
                        exit_px, reason = targ, "TP"; break
                else:
                    if hi[j] >= stop:
                        exit_px, reason = stop, "SL"; break
                    if lo[j] <= targ:
                        exit_px, reason = targ, "TP"; break
            if exit_px is None:  # timed out
                jj = min(i + max_hold, n - 1)
                exit_px, reason, held = cl[jj], "TIME", jj - i
            ret = (exit_px / entry - 1) if long else (entry / exit_px - 1)
            trades.append({"date": dates[i + 1], "ret": float(ret), "reason": reason, "held": held})
            i = i + 1 + held  # no overlapping trades per symbol

    out = {
        "name": name, "direction": direction,
        "conditions": conditions, "exit": {"sl_atr": sl_atr, "tp_atr": tp_atr, "max_hold": max_hold},
        "symbols_used": used,
        "methodology": "Enter next-open on signal, exit at stop / target / max_hold (stop wins "
                       "same-bar ties). 1-unit sizing, no fees or slippage — measures edge, not P&L. "
                       "recent_12m = same stats on the last 12 months (stability check).",
    }
    if not trades:
        out["trades"] = 0
        return out

    tdf = pd.DataFrame(trades)

    def _stats(d: pd.DataFrame) -> dict:
        wins = d[d["ret"] > 0]["ret"]; losses = d[d["ret"] <= 0]["ret"]
        gross_win = float(wins.sum()); gross_loss = float(-losses.sum())
        mu = float(d["ret"].mean()); sd = float(d["ret"].std(ddof=1)) if len(d) > 1 else 0.0
        return {
            "trades": int(len(d)),
            "win_rate": round(float((d["ret"] > 0).mean()), 3),
            "avg_return_pct": round(mu * 100, 3),
            "avg_win_pct": round(float(wins.mean()) * 100, 3) if len(wins) else None,
            "avg_loss_pct": round(float(losses.mean()) * 100, 3) if len(losses) else None,
            "expectancy_pct": round(mu * 100, 3),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else ("inf" if gross_win > 0 else None),
            "t_stat": round(mu / (sd / np.sqrt(len(d))), 2) if len(d) > 2 and sd > 0 else None,
            "median_hold_bars": int(d["held"].median()),
            "exit_mix": {k: int(v) for k, v in d["reason"].value_counts().items()},
        }

    out["overall"] = _stats(tdf)
    cutoff = tdf["date"].max() - pd.Timedelta(days=365)
    recent = tdf[tdf["date"] >= cutoff]
    out["recent_12m"] = _stats(recent) if len(recent) >= 10 else {"trades": int(len(recent)),
                                                                   "note": "too few recent trades to judge stability"}
    return out
