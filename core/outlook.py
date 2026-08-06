# core/outlook.py
"""
Directional outlook the honest way: look up today's technical state in the
empirical base-rate table (build_baserates.py) and report P(up), typical move
and a bias verdict per horizon — plus the scenario frame pros actually use:
nearest support/resistance as trigger/invalidation levels.

A verdict is only "up"/"down" when the base rate clears a threshold; otherwise
it's "sideways / no edge". Day-ahead direction is ~coin-flip in every state —
the table says so, and we show that rather than hide it.
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd

_CORE_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_baserates.json")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")
_DATA_TABLE = os.path.join(_DATA_DIR, "state_baserates.json")      # rebuilt (current data)
_CAL_PATH = os.path.join(_DATA_DIR, "calibration.json")            # fitted shrink factors
_table_cache: dict | None = None
_table_mtime: float = 0.0
_cal_cache: dict | None = None
_cal_mtime: float = 0.0

UP_THRESHOLD = 0.55
DOWN_THRESHOLD = 0.45
MIN_N = 150  # below this the state sample is too thin; fall back to _all


def _load_table() -> dict:
    """Prefer the rebuilt table on the volume (current data + calibrated);
    fall back to the committed default. Reload if the file changed on disk."""
    global _table_cache, _table_mtime
    path = _DATA_TABLE if os.path.exists(_DATA_TABLE) else _CORE_TABLE
    m = os.path.getmtime(path)
    if _table_cache is None or _table_cache.get("_path") != path or m != _table_mtime:
        with open(path) as f:
            t = json.load(f)
        t["_path"] = path
        _table_cache, _table_mtime = t, m
    return _table_cache


def _load_calibration() -> dict:
    """Per-horizon shrink factors {name: {k: ...}}; empty if never fit."""
    global _cal_cache, _cal_mtime
    if not os.path.exists(_CAL_PATH):
        return {}
    m = os.path.getmtime(_CAL_PATH)
    if _cal_cache is None or m != _cal_mtime:
        try:
            with open(_CAL_PATH) as f:
                _cal_cache = json.load(f)
            _cal_mtime = m
        except Exception:
            return {}
    return _cal_cache or {}


def _calibrate(p_up: float | None, name: str) -> tuple[float | None, float | None]:
    """Return (calibrated_p, k). Shrinks toward 0.5 by the fitted factor so the
    displayed probability matches what actually happens out-of-sample."""
    if p_up is None:
        return None, None
    k = ((_load_calibration().get("horizons") or {}).get(name) or {}).get("k")
    if k is None:
        return p_up, None
    return round(min(1.0, max(0.0, 0.5 + float(k) * (p_up - 0.5))), 4), float(k)


def current_state(ind_df: pd.DataFrame, ctx: pd.DataFrame | None,
                  use_market_context: bool = True) -> dict:
    """State vector for the latest bar, mirroring build_baserates.state_series.

    With `use_market_context=False` the market dimension is MARGINALIZED, not
    dropped: the key becomes `mkt=any|…`, which build_baserates/calibrate emit
    by pooling the same stock state across both NIFTY regimes. Dropping ctx
    instead would fall through to the `mkt="up"` default below and quote
    bull-market base rates — cheerier numbers than any evidence supports.
    """
    ma20 = float(ind_df["MA20"].iloc[-1])
    ma50 = float(ind_df["MA50"].iloc[-1])
    trend = "up" if ma20 > ma50 else "dn"
    rsi_v = float(ind_df["RSI"].iloc[-1]) if np.isfinite(ind_df["RSI"].iloc[-1]) else 50.0
    band = "low" if rsi_v <= 40 else ("mid" if rsi_v <= 60 else "high")
    if ctx is not None and len(ctx):
        last = ctx.iloc[-1]
        mkt = "up" if bool(last.get("mkt_up")) else "dn"
        rs_v = last.get("rs_20")
        rs = "+" if (rs_v is not None and np.isfinite(rs_v) and rs_v >= 0) else "-"
    else:
        mkt, rs = "up", "+"
    if not use_market_context:
        mkt = "any"   # relative strength stays: it is stock-vs-sector, not tape
    return {
        "key": f"mkt={mkt}|trend={trend}|rs={rs}|rsi={band}",
        "market": mkt, "trend": trend, "rs": rs, "rsi_band": band,
        "rsi": round(rsi_v, 1),
    }


def _verdict(p_up: float | None) -> str:
    if p_up is None:
        return "unknown"
    if p_up >= UP_THRESHOLD:
        return "up"
    if p_up <= DOWN_THRESHOLD:
        return "down"
    return "sideways"


def build_outlook(
    ind_df: pd.DataFrame,
    ctx: pd.DataFrame | None,
    levels: list[dict] | None,
    close: float,
    use_market_context: bool = True,
) -> dict:
    state = current_state(ind_df, ctx, use_market_context=use_market_context)
    table = _load_table()
    row = table.get(state["key"]) or {}
    fallback = table.get("_all", {})

    horizons = {}
    for name in ("1d", "1w", "1m", "3m"):
        st = row.get(name) or {}
        used_fallback = not st or (st.get("n") or 0) < MIN_N
        if used_fallback:
            st = fallback.get(name) or {}
        raw_p = st.get("p_up")
        p_up, k = _calibrate(raw_p, name)  # honest, calibrated probability
        horizons[name] = {
            "p_up": p_up,
            "raw_p_up": raw_p,
            "calibrated": k is not None and k != 1.0,
            "median_move": st.get("median"),
            "typical_range": [st.get("q25"), st.get("q75")],
            "n": st.get("n"),
            "verdict": _verdict(p_up),
            "fallback": used_fallback,
        }

    # scenario frame: nearest support below / resistance above
    nearest_support = None
    nearest_resistance = None
    for lv in levels or []:
        if lv["kind"] == "support" and lv["price"] <= close:
            if nearest_support is None or lv["price"] > nearest_support["price"]:
                nearest_support = lv
        if lv["kind"] == "resistance" and lv["price"] >= close:
            if nearest_resistance is None or lv["price"] < nearest_resistance["price"]:
                nearest_resistance = lv

    return {
        "state": state,
        "horizons": horizons,
        "market_context_applied": bool(use_market_context),
        "nearest_support": None if nearest_support is None else nearest_support["price"],
        "nearest_resistance": None if nearest_resistance is None else nearest_resistance["price"],
        "note": (
            "How to read this: we found every day in the past 4 years (across 35 NSE stocks) "
            "that looked like today — same "
            + ("market trend, stock trend" if use_market_context else "stock trend")
            + ", sector standing and RSI "
            "zone — and counted what happened next. \"57% closed higher\" means 57 of every 100 "
            "similar days ended up — a lean, not a forecast. No situation in this data ever "
            "predicted a week ahead better than ~58 in 100, so let the support/resistance "
            "levels decide your entries, not this card."
            + ("" if use_market_context else
               " STOCK-ONLY VIEW: these base rates pool days from BOTH rising and falling "
               "NIFTY regimes, so they include the bad tape you asked to look past — they are "
               "not the numbers for a good market.")
        ),
    }
