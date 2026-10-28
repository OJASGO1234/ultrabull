# core/audit_gates.py
"""
Diagnostic / audit of the gate stack in rules/gates.py::apply_gates.

PURPOSE: measure what each individual gate condition actually does to the
detection pool and to forward returns — NOT to tune thresholds. This script
does not modify rules/gates.py, evaluate_gates.py, or event_study.py; it
reproduces the gate *conditions* independently (duplicated math, same
formulas, same GATE_CFG thresholds imported unmodified from evaluate_gates.py)
so each rejection can be attributed to a specific named condition instead of
inferred after the fact from the fully-gated pool.

ATTRIBUTION METHOD (read before interpreting the output):
apply_gates() ANDs several sub-conditions together within each pattern
family (e.g. the breakout gate is trend & vol_surge & close>=MA20 as one
boolean expression). There is no way to assign a single rejected signal to
"trend" vs "volume" when both are false at once — any such assignment would
be an arbitrary sequencing choice, not something apply_gates() itself does.
So every gate below is measured MARGINALLY: for the sub-population it
actually applies to (e.g. the trend sub-condition of the breakout gate is
only evaluated on Breakout20/Breakout55/Breakdown20/Breakdown55-tagged
rows), independent of whether other sub-conditions in the same family also
reject that row. This means the per-condition "n_rejected" values within a
family do NOT sum to the family's combined AND rejection count — the
overlap is the point, not an error. A separate combined-family row is
reported alongside the standalone sub-conditions.

Gates that cannot be cleanly isolated at all under the CURRENT GATE_CFG
values are flagged as inactive/no-op rather than measured as if they did
something:
  - ADX gates: cfg["gates"]["adx"] = {"breakout_min": 0.0, "meanrev_max": 0.0}
    in GATE_CFG. gates.py only applies each ADX gate `if adx_bo_min > 0` /
    `if adx_mr_max > 0`, so at 0.0 both are permanently no-ops.
  - Sector regime: GATE_CFG["gates"]["market"]["require_sector_regime"] =
    False, so this branch never executes under the base config.
  - Breadth floor / distribution-day ceiling: both use threshold 0.0 in the
    base GATE_CFG (no-op, mirrors ADX's guard). evaluate_gates.py only
    exercises them via separate cfg_breadth/cfg_dd copies with breadth_min
    = 0.45 / dist_days_max = 5.0 respectively — this script reports those
    as explicit VARIANTS, not part of the base gate stack, exactly as
    evaluate_gates.py treats them.

Run:  python audit_gates.py <cache_dir>   (from the core/ directory)
Output: core/audit_gates_results.csv (+ printed summary)
"""
from __future__ import annotations
import copy
import os
import numpy as np
import pandas as pd

from event_study import UNIVERSE, prepare_symbol, compute_event_rows
from patterns.postprocess import enrich_detections, direction_of
from market_context import build_market_context
from stock_explorer import load_events
from evaluate_gates import GATE_CFG   # imported, not duplicated — exact thresholds

HORIZONS_REPORTED = (5, 10)


def _get(cfg, path, default=None):
    cur = cfg
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _load_events_cached(symbol: str, df_ohlcv, cache_dir: str | None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"events_{symbol.replace('&','_')}.pkl")
        if os.path.exists(path):
            return pd.read_pickle(path)
    ev = load_events(symbol, df_ohlcv.rename(columns=str.lower))
    if cache_dir:
        ev.to_pickle(path)
    return ev


def _bar_idx(det: pd.DataFrame, df_ohlcv: pd.DataFrame) -> np.ndarray:
    """Independent reproduction of gates.py::_ensure_bar_index's datetime->positional
    mapping (detections here always carry a 'date' column, no integer bar column)."""
    dt = pd.to_datetime(det["date"])
    lookup = pd.Series(np.arange(len(df_ohlcv), dtype="int64"),
                        index=pd.to_datetime(df_ohlcv.index))
    mapped = lookup.reindex(dt).to_numpy()
    return np.clip(np.nan_to_num(mapped, nan=0).astype("int64"), 0, len(df_ohlcv) - 1)


def _symbol_frame(sym: str, years: float, cache_dir: str | None) -> pd.DataFrame | None:
    """Build one row per (raw) detection for `sym`, with every gate condition
    evaluated as its own boolean column, plus 5-/10-bar signed & excess
    forward returns via the exact event_study methodology."""
    prep = prepare_symbol(sym, years=years, cache_dir=cache_dir)
    if prep is None:
        return None
    df_ohlcv, ind_df, dets = prep
    det = enrich_detections(dets, df_ohlcv, ind_df, sym)
    if det is None or det.empty:
        return None

    # ---- forward returns (identical methodology to event_study.compute_event_rows) ----
    ev = compute_event_rows(det, df_ohlcv, sym)
    ret_cols = {}
    for h in HORIZONS_REPORTED:
        ret_cols[h] = ({}, {})  # (signed_ret_by(date,pattern), excess_by(date,pattern))
    if ev is not None and len(ev):
        for h in HORIZONS_REPORTED:
            sub = ev[ev["horizon"] == h]
            ret_cols[h] = (
                dict(zip(zip(sub["date"], sub["pattern"]), sub["signed_ret"])),
                dict(zip(zip(sub["date"], sub["pattern"]), sub["excess"])),
            )
    for h in HORIZONS_REPORTED:
        signed_map, excess_map = ret_cols[h]
        key = list(zip(det["date"], det["pattern"]))
        det[f"ret{h}_signed"] = [signed_map.get(k, np.nan) for k in key]
        det[f"ret{h}_excess"] = [excess_map.get(k, np.nan) for k in key]

    # ---- indicator / price arrays, indexed by bar position ----
    idx = _bar_idx(det, df_ohlcv)
    adx = ind_df["ADX"].to_numpy() if "ADX" in ind_df.columns else None
    rsi = ind_df["RSI"].to_numpy() if "RSI" in ind_df.columns else None
    ma20 = ind_df["MA20"].to_numpy() if "MA20" in ind_df.columns else None
    ma50 = ind_df["MA50"].to_numpy() if "MA50" in ind_df.columns else None
    bb_u = ind_df["BB_upper"].to_numpy() if "BB_upper" in ind_df.columns else None
    bb_l = ind_df["BB_lower"].to_numpy() if "BB_lower" in ind_df.columns else None
    cols = {str(c).lower(): c for c in df_ohlcv.columns}
    close = df_ohlcv[cols["close"]].to_numpy() if "close" in cols else None
    vol = df_ohlcv[cols["volume"]].to_numpy() if "volume" in cols else None
    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().to_numpy() if vol is not None else None
    volx20 = (vol / (vol_ma20 + 1e-12)) if vol is not None else None

    rsi_low, rsi_high = (_get(GATE_CFG, "momentum.rsi_levels", [40, 60]) or [40, 60])[:2]
    vol_k = float(_get(GATE_CFG, "breakout.vol_k", 1.5) or 1.5)
    bb_eps = float(_get(GATE_CFG, "gates.bb_touch_eps", 0.0) or 0.0)
    ma_tol = float(_get(GATE_CFG, "gates.pullback_ma_tol", 0.01) or 0.01)

    def finite(a, i):
        return np.isfinite(a[i]) if a is not None else np.zeros(len(i), dtype=bool)

    tags = det["tag"].astype(str).to_numpy() if "tag" in det.columns else det["pattern"].astype(str).to_numpy()
    is_brk = np.isin(tags, ["Breakout20", "Breakout55"])
    is_brd = np.isin(tags, ["Breakdown20", "Breakdown55"])
    is_buydip = tags == "BB_BuyDip"
    is_sellfade = tags == "BB_SellFade"
    is_pb = tags == "PullbackToMA20"
    is_hammer = tags == "Hammer"
    is_bulleng = tags == "BullishEngulfing"
    is_beareng = tags == "BearishEngulfing"
    is_mr = is_buydip | is_sellfade | is_pb

    det["_applicable_breakout"] = is_brk | is_brd
    det["_cond_trend"] = np.where(is_brk, (ma20 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] > ma50[idx]) if ma20 is not None else True,
                          np.where(is_brd, (ma20 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] < ma50[idx]) if ma20 is not None else True, np.nan))
    det["_cond_vol_surge"] = np.where(is_brk | is_brd,
                                       (volx20 is not None) & finite(volx20, idx) & (volx20[idx] >= vol_k) if volx20 is not None else True,
                                       np.nan)
    det["_cond_ma20_pos"] = np.where(is_brk, (close is not None) & (ma20 is not None) & finite(close, idx) & finite(ma20, idx) & (close[idx] >= ma20[idx]) if close is not None else True,
                             np.where(is_brd, (close is not None) & (ma20 is not None) & finite(close, idx) & finite(ma20, idx) & (close[idx] <= ma20[idx]) if close is not None else True, np.nan))
    det["_cond_breakout_family_all"] = np.where(
        is_brk | is_brd,
        det["_cond_trend"].astype(bool) & det["_cond_vol_surge"].astype(bool) & det["_cond_ma20_pos"].astype(bool),
        np.nan)

    det["_applicable_meanrev"] = is_buydip | is_sellfade
    bb_touch = np.where(is_buydip, (close is not None) & (bb_l is not None) & finite(close, idx) & finite(bb_l, idx) & (close[idx] <= bb_l[idx] * (1 + bb_eps)) if close is not None else True,
                np.where(is_sellfade, (close is not None) & (bb_u is not None) & finite(close, idx) & finite(bb_u, idx) & (close[idx] >= bb_u[idx] * (1 - bb_eps)) if close is not None else True, np.nan))
    rsi_extreme = np.where(is_buydip, (rsi is not None) & finite(rsi, idx) & (rsi[idx] <= rsi_low) if rsi is not None else True,
                   np.where(is_sellfade, (rsi is not None) & finite(rsi, idx) & (rsi[idx] >= rsi_high) if rsi is not None else True, np.nan))
    mr_trend = np.where(is_buydip, (ma20 is not None) & (ma50 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] > ma50[idx]) if ma20 is not None else True,
                np.where(is_sellfade, (ma20 is not None) & (ma50 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] < ma50[idx]) if ma20 is not None else True, np.nan))
    det["_cond_bb_touch"] = bb_touch
    det["_cond_rsi_extreme"] = rsi_extreme
    det["_cond_meanrev_trend"] = mr_trend
    det["_cond_meanrev_family_all"] = np.where(
        is_buydip | is_sellfade,
        pd.Series(bb_touch).astype(bool) & pd.Series(rsi_extreme).astype(bool) & pd.Series(mr_trend).astype(bool),
        np.nan)

    det["_applicable_pullback"] = is_pb
    pb_trend = (ma20 is not None) & (ma50 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] > ma50[idx]) if ma20 is not None else np.ones(len(det), dtype=bool)
    pb_ma_tol = (close is not None) & (ma20 is not None) & finite(close, idx) & finite(ma20, idx) & (close[idx] <= ma20[idx] * (1 + ma_tol)) if close is not None else np.ones(len(det), dtype=bool)
    det["_cond_pullback_trend"] = np.where(is_pb, pb_trend, np.nan)
    det["_cond_pullback_ma_tol"] = np.where(is_pb, pb_ma_tol, np.nan)
    det["_cond_pullback_family_all"] = np.where(is_pb, pb_trend.astype(bool) & pb_ma_tol.astype(bool), np.nan)

    det["_applicable_candle"] = is_hammer | is_bulleng | is_beareng
    up_trend = (ma20 is not None) & (ma50 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] > ma50[idx]) if ma20 is not None else np.ones(len(det), dtype=bool)
    dn_trend = (ma20 is not None) & (ma50 is not None) & finite(ma20, idx) & finite(ma50, idx) & (ma20[idx] < ma50[idx]) if ma20 is not None else np.ones(len(det), dtype=bool)
    det["_cond_candle_trend"] = np.where(is_hammer | is_bulleng, up_trend, np.where(is_beareng, dn_trend, np.nan))

    # ADX gates — informational only; both thresholds are 0.0 in GATE_CFG => no-op.
    adx_bo_min = float(_get(GATE_CFG, "gates.adx.breakout_min", 0.0) or 0.0)
    adx_mr_max = float(_get(GATE_CFG, "gates.adx.meanrev_max", 0.0) or 0.0)
    # gates.py guards each ADX gate with `if adx_bo_min > 0` / `if adx_mr_max > 0` —
    # at 0.0 (current GATE_CFG) the gate never runs, i.e. it is a true pass-through,
    # NOT "adx >= 0" / "adx <= 0" evaluated literally. Reproduce the guard exactly.
    if adx is not None and adx_bo_min > 0:
        det["_cond_adx_breakout_min"] = np.where(is_brk | is_brd, (~np.isfinite(adx[idx])) | (adx[idx] >= adx_bo_min), np.nan)
    else:
        det["_cond_adx_breakout_min"] = np.where(is_brk | is_brd, True, np.nan)
    if adx is not None and adx_mr_max > 0:
        det["_cond_adx_meanrev_max"] = np.where(is_mr, (~np.isfinite(adx[idx])) | (adx[idx] <= adx_mr_max), np.nan)
    else:
        det["_cond_adx_meanrev_max"] = np.where(is_mr, True, np.nan)

    # Liquidity floor — applies to all rows.
    min_turnover = float(_get(GATE_CFG, "gates.liquidity.min_turnover_20d", 0.0) or 0.0)
    if min_turnover > 0 and close is not None and vol is not None:
        turnover20 = pd.Series(close * vol).rolling(20, min_periods=5).median().to_numpy()
        t_i = turnover20[idx]
        det["_cond_liquidity"] = (~np.isfinite(t_i)) | (t_i >= min_turnover)
    else:
        det["_cond_liquidity"] = True

    # Qualification pre-filter (add_qualification), and pattern-disable — universe-wide.
    det["_cond_qualified"] = det["qualified"].astype(bool) if "qualified" in det.columns else True
    disabled = set(_get(GATE_CFG, "disable_patterns", []) or [])
    tag_col = "tag" if "tag" in det.columns else "pattern"
    det["_cond_not_disabled"] = ~det[tag_col].isin(disabled)

    # ---- market/sector/RS/breadth/dist-days (needs ctx) ----
    ctx = build_market_context(sym, df_ohlcv["Close"], years=years, cache_dir=cache_dir)
    mgate = _get(GATE_CFG, "gates.market", {}) or {}
    direction = det["direction"].astype(str).to_numpy()
    bull = direction == "bullish"
    bear = direction == "bearish"
    exempt = set(mgate.get("exempt_patterns", ["Hammer"]) or [])
    nonexempt = ~det[tag_col].astype(str).isin(exempt).to_numpy()
    bull_ne = bull & nonexempt
    bear_ne = bear & nonexempt
    det["_applicable_market_regime"] = bull_ne | bear_ne

    mkt_up = ctx["mkt_up"].to_numpy() if "mkt_up" in ctx.columns else None
    mkt_dn = ctx["mkt_dn"].to_numpy() if "mkt_dn" in ctx.columns else None
    if bool(mgate.get("require_market_regime", True)) and mkt_up is not None and mkt_dn is not None:
        cond = np.where(bull_ne, mkt_up[idx].astype(bool), np.where(bear_ne, mkt_dn[idx].astype(bool), np.nan))
        det["_cond_market_regime"] = cond
    else:
        det["_cond_market_regime"] = np.nan

    # Sector regime — inactive under GATE_CFG (require_sector_regime=False); reported as such.
    det["_applicable_sector_regime"] = bull_ne | bear_ne
    det["_cond_sector_regime"] = np.nan  # not evaluated: gate never runs under base cfg

    rs_col = str(mgate.get("rs_col", "rs_20"))
    tags_s = det[tag_col].astype(str)
    is_bo = tags_s.str.startswith("Breakout").to_numpy()
    is_bd = tags_s.str.startswith("Breakdown").to_numpy()
    det["_applicable_rel_strength"] = is_bo | is_bd
    if rs_col in ctx.columns:
        rs_i = ctx[rs_col].to_numpy()[idx]
        rs_min = float(mgate.get("breakout_rs_min", 0.0))
        rs_max = float(mgate.get("breakdown_rs_max", 0.0))
        ok_bo = (~np.isfinite(rs_i)) | (rs_i >= rs_min)
        ok_bd = (~np.isfinite(rs_i)) | (rs_i <= rs_max)
        det["_cond_rel_strength"] = np.where(is_bo, ok_bo, np.where(is_bd, ok_bd, np.nan))
    else:
        det["_cond_rel_strength"] = np.nan

    # Breadth / distribution-days — inactive in base GATE_CFG (threshold 0.0); computed here
    # as VARIANTS (breadth_min=0.45, dist_days_max=5.0) mirroring evaluate_gates.py's
    # cfg_breadth / cfg_dd, applicable to bullish (non-exempt) rows only, per gates.py.
    det["_applicable_breadth_variant"] = bull_ne
    det["_applicable_distday_variant"] = bull_ne
    if "breadth" in ctx.columns:
        b = ctx["breadth"].to_numpy()[idx]
        det["_cond_breadth_variant045"] = np.where(bull_ne, (~np.isfinite(b)) | (b >= 0.45), np.nan)
    else:
        det["_cond_breadth_variant045"] = np.nan
    if "dist_days" in ctx.columns:
        d = ctx["dist_days"].to_numpy()[idx]
        det["_cond_distday_variant5"] = np.where(bull_ne, (~np.isfinite(d)) | (d < 5.0), np.nan)
    else:
        det["_cond_distday_variant5"] = np.nan

    # ---- earnings blackout ----
    egate = _get(GATE_CFG, "gates.events", {}) or {}
    events_ok = None
    events_available = False
    try:
        sym_events = _load_events_cached(sym, df_ohlcv, None)
        if sym_events is not None and len(sym_events) and bool(egate.get("enabled", True)):
            events_available = True
            before = int(egate.get("earnings_blackout_before", 3))
            after = int(egate.get("earnings_blackout_after", 1))
            n = len(df_ohlcv)
            blackout = np.zeros(n, dtype=bool)
            etype = sym_events["type"].astype(str).str.upper()
            edates = pd.to_datetime(sym_events.loc[etype == "EARNINGS", "date"])
            ohlc_idx_dt = pd.to_datetime(df_ohlcv.index)
            for d in edates:
                p = int(ohlc_idx_dt.searchsorted(d))
                lo, hi = max(0, p - before), min(n, p + after + 1)
                blackout[lo:hi] = True
            events_ok = ~blackout[idx]
    except Exception:
        events_ok = None
    det["_events_data_available"] = events_available
    det["_cond_earnings_blackout"] = events_ok if events_ok is not None else np.nan
    det["_applicable_earnings_blackout"] = True

    det["symbol"] = sym
    return det


GATE_SPECS = [
    # (report_label, stage, applicable_col, cond_col)
    ("Disabled patterns (RSI_60_Down, RSI_40_Up, Doji, MACD_Cross_Zeroish_Up)",
     "pattern_disable", None, "_cond_not_disabled"),
    ("Qualification pre-filter (alignment / counter-trend-reversal + vol confirm)",
     "pre_filter", None, "_cond_qualified"),
    ("Breakout/Breakdown: trend (MA20 vs MA50)", "conditional_gate", "_applicable_breakout", "_cond_trend"),
    ("Breakout/Breakdown: volume surge (>=1.5x 20d avg)", "conditional_gate", "_applicable_breakout", "_cond_vol_surge"),
    ("Breakout/Breakdown: close vs MA20 position", "conditional_gate", "_applicable_breakout", "_cond_ma20_pos"),
    ("Breakout/Breakdown: COMBINED (trend & volume & MA20, as ANDed in gates.py)",
     "conditional_gate_combined", "_applicable_breakout", "_cond_breakout_family_all"),
    ("Mean-reversion (BB_BuyDip/BB_SellFade): Bollinger-band touch", "conditional_gate", "_applicable_meanrev", "_cond_bb_touch"),
    ("Mean-reversion: RSI extreme (<=40 / >=60)", "conditional_gate", "_applicable_meanrev", "_cond_rsi_extreme"),
    ("Mean-reversion: trend bias", "conditional_gate", "_applicable_meanrev", "_cond_meanrev_trend"),
    ("Mean-reversion: COMBINED (touch & RSI & trend, as ANDed in gates.py)",
     "conditional_gate_combined", "_applicable_meanrev", "_cond_meanrev_family_all"),
    ("Pullback: trend (uptrend)", "conditional_gate", "_applicable_pullback", "_cond_pullback_trend"),
    ("Pullback: MA20 tolerance (within 1%)", "conditional_gate", "_applicable_pullback", "_cond_pullback_ma_tol"),
    ("Pullback: COMBINED (trend & MA tolerance)", "conditional_gate_combined", "_applicable_pullback", "_cond_pullback_family_all"),
    ("Candles (Hammer/BullishEngulfing/BearishEngulfing): trend bias", "conditional_gate", "_applicable_candle", "_cond_candle_trend"),
    ("ADX: breakout_min (INACTIVE, threshold=0.0 in GATE_CFG)", "inactive", "_applicable_breakout", "_cond_adx_breakout_min"),
    ("ADX: meanrev_max (INACTIVE, threshold=0.0 in GATE_CFG)", "inactive", "_applicable_meanrev", "_cond_adx_meanrev_max"),
    ("Liquidity floor (median 20d turnover >= 2e7)", "conditional_gate", None, "_cond_liquidity"),
    ("Market regime (index MA20 vs MA50, direction-matched; Hammer exempt)", "conditional_gate", "_applicable_market_regime", "_cond_market_regime"),
    ("Sector regime (INACTIVE: require_sector_regime=False in GATE_CFG)", "inactive", "_applicable_sector_regime", "_cond_sector_regime"),
    ("Relative strength vs sector/NIFTY (breakout_rs_min=0.0, breakdown_rs_max=0.0)", "conditional_gate", "_applicable_rel_strength", "_cond_rel_strength"),
    ("VARIANT (not in base GATE_CFG): breadth >= 0.45", "variant_not_in_base_stack", "_applicable_breadth_variant", "_cond_breadth_variant045"),
    ("VARIANT (not in base GATE_CFG): distribution days < 5", "variant_not_in_base_stack", "_applicable_distday_variant", "_cond_distday_variant5"),
    ("Earnings blackout (-3/+1 trading days around EARNINGS events)", "conditional_gate", "_applicable_earnings_blackout", "_cond_earnings_blackout"),
]


def _stats_for(sub: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS_REPORTED:
        out[f"ret{h}_signed_mean"] = sub[f"ret{h}_signed"].mean() if len(sub) else np.nan
        out[f"ret{h}_excess_mean"] = sub[f"ret{h}_excess"].mean() if len(sub) else np.nan
    return out


def build_report(all_det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, stage, applic_col, cond_col in GATE_SPECS:
        if applic_col is not None:
            pool = all_det[all_det[applic_col].astype(object).fillna(False).astype(bool) if applic_col else True]
        else:
            pool = all_det

        if cond_col not in pool.columns:
            continue
        evaluated = pool[pool[cond_col].notna()]
        n_before = len(evaluated)
        if n_before == 0:
            rows.append({"gate": label, "stage": stage, "n_before": 0, "n_rejected": 0,
                         "n_surviving": 0, "pct_rejected": np.nan,
                         **{f"ret{h}_{k}_survive": np.nan for h in HORIZONS_REPORTED for k in ("signed_mean", "excess_mean")},
                         **{f"ret{h}_{k}_reject": np.nan for h in HORIZONS_REPORTED for k in ("signed_mean", "excess_mean")}})
            continue

        cond = evaluated[cond_col].astype(bool)
        survive = evaluated[cond]
        reject = evaluated[~cond]
        n_rej = len(reject)

        row = {
            "gate": label,
            "stage": stage,
            "n_before": n_before,
            "n_rejected": n_rej,
            "n_surviving": n_before - n_rej,
            "pct_rejected": round(100.0 * n_rej / n_before, 2),
        }
        for k, v in _stats_for(survive).items():
            row[f"{k}_survive_bps" if "excess" in k or "signed" in k else k] = v * 1e4 if pd.notna(v) else v
        for k, v in _stats_for(reject).items():
            row[f"{k}_reject_bps" if "excess" in k or "signed" in k else k] = v * 1e4 if pd.notna(v) else v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else None
    years = 4
    frames = []
    events_avail_count = 0
    events_total = 0
    for sym in UNIVERSE:
        try:
            f = _symbol_frame(sym, years=years, cache_dir=cache)
        except Exception as e:
            print(f"  {sym}: FAILED ({e})")
            continue
        if f is None or f.empty:
            continue
        events_total += 1
        if f["_events_data_available"].any():
            events_avail_count += 1
        frames.append(f)
        print(f"  {sym}: {len(f)} raw detections")

    if not frames:
        print("No data collected — aborting.")
        return
    all_det = pd.concat(frames, ignore_index=True)

    # directional rows only (neutral-direction patterns carry no forward-return
    # comparison under event_study's methodology and are excluded there too)
    all_det = all_det[all_det["direction"].isin(["bullish", "bearish"])].copy()

    n_raw = len(all_det)
    n_disabled_rows = int((~all_det["_cond_not_disabled"]).sum())
    disabled_patterns = sorted(set(_get(GATE_CFG, "disable_patterns", []) or []))

    report = build_report(all_det)
    report.to_csv("audit_gates_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)

    print("\n" + "=" * 100)
    print(f"Raw enriched detection pool (directional patterns only): {n_raw} signals across {len(frames)} symbols")
    print(f"Disabled patterns under GATE_CFG: {disabled_patterns}  -> {n_disabled_rows} rows carry a disabled tag ({100*n_disabled_rows/n_raw:.1f}%)")
    print(f"Earnings-events data available for {events_avail_count}/{events_total} symbols")
    print("=" * 100)

    cols_show = ["gate", "stage", "n_before", "n_rejected", "n_surviving", "pct_rejected",
                 "ret5_excess_mean_survive_bps", "ret5_excess_mean_reject_bps",
                 "ret10_excess_mean_survive_bps", "ret10_excess_mean_reject_bps"]
    cols_show = [c for c in cols_show if c in report.columns]
    print(report[cols_show].round(2).to_string(index=False))

    print("\nSaved: audit_gates_results.csv")

    print("""
LIMITATIONS (see module docstring for the full attribution-method caveat):
  - Forward-return windows overlap across nearby detections of the same
    symbol/pattern (a signal on day t and t+2 both look out to day t+5,
    t+7) — reported means are NOT independent observations; treat t-stats
    from event_study/evaluate_gates as directional, not exact.
  - "Excess" = signed forward return minus the symbol's own unconditional
    mean forward return over the same horizon and window (event_study's
    baseline), i.e. Excess is relative to that stock's own drift, not to
    the market or to zero.
  - No execution costs, slippage, or signal-to-fill delay are modeled;
    all returns use the same-day close-to-close path event_study uses.
  - Earnings-blackout coverage depends on yfinance earnings-date
    availability per symbol; see the "data available" count above — gates
    for symbols without events data are permissive by construction
    (nothing gets blocked), which silently overstates that gate's
    coverage rate if taken as "N% of the universe".
  - Per-condition "n_rejected" values inside an ANDed family
    (breakout/breakdown, mean-reversion, pullback) are MARGINAL and
    overlap with each other; do not sum them and compare to the combined
    row's n_rejected.
  - ADX gates and sector regime are inactive (no-op) under the CURRENT
    GATE_CFG values and are reported for completeness only, not as
    evidence of an effect.
""")


if __name__ == "__main__":
    main()
