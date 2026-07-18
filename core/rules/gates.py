from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

def _get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _ensure_bar_index(det: pd.DataFrame, df_ohlcv: pd.DataFrame, bar_idx_col: Optional[str] = None) -> pd.DataFrame:
    # Prefer any existing integer bar column
    for c in [bar_idx_col, "bar_idx", "bar", "i", "idx", "index"]:
        if c and c in det.columns and pd.api.types.is_integer_dtype(det[c]):
            if c != "bar_idx":
                det = det.copy()
                det["bar_idx"] = det[c].astype("int64")
            return det

    # Map datetime → positional
    dt = None
    if isinstance(det.index, pd.DatetimeIndex):
        dt = pd.to_datetime(det.index)
    else:
        for c in ["date", "Date", "dt", "timestamp", "Timestamp"]:
            if c in det.columns:
                dt = pd.to_datetime(det[c])
                break
    if dt is None:
        raise ValueError("apply_gates: need an integer bar index or a datetime to map")

    # Build lookup on ohlcv index
    lookup = pd.Series(np.arange(len(df_ohlcv), dtype="int64"),
                       index=pd.to_datetime(df_ohlcv.index))

    # Reindex to the detections' datetimes, then **realign to detections' index**
    mapped = lookup.reindex(pd.to_datetime(dt))
    mapped = pd.Series(mapped.values, index=det.index)   # <- key line

    keep = mapped.notna()
    det = det.loc[keep].copy()
    det["bar_idx"] = mapped.loc[keep].astype("int64").to_numpy()
    return det


def apply_gates(
    detections_enriched: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    bar_idx_col: Optional[str] = None,
    ctx: Optional[pd.DataFrame] = None,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Filter detections with simple, robust gates.
    - Uses MA20/MA50/RSI from ind_df when available
    - Uses volume/20d average from df_ohlcv
    - Honors cfg['disable_patterns'] if present
    - If `ctx` (market_context.build_market_context, aligned to df_ohlcv) is
      given: longs require market uptrend, shorts market downtrend, and
      breakouts/breakdowns additionally require positive/negative relative
      strength vs their sector. Tunable under cfg['gates']['market'].
    - If `events` (stock_explorer.load_events tidy frame) is given: suppress
      signals in a blackout window around earnings announcements — the gap
      risk of holding a fresh position through results dwarfs any pattern
      edge. Tunable under cfg['gates']['events'].
    Returns a filtered copy of detections_enriched.
    """
    if detections_enriched is None or len(detections_enriched) == 0:
        return detections_enriched

    det = _ensure_bar_index(detections_enriched, df_ohlcv, bar_idx_col).copy()

    # Base mask = qualified (if present)
    mask = det["qualified"].astype(bool) if "qualified" in det.columns else pd.Series(True, index=det.index)

    # Disabled patterns
    disabled = set(_get(cfg, "disable_patterns", []) or [])
    if "tag" in det.columns:
        mask &= ~det["tag"].isin(disabled)
    elif "pattern" in det.columns:
        mask &= ~det["pattern"].isin(disabled)

    # Index mapping → arrays
    idx = det["bar_idx"].astype(int).clip(0, len(df_ohlcv) - 1).to_numpy()

    # Indicators (optional)
    adx   = ind_df["ADX"].to_numpy()       if "ADX" in ind_df.columns else None
    rsi   = ind_df["RSI"].to_numpy()       if "RSI" in ind_df.columns else None
    ma20  = ind_df["MA20"].to_numpy()      if "MA20" in ind_df.columns else None
    ma50  = ind_df["MA50"].to_numpy()      if "MA50" in ind_df.columns else None
    bb_u  = ind_df["BB_upper"].to_numpy()  if "BB_upper" in ind_df.columns else None
    bb_l  = ind_df["BB_lower"].to_numpy()  if "BB_lower" in ind_df.columns else None
    # case-insensitive OHLCV lookup (df_ohlcv may be lower/title/upper case)
    _cols = {str(c).lower(): c for c in df_ohlcv.columns}
    close = df_ohlcv[_cols["close"]].to_numpy() if "close" in _cols else None
   # Volume 20d ratio
    vol = df_ohlcv[_cols["volume"]].to_numpy() if "volume" in _cols else None
    vol_ma20 = pd.Series(vol).rolling(20, min_periods=1).mean().to_numpy() if vol is not None else None
    volx20 = (vol / (vol_ma20 + 1e-12)) if vol is not None else None

    # Thresholds from cfg / Phase-1
    rsi_low, rsi_high = (_get(cfg, "momentum.rsi_levels", [40, 60]) or [40, 60])[:2]
    vol_k = float(_get(cfg, "breakout.vol_k", 1.5) or 1.5)
    bb_eps = float(_get(cfg, "gates.bb_touch_eps", 0.0) or 0.0)          # slack for band touch (e.g., 0.002 = 0.2%)
    ma_tol = float(_get(cfg, "gates.pullback_ma_tol", 0.01) or 0.01)      # 1% within MA20

    # Helpers (True if data missing)
    def _trend_up(i):
        if ma20 is None or ma50 is None: return np.ones_like(i, dtype=bool)
        return ma20[i] > ma50[i]
    def _trend_dn(i):
        if ma20 is None or ma50 is None: return np.ones_like(i, dtype=bool)
        return ma20[i] < ma50[i]
    def _rsi_le(i, thr):
        if rsi is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(rsi[i]) & (rsi[i] <= thr)
    def _rsi_ge(i, thr):
        if rsi is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(rsi[i]) & (rsi[i] >= thr)
    def _vol_surge(i, k):
        if volx20 is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(volx20[i]) & (volx20[i] >= k)
    def _close_ge_ma20(i):
        if close is None or ma20 is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(close[i]) & np.isfinite(ma20[i]) & (close[i] >= ma20[i])
    def _close_le_ma20(i):
        if close is None or ma20 is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(close[i]) & np.isfinite(ma20[i]) & (close[i] <= ma20[i])
    def _near_or_below_ma20(i, tol):
        if close is None or ma20 is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(close[i]) & np.isfinite(ma20[i]) & (close[i] <= ma20[i] * (1.0 + tol))
    def _touch_lower(i, eps):
        if close is None or bb_l is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(close[i]) & np.isfinite(bb_l[i]) & (close[i] <= bb_l[i] * (1.0 + eps))
    def _touch_upper(i, eps):
        if close is None or bb_u is None: return np.ones_like(i, dtype=bool)
        return np.isfinite(close[i]) & np.isfinite(bb_u[i]) & (close[i] >= bb_u[i] * (1.0 - eps))
    def _adx_ge(i, thr):
        if adx is None: return np.ones_like(i, dtype=bool)
        return ~np.isfinite(adx[i]) | (adx[i] >= thr)   # NaN → permissive
    def _adx_le(i, thr):
        if adx is None: return np.ones_like(i, dtype=bool)
        return ~np.isfinite(adx[i]) | (adx[i] <= thr)

    # Tag column (fallback to 'pattern')
    tag_col = "tag" if "tag" in det.columns else ("pattern" if "pattern" in det.columns else None)

    if tag_col:
        tags = det[tag_col].astype(str).to_numpy()

        gates = np.ones(len(det), dtype=bool)

        # Breakouts/downs: trend + vol surge + price position wrt MA20
        is_brk20 = tags == "Breakout20"
        is_brk55 = tags == "Breakout55"
        is_brd20 = tags == "Breakdown20"
        is_brd55 = tags == "Breakdown55"
        gates &= ~(is_brk20 | is_brk55 | is_brd20 | is_brd55) | (
            ((is_brk20 | is_brk55) & _trend_up(idx) & _vol_surge(idx, vol_k) & _close_ge_ma20(idx)) |
            ((is_brd20 | is_brd55) & _trend_dn(idx) & _vol_surge(idx, vol_k) & _close_le_ma20(idx))
        )

        # ADX: breakouts need a trending tape; mean-reversion wants a calm one
        adx_bo_min = float(_get(cfg, "gates.adx.breakout_min", 0.0) or 0.0)
        adx_mr_max = float(_get(cfg, "gates.adx.meanrev_max", 0.0) or 0.0)
        if adx_bo_min > 0:
            gates &= ~(is_brk20 | is_brk55 | is_brd20 | is_brd55) | _adx_ge(idx, adx_bo_min)
        if adx_mr_max > 0:
            is_mr = (tags == "BB_BuyDip") | (tags == "BB_SellFade") | (tags == "PullbackToMA20")
            gates &= ~is_mr | _adx_le(idx, adx_mr_max)

        # Mean reversion: require band touch + RSI extreme + trend bias
        is_buydip   = tags == "BB_BuyDip"
        is_sellfade = tags == "BB_SellFade"
        gates &= ~is_buydip   | (_touch_lower(idx, bb_eps) & _rsi_le(idx, rsi_low)  & _trend_up(idx))
        gates &= ~is_sellfade | (_touch_upper(idx, bb_eps) & _rsi_ge(idx, rsi_high) & _trend_dn(idx))


        # Pullback longs: uptrend AND close near/below MA20 (within tolerance)
        is_pb = tags == "PullbackToMA20"
        gates &= ~is_pb | (_trend_up(idx) & _near_or_below_ma20(idx, ma_tol))

        # Candles: trend-biased
        is_hammer   = tags == "Hammer"
        is_bulleng  = tags == "BullishEngulfing"
        is_beareng  = tags == "BearishEngulfing"
        gates &= ~is_hammer   | _trend_up(idx)
        gates &= ~is_bulleng  | _trend_up(idx)
        gates &= ~is_beareng  | _trend_dn(idx)

        mask &= gates

    # ---- Liquidity floor: median 20d turnover (Phase 5) ----
    min_turnover = float(_get(cfg, "gates.liquidity.min_turnover_20d", 0.0) or 0.0)
    if min_turnover > 0 and close is not None and vol is not None:
        turnover20 = pd.Series(close * vol).rolling(20, min_periods=5).median().to_numpy()
        t_i = turnover20[idx]
        mask &= ~np.isfinite(t_i) | (t_i >= min_turnover)

    # ---- Market/sector regime + relative-strength gates (Phase 3) ----
    mgate = _get(cfg, "gates.market", {}) or {}
    if ctx is not None and bool(mgate.get("enabled", True)) and "direction" in det.columns:
        bull = det["direction"].astype(str).eq("bullish").to_numpy()
        bear = det["direction"].astype(str).eq("bearish").to_numpy()

        # Counter-trend reversal patterns (e.g. Hammer) fire best exactly when
        # the regime is down — exempt them from the regime requirement
        exempt = set(mgate.get("exempt_patterns", ["Hammer"]) or [])
        if exempt and tag_col:
            nonexempt = ~det[tag_col].astype(str).isin(exempt).to_numpy()
            bull = bull & nonexempt
            bear = bear & nonexempt

        def _ctx_col(name):
            return ctx[name].to_numpy() if name in ctx.columns else None

        if bool(mgate.get("require_market_regime", True)):
            mkt_up, mkt_dn = _ctx_col("mkt_up"), _ctx_col("mkt_dn")
            if mkt_up is not None:
                mask &= ~bull | mkt_up[idx].astype(bool)
            if mkt_dn is not None:
                mask &= ~bear | mkt_dn[idx].astype(bool)

        if bool(mgate.get("require_sector_regime", False)):
            sect_up, sect_dn = _ctx_col("sect_up"), _ctx_col("sect_dn")
            if sect_up is not None:
                mask &= ~bull | sect_up[idx].astype(bool)
            if sect_dn is not None:
                mask &= ~bear | sect_dn[idx].astype(bool)

        # Market health: breadth floor and distribution-day ceiling for longs
        breadth_min = float(mgate.get("breadth_min", 0.0) or 0.0)
        if breadth_min > 0 and "breadth" in ctx.columns:
            b = ctx["breadth"].to_numpy()[idx]
            mask &= ~bull | ~np.isfinite(b) | (b >= breadth_min)
        dd_max = float(mgate.get("dist_days_max", 0.0) or 0.0)
        if dd_max > 0 and "dist_days" in ctx.columns:
            d = ctx["dist_days"].to_numpy()[idx]
            mask &= ~bull | ~np.isfinite(d) | (d < dd_max)

        # Relative strength: breakouts should be leaders, breakdowns laggards
        rs = _ctx_col(str(mgate.get("rs_col", "rs_20")))
        if rs is not None and tag_col:
            tags_s = det[tag_col].astype(str)
            is_bo = tags_s.str.startswith("Breakout").to_numpy()
            is_bd = tags_s.str.startswith("Breakdown").to_numpy()
            rs_i = rs[idx]
            rs_min = float(mgate.get("breakout_rs_min", 0.0))
            rs_max = float(mgate.get("breakdown_rs_max", 0.0))
            ok_bo = ~np.isfinite(rs_i) | (rs_i >= rs_min)   # NaN → permissive
            ok_bd = ~np.isfinite(rs_i) | (rs_i <= rs_max)
            mask &= ~is_bo | ok_bo
            mask &= ~is_bd | ok_bd

    # ---- Earnings blackout (Phase 4) ----
    egate = _get(cfg, "gates.events", {}) or {}
    if events is not None and len(events) and bool(egate.get("enabled", True)):
        before = int(egate.get("earnings_blackout_before", 3))
        after = int(egate.get("earnings_blackout_after", 1))
        n = len(df_ohlcv)
        blackout = np.zeros(n, dtype=bool)
        etype = events["type"].astype(str).str.upper()
        edates = pd.to_datetime(events.loc[etype == "EARNINGS", "date"])
        ohlc_idx = pd.to_datetime(df_ohlcv.index)
        for d in edates:
            p = int(ohlc_idx.searchsorted(d))   # earnings day or next trading day
            lo, hi = max(0, p - before), min(n, p + after + 1)
            blackout[lo:hi] = True
        mask &= ~blackout[idx]

    return det[mask].copy()
