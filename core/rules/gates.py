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
) -> pd.DataFrame:
    """
    Filter detections with simple, robust gates.
    - Uses MA20/MA50/RSI from ind_df when available
    - Uses volume/20d average from df_ohlcv
    - Honors cfg['disable_patterns'] if present
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
    rsi   = ind_df["RSI"].to_numpy()       if "RSI" in ind_df.columns else None
    ma20  = ind_df["MA20"].to_numpy()      if "MA20" in ind_df.columns else None
    ma50  = ind_df["MA50"].to_numpy()      if "MA50" in ind_df.columns else None
    bb_u  = ind_df["BB_upper"].to_numpy()  if "BB_upper" in ind_df.columns else None
    bb_l  = ind_df["BB_lower"].to_numpy()  if "BB_lower" in ind_df.columns else None
    close = df_ohlcv["Close"].to_numpy()   if "close"  in df_ohlcv.columns else None
   # Volume 20d ratio
    vol = df_ohlcv["volume"].to_numpy() if "volume" in df_ohlcv.columns else None
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

    return det[mask].copy()
