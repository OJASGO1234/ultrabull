from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Atomic trading signal produced from a pattern detection."""
    bar_idx: int              # integer bar index aligned to df_ohlcv
    side: str                 # 'long' | 'short'
    sl_atr: float             # stop distance in ATRs
    tp_atr: float             # take-profit distance in ATRs
    trail_atr: float = 0.0    # trailing distance in ATRs (0 = off)
    tag: str = ""             # e.g., 'Breakout20'
    size_cash: Optional[float] = None  # (engine computes shares from ATR risk)
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["meta"] = d.get("meta", {})  # flatten for DataFrame
        return d


def _get(cfg: Dict[str, Any], path: str, default=None):
    """Small helper to read dotted keys from dicts."""
    cur = cfg
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _infer_bar_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    """Pick an existing integer bar-index column if present."""
    candidates = [preferred, "bar_idx", "bar", "i", "idx", "index"]
    for c in candidates:
        if c and c in df.columns and pd.api.types.is_integer_dtype(df[c]):
            return c
    return None


def _ensure_bar_index(
    detections_enriched: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    bar_idx_col: Optional[str]
) -> pd.DataFrame:
    """
    Ensure detections_enriched has an integer 'bar_idx' column aligned to df_ohlcv.
    Accepts either an existing integer column or any datetime (index/column).
    """
    det = detections_enriched

    # 1) If an integer bar column exists, standardize name and return
    bar_col = _infer_bar_col(det, bar_idx_col)
    if bar_col is not None:
        if bar_col != "bar_idx":
            det = det.copy()
            det["bar_idx"] = det[bar_col].astype("int64")
        return det

    # 2) Otherwise, map datetime → positional index
    dt_series = None
    if isinstance(det.index, pd.DatetimeIndex):
        dt_series = pd.to_datetime(det.index)
    else:
        for c in ["date", "Date", "dt", "timestamp", "Timestamp"]:
            if c in det.columns:
                dt_series = pd.to_datetime(det[c])
                break
    if dt_series is None:
        raise ValueError(
            "compile_signals: Need an integer bar index column "
            "or a datetime index/column to map onto df_ohlcv."
        )

    # Build lookup from df_ohlcv index (assumed datetime-like) → integer position
    ohlc_dt = pd.to_datetime(df_ohlcv.index)
    lookup = pd.Series(np.arange(len(df_ohlcv), dtype="int64"), index=ohlc_dt)

    mapped = lookup.reindex(pd.to_datetime(dt_series))
    if mapped.isna().any():
        # Drop rows we cannot map (e.g., mismatched sessions)
        keep_mask = ~mapped.isna()
        det = det.loc[keep_mask].copy()
        mapped = mapped.loc[keep_mask]

    det["bar_idx"] = mapped.astype("int64").to_numpy()
    return det


def compile_signals(
    detections_enriched: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    pattern_col: str = "pattern",
    dir_col: str = "direction",
    qualified_col: str = "qualified",
    bar_idx_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert Phase-1/2 detections into executable signals.

    Inputs:
      - detections_enriched: must include 'qualified' (bool) and either an integer bar index
        (bar/bar_idx/i/…) or a datetime index/column (date/Date/dt/timestamp).
      - df_ohlcv: lowercase ['open','high','low','close','volume'] indexed by datetime (or same length).
      - ind_df: must be row-aligned to df_ohlcv; 'ATR' used if present.

    Config keys used:
      filters.warmup_cutoff_bars, filters.min_atr,
      execution.cooldown_bars,
      patterns.<tag>.{side,sl_atr,tp_atr,trail_atr},
      defaults.{sl_atr,tp_atr,trail_atr}

    Returns:
      DataFrame with columns:
        ['bar_idx','side','sl_atr','tp_atr','trail_atr','tag','size_cash','meta']
    """
    if detections_enriched is None or len(detections_enriched) == 0:
        return pd.DataFrame(
            columns=["bar_idx", "side", "sl_atr", "tp_atr", "trail_atr", "tag", "size_cash", "meta"]
        )

    det = _ensure_bar_index(detections_enriched, df_ohlcv, bar_idx_col)

    # Basic filters: qualified + warmup + optional ATR floor
    if qualified_col in det.columns:
        det = det[det[qualified_col].astype(bool)]
    warmup_cutoff = int(_get(cfg, "filters.warmup_cutoff_bars", 0) or 0)
    det = det[det["bar_idx"] >= warmup_cutoff]

    min_atr = float(_get(cfg, "filters.min_atr", 0.0) or 0.0)
    if min_atr > 0.0 and "ATR" in ind_df.columns:
        atr_np = ind_df["ATR"].to_numpy()
        # Clip indices to safe range
        b = det["bar_idx"].astype(int).clip(0, len(atr_np) - 1).to_numpy()
        det = det[atr_np[b] >= min_atr]

    # Local cooldown (engine also enforces; this keeps signals clean)
    cooldown = int(_get(cfg, "execution.cooldown_bars", 0) or 0)
    det = det.sort_values("bar_idx").reset_index(drop=True)

    out: List[Signal] = []
    last_emit = -10_000

    for row in det.itertuples(index=False):
        b = int(getattr(row, "bar_idx"))
        if b - last_emit < cooldown:
            continue

        # Identify the pattern/tag and direction
        pattern = getattr(row, pattern_col, None) if pattern_col in det.columns else None
        direction = getattr(row, dir_col, None) if dir_col in det.columns else None
        tag = getattr(row, "tag", None) if "tag" in det.columns else None
        tag = tag or (pattern or "")

        # Pull per-pattern overrides, then fall back to defaults
        p_cfg = (_get(cfg, f"patterns.{tag}", {}) or
                 _get(cfg, f"patterns.{pattern}", {}) or {})

        side = (
            p_cfg.get("side")
            or {"up": "long", "down": "short", "bull": "long", "bear": "short"}.get(str(direction).lower())
            or "long"
        )
        sl_atr = float(p_cfg.get("sl_atr", _get(cfg, "defaults.sl_atr", 1.5)))
        tp_atr = float(p_cfg.get("tp_atr", _get(cfg, "defaults.tp_atr", 3.0)))
        trail_atr = float(p_cfg.get("trail_atr", _get(cfg, "defaults.trail_atr", 0.0)))

        # Pack any extra detection fields into meta (handy for debugging/analysis)
        fields = getattr(row, "_fields", [])
        exclude = {"bar_idx", pattern_col, dir_col, qualified_col, "tag"}
        meta = {k: getattr(row, k) for k in fields if k not in exclude and hasattr(row, k)}

        out.append(
            Signal(
                bar_idx=b,
                side=side,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                trail_atr=trail_atr,
                tag=tag or (pattern or side),
                meta=meta or None,
            )
        )
        last_emit = b

    return pd.DataFrame([s.to_dict() for s in out])
