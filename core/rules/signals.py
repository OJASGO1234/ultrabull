from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class Signal:
    bar_idx: int
    side: str
    sl_atr: float
    tp_atr: float
    trail_atr: float
    tag: str
    size_cash: Optional[float] = None
    enter_at: str = "open"      # NEW
    delay_bars: int = 1         # NEW
    priority: int = 9           # NEW
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_idx": self.bar_idx,
            "side": self.side,
            "sl_atr": self.sl_atr,
            "tp_atr": self.tp_atr,
            "trail_atr": self.trail_atr,
            "tag": self.tag,
            "size_cash": self.size_cash,
            "enter_at": self.enter_at,         # NEW
            "delay_bars": self.delay_bars,     # NEW
            "priority": self.priority,         # NEW
            "meta": self.meta,
        }

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
    # --- add near the top of compile_signals, before iterating rows ---
    execution_defaults = {
        "enter_at": str(cfg.get("execution", {}).get("enter_at", "open")).lower(),
        "delay_bars": int(cfg.get("execution", {}).get("delay_bars", 1)),
    }

    # Breakouts should be immediate; others at next open by default
    enter_overrides = {
        "Breakout20": {"enter_at": "close", "delay_bars": 0},
        "Breakout55": {"enter_at": "close", "delay_bars": 0},
        # everything else inherits defaults unless you specify here
        "BB_BuyDip": {"enter_at": "open", "delay_bars": 1},
        "BB_SellFade": {"enter_at": "open", "delay_bars": 1},
        "PullbackToMA20": {"enter_at": "open", "delay_bars": 1},
        "Hammer": {"enter_at": "open", "delay_bars": 1},
        "BullishEngulfing": {"enter_at": "open", "delay_bars": 1},
        "BearishEngulfing": {"enter_at": "open", "delay_bars": 1},
        # RSI tags are disabled in your cfg anyway
    }

    # Simple priority ladder (lower = earlier)
    priority = {
        "Breakout55": 0,
        "Breakout20": 1,
        "Hammer": 2,
        "BullishEngulfing": 3,
        "BearishEngulfing": 3,
        "PullbackToMA20": 4,
        "BB_BuyDip": 5,
        "BB_SellFade": 5,
    }


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
    # Sort by priority/score WITHIN each bar before applying the cooldown, so a
    # low-priority signal can't consume the slot ahead of a better one
    tag_src = det["tag"] if "tag" in det.columns else (det[pattern_col] if pattern_col in det.columns else None)
    det = det.copy()
    det["_prio"] = tag_src.map(priority).fillna(9).astype(int) if tag_src is not None else 9
    sort_keys, sort_asc = ["bar_idx", "_prio"], [True, True]
    if "score" in det.columns:
        sort_keys.append("score"); sort_asc.append(False)
    det = det.sort_values(sort_keys, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

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

        ov = enter_overrides.get(tag, {})
        delay_bars = int(ov.get("delay_bars", execution_defaults["delay_bars"]))
        enter_at   = str(ov.get("enter_at",   execution_defaults["enter_at"])).lower()
        prio       = int(priority.get(tag, 9))

        # Pull per-pattern overrides, then fall back to defaults
        p_cfg = (_get(cfg, f"patterns.{tag}", {}) or
                 _get(cfg, f"patterns.{pattern}", {}) or {})

        side = (
            p_cfg.get("side")
            or {"up": "long", "down": "short", "bull": "long", "bear": "short",
                "bullish": "long", "bearish": "short"}.get(str(direction).lower())
            or "long"
        )
        sl_atr = float(p_cfg.get("sl_atr", _get(cfg, "defaults.sl_atr", 1.5)))
        tp_atr = float(p_cfg.get("tp_atr", _get(cfg, "defaults.tp_atr", 3.0)))
        trail_atr = float(p_cfg.get("trail_atr", _get(cfg, "defaults.trail_atr", 0.0)))

        # Pack any extra detection fields into meta (handy for debugging/analysis)
        fields = getattr(row, "_fields", [])
        exclude = {"bar_idx", pattern_col, dir_col, qualified_col, "tag", "_prio"}
        meta = {k: getattr(row, k) for k in fields if k not in exclude and hasattr(row, k)}

        out.append(
            Signal(
                bar_idx=b,
                side=side,
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                trail_atr=trail_atr,
                tag=tag or (pattern or side),
                size_cash=None,
                enter_at=enter_at,          # NEW
                delay_bars=delay_bars,      # NEW
                priority=prio,              # NEW
                meta=meta or None,
            )
        )
        last_emit = b

    df = pd.DataFrame([s.to_dict() for s in out])
    
    # stable; prefer earlier bars, higher-priority tags, higher score, then faster entry
    keys, asc = ["bar_idx", "priority"], [True, True]
    if "score" in detections_enriched.columns:
        sc = (
            detections_enriched
            .assign(tag=detections_enriched.get("tag", detections_enriched.get(pattern_col)))
            .groupby(["bar_idx","tag"])["score"].max()
        ).rename("score").reset_index()
        df = df.merge(sc, on=["bar_idx","tag"], how="left")
        keys += ["score"]; asc += [False]
    keys += ["delay_bars"]; asc += [True]

    df = df.sort_values(keys, ascending=asc, kind="mergesort").reset_index(drop=True)
    return df