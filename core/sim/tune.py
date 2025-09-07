from __future__ import annotations
from typing import Dict, Any, List, Tuple
import copy
import numpy as np
import pandas as pd

from rules.signals import compile_signals
from sim.engine import simulate
from sim.metrics import summarize

def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _preset_overrides(name: str, base_cfg: Dict[str, Any], df_ohlcv: pd.DataFrame, ind_df: pd.DataFrame) -> Dict[str, Any]:
    """Return overrides dict for a given preset name."""
    if name == "baseline":
        return {}

    if name == "time_stop_10":
        return {"execution": {"max_hold_bars": 10}}

    if name == "min_atr_p20":
        if "ATR" in ind_df.columns:
            thr = float(np.nanpercentile(ind_df["ATR"].to_numpy(), 20))
        else:
            thr = 0.0
        return {"filters": {"min_atr": thr}}

    if name == "mr_tighter_tp":
        # Mean-reversion patterns: take profits faster, keep SL modest, keep trail on
        return {"patterns": {
            "BB_BuyDip":   {"sl_atr": 1.2, "tp_atr": 1.6, "trail_atr": 0.8},
            "BB_SellFade": {"sl_atr": 1.2, "tp_atr": 1.6, "trail_atr": 1.0},
            "PullbackToMA20": {"sl_atr": 1.2, "tp_atr": 2.0, "trail_atr": 1.0},
        }}

    if name == "trend_wider_sl":
        # Trend-following patterns: give room, trail profits
        return {"patterns": {
            "Breakout20":  {"sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.5},
            "Breakout55":  {"sl_atr": 2.5, "tp_atr": 5.0, "trail_atr": 2.0},
            "RSI_60_Down": {"sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 1.0},
            "Hammer":      {"sl_atr": 1.8, "tp_atr": 3.6, "trail_atr": 1.0},
        }}

    if name == "cooldown_plus2":
        cd = int(base_cfg.get("execution", {}).get("cooldown_bars", 0)) + 2
        return {"execution": {"cooldown_bars": cd}}

    if name == "enter_open":
        return {"execution": {"enter_at": "open"}}

    if name == "delay0_samebarclose":
        # Only for backtesting exploration; may be optimistic
        return {"execution": {"delay_bars": 0, "enter_at": "close"}}

    return {}

def try_presets(
    detections_enriched: pd.DataFrame,
    df_ohlcv: pd.DataFrame,
    ind_df: pd.DataFrame,
    base_cfg: Dict[str, Any],
    names: List[str] = None,
) -> pd.DataFrame:
    """
    Run a small sweep of presets; return a metrics table sorted by profit factor then return.
    """
    if names is None:
        names = ["baseline", "mr_tighter_tp", "time_stop_10", "min_atr_p20",
                 "trend_wider_sl", "cooldown_plus2", "enter_open"]

    rows = []
    for name in names:
        overrides = _preset_overrides(name, base_cfg, df_ohlcv, ind_df)
        cfg = _deep_update(base_cfg, overrides)

        sig = compile_signals(detections_enriched, df_ohlcv, ind_df, cfg)
        trades, equity, _ = simulate(df_ohlcv, sig, ind_df, cfg)
        m = summarize(trades, equity)
        rows.append({
            "preset": name,
            "trades": m.get("trades", 0),
            "win_rate": round(m.get("win_rate", float("nan")), 3),
            "pf": round(m.get("profit_factor", 0.0), 3),
            "exp": round(m.get("expectancy", 0.0), 2),
            "ret": round(m.get("return_total", 0.0), 3),
            "mdd": round(m.get("max_drawdown", 0.0), 3),
            "sharpe": round(m.get("sharpe_daily", float("nan")), 2),
        })

    out = pd.DataFrame(rows).sort_values(["pf","ret"], ascending=[False, False]).reset_index(drop=True)
    return out
