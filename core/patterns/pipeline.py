# patterns/pipeline.py
import pandas as pd
from . import candles, breakouts, pullbacks, momentum, cluster
from .primitives import regime_from

def run_all_patterns(df: pd.DataFrame, ind: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = ind.copy()
    ind["_regime"] = regime_from(ind)
    print(ind.head())
    ind["Close"] = df["Close"]  # used for MACD normalization

    frames = []
    frames.append(candles.engulfing(df, ind, ratio=cfg["candles"]["engulfing_ratio"]))
    frames.append(candles.hammer_doji(df, ind,
                    hammer_shadow_ratio=cfg["candles"]["hammer_shadow_ratio"],
                    doji_body_pct=cfg["candles"]["doji_body_pct"]))
    frames.append(breakouts.breakouts(df, ind,
                    lookbacks=tuple(cfg["breakout"]["lookbacks"]),
                    vol_k=cfg["breakout"]["vol_k"]))
    frames.append(pullbacks.pullback_ma20(df, ind, atr_tol_k=cfg["pullback"]["atr_tol_k"]))
    frames.append(pullbacks.bb_reversion(df, ind))
    frames.append(momentum.rsi_regime_cross(ind, *cfg["momentum"]["rsi_levels"]))
    frames.append(momentum.macd_cross_near_zero(ind, cfg["momentum"]["macd_zero_eps_atr_pct"]))

    dets = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    if dets.empty: return dets

    dets = dets.sort_values(["date","pattern"]).reset_index(drop=True)
    dets = cluster.apply_cooldown(dets, bars=cfg["cooldowns"]["bars"])
    dets = cluster.add_strength_and_gate(dets, emit_min=cfg["emit_min"])
    return dets
